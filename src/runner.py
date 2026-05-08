"""Phase D — Single-run pipeline driver and CLI.

Orchestrates the end-to-end measurement pipeline:

1. Open / init the SQLite cache.
2. Read the candidate universe (one ticker per line, stripped, uppercased).
3. For each enabled tier: select tickers via :func:`src.selection.select_tickers_for_tier`
   (deterministic when ``seed`` is supplied; per-tier seeds are derived from
   the global seed so different tiers don't sample the same prefix).
4. For each (tier, ticker, window): run the rolling backtest and persist
   per-(ticker, window, model) prediction CSVs into the per-tier
   ``predictions/`` directory plus a per-(ticker, window) figure into
   ``individual/``.
5. Aggregate metric rows into ``metrics.csv``, write ``ticker_tested.csv``
   and ``manifest.json``.

Approach for the per-ticker driver: option (a) — a small in-runner helper
``_run_one_ticker_into_tier`` mirrors the body of
:func:`src.evaluate._run_one_ticker` but writes prediction CSVs directly into
the per-tier ``predictions/`` directory rather than the legacy hard-coded
``results/predictions/`` path.  This keeps the legacy module untouched for any
remaining callers while giving the runner full control over its output tree.

CLI: ``python -m src.runner [--target N] [--seed S] [--refresh-cache]
[--tiers tier1,tier2] [--force]``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless-safe; must precede pyplot

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config as cfg  # noqa: E402
from src import evaluate as _evaluate  # noqa: E402  (library reuse)
from src import models as _models  # noqa: E402
from src.metrics import mae, rmse  # noqa: E402
from src.models import ForecasterProtocol  # noqa: E402
from src.plots import (  # noqa: E402
    _PER_PAIR_FIGURE_REGISTRY,
    color_for,
    figure_dirs,
    ordered_models,
)
from src.selection import TierSpec, select_tickers_for_tier  # noqa: E402
from src.storage.db import init_schema, open_db  # noqa: E402

# ---------------------------------------------------------------------------
# Defaults — these mirror the constants Phase H will add to ``src.config``.
# Until then we read via ``getattr`` so the runner works either way.
# ---------------------------------------------------------------------------

_DEFAULT_TIERS: Dict[str, TierSpec] = {
    "tier1": TierSpec(
        start_price=10.0,
        end_price=20.0,
        below_threshold=5.0,
        upper_threshold=40.0,
        target_count=100,
    ),
    "tier2": TierSpec(
        start_price=20.0,
        end_price=50.0,
        below_threshold=10.0,
        upper_threshold=100.0,
        target_count=100,
    ),
    "tier3": TierSpec(
        start_price=50.0,
        end_price=1e9,
        below_threshold=0.0,
        upper_threshold=1e9,
        target_count=100,
    ),
}

_DEFAULT_UNIVERSE_FILE = "data/universe/russell3000.txt"
_DEFAULT_DB_PATH = "ticker_data/cache.db"
_DEFAULT_TEST_RUN_ROOT = "results/test_runs"

# Hard refusal threshold — if the projected ARMA cost across all selected
# (ticker, window) pairs exceeds this many seconds and ``--force`` was not
# passed, the runner aborts before any rolling work begins.
_BUDGET_SECONDS: float = 3600.0  # 1 hour

ENSEMBLE_NAME = "ensemble"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _setup_logger(level: int = logging.INFO) -> logging.Logger:
    """Idempotent stdout logger named ``runner``."""
    logger = logging.getLogger("runner")
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
        )
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def _iso_timestamp() -> str:
    """UTC ISO timestamp safe for use in directory names (no colons)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _read_universe(path: Path) -> List[str]:
    """Read a one-symbol-per-line file. Strip, drop blanks, uppercase, dedup."""
    if not path.exists():
        raise FileNotFoundError(f"universe file not found: {path}")
    seen: set = set()
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            sym = line.strip().upper()
            if not sym:
                continue
            if sym in seen:
                continue
            seen.add(sym)
            out.append(sym)
    return out


def _make_tier_dir(test_root: Path, tier_name: str) -> Path:
    """Create the per-tier directory tree and return the tier root."""
    tier_dir = test_root / tier_name
    for sub in ("individual", "grouped", "predictions", "analysis"):
        (tier_dir / sub).mkdir(parents=True, exist_ok=True)
    return tier_dir


def _save_predictions_csv(
    path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    window: int,
) -> None:
    """Write a (idx, y_true, y_pred) prediction CSV — same schema as legacy."""
    n = int(y_true.size)
    df = pd.DataFrame(
        {
            "idx": np.arange(window, window + n, dtype=int),
            "y_true": np.asarray(y_true, dtype=float),
            "y_pred": np.asarray(y_pred, dtype=float),
        }
    )
    df.to_csv(path, index=False)


def _ensemble_predictions(
    per_model: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Per-step mean across per-model predictions (NaN-safe)."""
    keys = sorted(per_model.keys())
    stacked = np.vstack([per_model[k][1] for k in keys])
    return np.nanmean(stacked, axis=0)


def _individual_figure(
    tier_dir: Path,
    ticker: str,
    window: int,
    per_model: Dict[str, Tuple[np.ndarray, np.ndarray]],
    ensemble_pred: Optional[np.ndarray] = None,
    *,
    tail: int = 200,
) -> None:
    """Write ``<tier_dir>/individual/<TICKER>_w<WIN>.{png,csv}``.

    Single PNG showing the actual log returns vs each model's predictions
    (last ``min(n_steps, tail)`` steps) plus a sidecar CSV containing the
    same data.  Self-contained so we don't depend on ``src.plots``' current
    output paths.
    """
    if not per_model:
        return
    # Build the wide frame: idx, y_true, then one column per model + ensemble.
    keys = ordered_models({k: None for k in per_model})  # type: ignore[arg-type]
    base_key = keys[0]
    y_true_full = per_model[base_key][0]
    n_steps = int(y_true_full.size)
    take = min(int(tail), n_steps)
    start = n_steps - take
    idx = np.arange(window + start, window + n_steps, dtype=int)

    data: Dict[str, np.ndarray] = {"idx": idx, "y_true": y_true_full[start:]}
    for name in keys:
        _, yp = per_model[name]
        data[f"y_pred_{name}"] = yp[start:]
    if ensemble_pred is not None:
        data[f"y_pred_{ENSEMBLE_NAME}"] = ensemble_pred[start:]

    df = pd.DataFrame(data)
    csv_path = tier_dir / "individual" / f"{ticker}_w{window}.csv"
    df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["idx"], df["y_true"], color="black", linestyle="--",
            linewidth=1.8, label="actual")
    for name in keys:
        col = f"y_pred_{name}"
        ax.plot(df["idx"], df[col], color=color_for(name), linewidth=1.0,
                alpha=0.85, label=name)
    if ensemble_pred is not None:
        ax.plot(
            df["idx"],
            df[f"y_pred_{ENSEMBLE_NAME}"],
            color=color_for(ENSEMBLE_NAME),
            linewidth=1.2,
            linestyle=":",
            label=ENSEMBLE_NAME,
        )
    ax.set_title(f"{ticker} — window={window} (last {take} steps)")
    ax.set_xlabel("Step index")
    ax.set_ylabel("Log return")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=False, fontsize=8)
    fig.tight_layout()
    png_path = tier_dir / "individual" / f"{ticker}_w{window}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _run_one_ticker_into_tier(
    tier_name: str,
    ticker: str,
    tier_dir: Path,
    logger: logging.Logger,
    *,
    conn,
    models_factory: Callable[[], List[ForecasterProtocol]],
) -> List[Dict[str, Any]]:
    """Run the new-API per-model-lookback evaluation for one ticker.

    Persists per-model prediction CSVs under ``tier_dir/predictions/`` with
    schema ``<TICKER>_<MODEL>.csv`` (no window component) and writes per-pair
    figures into ``tier_dir/individual/``. Returns one metric row per model
    (including the post-hoc ensemble).
    """
    from src.data import load_returns  # local import — avoids cycles

    df = load_returns(ticker, conn=conn)
    if df.empty or "log_return" not in df.columns:
        logger.warning("[%s/%s] empty or malformed return series — skip",
                       tier_name, ticker)
        return []

    active_models = models_factory()
    predictions_dir = tier_dir / "predictions"
    os.makedirs(predictions_dir, exist_ok=True)

    t0 = time.time()
    rows, per_model = _evaluate.run_one_ticker_eval(
        tier_name, ticker,
        df=df,
        test_start=cfg.TEST_START_DATE,
        test_end=cfg.TEST_END_DATE,
        models=active_models,
        predictions_dir=str(predictions_dir),
    )
    elapsed = time.time() - t0
    logger.info("[%s/%s] %d models, %.2fs", tier_name, ticker, len(per_model), elapsed)

    # Per-pair figures via the shared registry. The "window" passed through
    # is now nominal — the registry uses it only in filenames; we pass 0 so
    # the schema becomes ``<kind>_<TICKER>_0.png`` (then we strip the suffix
    # in a downstream pass if desired).
    if per_model:
        _render_per_pair_figures(tier_dir, ticker, per_model)

    return rows


def _build_model_dict(
    per_model: Dict[str, Tuple[np.ndarray, np.ndarray]],
    window: int,
    *,
    ensemble_pred: Optional[np.ndarray] = None,
    y_true_ref: Optional[np.ndarray] = None,
) -> Dict[str, pd.DataFrame]:
    """Convert the runner's per-model arrays into the registry's input shape.

    The per-pair figure registry expects ``{model_name: DataFrame}`` where
    each DataFrame carries ``(idx, y_true, y_pred)`` — exactly what the
    on-disk prediction CSVs use. We build the same shape in memory to avoid
    round-tripping through disk.
    """
    out: Dict[str, pd.DataFrame] = {}
    for name, (yt, yp) in per_model.items():
        n = int(yt.size)
        idx = np.arange(window, window + n, dtype=int)
        out[name] = pd.DataFrame({
            "idx": idx,
            "y_true": np.asarray(yt, dtype=float),
            "y_pred": np.asarray(yp, dtype=float),
        })
    if ensemble_pred is not None and y_true_ref is not None:
        n = int(y_true_ref.size)
        idx = np.arange(window, window + n, dtype=int)
        out[ENSEMBLE_NAME] = pd.DataFrame({
            "idx": idx,
            "y_true": np.asarray(y_true_ref, dtype=float),
            "y_pred": np.asarray(ensemble_pred, dtype=float),
        })
    return out


def _render_per_pair_figures(
    tier_dir: Path,
    ticker: str,
    per_model: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> None:
    """Drive every renderer in ``_PER_PAIR_FIGURE_REGISTRY`` for one ticker.

    The new pipeline has no shared window axis, so the registry's ``window``
    argument is passed as 0 (the renderer uses it only in the filename).
    """
    individual_dir = tier_dir / "individual"
    individual_data_dir = individual_dir / "data"
    os.makedirs(individual_dir, exist_ok=True)
    os.makedirs(individual_data_dir, exist_ok=True)

    model_dict: Dict[str, pd.DataFrame] = {}
    for name, (yt, yp) in per_model.items():
        n = int(yt.size)
        idx = np.arange(n, dtype=int)
        model_dict[name] = pd.DataFrame({
            "idx": idx,
            "y_true": np.asarray(yt, dtype=float),
            "y_pred": np.asarray(yp, dtype=float),
        })
    if not model_dict:
        return

    with figure_dirs(str(individual_dir), str(individual_data_dir)):
        for renderer in _PER_PAIR_FIGURE_REGISTRY.values():
            try:
                renderer(model_dict, ticker, 0)
            except Exception:  # noqa: BLE001
                continue


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_test(
    tier_specs: Dict[str, TierSpec],
    universe_file: Path,
    db_path: Path,
    out_root: Path,
    *,
    seed: Optional[int] = None,
    refresh_cache: bool = False,
    tiers_subset: Optional[List[str]] = None,
    target_override: Optional[int] = None,
    windows: Optional[Tuple[int, ...]] = None,
    models_factory: Optional[Callable[[], List[ForecasterProtocol]]] = None,
    force: bool = False,
) -> Path:
    """Run the full single-pass pipeline and return the test-run root path.

    See module docstring for the high-level behaviour.  All paths are written
    under ``out_root / f"test_<ISO_TIMESTAMP>"``.
    """
    logger = _setup_logger()
    t_start = time.time()

    # 1) Open / init the cache.
    db_path = Path(db_path)
    conn = open_db(str(db_path))
    init_schema(conn)

    # 2) Universe.
    universe_file = Path(universe_file)
    universe = _read_universe(universe_file)
    logger.info("Loaded universe: %d tickers from %s", len(universe), universe_file)

    # 3) Resolve tier subset and per-tier specs.
    all_tier_names = sorted(tier_specs.keys())
    if tiers_subset is None:
        active_tier_names = list(all_tier_names)
    else:
        unknown = [t for t in tiers_subset if t not in tier_specs]
        if unknown:
            raise ValueError(
                f"Unknown tier names {unknown}; known tiers: {all_tier_names}"
            )
        active_tier_names = sorted(tiers_subset)

    effective_specs: Dict[str, TierSpec] = {}
    for tname in active_tier_names:
        spec = tier_specs[tname]
        if target_override is not None:
            spec = replace(spec, target_count=int(target_override))
        effective_specs[tname] = spec

    # ``windows`` is retained as a kwarg for backwards compatibility but is
    # ignored under the new per-model-lookback API. The lookbacks are now
    # baked into each Forecaster instance returned by ``models_factory``.
    _ = windows

    if models_factory is None:
        models_factory = _models.default_models

    # 4) Selection.
    per_tier_selected: Dict[str, List[str]] = {}
    for tname in active_tier_names:
        tier_index = all_tier_names.index(tname)
        tier_seed = None if seed is None else int(seed) + tier_index
        spec = effective_specs[tname]
        selected = select_tickers_for_tier(
            tname, spec, universe, conn, seed=tier_seed,
        )
        per_tier_selected[tname] = selected
        logger.info(
            "Selected %d tickers for %s (target=%d)",
            len(selected), tname, spec.target_count,
        )

    # 5) Build the test root + per-tier subdirs.
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    iso_ts = _iso_timestamp()
    test_root = out_root / f"test_{iso_ts}"
    test_root.mkdir(parents=True, exist_ok=False)
    (test_root / "analysis").mkdir(parents=True, exist_ok=True)
    tier_dirs: Dict[str, Path] = {
        tname: _make_tier_dir(test_root, tname) for tname in active_tier_names
    }

    # 6) Cost gating — refuse to start if projected ARMA cost > 1h budget.
    # Test set is ~252 trading days; ARMA runs at each test step for both
    # ``arma60`` and ``arma90``.
    total_arma_est = 0.0
    n_steps_test = 252
    arma_lookbacks = tuple(getattr(cfg, "ARMA_LOOKBACKS", (60, 90)))
    for tname in active_tier_names:
        n_tickers = len(per_tier_selected[tname])
        if n_tickers == 0:
            continue
        for L in arma_lookbacks:
            total_arma_est += n_tickers * _evaluate._estimate_arma_cost(
                n_steps_test, int(L),
            )
    if total_arma_est > _BUDGET_SECONDS and not force:
        raise RuntimeError(
            "Estimated runtime %.1fh exceeds 1h budget — pass --force to proceed."
            % (total_arma_est / 3600.0)
        )
    logger.info(
        "Estimated total ARMA cost: %.1fs (%.2fh)",
        total_arma_est, total_arma_est / 3600.0,
    )

    # 7) Per-(tier, ticker) backtest + per-tier summaries.
    from src.summary import summarise_overall, summarise_tier

    all_rows: List[Dict[str, Any]] = []
    ticker_rows: List[Dict[str, str]] = []
    for tname in active_tier_names:
        tier_dir = tier_dirs[tname]
        selected = per_tier_selected[tname]
        # Per-tier banner + ticker-list text dump for inclusion in the report.
        logger.info("=" * 60)
        logger.info(
            "=== %s START === %d stocks being tested",
            tname.upper(), len(selected),
        )
        logger.info("=" * 60)
        try:
            list_path = tier_dir / f"{tname}_tickers.txt"
            list_path.write_text(
                "\n".join(selected) + ("\n" if selected else ""),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not write %s_tickers.txt: %s", tname, exc)

        for ticker in selected:
            ticker_rows.append({"tier": tname, "ticker": ticker})
            try:
                rows = _run_one_ticker_into_tier(
                    tname, ticker, tier_dir, logger,
                    conn=conn, models_factory=models_factory,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("[%s/%s] FAILED: %s", tname, ticker, exc)
                continue
            all_rows.extend(rows)
        logger.info("=== %s DONE ===", tname.upper())

        # Tier-scoped summary outputs. Failures are non-fatal: derivative
        # artifacts must not abort the rest of the run.
        try:
            summarise_tier(tname, test_root)
        except Exception as exc:  # noqa: BLE001
            logger.warning("summary for %s failed: %s", tname, exc, exc_info=True)

    # 8) Aggregate metrics.csv.
    if all_rows:
        metrics = pd.DataFrame(all_rows)[
            ["tier", "ticker", "model", "rmse", "mae", "n"]
        ]
    else:
        metrics = pd.DataFrame(
            columns=["tier", "ticker", "model", "rmse", "mae", "n"]
        )
    metrics.to_csv(test_root / "metrics.csv", index=False)

    # 9) ticker_tested.csv (sorted by tier then ticker).
    if ticker_rows:
        tt = pd.DataFrame(ticker_rows).sort_values(["tier", "ticker"]).reset_index(drop=True)
    else:
        tt = pd.DataFrame(columns=["tier", "ticker"])
    tt.to_csv(test_root / "ticker_tested.csv", index=False)

    # 10) manifest.json.
    runtime = time.time() - t_start
    manifest = {
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": seed,
        "tiers": list(active_tier_names),
        "target_per_tier_effective": {
            tname: len(per_tier_selected[tname]) for tname in active_tier_names
        },
        "universe_file": str(universe_file),
        "n_tickers_total": int(sum(len(v) for v in per_tier_selected.values())),
        "test_start_date": cfg.TEST_START_DATE,
        "test_end_date": cfg.TEST_END_DATE,
        "ma_lookbacks": list(getattr(cfg, "MA_LOOKBACKS", ())),
        "arma_lookbacks": list(getattr(cfg, "ARMA_LOOKBACKS", ())),
        "runtime_seconds": float(runtime),
        "python_version": sys.version.split()[0],
        "refresh_cache": bool(refresh_cache),
    }
    with open(test_root / "manifest.json", "w", encoding="utf-8") as fh:
        fh.write(json.dumps(manifest, indent=2, sort_keys=True))

    conn.close()

    # Cross-tier summary + score histogram.
    try:
        summarise_overall(test_root, list(active_tier_names))
    except Exception as exc:  # noqa: BLE001
        logger.warning("overall summary failed: %s", exc, exc_info=True)

    # Optional: per-stock dotplot analysis (kept from earlier phase).
    try:
        from src.analysis.runner import analyse_test_run
        analyse_test_run(test_root)
    except Exception as exc:  # noqa: BLE001
        logger.warning("analysis stage failed: %s", exc, exc_info=True)

    logger.info("Wrote test run to %s (%.1fs)", test_root, runtime)
    return test_root


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _resolve_default_tiers() -> Dict[str, TierSpec]:
    """Pull tier specs from ``src.config.TIERS``."""
    return cfg.TIERS


def _resolve_default_paths() -> Tuple[Path, Path, Path]:
    """Project-root-anchored defaults for ``universe_file``, ``db_path``, ``out_root``."""
    project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    universe_file = Path(cfg.CANDIDATE_UNIVERSE_FILE)
    if not universe_file.is_absolute():
        universe_file = project_root / universe_file
    db_path = Path(cfg.TICKER_DB_PATH)
    if not db_path.is_absolute():
        db_path = project_root / db_path
    out_root = Path(cfg.TEST_RUN_ROOT)
    if not out_root.is_absolute():
        out_root = project_root / out_root
    return universe_file, db_path, out_root


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Single-run measurement pipeline")
    parser.add_argument("--target", type=int, default=None,
                        help="Override per-tier target_count.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Reproducibility seed for ticker selection.")
    parser.add_argument("--refresh-cache", action="store_true",
                        help="Bypass DB cache for this run.")
    parser.add_argument("--tiers", type=str, default=None,
                        help="Comma-separated tier names (default: all tiers).")
    parser.add_argument("--force", action="store_true",
                        help="Proceed even if estimated runtime exceeds 1h.")
    args = parser.parse_args(argv)

    tier_specs = _resolve_default_tiers()
    universe_file, db_path, out_root = _resolve_default_paths()

    tiers_subset: Optional[List[str]] = None
    if args.tiers:
        tiers_subset = [t.strip() for t in args.tiers.split(",") if t.strip()]

    test_root = run_test(
        tier_specs=tier_specs,
        universe_file=universe_file,
        db_path=db_path,
        out_root=out_root,
        seed=args.seed,
        refresh_cache=bool(args.refresh_cache),
        tiers_subset=tiers_subset,
        target_override=args.target,
        force=bool(args.force),
    )
    print(str(test_root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
