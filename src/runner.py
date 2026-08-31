"""Phase D — Single-run pipeline driver and CLI.

Orchestrates the end-to-end measurement pipeline:

1. Open / init the SQLite cache.
2. Read the candidate universe (one ticker per line, stripped, uppercased).
3. For each enabled tier: select tickers via :func:`src.selection.select_tickers_for_tier`
   (deterministic when ``seed`` is supplied; per-tier seeds are derived from
   the global seed so different tiers don't sample the same prefix).
4. For each (tier, ticker): run the backtest via
   :func:`src.evaluate.run_one_ticker_eval` and persist per-(ticker, model)
   prediction CSVs into the per-tier ``predictions/`` directory, plus per-pair
   figures into ``individual/``.
5. Aggregate metric rows into ``metrics.csv``, write ``ticker_tested.csv``
   and ``manifest.json``, then render the cross-tier summaries.

Each model carries its own lookback, so there is no shared rolling window ``W``
in this pipeline; ``_run_one_ticker_into_tier`` writes ``<TICKER>_<MODEL>.csv``
with no window component in the name.

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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config as cfg  # noqa: E402
from src import evaluate as _evaluate  # noqa: E402  (library reuse)
from src import models as _models  # noqa: E402
from src.models import ForecasterProtocol  # noqa: E402
from src.plots import _PER_PAIR_FIGURE_REGISTRY, figure_dirs  # noqa: E402
from src.selection import TierSpec, select_tickers_for_tier  # noqa: E402
from src.logging_setup import get_logger  # noqa: E402
from src.storage.db import init_schema, open_db  # noqa: E402

# Hard refusal threshold — if the projected ARMA cost across all selected
# (ticker, window) pairs exceeds this many seconds and ``--force`` was not
# passed, the runner aborts before any rolling work begins.
_BUDGET_SECONDS: float = 3600.0  # 1 hour


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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


def _resolve_active_tiers(
    tier_specs: Dict[str, TierSpec],
    tiers_subset: Optional[List[str]],
    target_override: Optional[int],
) -> Tuple[List[str], List[str], Dict[str, TierSpec]]:
    """Resolve which tiers run, and their specs after ``--target`` is applied.

    Returns ``(all_tier_names, active_tier_names, effective_specs)``.
    ``all_tier_names`` is returned because the per-tier seed derives from a
    tier's index in the *full* set, so that adding ``--tiers`` to a run does
    not change which tickers an included tier selects.
    """
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
    return all_tier_names, active_tier_names, effective_specs


def _select_per_tier(
    all_tier_names: List[str],
    active_tier_names: List[str],
    effective_specs: Dict[str, TierSpec],
    universe: List[str],
    conn,
    logger: logging.Logger,
    *,
    seed: Optional[int],
) -> Dict[str, List[str]]:
    """Run ticker selection for each active tier.

    Per-tier seeds are offset by the tier's index in the full tier set so two
    tiers drawing from the same universe do not sample the same prefix.
    """
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
    return per_tier_selected


def _make_run_tree(out_root: Path, active_tier_names: List[str]) -> Tuple[Path, Dict[str, Path]]:
    """Create ``<out_root>/test_<ISO>/`` and its per-tier subdirs.

    The test root is created with ``exist_ok=False`` so two runs starting in
    the same second cannot silently interleave their output.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    test_root = out_root / f"test_{_iso_timestamp()}"
    test_root.mkdir(parents=True, exist_ok=False)
    (test_root / "analysis").mkdir(parents=True, exist_ok=True)
    tier_dirs = {
        tname: _make_tier_dir(test_root, tname) for tname in active_tier_names
    }
    return test_root, tier_dirs


def _enforce_arma_budget(
    active_tier_names: List[str],
    per_tier_selected: Dict[str, List[str]],
    logger: logging.Logger,
    *,
    force: bool,
) -> None:
    """Refuse to start if the projected ARMA cost exceeds the budget.

    ARMA is fitted at every test step for every ARMA lookback, so the cost
    scales with tickers x lookbacks and is worth estimating before any rolling
    work begins rather than discovering an hour in.
    """
    total_arma_est = 0.0
    n_steps_test = 252  # ~one year of trading days
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


def _backtest_tier(
    tname: str,
    tier_dir: Path,
    selected: List[str],
    test_root: Path,
    conn,
    logger: logging.Logger,
    *,
    models_factory: Callable[[], List[ForecasterProtocol]],
) -> List[Dict[str, Any]]:
    """Backtest every ticker in one tier and write that tier's summary.

    A failure on one ticker is logged and skipped rather than aborting the
    tier, and a failure in the tier summary is logged rather than aborting the
    run: derivative artifacts must not cost us the measurements already made.
    """
    logger.info("=" * 60)
    logger.info(
        "=== %s START === %d stocks being tested", tname.upper(), len(selected),
    )
    logger.info("=" * 60)
    try:
        list_path = tier_dir / f"{tname}_tickers.txt"
        list_path.write_text(
            "\n".join(selected) + ("\n" if selected else ""), encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not write %s_tickers.txt: %s", tname, exc)

    rows_out: List[Dict[str, Any]] = []
    for ticker in selected:
        try:
            rows_out.extend(_run_one_ticker_into_tier(
                tname, ticker, tier_dir, logger,
                conn=conn, models_factory=models_factory,
            ))
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s/%s] FAILED: %s", tname, ticker, exc)
            continue
    logger.info("=== %s DONE ===", tname.upper())

    from src.summary import summarise_tier
    try:
        summarise_tier(tname, test_root)
    except Exception as exc:  # noqa: BLE001
        logger.warning("summary for %s failed: %s", tname, exc, exc_info=True)
    return rows_out


def _write_run_tables(
    test_root: Path,
    all_rows: List[Dict[str, Any]],
    ticker_rows: List[Dict[str, str]],
) -> None:
    """Write ``metrics.csv`` and ``ticker_tested.csv``.

    Both are written even when empty so a failed run still produces a tree with
    the expected shape rather than a missing file.
    """
    metrics_cols = ["tier", "ticker", "model", "rmse", "mae", "n"]
    metrics = (
        pd.DataFrame(all_rows)[metrics_cols] if all_rows
        else pd.DataFrame(columns=metrics_cols)
    )
    metrics.to_csv(test_root / "metrics.csv", index=False)

    tt = (
        pd.DataFrame(ticker_rows).sort_values(["tier", "ticker"]).reset_index(drop=True)
        if ticker_rows else pd.DataFrame(columns=["tier", "ticker"])
    )
    tt.to_csv(test_root / "ticker_tested.csv", index=False)


def _write_manifest(
    test_root: Path,
    active_tier_names: List[str],
    per_tier_selected: Dict[str, List[str]],
    universe_file: Path,
    *,
    seed: Optional[int],
    refresh_cache: bool,
    runtime: float,
) -> None:
    """Write ``manifest.json`` — the record of what this run actually did."""
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


def _render_cross_tier_outputs(
    test_root: Path,
    active_tier_names: List[str],
    logger: logging.Logger,
) -> None:
    """Cross-tier summary, score histogram and per-stock dotplots.

    Every stage here is derivative of ``metrics.csv`` and the prediction CSVs,
    which are already on disk by this point, so each failure is logged and the
    run still succeeds.
    """
    from src.summary import summarise_overall
    try:
        summarise_overall(test_root, list(active_tier_names))
    except Exception as exc:  # noqa: BLE001
        logger.warning("overall summary failed: %s", exc, exc_info=True)

    try:
        from src.analysis.runner import analyse_test_run
        analyse_test_run(test_root)
    except Exception as exc:  # noqa: BLE001
        logger.warning("analysis stage failed: %s", exc, exc_info=True)


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
    models_factory: Optional[Callable[[], List[ForecasterProtocol]]] = None,
    force: bool = False,
) -> Path:
    """Run the full single-pass pipeline and return the test-run root path.

    See module docstring for the high-level behaviour.  All paths are written
    under ``out_root / f"test_<ISO_TIMESTAMP>"``.
    """
    logger = get_logger("runner")
    t_start = time.time()

    conn = open_db(str(Path(db_path)))
    init_schema(conn)

    universe_file = Path(universe_file)
    universe = _read_universe(universe_file)
    logger.info("Loaded universe: %d tickers from %s", len(universe), universe_file)

    all_tier_names, active_tier_names, effective_specs = _resolve_active_tiers(
        tier_specs, tiers_subset, target_override,
    )
    if models_factory is None:
        models_factory = _models.default_models

    per_tier_selected = _select_per_tier(
        all_tier_names, active_tier_names, effective_specs,
        universe, conn, logger, seed=seed,
    )
    test_root, tier_dirs = _make_run_tree(Path(out_root), active_tier_names)
    _enforce_arma_budget(
        active_tier_names, per_tier_selected, logger, force=force,
    )

    all_rows: List[Dict[str, Any]] = []
    ticker_rows: List[Dict[str, str]] = []
    for tname in active_tier_names:
        selected = per_tier_selected[tname]
        ticker_rows.extend({"tier": tname, "ticker": t} for t in selected)
        all_rows.extend(_backtest_tier(
            tname, tier_dirs[tname], selected, test_root, conn, logger,
            models_factory=models_factory,
        ))

    _write_run_tables(test_root, all_rows, ticker_rows)
    runtime = time.time() - t_start
    _write_manifest(
        test_root, active_tier_names, per_tier_selected, universe_file,
        seed=seed, refresh_cache=refresh_cache, runtime=runtime,
    )
    conn.close()

    _render_cross_tier_outputs(test_root, active_tier_names, logger)

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
