"""The contract between what the pipeline writes and what its readers expect.

Every test here failed before the ``window`` removal. The pipeline retired the
unified rolling window ``W`` in favour of per-model lookbacks: ``evaluate.py``
stopped putting a window component in prediction filenames and stopped writing a
``window`` column into ``metrics.csv``. Three readers were never updated, so the
analysis stage and the plots CLI both silently processed nothing.

These assert the contract from the writer's side — the filenames and columns the
pipeline actually produces — so a reader drifting from it fails here rather than
degrading into a no-op.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.runner import analyse_test_run
from src.evaluate import run_one_ticker_eval
from src.models import MODEL_ORDER, NaiveModel
from src.summary import _scan_predictions


# ---------------------------------------------------------------------------
# Fixtures shaped like real pipeline output
# ---------------------------------------------------------------------------


def _write_pred_csv(path: Path, n: int = 60, seed: int = 0) -> None:
    """Write a prediction CSV with the schema the pipeline emits."""
    rng = np.random.default_rng(seed)
    y_true = rng.normal(0, 0.01, size=n)
    y_pred = y_true + rng.normal(0, 0.005, size=n)
    pd.DataFrame({
        "idx": np.arange(n, dtype=int),
        "y_true": y_true,
        "y_pred": y_pred,
    }).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# What the writer actually produces
# ---------------------------------------------------------------------------


def test_evaluate_writes_ticker_model_filenames(tmp_path: Path) -> None:
    """The writer's naming contract: ``<TICKER>_<MODEL>.csv``, no window part.

    This is the fact the broken readers disagreed with, so it is asserted
    directly rather than assumed.
    """
    dates = pd.date_range("2023-01-01", periods=40, freq="B")
    df = pd.DataFrame({
        "date": dates,
        "log_return": np.random.default_rng(0).normal(0, 0.01, len(dates)),
    })
    rows, _ = run_one_ticker_eval(
        "tier1", "AAA",
        df=df,
        test_start="2023-01-10", test_end="2023-02-20",
        models=[NaiveModel()],
        predictions_dir=str(tmp_path),
    )
    written = sorted(p.name for p in tmp_path.glob("*.csv"))
    assert written == ["AAA_naive.csv"], written
    assert rows and rows[0]["model"] == "naive"


def test_metrics_csv_has_no_window_column() -> None:
    """``metrics.csv``'s columns are fixed by the runner; ``window`` is not one."""
    from src.runner import _write_run_tables
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _write_run_tables(
            root,
            [{"tier": "tier1", "ticker": "AAA", "model": "naive",
              "rmse": 0.1, "mae": 0.08, "n": 60}],
            [{"tier": "tier1", "ticker": "AAA"}],
        )
        cols = list(pd.read_csv(root / "metrics.csv").columns)
    assert cols == ["tier", "ticker", "model", "rmse", "mae", "n"]
    assert "window" not in cols


# ---------------------------------------------------------------------------
# Readers must accept it
# ---------------------------------------------------------------------------


def test_analysis_stage_persists_the_files_the_pipeline_writes(tmp_path: Path) -> None:
    """The analysis stage must process real pipeline output.

    Before the fix its filename regex required a window component, matched zero
    files, and persisted zero rows — while still returning a DB path and
    reporting no error.
    """
    test_x = tmp_path / "test_x"
    pred_dir = test_x / "tier1" / "predictions"
    pred_dir.mkdir(parents=True)
    models = ["naive", "expanding", "ma30"]
    for i, ticker in enumerate(["AAA", "BBB"]):
        for j, model in enumerate(models):
            _write_pred_csv(pred_dir / f"{ticker}_{model}.csv", seed=10 * i + j)

    db_path = analyse_test_run(test_x)

    conn = sqlite3.connect(str(db_path))
    n_step = conn.execute("SELECT COUNT(*) FROM analysis_per_step").fetchone()[0]
    n_sum = conn.execute("SELECT COUNT(*) FROM analysis_summary").fetchone()[0]
    conn.close()

    assert n_sum == 2 * len(models), "summary rows missing — the scan matched nothing"
    assert n_step == 2 * len(models) * 60

    assert (test_x / "tier1" / "analysis" / "dotplot_rmse.png").exists()
    assert (test_x / "analysis" / "all_tiers_dotplot_rmse.png").exists()


def test_analysis_stage_is_idempotent(tmp_path: Path) -> None:
    """Re-analysing the same run must not duplicate rows.

    The primary key is what enforces this, so it is worth pinning after the
    key changed shape.
    """
    test_x = tmp_path / "test_x"
    pred_dir = test_x / "tier1" / "predictions"
    pred_dir.mkdir(parents=True)
    _write_pred_csv(pred_dir / "AAA_naive.csv")

    db_path = analyse_test_run(test_x)
    analyse_test_run(test_x)

    conn = sqlite3.connect(str(db_path))
    n_sum = conn.execute("SELECT COUNT(*) FROM analysis_summary").fetchone()[0]
    n_step = conn.execute("SELECT COUNT(*) FROM analysis_per_step").fetchone()[0]
    conn.close()
    assert (n_sum, n_step) == (1, 60)


def test_prediction_scan_finds_pipeline_output(tmp_path: Path) -> None:
    """The scanner must match what the pipeline writes.

    Two scanners used to exist — ``summary._scan_predictions`` (live) and
    ``plots.discover_predictions`` (reachable only from a CLI whose input
    directory nothing populated). The dead one is gone; this covers the
    survivor.
    """
    for model in ("naive", "ma30"):
        _write_pred_csv(tmp_path / f"AAA_{model}.csv")
    _write_pred_csv(tmp_path / "BBB_naive.csv")

    found = _scan_predictions(tmp_path)

    assert set(found) == {"AAA", "BBB"}
    assert set(found["AAA"]) == {"naive", "ma30"}
    assert list(found["AAA"]["naive"].columns) == ["idx", "y_true", "y_pred"]


def test_unknown_prediction_filenames_are_ignored(tmp_path: Path) -> None:
    """Stray files must be skipped, not crash the scan — including the legacy
    three-part names, which no longer describe anything the pipeline writes."""
    _write_pred_csv(tmp_path / "AAA_naive.csv")
    (tmp_path / "notes.txt").write_text("ignore me")
    (tmp_path / "AAA_60_naive.csv").write_text("legacy,shape\n1,2\n")

    found = _scan_predictions(tmp_path)
    assert set(found) == {"AAA"}
    assert set(found["AAA"]) == {"naive"}


# ---------------------------------------------------------------------------
# The cache-refresh flag
# ---------------------------------------------------------------------------


def test_refresh_cache_reaches_load_returns(monkeypatch, tmp_path: Path) -> None:
    """``--refresh-cache`` was recorded in manifest.json and never acted on, so a
    run whose manifest claimed a refreshed cache had not refreshed it."""
    from src import runner as runner_mod
    import src.data as data_mod

    seen = {}

    def _fake_load_returns(ticker, *args, **kwargs):
        seen["refresh"] = kwargs.get("refresh")
        return pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=30, freq="B"),
            "log_return": np.zeros(30),
        })

    monkeypatch.setattr(data_mod, "load_returns", _fake_load_returns)

    tier_dir = tmp_path / "tier1"
    (tier_dir / "predictions").mkdir(parents=True)
    (tier_dir / "individual").mkdir(parents=True)

    import logging
    runner_mod._run_one_ticker_into_tier(
        "tier1", "AAA", tier_dir, logging.getLogger("t"),
        conn=None, models_factory=lambda: [NaiveModel()],
        refresh=True,
    )
    assert seen["refresh"] is True, "refresh never reached load_returns"


@pytest.mark.parametrize("model", [m for m in MODEL_ORDER if m != "ensemble"])
def test_every_model_name_survives_a_filename_round_trip(
    model: str, tmp_path: Path
) -> None:
    """Each live model name must survive write -> discover unchanged.

    Model names carry digits (``ma30``, ``arma60``); a reader whose pattern
    mis-splits on them would silently drop those models from every figure.
    """
    _write_pred_csv(tmp_path / f"AAA_{model}.csv")
    found = _scan_predictions(tmp_path)
    assert set(found["AAA"]) == {model}


# ---------------------------------------------------------------------------
# Per-pair figure rendering
# ---------------------------------------------------------------------------


def test_every_registered_per_pair_figure_renders(tmp_path: Path) -> None:
    """All five registered renderers must produce a figure.

    ``rolling_rmse`` and ``rolling_mae`` never did: they capped the smoothing
    span at ``min(ROLLING_ERROR_WINDOW, window)``, the runner passes no real
    window, and ``rolling(0)`` raises — which the runner's blanket
    ``except Exception: continue`` swallowed. Two of five figures were missing
    from every run and nothing said so.
    """
    from src.plots import _PER_PAIR_FIGURE_REGISTRY, figure_dirs

    rng = np.random.default_rng(0)
    model_dict = {}
    for model in ("naive", "ma30"):
        y_true = rng.normal(0, 0.01, 120)
        model_dict[model] = pd.DataFrame({
            "idx": np.arange(120),
            "y_true": y_true,
            "y_pred": y_true + rng.normal(0, 0.004, 120),
        })

    figs, data = tmp_path / "f", tmp_path / "f" / "data"
    rendered = {}
    with figure_dirs(str(figs), str(data)):
        for key, renderer in _PER_PAIR_FIGURE_REGISTRY.items():
            rendered[key] = renderer(model_dict, "AAA", 0)

    missing = [k for k, paths in rendered.items() if not paths]
    assert missing == [], f"renderers produced nothing: {missing}"
    assert len(sorted(figs.glob("*.png"))) == len(_PER_PAIR_FIGURE_REGISTRY)


def test_rolling_smoothing_span_adapts_to_short_series(tmp_path: Path) -> None:
    """The smoothing span is capped by the data available, not by a fixed
    window — a series shorter than ``ROLLING_ERROR_WINDOW`` must still render."""
    from src.plots import _rolling_error_frame

    short = {"naive": pd.DataFrame({
        "idx": np.arange(5), "y_true": np.zeros(5), "y_pred": np.ones(5),
    })}
    frame, smooth = _rolling_error_frame(short, "rmse")
    assert 1 <= smooth <= 5
    assert len(frame) == 5
