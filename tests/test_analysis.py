"""Tests for Phase I — measurement-run analysis."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.compute import compute_per_step, compute_summary
from src.analysis.db import init_analysis_schema, open_analysis_db
from src.analysis.dotplot import render_dotplot
from src.analysis.persist import persist_summary
from src.analysis.runner import analyse_test_run


# ---------------------------------------------------------------------------
# compute_per_step / compute_summary
# ---------------------------------------------------------------------------


def test_compute_per_step_correct(tmp_path: Path) -> None:
    df = pd.DataFrame({
        "idx": [60, 61, 62, 63, 64],
        "y_true": [1.0, 2.0, 3.0, 4.0, 5.0],
        "y_pred": [1.5, 1.5, 3.0, 3.5, 4.0],
    })
    csv = tmp_path / "AAA_60_naive.csv"
    df.to_csv(csv, index=False)

    out = compute_per_step(csv)

    assert list(out.columns) == ["step_idx", "sq_err", "abs_err"]
    np.testing.assert_array_equal(out["step_idx"].to_numpy(), np.arange(5))
    expected_err = np.array([-0.5, 0.5, 0.0, 0.5, 1.0])
    np.testing.assert_allclose(out["sq_err"].to_numpy(), expected_err ** 2)
    np.testing.assert_allclose(out["abs_err"].to_numpy(), np.abs(expected_err))


def test_compute_summary_matches_handcalc() -> None:
    err = np.array([0.1, -0.2, 0.3, -0.1, 0.0, 0.5, -0.4, 0.2, 0.1, 0.0])
    per_step = pd.DataFrame({
        "step_idx": np.arange(err.size),
        "sq_err": err ** 2,
        "abs_err": np.abs(err),
    })
    s = compute_summary(per_step)

    assert s["n_steps"] == 10
    assert s["mse"] == pytest.approx(float((err ** 2).mean()), abs=1e-12)
    assert s["rmse"] == pytest.approx(float(np.sqrt((err ** 2).mean())), abs=1e-12)
    assert s["mae"] == pytest.approx(float(np.abs(err).mean()), abs=1e-12)
    assert s["sq_err_var"] == pytest.approx(float((err ** 2).var(ddof=0)), abs=1e-12)
    assert s["sq_err_median"] == pytest.approx(float(np.median(err ** 2)), abs=1e-12)
    assert s["sq_err_max"] == pytest.approx(float((err ** 2).max()), abs=1e-12)
    assert s["sq_err_min"] == pytest.approx(float((err ** 2).min()), abs=1e-12)
    assert s["abs_err_max"] == pytest.approx(float(np.abs(err).max()), abs=1e-12)


# ---------------------------------------------------------------------------
# DB round-trip
# ---------------------------------------------------------------------------


def test_db_round_trip_summary(tmp_path: Path) -> None:
    db_path = tmp_path / "analysis.db"
    conn = open_analysis_db(str(db_path))
    init_analysis_schema(conn)

    row = {
        "tier": "tier1", "ticker": "AAA", "window": 60, "model": "naive",
        "n_steps": 100, "mse": 0.04, "rmse": 0.2, "mae": 0.18,
        "sq_err_var": 0.001, "sq_err_median": 0.01,
        "sq_err_max": 0.5, "sq_err_min": 0.0,
        "abs_err_var": 0.005, "abs_err_median": 0.1,
        "abs_err_max": 0.7, "abs_err_min": 0.0,
    }
    persist_summary(conn, "test_2026-05-08T00-00-00Z", [row])

    cur = conn.execute(
        "SELECT tier, ticker, window, model, n_steps, rmse, mae "
        "FROM analysis_summary WHERE test_run = ?",
        ("test_2026-05-08T00-00-00Z",),
    )
    fetched = cur.fetchall()
    conn.close()

    assert len(fetched) == 1
    assert fetched[0] == ("tier1", "AAA", 60, "naive", 100, 0.2, 0.18)


# ---------------------------------------------------------------------------
# render_dotplot
# ---------------------------------------------------------------------------


def test_render_dotplot_writes_png_and_csv(tmp_path: Path) -> None:
    df = pd.DataFrame({
        "ticker": ["AAA", "BBB", "AAA", "BBB"],
        "window": [60, 60, 60, 60],
        "model": ["naive", "naive", "average", "average"],
        "rmse": [0.10, 0.12, 0.09, 0.13],
        "mae":  [0.08, 0.09, 0.07, 0.10],
    })
    out_png = tmp_path / "dotplot_rmse.png"
    render_dotplot(df, metric="rmse", title="tier1 — RMSE", out_path=out_png)

    assert out_png.exists() and out_png.stat().st_size > 0
    assert out_png.with_suffix(".csv").exists()


# ---------------------------------------------------------------------------
# End-to-end smoke
# ---------------------------------------------------------------------------


def _write_pred_csv(path: Path, n: int = 60, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    y_true = rng.normal(0, 0.01, size=n)
    y_pred = y_true + rng.normal(0, 0.005, size=n)
    pd.DataFrame({
        "idx": np.arange(60, 60 + n, dtype=int),
        "y_true": y_true,
        "y_pred": y_pred,
    }).to_csv(path, index=False)


def test_analyse_test_run_smoke(tmp_path: Path) -> None:
    test_x = tmp_path / "test_x"
    pred_dir = test_x / "tier1" / "predictions"
    pred_dir.mkdir(parents=True)
    for i, ticker in enumerate(["AAA", "BBB"]):
        for j, model in enumerate(["naive", "average"]):
            _write_pred_csv(pred_dir / f"{ticker}_60_{model}.csv",
                            seed=10 * i + j)

    db_path = analyse_test_run(test_x)

    assert db_path.exists()

    conn = sqlite3.connect(str(db_path))
    n_step = conn.execute("SELECT COUNT(*) FROM analysis_per_step").fetchone()[0]
    n_sum = conn.execute("SELECT COUNT(*) FROM analysis_summary").fetchone()[0]
    conn.close()
    assert n_step == 4 * 60          # 2 tickers × 2 models × 60 steps
    assert n_sum == 4                 # 2 tickers × 2 models

    assert (test_x / "tier1" / "analysis" / "dotplot_rmse.png").exists()
    assert (test_x / "tier1" / "analysis" / "dotplot_mae.png").exists()
    assert (test_x / "analysis" / "all_tiers_dotplot_rmse.png").exists()
    assert (test_x / "analysis" / "all_tiers_dotplot_mae.png").exists()
