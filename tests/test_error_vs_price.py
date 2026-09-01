"""Tests for the best-predictor-error-vs-price figure.

The README publishes this figure as a result, but nothing in the repo generated
it — it was made outside the project, so the headline scaling claim could not be
reproduced from a run. These cover the generator that closes that gap, and the
mean-price plumbing it needs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.storage.db import get_mean_prices, init_schema, open_db, put_history
from tests.helpers import mean_predictions, write_prediction_csv
from src.summary import (
    _best_causal_rmse,
    _loglog_slope,
    _plot_error_vs_price,
    _read_ticker_prices,
    summarise_overall,
)


# ---------------------------------------------------------------------------
# Picking the ticker's best causal model
# ---------------------------------------------------------------------------


def _per_ticker(rows) -> pd.DataFrame:
    return pd.DataFrame(
        [{"ticker": t, "model": m, "rmse": r, "mae": r * 0.8, "n": 250}
         for t, m, r in rows]
    )


def test_best_causal_rmse_excludes_the_three_non_candidates() -> None:
    """``global`` peeks at the test set, ``naive`` is the trivial benchmark and
    ``ensemble`` is derived from the others — none may be picked as "best"."""
    best = _best_causal_rmse(_per_ticker([
        ("AAA", "global", 0.001),     # lowest, but future-leaking
        ("AAA", "naive", 0.002),      # trivial
        ("AAA", "ensemble", 0.003),   # derived
        ("AAA", "ma30", 0.010),
        ("AAA", "expanding", 0.009),  # the answer
    ]))
    assert list(best["model"]) == ["expanding"]
    assert best.iloc[0]["rmse"] == pytest.approx(0.009)


def test_best_causal_rmse_picks_one_row_per_ticker() -> None:
    best = _best_causal_rmse(_per_ticker([
        ("AAA", "ma30", 0.010), ("AAA", "ma60", 0.008),
        ("BBB", "ma30", 0.020), ("BBB", "ma60", 0.030),
    ]))
    assert dict(zip(best["ticker"], best["model"])) == {"AAA": "ma60", "BBB": "ma30"}


def test_best_causal_rmse_on_empty_input_returns_an_empty_frame() -> None:
    assert _best_causal_rmse(pd.DataFrame()).empty
    only_excluded = _per_ticker([("AAA", "naive", 0.01), ("AAA", "global", 0.02)])
    assert _best_causal_rmse(only_excluded).empty


# ---------------------------------------------------------------------------
# The power-law slope the figure reports
# ---------------------------------------------------------------------------


def test_loglog_slope_recovers_a_known_power_law() -> None:
    """A slope here is an exponent: rmse ~ price ** slope."""
    x = np.array([1.0, 10.0, 100.0, 1000.0])
    assert _loglog_slope(x, 0.05 * x ** -0.25) == pytest.approx(-0.25, abs=1e-9)
    assert _loglog_slope(x, 0.05 * x ** 0.5) == pytest.approx(0.5, abs=1e-9)


def test_loglog_slope_is_undefined_rather_than_wrong_on_bad_input() -> None:
    """Non-positive and non-finite values have no logarithm; too few points
    have no line. Both return None instead of a fabricated number."""
    assert _loglog_slope(np.array([1.0]), np.array([1.0])) is None
    assert _loglog_slope(np.array([0.0, -1.0]), np.array([1.0, 2.0])) is None
    assert _loglog_slope(np.array([1.0, np.nan]), np.array([1.0, 2.0])) is None


def test_loglog_slope_ignores_unusable_points_but_uses_the_rest() -> None:
    x = np.array([1.0, 10.0, 100.0, 0.0])
    y = np.array([1.0, 0.1, 0.01, 5.0])
    assert _loglog_slope(x, y) == pytest.approx(-1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Reading the price column back out of a run
# ---------------------------------------------------------------------------


def test_read_ticker_prices_returns_the_recorded_means(tmp_path: Path) -> None:
    pd.DataFrame([
        {"tier": "tier1", "ticker": "AAA", "mean_price": 12.5},
        {"tier": "tier2", "ticker": "BBB", "mean_price": 80.0},
    ]).to_csv(tmp_path / "ticker_tested.csv", index=False)

    out = _read_ticker_prices(tmp_path)
    assert dict(zip(out["ticker"], out["mean_price"])) == {"AAA": 12.5, "BBB": 80.0}


def test_ticker_symbols_that_look_numeric_still_join(tmp_path: Path) -> None:
    """A symbol like ``600`` is read back from CSV as int64 while the same
    symbol parsed out of a filename is a str, so the join has to normalise.
    Caught by a full pipeline run, not by the unit tests above, whose tickers
    were all alphabetic."""
    pd.DataFrame([{"tier": "tier1", "ticker": 600, "mean_price": 42.0}]).to_csv(
        tmp_path / "ticker_tested.csv", index=False)
    prices = _read_ticker_prices(tmp_path)
    assert prices.iloc[0]["ticker"] == "600", "symbol must come back as text"

    best = pd.DataFrame({"ticker": ["600"], "model": ["ma30"], "rmse": [0.03]})
    out = _plot_error_vs_price(best, prices, out_path=tmp_path / "f.png")
    assert out.exists() and out.stat().st_size > 0


def test_read_ticker_prices_tolerates_an_older_run_tree(tmp_path: Path) -> None:
    """A run made before ``mean_price`` existed must not break the summary —
    the figure is skipped, every other output is already written by then."""
    assert _read_ticker_prices(tmp_path).empty          # no file at all
    pd.DataFrame([{"tier": "tier1", "ticker": "AAA"}]).to_csv(
        tmp_path / "ticker_tested.csv", index=False)
    assert _read_ticker_prices(tmp_path).empty          # file, but no column


# ---------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------


def test_plot_error_vs_price_writes_a_figure(tmp_path: Path) -> None:
    best = pd.DataFrame({
        "ticker": ["AAA", "BBB", "CCC"],
        "model": ["expanding"] * 3,
        "rmse": [0.05, 0.02, 0.012],
    })
    prices = pd.DataFrame({
        "tier": ["tier1", "tier2", "tier3"],
        "ticker": ["AAA", "BBB", "CCC"],
        "mean_price": [5.0, 60.0, 400.0],
    })
    out = _plot_error_vs_price(best, prices, out_path=tmp_path / "f.png")
    assert out.exists() and out.stat().st_size > 0


def test_plot_error_vs_price_survives_unusable_prices(tmp_path: Path) -> None:
    """A zero or NaN price has no logarithm. The figure must still be written
    rather than taking the summary stage down with it."""
    best = pd.DataFrame({"ticker": ["AAA", "BBB"], "model": ["ma30"] * 2,
                         "rmse": [0.05, 0.02]})
    prices = pd.DataFrame({"tier": ["tier1", "tier1"], "ticker": ["AAA", "BBB"],
                           "mean_price": [0.0, np.nan]})
    out = _plot_error_vs_price(best, prices, out_path=tmp_path / "f.png")
    assert out.exists() and out.stat().st_size > 0


# ---------------------------------------------------------------------------
# End to end through the summary stage
# ---------------------------------------------------------------------------


def _write_pred(path: Path, sigma: float, n: int = 80, seed: int = 0) -> None:
    """Realised volatility ``sigma``, predicted by the series' own mean — so the
    best model's RMSE is ``sigma``, which is the identity the figure plots."""
    write_prediction_csv(path, *mean_predictions(n, sigma=sigma, seed=seed))


def test_summarise_overall_writes_the_price_figure(tmp_path: Path) -> None:
    """The figure the README publishes is now produced by a run."""
    root = tmp_path / "test_x"
    specs = [("tier1", "AAA", 5.0, 0.05), ("tier2", "BBB", 60.0, 0.02),
             ("tier3", "CCC", 400.0, 0.012)]
    for i, (tier, ticker, _px, sigma) in enumerate(specs):
        pred = root / tier / "predictions"
        pred.mkdir(parents=True, exist_ok=True)
        for model in ("expanding", "ma30"):
            _write_pred(pred / f"{ticker}_{model}.csv", sigma, seed=i)
    pd.DataFrame([
        {"tier": t, "ticker": k, "mean_price": p} for t, k, p, _ in specs
    ]).to_csv(root / "ticker_tested.csv", index=False)

    written = summarise_overall(root, [t for t, _, _, _ in specs])

    fig = root / "analysis" / "best_predictor_vs_price.png"
    assert fig.exists() and fig.stat().st_size > 0
    assert str(fig) in written


def test_summary_still_completes_without_a_price_column(tmp_path: Path) -> None:
    """The price figure is derivative; its absence must not cost the run its
    other cross-tier outputs."""
    root = tmp_path / "test_x"
    pred = root / "tier1" / "predictions"
    pred.mkdir(parents=True)
    _write_pred(pred / "AAA_expanding.csv", 0.02)

    written = summarise_overall(root, ["tier1"])

    assert (root / "analysis" / "summary_overall.csv").exists()
    assert (root / "analysis" / "score_histogram.png").exists()
    assert not (root / "analysis" / "best_predictor_vs_price.png").exists()
    assert any("summary_overall" in w for w in written)


# ---------------------------------------------------------------------------
# The mean-price source
# ---------------------------------------------------------------------------


def test_get_mean_prices_reads_back_what_put_history_stored(tmp_path: Path) -> None:
    conn = open_db(str(tmp_path / "c.db"))
    init_schema(conn)
    dates = pd.date_range("2023-01-01", periods=10, freq="B")
    for sym, level in (("AAA", 10.0), ("BBB", 100.0)):
        put_history(conn, sym, tier="tier1", start="2023-01-01", end="2023-01-31",
                    prices=pd.Series(np.full(10, level), index=dates, name="adj_close"))

    assert get_mean_prices(conn) == {"AAA": 10.0, "BBB": 100.0}
    assert get_mean_prices(conn, ["AAA"]) == {"AAA": 10.0}
    assert get_mean_prices(conn, ["ZZZ"]) == {}
    conn.close()
