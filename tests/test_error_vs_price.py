"""Tests for the best-predictor-error-vs-price figure.

The README publishes this figure as a result, but nothing in the repo generated
it — it was made outside the project, so the headline scaling claim could not be
reproduced from a run. These cover the generator that closes that gap, and the
mean-price plumbing it needs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tests.helpers import mean_predictions, write_prediction_csv
from src.summary import (
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


# ---------------------------------------------------------------------------
# The power-law slope the figure reports
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Reading the price column back out of a run
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# The mean-price source
# ---------------------------------------------------------------------------


