"""One NaN policy, applied everywhere metrics are computed.

A model may legitimately return ``nan``: ``src.models`` documents it for a
prediction made before any fit and for an ARMA whose every candidate order
failed, and the post-hoc ensemble already averages with ``np.nanmean`` so that
"one nan child does not poison the result".

Three call sites answered "what does a nan prediction mean for a metric?"
differently — ``evaluate`` and ``analysis.compute`` propagated it, ``summary``
masked it — so the same (ticker, model) was reported three ways across
metrics.csv, summary_<tier>.csv and analysis.db. These tests pin the single
answer: score the finite pairs, and report how many survived in ``n``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.compute import compute_per_step, compute_summary
from src.metrics import mae, rmse
from src.summary import _per_ticker_metrics


#: 5 steps, one unpredictable — the shape a windowed model's guard produces.
_Y_TRUE = np.array([0.01, -0.02, 0.03, 0.00, 0.02])
_Y_PRED = np.array([np.nan, -0.01, 0.02, 0.01, 0.00])
_FINITE = np.isfinite(_Y_PRED)
_EXPECTED_RMSE = float(np.sqrt(np.mean((_Y_TRUE[_FINITE] - _Y_PRED[_FINITE]) ** 2)))
_EXPECTED_MAE = float(np.mean(np.abs(_Y_TRUE[_FINITE] - _Y_PRED[_FINITE])))


# ---------------------------------------------------------------------------
# The policy itself
# ---------------------------------------------------------------------------


def test_all_nan_raises_rather_than_returning_a_meaningless_zero() -> None:
    """With nothing finite there is no metric to report. Raising keeps that
    distinct from "the model was perfect", which a silent 0.0 would not."""
    nan3 = np.full(3, np.nan)
    with pytest.raises(ValueError, match="no finite"):
        rmse(nan3, nan3)
    with pytest.raises(ValueError, match="no finite"):
        mae(np.array([1.0, 2.0, 3.0]), nan3)


# ---------------------------------------------------------------------------
# All three writers must now agree
# ---------------------------------------------------------------------------


def _analysis_answer(tmp: Path) -> dict:
    path = tmp / "AAA_naive.csv"
    pd.DataFrame({
        "idx": np.arange(_Y_TRUE.size), "y_true": _Y_TRUE, "y_pred": _Y_PRED,
    }).to_csv(path, index=False)
    return compute_summary(compute_per_step(path))


def _summary_answer() -> pd.Series:
    df = pd.DataFrame({
        "idx": np.arange(_Y_TRUE.size), "y_true": _Y_TRUE, "y_pred": _Y_PRED,
    })
    return _per_ticker_metrics({"AAA": {"naive": df}}).iloc[0]


def test_the_three_writers_report_the_same_number(tmp_path: Path) -> None:
    """metrics.csv, summary_<tier>.csv and analysis.db described the same
    (ticker, model) with three different values. They must not."""
    direct = rmse(_Y_TRUE, _Y_PRED)
    summary = float(_summary_answer()["rmse"])
    analysis = float(_analysis_answer(tmp_path)["rmse"])

    assert direct == pytest.approx(summary) == pytest.approx(analysis), (
        f"still disagreeing: metrics={direct}, summary={summary}, "
        f"analysis={analysis}"
    )
    assert direct == pytest.approx(_EXPECTED_RMSE)


