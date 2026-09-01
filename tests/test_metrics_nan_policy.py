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

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.compute import compute_per_step, compute_summary
from src.evaluate import run_one_ticker_eval
from src.metrics import mae, rmse
from src.models import NaiveModel
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


def test_rmse_and_mae_score_only_the_finite_pairs() -> None:
    assert rmse(_Y_TRUE, _Y_PRED) == pytest.approx(_EXPECTED_RMSE)
    assert mae(_Y_TRUE, _Y_PRED) == pytest.approx(_EXPECTED_MAE)


def test_a_nan_in_y_true_drops_the_pair_too() -> None:
    """Both sides are checked — a missing observation is as unusable as a
    missing forecast."""
    yt = np.array([0.01, np.nan, 0.03])
    yp = np.array([0.02, 0.05, 0.03])
    assert rmse(yt, yp) == pytest.approx(float(np.sqrt(np.mean([0.01 ** 2, 0.0]))))


def test_all_nan_raises_rather_than_returning_a_meaningless_zero() -> None:
    """With nothing finite there is no metric to report. Raising keeps that
    distinct from "the model was perfect", which a silent 0.0 would not."""
    nan3 = np.full(3, np.nan)
    with pytest.raises(ValueError, match="no finite"):
        rmse(nan3, nan3)
    with pytest.raises(ValueError, match="no finite"):
        mae(np.array([1.0, 2.0, 3.0]), nan3)


def test_existing_guards_still_hold() -> None:
    """Shape and emptiness checks predate this change and must survive it."""
    with pytest.raises(ValueError, match="shape mismatch"):
        rmse(np.array([1.0, 2.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="at least one observation"):
        rmse(np.array([]), np.array([]))


def test_finite_input_is_unaffected() -> None:
    """The overwhelmingly common case must be byte-identical to before."""
    yt = np.array([1.0, 2.0, 3.0, 4.0])
    yp = np.array([1.5, 1.5, 3.5, 3.5])
    assert rmse(yt, yp) == pytest.approx(0.5)
    assert mae(yt, yp) == pytest.approx(0.5)


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


def test_n_records_how_many_pairs_were_actually_scored(tmp_path: Path) -> None:
    """The dropped step must stay visible rather than silently vanishing."""
    assert int(_summary_answer()["n"]) == 4
    assert int(_analysis_answer(tmp_path)["n_steps"]) == 4


def test_evaluate_reports_a_finite_metric_for_a_partly_nan_model(
    tmp_path: Path,
) -> None:
    """End-to-end through the writer: a model that could not predict every step
    still gets a usable score, with ``n`` showing the shortfall."""
    dates = pd.date_range("2023-01-01", periods=40, freq="B")
    df = pd.DataFrame({
        "date": dates,
        "log_return": np.random.default_rng(0).normal(0, 0.01, len(dates)),
    })

    class SometimesNaN(NaiveModel):
        """Fails its first prediction, as a windowed model would at short t."""

        name = "flaky"
        kind = "naive_then_nan"   # not a known kind -> treated as windowed
        lookback = 2

        def __init__(self) -> None:
            super().__init__()
            self._calls = 0

        def fit(self, y: np.ndarray) -> None:
            self._calls += 1
            super().fit(y)

        def predict_one(self) -> float:
            return float("nan") if self._calls == 1 else super().predict_one()

    rows, _ = run_one_ticker_eval(
        "tier1", "AAA", df=df,
        test_start="2023-01-10", test_end="2023-02-20",
        models=[SometimesNaN()], predictions_dir=str(tmp_path),
    )
    row = next(r for r in rows if r["model"] == "flaky")
    assert np.isfinite(row["rmse"]), "a single failed step must not void the series"
    assert np.isfinite(row["mae"])


def test_ensemble_children_with_nan_still_produce_a_finite_ensemble() -> None:
    """The ensemble's np.nanmean tolerance and the metric's must agree — this
    is the inconsistency that sat two lines apart in evaluate.py."""
    dates = pd.date_range("2023-01-01", periods=40, freq="B")
    df = pd.DataFrame({
        "date": dates,
        "log_return": np.random.default_rng(1).normal(0, 0.01, len(dates)),
    })
    with tempfile.TemporaryDirectory() as d:
        rows, per_model = run_one_ticker_eval(
            "tier1", "AAA", df=df,
            test_start="2023-01-10", test_end="2023-02-20",
            predictions_dir=d,
        )
    for row in rows:
        assert np.isfinite(row["rmse"]), f"{row['model']} scored nan"
        assert row["n"] > 0
