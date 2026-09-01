"""Characterisation tests for ``src.rolling.run_eval``.

These pin what the engine *currently does*, not what it ideally should. The
module had no direct test: its only end-to-end caller drives it with a single
``NaiveModel``, so three of its four dispatch branches, the expanding-mean
running-sum path and the short-lookback guard were executed by nothing.

That is a prerequisite for touching the ``kind`` dispatch (backlog R9/R10), so
every assertion here was derived by running the current engine and recording
its answer. One of them recorded a latent bug rather than intended behaviour —
the ``t = 0`` look-ahead leak in the naive branch — and pinning it is what made
the fix safe to make later; that test now asserts the fixed behaviour.

The engine's contract, as observed:

* ``naive``     — reads ``y_full[t-1]`` directly; the model object is bypassed
* ``global``    — fitted once on the whole series before the loop, then polled
* ``expanding`` — running mean of ``y_full[train_start:t]``, via an O(1)
                  running sum for ``ExpandingMeanModel`` and a re-fit otherwise
* ``windowed``  — ``fit(y_full[t-L:t])`` then ``predict_one()``, NaN when the
                  window would run off the front of the series
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest

from src.models import (
    ExpandingMeanModel,
    GlobalMeanModel,
    MovingAverageModel,
    NaiveModel,
)
from src.rolling import run_eval


# ---------------------------------------------------------------------------
# Spies — record how the engine drives a model, not just what it returns
# ---------------------------------------------------------------------------


class _SpyMixin:
    """Records every ``fit`` / ``predict_one`` call and the window each saw."""

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.fit_calls: List[np.ndarray] = []
        self.predict_calls: int = 0

    def fit(self, y: np.ndarray) -> None:
        self.fit_calls.append(np.asarray(y).copy())
        super().fit(y)

    def predict_one(self) -> float:
        self.predict_calls += 1
        return super().predict_one()


class SpyNaive(_SpyMixin, NaiveModel):
    pass


class SpyGlobal(_SpyMixin, GlobalMeanModel):
    pass


class SpyMA(_SpyMixin, MovingAverageModel):
    pass


class SpyExpanding(_SpyMixin, ExpandingMeanModel):
    def __init__(self) -> None:
        super().__init__()
        self.set_state_calls: int = 0

    def set_state(self, running_sum: float, count: int) -> None:
        self.set_state_calls += 1
        super().set_state(running_sum, count)


class DuckExpanding:
    """An ``expanding`` model with no ``set_state``.

    Exercises the fallback branch, which re-fits on the whole prefix instead of
    using the running sum.
    """

    name = "duck"
    kind = "expanding"
    lookback = -1

    def __init__(self) -> None:
        self.fit_calls: List[np.ndarray] = []
        self._mean = float("nan")

    def fit(self, y: np.ndarray) -> None:
        self.fit_calls.append(np.asarray(y).copy())
        self._mean = float(np.mean(y)) if len(y) else float("nan")

    def predict_one(self) -> float:
        return self._mean


class NoKindModel:
    """A model that declares no ``kind`` at all."""

    name = "nokind"
    lookback = 3

    def __init__(self) -> None:
        self._mean = float("nan")

    def fit(self, y: np.ndarray) -> None:
        self._mean = float(np.mean(y))

    def predict_one(self) -> float:
        return self._mean


@pytest.fixture
def y() -> np.ndarray:
    """``[0., 1., ..., 9.]`` — every expected value below is hand-checkable."""
    return np.arange(10, dtype=float)


# ---------------------------------------------------------------------------
# naive
# ---------------------------------------------------------------------------


def test_naive_reads_the_series_directly_and_never_fits(y: np.ndarray) -> None:
    """The engine inlines ``y_full[t-1]``; the model object is not consulted.

    Worth pinning because it makes the naive branch independent of
    ``NaiveModel`` entirely — a refactor routing it back through ``fit`` would
    be a behaviour change even though the numbers would agree here.
    """
    m = SpyNaive()
    out = run_eval(y, [m], [3, 5, 7])

    np.testing.assert_array_equal(out["naive"][1], [2.0, 4.0, 6.0])
    assert m.fit_calls == []
    assert m.predict_calls == 0


def test_naive_at_index_zero_is_nan_not_the_last_observation(y: np.ndarray) -> None:
    """At ``t = 0`` there is no prior observation, so there is no forecast.

    This pinned a bug until it was fixed: the engine evaluated ``y_full[t - 1]``
    unguarded, and numpy resolves ``y_full[-1]`` to the *last* element — so the
    forecast for the first observation was the final one, a look-ahead leak that
    would have scored as a perfect prediction of a future the model cannot see.

    NaN is not a new policy; it is the one the windowed branch already applies
    when ``t - L < 0``. The same condition now gets the same answer.
    """
    out = run_eval(y, [NaiveModel()], [0])
    assert np.isnan(out["naive"][1][0]), "t=0 must not reach back to y_full[-1]"

    # Every later step is unaffected.
    later = run_eval(y, [NaiveModel()], [1, 5, 9])
    np.testing.assert_array_equal(later["naive"][1], [0.0, 4.0, 8.0])


# ---------------------------------------------------------------------------
# global
# ---------------------------------------------------------------------------


def test_global_is_fitted_once_on_the_whole_series_including_the_test_set(
    y: np.ndarray,
) -> None:
    """Future-leaking by design — it is the benchmark, not a candidate.

    One fit before the loop, over the entire series, then polled per step.
    """
    m = SpyGlobal()
    out = run_eval(y, [m], [3, 4, 5])

    assert len(m.fit_calls) == 1, "global must be fitted exactly once"
    np.testing.assert_array_equal(m.fit_calls[0], y)
    np.testing.assert_allclose(out["global"][1], [y.mean()] * 3)
    assert m.predict_calls == 3


# ---------------------------------------------------------------------------
# expanding — the branch carrying the O(1) optimisation
# ---------------------------------------------------------------------------


def test_expanding_mean_uses_the_running_sum_and_never_refits(y: np.ndarray) -> None:
    """``ExpandingMeanModel`` is driven by ``set_state``, never by ``fit``.

    This is the optimisation the backlog warns a naive polymorphic rewrite
    would undo: re-fitting on ``y[train_start:t]`` each step is O(t) per step
    and O(n^2) over the run. Counting ``fit`` calls is the observable proxy —
    if this assertion starts failing, the complexity changed.
    """
    m = SpyExpanding()
    out = run_eval(y, [m], [3, 4, 5])

    assert m.fit_calls == [], "expanding must not re-fit; it carries a running sum"
    assert m.set_state_calls == 3
    # mean of y[0:3], y[0:4], y[0:5]
    np.testing.assert_allclose(out["expanding"][1], [1.0, 1.5, 2.0])


def test_expanding_fallback_refits_but_agrees_numerically(y: np.ndarray) -> None:
    """A duck-typed expanding model takes the re-fit path and must agree.

    The two paths computing the same numbers is what makes the optimisation
    safe, so it is pinned rather than assumed.
    """
    duck = DuckExpanding()
    fast = run_eval(y, [ExpandingMeanModel()], [3, 4, 5])["expanding"][1]
    slow = run_eval(y, [duck], [3, 4, 5])["duck"][1]

    assert len(duck.fit_calls) == 3, "the fallback re-fits once per step"
    np.testing.assert_allclose(slow, fast)


def test_expanding_routing_is_decided_by_set_state_not_by_class(
    y: np.ndarray,
) -> None:
    """Any model exposing ``set_state`` gets the running sum.

    NOTE — this is the one place the engine's behaviour was deliberately
    changed rather than pinned. The branch used to test
    ``isinstance(m, ExpandingMeanModel)``, which made ``src.rolling`` import a
    concrete model class and contradicted the invariant stated on
    ``Forecaster``: that this module depends on the abstraction, "never on a
    concrete model". Asking structurally honours that.

    The difference is reachable only by a model that exposes ``set_state``
    without subclassing ``ExpandingMeanModel`` — none exists in this codebase.
    Such a model previously took the re-fit path and now takes the fast one.
    The forecasts are identical either way; only the route changes.
    """
    class DuckWithSetState(DuckExpanding):
        name = "duck2"

        def __init__(self) -> None:
            super().__init__()
            self.set_state_calls = 0

        def set_state(self, running_sum: float, count: int) -> None:
            self.set_state_calls += 1
            self._mean = float(running_sum / count) if count else float("nan")

    fast = DuckWithSetState()
    out = run_eval(y, [fast], [3, 4, 5])

    assert fast.set_state_calls == 3
    assert fast.fit_calls == [], "a model advertising set_state must not re-fit"
    np.testing.assert_allclose(out["duck2"][1], [1.0, 1.5, 2.0])


def test_expanding_window_excludes_the_step_being_predicted(y: np.ndarray) -> None:
    """The running sum advances *after* each prediction, so step ``t`` is
    predicted from ``[train_start, t)`` and never sees its own value."""
    duck = DuckExpanding()
    run_eval(y, [duck], [4, 5])
    np.testing.assert_array_equal(duck.fit_calls[0], y[0:4])
    np.testing.assert_array_equal(duck.fit_calls[1], y[0:5])


def test_train_start_offsets_the_expanding_window(y: np.ndarray) -> None:
    """``train_start`` moves the left edge of the expanding mean."""
    out = run_eval(y, [ExpandingMeanModel()], [5, 6], train_start=2)
    # mean of y[2:5] and y[2:6]
    np.testing.assert_allclose(out["expanding"][1], [3.0, 3.5])


# ---------------------------------------------------------------------------
# windowed
# ---------------------------------------------------------------------------


def test_windowed_fits_exactly_the_preceding_lookback(y: np.ndarray) -> None:
    m = SpyMA(3)
    out = run_eval(y, [m], [5, 8])

    np.testing.assert_array_equal(m.fit_calls[0], y[2:5])
    np.testing.assert_array_equal(m.fit_calls[1], y[5:8])
    np.testing.assert_allclose(out["ma3"][1], [3.0, 6.0])


def test_windowed_yields_nan_when_the_window_runs_off_the_front(
    y: np.ndarray,
) -> None:
    """``t - L < 0`` produces NaN and skips the fit entirely — the model is not
    called at all for that step, so no partial window ever reaches it."""
    m = SpyMA(5)
    out = run_eval(y, [m], [2, 7])

    assert np.isnan(out["ma5"][1][0])
    assert out["ma5"][1][1] == pytest.approx(y[2:7].mean())
    assert len(m.fit_calls) == 1, "the guarded step must not fit"


def test_non_positive_lookback_yields_nan(y: np.ndarray) -> None:
    """``L <= 0`` is guarded the same way. ``GlobalMeanModel`` and
    ``ExpandingMeanModel`` carry ``lookback = -1`` as a sentinel, so a model
    reaching the windowed branch with one must not be fitted on a reversed
    slice."""
    class ZeroLookback(NoKindModel):
        name = "zero"
        lookback = 0

    out = run_eval(y, [ZeroLookback()], [5])
    assert np.isnan(out["zero"][1][0])


def test_a_model_without_a_kind_attribute_is_treated_as_windowed(
    y: np.ndarray,
) -> None:
    out = run_eval(y, [NoKindModel()], [5])
    assert out["nokind"][1][0] == pytest.approx(y[2:5].mean())


# ---------------------------------------------------------------------------
# Shape and aliasing of the result
# ---------------------------------------------------------------------------


def test_all_four_kinds_dispatch_together_in_one_pass(y: np.ndarray) -> None:
    """The mixed case: one model of each kind, driven in a single call."""
    models = [NaiveModel(), GlobalMeanModel(), ExpandingMeanModel(),
              MovingAverageModel(3)]
    out = run_eval(y, models, [5, 6, 7])

    assert set(out) == {"naive", "global", "expanding", "ma3"}
    np.testing.assert_array_equal(out["naive"][1], [4.0, 5.0, 6.0])
    np.testing.assert_allclose(out["global"][1], [y.mean()] * 3)
    np.testing.assert_allclose(out["expanding"][1], [2.0, 2.5, 3.0])
    np.testing.assert_allclose(out["ma3"][1], [3.0, 4.0, 5.0])


def test_y_true_is_the_same_array_shared_by_every_model(y: np.ndarray) -> None:
    """Every entry aliases one ``y_true`` array rather than holding a copy.

    Pinned because a refactor that copies per model would raise memory use
    across a 300-ticker run, and one that mutates it would corrupt every model
    at once.
    """
    out = run_eval(y, [NaiveModel(), GlobalMeanModel()], [3, 4])
    assert out["naive"][0] is out["global"][0]
    np.testing.assert_array_equal(out["naive"][0], [3.0, 4.0])


def test_empty_test_indices_returns_empty_arrays_without_raising(
    y: np.ndarray,
) -> None:
    """The seeding of the running sum reads ``test_indices[0]``, so the empty
    case is guarded; it must stay guarded."""
    out = run_eval(y, [NaiveModel(), ExpandingMeanModel()], [])
    assert set(out) == {"naive", "expanding"}
    for y_true, y_pred in out.values():
        assert y_true.shape == (0,) and y_pred.shape == (0,)


def test_predictions_are_aligned_to_test_indices_in_order(y: np.ndarray) -> None:
    """Results are positional: entry ``i`` corresponds to ``test_indices[i]``,
    whatever order those indices arrive in."""
    out = run_eval(y, [NaiveModel()], [7, 3, 5])
    np.testing.assert_array_equal(out["naive"][0], [7.0, 3.0, 5.0])
    np.testing.assert_array_equal(out["naive"][1], [6.0, 2.0, 4.0])


def test_results_are_keyed_by_model_name_so_duplicates_collapse(
    y: np.ndarray,
) -> None:
    """Two models sharing a name yield one entry — the later one wins.

    Not obviously desirable, but it is what the engine does, and a refactor
    changing it would silently alter the number of rows a run produces.
    """
    out = run_eval(y, [MovingAverageModel(3), MovingAverageModel(3)], [5])
    assert set(out) == {"ma3"}
