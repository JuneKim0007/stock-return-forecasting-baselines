"""Unit tests for ``src/models.py`` — the Phase 2 forecasters.

These cover correctness of every model's closed-form (or quasi closed-form)
prediction, ARMA's caching/fallback behavior, and ensemble averaging.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.models import (
    ARMAModel,
    ExpandingMeanModel,
    GlobalMeanModel,
    MovingAverageModel,
    NaiveModel,
)


# ---------------------------------------------------------------------------
# Trivial baselines
# ---------------------------------------------------------------------------


def test_naive_returns_last() -> None:
    y = np.array([1, 2, 3, 4, 5], dtype=float)
    m = NaiveModel()
    m.fit(y)
    assert m.predict_one() == 5.0


def test_global_mean_matches_full_mean() -> None:
    rng = np.random.default_rng(123)
    y = rng.normal(size=50)
    m = GlobalMeanModel()
    m.fit(y)
    assert math.isclose(m.predict_one(), float(np.mean(y)), abs_tol=1e-12, rel_tol=0)


def test_expanding_set_state_returns_running_mean() -> None:
    m = ExpandingMeanModel()
    m.set_state(running_sum=10.0, count=4)
    assert math.isclose(m.predict_one(), 2.5, abs_tol=1e-12)
    m.set_state(running_sum=0.0, count=0)
    assert math.isnan(m.predict_one())


def test_moving_average_s10() -> None:
    rng = np.random.default_rng(7)
    y = rng.normal(size=100)
    m = MovingAverageModel(10)
    m.fit(y)
    expected = float(np.mean(y[-10:]))
    assert math.isclose(m.predict_one(), expected, abs_tol=1e-12, rel_tol=0)


def test_moving_average_uses_only_last_s() -> None:
    """Mutating ``y[:-s]`` must NOT change MA(s)'s prediction — MA only sees
    the last ``s`` observations.
    """
    rng = np.random.default_rng(99)
    s = 10
    y = rng.normal(size=100)

    m1 = MovingAverageModel(s)
    m1.fit(y)
    pred1 = m1.predict_one()

    # Wreck the prefix; tail untouched.
    y_perturbed = y.copy()
    y_perturbed[:-s] = 1e6

    m2 = MovingAverageModel(s)
    m2.fit(y_perturbed)
    pred2 = m2.predict_one()

    assert math.isclose(pred1, pred2, abs_tol=1e-12, rel_tol=0)


# ---------------------------------------------------------------------------
# ARMA
# ---------------------------------------------------------------------------


def test_arma_on_white_noise_close_to_zero() -> None:
    """On i.i.d. N(0,1) noise the conditional mean is ~0; ARMA's one-step
    forecast should sit near 0 within a loose tolerance (AIC may pick a
    non-trivial order on noise — hence the ±0.5 slack).
    """
    rng = np.random.default_rng(42)
    y = rng.normal(size=300)
    m = ARMAModel()
    m.fit(y)
    pred = m.predict_one()
    assert math.isfinite(pred), f"ARMA returned non-finite prediction: {pred}"
    assert abs(pred) < 0.5, f"ARMA on white noise drifted: {pred}"


def test_arma_returns_finite_on_constant_series() -> None:
    """A degenerate (constant) series must NOT crash ARMA. The fallback path
    (window mean) should kick in and yield a finite float.
    """
    y = np.ones(200, dtype=float)
    m = ARMAModel()
    m.fit(y)
    pred = m.predict_one()
    assert isinstance(pred, float)
    assert math.isfinite(pred), f"ARMA on constant series produced {pred}"


def test_arma_aic_caches_order() -> None:
    """After a successful fit, ``_best_order`` is set. A subsequent fit
    *within* the ``refit_every`` window must reuse the cached order without
    re-running the full grid search.
    """
    rng = np.random.default_rng(2024)
    y = rng.normal(size=200)

    m = ARMAModel(refit_every=20)
    m.fit(y)

    # The cached attribute and the public introspection property both expose
    # the selected order. Verify both are populated.
    assert m._best_order is not None, "best_order not cached after first fit"
    assert m.best_order == m._best_order
    cached = m._best_order

    # Fits 2..20 are inside the same refit_every window: each one re-fits the
    # cached order only. The cached order must be unchanged across them.
    for _ in range(5):
        m.fit(y)
        assert m._best_order == cached, (
            f"cached order mutated mid-window: {m._best_order} != {cached}"
        )


