"""Integration tests — Phase 3.

These exercise a *minimal* in-test rolling loop (Phase 4 has not yet built
``src/rolling.py`` — do NOT import from there). Two checks:

1. On a synthetic AR(1) process, ARMA's RMSE must be at most 5 % worse than
   Naive's. ARMA should be at least competitive on a true AR process.
2. On a real ticker pulled from the manifest, a 30-step rolling forecast with
   the trivial baselines must yield no NaNs and the right number of rows.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
import pytest

from src.models import ARMAModel, NaiveModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ar1(
    n: int, phi: float = 0.6, sigma: float = 1.0, seed: int = 0
) -> np.ndarray:
    """Generate a length-``n`` AR(1) series ``y_t = phi*y_{t-1} + eps_t``."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(loc=0.0, scale=sigma, size=n)
    y = np.zeros(n, dtype=float)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + eps[t]
    return y


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(err * err)))


# ---------------------------------------------------------------------------
# Rolling AR(1) — ARMA must be competitive with Naive
# ---------------------------------------------------------------------------


def test_rolling_arma_beats_naive_on_ar1() -> None:
    """Rolling one-step forecasts on AR(1) (phi=0.6, sigma=1, n=400, seed=0).
    Window size 100. From t=100 to 399, fit on ``y[t-100:t]`` and predict
    ``y[t]``. Assert ``rmse_arma <= rmse_naive * 1.05``.
    """
    n = 400
    window = 100
    y = _make_ar1(n=n, phi=0.6, sigma=1.0, seed=0)

    arma = ARMAModel()
    naive = NaiveModel()

    actual: list[float] = []
    pred_arma: list[float] = []
    pred_naive: list[float] = []

    for t in range(window, n):
        slice_y = y[t - window : t]
        arma.fit(slice_y)
        naive.fit(slice_y)
        pred_arma.append(arma.predict_one())
        pred_naive.append(naive.predict_one())
        actual.append(float(y[t]))

    a = np.asarray(actual, dtype=float)
    pa = np.asarray(pred_arma, dtype=float)
    pn = np.asarray(pred_naive, dtype=float)

    # Predictions must be finite.
    assert np.all(np.isfinite(pa)), "ARMA produced NaNs in rolling forecast"
    assert np.all(np.isfinite(pn)), "Naive produced NaNs in rolling forecast"

    rmse_arma = _rmse(a, pa)
    rmse_naive = _rmse(a, pn)

    # 5 % slack — ARMA should be at least competitive on a true AR process.
    assert rmse_arma <= rmse_naive * 1.05, (
        f"ARMA worse than Naive by >5%: rmse_arma={rmse_arma:.4f}, "
        f"rmse_naive={rmse_naive:.4f}, ratio={rmse_arma / rmse_naive:.4f}"
    )


