"""Fixtures shared across test modules.

Only things several modules genuinely need: the on-disk prediction schema and
the synthetic series the model tests fit. Anything used by one module stays in
that module, where it can be read next to the assertion it serves.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def write_prediction_csv(
    path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Path:
    """Write one prediction CSV in the schema ``src.evaluate`` emits.

    The schema — ``idx, y_true, y_pred``, one row per scored step — is the part
    every caller shares. How the two series are generated is the part that
    differs per test, so it stays with the test.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    pd.DataFrame({
        "idx": np.arange(y_true.size, dtype=int),
        "y_true": y_true,
        "y_pred": y_pred,
    }).to_csv(path, index=False)
    return path


def noisy_predictions(
    n: int = 60, *, sigma: float = 0.01, noise: float = 0.005, seed: int = 0
):
    """A series and a forecast of it that is right on average but not exactly.

    Returns ``(y_true, y_pred)``. Good enough for any test that needs plausible
    prediction data and does not care about the numbers themselves.
    """
    rng = np.random.default_rng(seed)
    y_true = rng.normal(0.0, sigma, size=n)
    return y_true, y_true + rng.normal(0.0, noise, size=n)


def mean_predictions(n: int = 80, *, sigma: float = 0.02, seed: int = 0):
    """A series predicted by its own mean, so RMSE equals its standard deviation.

    Returns ``(y_true, y_pred)``. Used where a test needs the error to be a
    known quantity rather than an arbitrary one.
    """
    rng = np.random.default_rng(seed)
    y = rng.normal(0.0, sigma, n)
    return y, np.full(n, y.mean())


def make_ar1(
    n: int = 200, phi: float = 0.5, sigma: float = 1.0, seed: int = 42
) -> np.ndarray:
    """Length-``n`` AR(1) series ``y_t = phi * y_{t-1} + eps_t``.

    The one process in this project with real autocorrelation, so it is what
    the model tests use to show ARMA can beat a naive forecast at all.
    """
    rng = np.random.default_rng(seed)
    eps = rng.normal(loc=0.0, scale=sigma, size=n)
    y = np.zeros(n, dtype=float)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + eps[t]
    return y


__all__ = [
    "write_prediction_csv",
    "noisy_predictions",
    "mean_predictions",
    "make_ar1",
]
