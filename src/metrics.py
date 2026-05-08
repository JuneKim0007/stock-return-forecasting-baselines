"""Phase 4 — Error metrics.

Two metrics, both operating on equally-shaped 1-D numpy arrays of
``y_true`` and ``y_pred``:

* :func:`rmse` — root mean squared error.
* :func:`mae`  — mean absolute error.

Both raise :class:`ValueError` when the inputs disagree on shape.
"""

from __future__ import annotations

import numpy as np


def _check_shapes(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if yt.shape != yp.shape:
        raise ValueError(
            f"shape mismatch: y_true.shape={yt.shape} vs y_pred.shape={yp.shape}"
        )
    if yt.size == 0:
        raise ValueError("metrics require at least one observation; got empty arrays")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error.

    sqrt(mean((y_true - y_pred) ** 2))
    """
    _check_shapes(y_true, y_pred)
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error.

    mean(|y_true - y_pred|)
    """
    _check_shapes(y_true, y_pred)
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(yt - yp)))


__all__ = ["rmse", "mae"]
