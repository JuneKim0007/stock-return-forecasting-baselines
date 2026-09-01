"""Error metrics, and the project's one answer to what ``nan`` means for them.

Two metrics over equally-shaped 1-D arrays of ``y_true`` and ``y_pred``:
:func:`rmse` and :func:`mae`.

NaN policy
----------
A forecast may legitimately be ``nan``: ``src.models`` sanctions it for a
prediction made before any fit and for an ARMA whose every candidate order
failed, and the post-hoc ensemble already averages with ``np.nanmean`` so that
one ``nan`` child does not poison the result.

Both metrics therefore score **only the pairs where both sides are finite**,
and callers report how many survived — the ``n`` column in ``metrics.csv``,
``n_steps`` in the analysis tables. A model that could not predict every step
still gets a usable score, and the shortfall stays visible in the count rather
than being hidden inside the metric.

This lived in three places with three different answers, so the same
(ticker, model) was reported one way in ``metrics.csv``, another in
``summary_<tier>.csv`` and a third in ``analysis.db``. It is stated here once.

Both metrics raise :class:`ValueError` when the inputs disagree on shape, when
they are empty, and when no pair is finite — the last because "nothing could be
scored" must stay distinguishable from "the model was perfect", which a silent
``0.0`` would not.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _finite_pairs(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate the inputs and return only the pairs that can be scored.

    Raises
    ------
    ValueError
        If the shapes disagree, if the arrays are empty, or if no pair has a
        finite value on both sides.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError(
            f"shape mismatch: y_true.shape={yt.shape} vs y_pred.shape={yp.shape}"
        )
    if yt.size == 0:
        raise ValueError("metrics require at least one observation; got empty arrays")

    mask = np.isfinite(yt) & np.isfinite(yp)
    if not mask.any():
        raise ValueError(
            "metrics require at least one finite (y_true, y_pred) pair; "
            "no finite pairs in an array of size "
            f"{yt.size}"
        )
    return yt[mask], yp[mask]


def n_scored(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    """How many pairs :func:`rmse` and :func:`mae` would actually score.

    Callers record this alongside the metric so a shortfall is visible in the
    output rather than hidden inside it.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return int((np.isfinite(yt) & np.isfinite(yp)).sum())


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error over the finite pairs.

    sqrt(mean((y_true - y_pred) ** 2))
    """
    yt, yp = _finite_pairs(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error over the finite pairs.

    mean(|y_true - y_pred|)
    """
    yt, yp = _finite_pairs(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


__all__ = ["rmse", "mae", "n_scored"]
