"""Forecasting driver — per-model lookbacks over a fixed test set.

The new pipeline retires the unified rolling-window W. Each model declares
its own ``lookback`` and ``kind`` attributes and the engine dispatches:

* ``naive``     — ``y_full[t-1]``
* ``global``    — ``mean(y_full)`` computed once before the loop
* ``expanding`` — running mean of ``y_full[train_start : t]``
* ``windowed``  — ``y_full[t - L : t]`` fed to ``fit`` + ``predict_one``

The legacy ``run_rolling`` and ``run_rolling_all_models`` are retained as
thin wrappers so older tests keep passing.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from src.models import (
    ExpandingMeanModel,
    Forecaster,
    ForecasterProtocol,
    GlobalMeanModel,
    NaiveModel,
)


def run_eval(
    y_full: np.ndarray,
    models: List[ForecasterProtocol],
    test_indices: Sequence[int],
    *,
    train_start: int = 0,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Drive every model over ``test_indices`` using its own lookback.

    Parameters
    ----------
    y_full : np.ndarray
        Full series including pre-test history. No NaNs.
    models : list of forecasters
        Each must expose ``name``, ``kind``, and (for ``windowed``) ``lookback``.
    test_indices : sequence of int
        Absolute indices into ``y_full`` at which to score predictions.
    train_start : int
        First index expanding-mean should include. Default 0.

    Returns
    -------
    dict {name: (y_true, y_pred)} aligned to ``test_indices``.
    """
    y_full = np.asarray(y_full, dtype=float).ravel()
    test_indices = np.asarray(test_indices, dtype=int)
    n_test = test_indices.size

    y_true = y_full[test_indices]
    preds: Dict[str, np.ndarray] = {m.name: np.empty(n_test, dtype=float) for m in models}

    # Pre-compute global means once.
    for m in models:
        kind = getattr(m, "kind", "windowed")
        if kind == "global":
            m.fit(y_full)

    # Initialise expanding-mean state with the prefix up to test_indices[0].
    cum_sum = float(np.sum(y_full[train_start : test_indices[0]])) if n_test else 0.0
    cum_n = int(test_indices[0] - train_start) if n_test else 0

    for i, t in enumerate(test_indices):
        for m in models:
            kind = getattr(m, "kind", "windowed")
            if kind == "naive":
                preds[m.name][i] = float(y_full[t - 1])
            elif kind == "global":
                preds[m.name][i] = float(m.predict_one())
            elif kind == "expanding":
                if isinstance(m, ExpandingMeanModel):
                    m.set_state(cum_sum, cum_n)
                    preds[m.name][i] = float(m.predict_one())
                else:
                    m.fit(y_full[train_start:t])
                    preds[m.name][i] = float(m.predict_one())
            else:  # windowed
                L = int(getattr(m, "lookback", 0))
                if L <= 0 or t - L < 0:
                    preds[m.name][i] = float("nan")
                    continue
                m.fit(y_full[t - L : t])
                preds[m.name][i] = float(m.predict_one())
        cum_sum += float(y_full[t])
        cum_n += 1

    return {name: (y_true, p) for name, p in preds.items()}


__all__ = ["run_eval"]
