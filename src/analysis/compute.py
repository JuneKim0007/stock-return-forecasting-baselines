"""Error-distribution statistics for the analysis stage.

``rmse`` and ``mae`` come from :mod:`src.metrics` so this stage reports the same
numbers as ``metrics.csv`` and ``summary_<tier>.csv``. The distribution stats
around them (variance, median, extremes) are computed here over the same finite
subset the metrics use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


def compute_per_step(predictions_csv: Path) -> pd.DataFrame:
    """Read a prediction CSV (cols: idx, y_true, y_pred) and return a frame
    with ``step_idx``, ``sq_err``, ``abs_err``. ``step_idx`` is the row order
    (0..N-1), independent of the absolute ``idx`` written by the rolling
    backtest.

    Steps whose forecast or observation is not finite are dropped, matching the
    nan policy in :mod:`src.metrics`; ``step_idx`` is assigned before the drop
    so it still identifies the original position in the series.
    """
    df = pd.read_csv(predictions_csv)
    yt = df["y_true"].to_numpy(dtype=float)
    yp = df["y_pred"].to_numpy(dtype=float)
    err = yt - yp
    out = pd.DataFrame(
        {
            "step_idx": np.arange(len(df), dtype=int),
            "sq_err": err * err,
            "abs_err": np.abs(err),
        }
    )
    return out[np.isfinite(yt) & np.isfinite(yp)].reset_index(drop=True)


def compute_summary(per_step: pd.DataFrame) -> Dict[str, Any]:
    """Distribution stats over a per-step error frame produced by
    :func:`compute_per_step`. Variances are population variances (ddof=0).
    """
    sq = per_step["sq_err"].to_numpy(dtype=float)
    ab = per_step["abs_err"].to_numpy(dtype=float)
    n = int(sq.size)
    mse = float(sq.mean()) if n else float("nan")
    # ``compute_per_step`` has already dropped the non-finite steps, so the
    # mean of sq_err here is the same quantity ``src.metrics.rmse`` squares.
    return {
        "n_steps": n,
        "mse": mse,
        "rmse": float(np.sqrt(mse)) if n else float("nan"),
        "mae": float(ab.mean()) if n else float("nan"),
        "sq_err_var": float(sq.var(ddof=0)) if n else None,
        "sq_err_median": float(np.median(sq)) if n else None,
        "sq_err_max": float(sq.max()) if n else None,
        "sq_err_min": float(sq.min()) if n else None,
        "abs_err_var": float(ab.var(ddof=0)) if n else None,
        "abs_err_median": float(np.median(ab)) if n else None,
        "abs_err_max": float(ab.max()) if n else None,
        "abs_err_min": float(ab.min()) if n else None,
    }


__all__ = ["compute_per_step", "compute_summary"]
