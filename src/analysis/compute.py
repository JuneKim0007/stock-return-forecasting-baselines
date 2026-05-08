"""Pure-pandas error-distribution statistics for the analysis stage."""

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
    """
    df = pd.read_csv(predictions_csv)
    err = df["y_true"].astype(float) - df["y_pred"].astype(float)
    out = pd.DataFrame(
        {
            "step_idx": np.arange(len(df), dtype=int),
            "sq_err": (err * err).to_numpy(dtype=float),
            "abs_err": err.abs().to_numpy(dtype=float),
        }
    )
    return out


def compute_summary(per_step: pd.DataFrame) -> Dict[str, Any]:
    """Distribution stats over a per-step error frame produced by
    :func:`compute_per_step`. Variances are population variances (ddof=0).
    """
    sq = per_step["sq_err"].to_numpy(dtype=float)
    ab = per_step["abs_err"].to_numpy(dtype=float)
    n = int(sq.size)
    mse = float(sq.mean()) if n else float("nan")
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
