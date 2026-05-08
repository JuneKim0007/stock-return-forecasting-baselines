"""Per-ticker evaluation helper.

Provides the single function the runner actually needs:
``run_one_ticker_eval`` drives ``src.rolling.run_eval`` for one ticker,
computes the post-hoc ensemble (children = expanding + 3 MAs + 2 ARMAs),
writes per-model prediction CSVs, and returns one metric row per model.

``_estimate_arma_cost`` is reused by the runner's cost-gating block.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.metrics import mae, rmse
from src.models import ForecasterProtocol
from src.models import default_models as _default_models
from src.rolling import run_eval


ENSEMBLE_NAME: str = "ensemble"
#: Children that contribute to the post-hoc ensemble (excludes naive + global).
ENSEMBLE_CHILDREN: Tuple[str, ...] = (
    "expanding", "ma30", "ma60", "ma90", "arma60", "arma90",
)

# Empirical ARMA timing constants (seconds), used by the runner's cost gate.
ARMA_FULL_SEARCH_LARGE_WINDOW_SECONDS: float = 1.6   # window >= 90
ARMA_FULL_SEARCH_SMALL_WINDOW_SECONDS: float = 1.1   # window < 90
ARMA_CACHED_REFIT_STEP_SECONDS: float = 0.005


def _setup_logger(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("evaluate")
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def _estimate_arma_cost(n_steps: int, window: int) -> float:
    """Rough ARMA-fit cost estimate over ``n_steps`` rolling forecasts.

    Sized from earlier benchmarks: full AIC searches dominate, cached
    refits are cheap.
    """
    n_full_searches = max(1, n_steps // 20)
    per_search = (
        ARMA_FULL_SEARCH_LARGE_WINDOW_SECONDS
        if window >= 90
        else ARMA_FULL_SEARCH_SMALL_WINDOW_SECONDS
    )
    return n_full_searches * per_search + n_steps * ARMA_CACHED_REFIT_STEP_SECONDS


def run_one_ticker_eval(
    tier: str,
    ticker: str,
    *,
    df: pd.DataFrame,
    test_start: str,
    test_end: str,
    models: Optional[List[ForecasterProtocol]] = None,
    predictions_dir: Optional[str] = None,
    train_start_idx: int = 0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Drive the new-style evaluation for one ticker.

    ``df`` must have columns ``date`` (datetime64) and ``log_return``.
    Returns ``(metric_rows, per_model_arrays)``. If ``predictions_dir`` is
    given, persists ``<TICKER>_<MODEL>.csv`` files there.
    """
    if models is None:
        models = _default_models()

    dates = pd.to_datetime(df["date"]).values.astype("datetime64[D]")
    y_full = df["log_return"].to_numpy(dtype=float)

    test_start_dt = np.datetime64(test_start, "D")
    test_end_dt = np.datetime64(test_end, "D")
    test_mask = (dates >= test_start_dt) & (dates <= test_end_dt)
    test_indices = np.where(test_mask)[0]
    if test_indices.size == 0:
        return [], {}

    per_model = run_eval(
        y_full, models, test_indices, train_start=train_start_idx,
    )

    children = [n for n in ENSEMBLE_CHILDREN if n in per_model]
    if children:
        stacked = np.vstack([per_model[n][1] for n in children])
        ens_pred = np.nanmean(stacked, axis=0)
        ref_y_true = per_model[children[0]][0]
        per_model[ENSEMBLE_NAME] = (ref_y_true, ens_pred)

    rows: List[Dict[str, Any]] = []
    for name, (yt, yp) in per_model.items():
        rows.append({
            "tier": tier,
            "ticker": ticker,
            "model": name,
            "rmse": rmse(yt, yp),
            "mae": mae(yt, yp),
            "n": int(yt.size),
        })
        if predictions_dir is not None:
            os.makedirs(predictions_dir, exist_ok=True)
            csv_path = os.path.join(predictions_dir, f"{ticker}_{name}.csv")
            pd.DataFrame({
                "idx": test_indices,
                "y_true": yt.astype(float),
                "y_pred": yp.astype(float),
            }).to_csv(csv_path, index=False)
    return rows, per_model


__all__ = [
    "ENSEMBLE_NAME",
    "ENSEMBLE_CHILDREN",
    "run_one_ticker_eval",
    "_estimate_arma_cost",
    "_setup_logger",
]
