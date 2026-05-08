"""Bulk persistence helpers for the analysis tables."""

from __future__ import annotations

import sqlite3
from typing import Any, Dict, List

import pandas as pd


_SUMMARY_COLS = (
    "test_run", "tier", "ticker", "window", "model",
    "n_steps", "mse", "rmse", "mae",
    "sq_err_var", "sq_err_median", "sq_err_max", "sq_err_min",
    "abs_err_var", "abs_err_median", "abs_err_max", "abs_err_min",
)


def persist_summary(
    conn: sqlite3.Connection,
    test_run: str,
    summaries: List[Dict[str, Any]],
) -> None:
    """Idempotently INSERT OR REPLACE one row per (tier, ticker, window, model)."""
    if not summaries:
        return
    rows = []
    for s in summaries:
        row = (
            test_run,
            s["tier"], s["ticker"], int(s["window"]), s["model"],
            int(s["n_steps"]), s["mse"], s["rmse"], s["mae"],
            s.get("sq_err_var"), s.get("sq_err_median"),
            s.get("sq_err_max"), s.get("sq_err_min"),
            s.get("abs_err_var"), s.get("abs_err_median"),
            s.get("abs_err_max"), s.get("abs_err_min"),
        )
        rows.append(row)
    placeholders = ",".join(["?"] * len(_SUMMARY_COLS))
    sql = (
        f"INSERT OR REPLACE INTO analysis_summary "
        f"({','.join(_SUMMARY_COLS)}) VALUES ({placeholders})"
    )
    with conn:
        conn.executemany(sql, rows)


def persist_per_step(
    conn: sqlite3.Connection,
    test_run: str,
    tier: str,
    ticker: str,
    window: int,
    model: str,
    per_step: pd.DataFrame,
) -> None:
    """Bulk-insert per-step rows in 10k chunks, idempotent via PRIMARY KEY."""
    if per_step.empty:
        return
    sql = (
        "INSERT OR REPLACE INTO analysis_per_step "
        "(test_run, tier, ticker, window, model, step_idx, sq_err, abs_err) "
        "VALUES (?,?,?,?,?,?,?,?)"
    )
    rows_iter = (
        (test_run, tier, ticker, int(window), model,
         int(r.step_idx), float(r.sq_err), float(r.abs_err))
        for r in per_step.itertuples(index=False)
    )
    chunk: List[tuple] = []
    with conn:
        for row in rows_iter:
            chunk.append(row)
            if len(chunk) >= 10_000:
                conn.executemany(sql, chunk)
                chunk = []
        if chunk:
            conn.executemany(sql, chunk)


__all__ = ["persist_summary", "persist_per_step"]
