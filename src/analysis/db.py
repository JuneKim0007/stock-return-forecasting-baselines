"""SQLite schema + open helper for the analysis stage.

Schema diverges from the original spec by dropping ``analysis_cross_measurement``
since the pipeline no longer runs N measurements — there is exactly one pass
per ``test_run`` so the per-stock summary IS the cross-measurement view.
"""

from __future__ import annotations

import os
import sqlite3


def open_analysis_db(path: str) -> sqlite3.Connection:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_analysis_schema(conn: sqlite3.Connection) -> None:
    with conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS analysis_per_step (
                test_run    TEXT NOT NULL,
                tier        TEXT NOT NULL,
                ticker      TEXT NOT NULL,
                window      INTEGER NOT NULL,
                model       TEXT NOT NULL,
                step_idx    INTEGER NOT NULL,
                sq_err      REAL NOT NULL,
                abs_err     REAL NOT NULL,
                PRIMARY KEY (test_run, tier, ticker, window, model, step_idx)
            );

            CREATE TABLE IF NOT EXISTS analysis_summary (
                test_run         TEXT NOT NULL,
                tier             TEXT NOT NULL,
                ticker           TEXT NOT NULL,
                window           INTEGER NOT NULL,
                model            TEXT NOT NULL,
                n_steps          INTEGER NOT NULL,
                mse              REAL NOT NULL,
                rmse             REAL NOT NULL,
                mae              REAL NOT NULL,
                sq_err_var       REAL,
                sq_err_median    REAL,
                sq_err_max       REAL,
                sq_err_min       REAL,
                abs_err_var      REAL,
                abs_err_median   REAL,
                abs_err_max      REAL,
                abs_err_min      REAL,
                PRIMARY KEY (test_run, tier, ticker, window, model)
            );

            CREATE INDEX IF NOT EXISTS idx_summary_tier_model
                ON analysis_summary(tier, model);
            """
        )


__all__ = ["open_analysis_db", "init_analysis_schema"]
