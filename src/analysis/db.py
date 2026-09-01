"""SQLite schema + open helper for the analysis stage.

Two divergences from the original spec, both because the pipeline changed
under it:

* ``analysis_cross_measurement`` is dropped — there is exactly one pass per
  ``test_run``, so the per-stock summary IS the cross-measurement view.
* ``window`` is dropped. It recorded the unified rolling window ``W``, which
  the pipeline retired in favour of per-model lookbacks; each model now carries
  its own, so the column was derivable from ``model`` and was never read.
  A database written by the pre-lookback pipeline therefore has an
  incompatible schema — see :func:`init_analysis_schema`.
"""

from __future__ import annotations

import sqlite3

from src.storage.db import open_db


def open_analysis_db(path: str) -> sqlite3.Connection:
    """Open (or create) the analysis database.

    Opening is the same operation as for the ticker cache — create the parent
    directory, connect, enable foreign keys — so it is not restated here. Only
    the schema differs, and that is :func:`init_analysis_schema`'s job.
    """
    return open_db(path)


class LegacyAnalysisSchemaError(RuntimeError):
    """Raised when the database predates the removal of the ``window`` column."""


def _has_legacy_window_column(conn: sqlite3.Connection) -> bool:
    """True when ``analysis_summary`` exists and still carries ``window``."""
    exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='analysis_summary';"
    ).fetchone()
    if exists is None:
        return False
    cols = {row[1] for row in conn.execute("PRAGMA table_info(analysis_summary);")}
    return "window" in cols


def init_analysis_schema(conn: sqlite3.Connection) -> None:
    """Create the analysis tables if absent.

    ``CREATE TABLE IF NOT EXISTS`` cannot alter a table that already exists, so
    a database written before ``window`` was dropped would silently keep the old
    shape and fail every insert on the missing NOT NULL column. Detect that and
    say so, rather than dropping rows the caller may still want.
    """
    if _has_legacy_window_column(conn):
        raise LegacyAnalysisSchemaError(
            "This analysis database was written by the pre-lookback pipeline "
            "and still has a 'window' column in its primary key. Its rows "
            "describe a scoring scheme this pipeline no longer uses. Delete "
            "the file (it is rebuilt from the prediction CSVs) or point "
            "--db-path elsewhere."
        )
    with conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS analysis_per_step (
                test_run    TEXT NOT NULL,
                tier        TEXT NOT NULL,
                ticker      TEXT NOT NULL,
                model       TEXT NOT NULL,
                step_idx    INTEGER NOT NULL,
                sq_err      REAL NOT NULL,
                abs_err     REAL NOT NULL,
                PRIMARY KEY (test_run, tier, ticker, model, step_idx)
            );

            CREATE TABLE IF NOT EXISTS analysis_summary (
                test_run         TEXT NOT NULL,
                tier             TEXT NOT NULL,
                ticker           TEXT NOT NULL,
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
                PRIMARY KEY (test_run, tier, ticker, model)
            );

            CREATE INDEX IF NOT EXISTS idx_summary_tier_model
                ON analysis_summary(tier, model);
            """
        )


__all__ = ["open_analysis_db", "init_analysis_schema", "LegacyAnalysisSchemaError"]
