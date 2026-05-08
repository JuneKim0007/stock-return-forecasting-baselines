"""SQLite-backed cache for ticker price history.

Implements the storage layer described in
``measurement_pipeline.md`` §7. The cache holds two tables:

- ``ticker_meta`` — one row per cached symbol with the period it covers
  and basic price summary stats.
- ``ticker_prices`` — one row per (symbol, ISO date) with the adjusted
  close price.

The module deliberately keeps its surface small. It exposes only the
five functions documented in the spec; everything else is private.

Constraints (Phase A):
- stdlib only (``sqlite3``, ``datetime``, ``os``) plus pandas / numpy
- no prints, no I/O outside the path passed in
- ``init_schema`` is idempotent
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Schema (verbatim from measurement_pipeline.md §7.3, with ``IF NOT EXISTS``
# so init_schema is idempotent).
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ticker_meta (
    symbol           TEXT PRIMARY KEY,
    tier             TEXT NOT NULL,
    period_start     TEXT NOT NULL,
    period_end       TEXT NOT NULL,
    mean_price       REAL,
    min_price        REAL,
    max_price        REAL,
    fetched_at       TEXT NOT NULL,
    source           TEXT NOT NULL DEFAULT 'yfinance'
);

CREATE TABLE IF NOT EXISTS ticker_prices (
    symbol           TEXT NOT NULL,
    date             TEXT NOT NULL,
    adj_close        REAL NOT NULL,
    PRIMARY KEY (symbol, date),
    FOREIGN KEY (symbol) REFERENCES ticker_meta(symbol)
);

CREATE INDEX IF NOT EXISTS idx_prices_symbol ON ticker_prices(symbol);
"""


# ---------------------------------------------------------------------------
# Connection / schema helpers
# ---------------------------------------------------------------------------


def open_db(path: str) -> sqlite3.Connection:
    """Open (or create) the SQLite database at ``path``.

    Ensures the parent directory exists so the caller doesn't have to
    pre-create ``ticker_data/``. Foreign keys are enabled because
    ``ticker_prices`` references ``ticker_meta``.
    """
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Create the cache tables and index if they do not already exist.

    Safe to call repeatedly; uses ``CREATE ... IF NOT EXISTS`` for every
    object so re-initializing an existing DB is a no-op.
    """
    with conn:
        conn.executescript(_SCHEMA_SQL)


# ---------------------------------------------------------------------------
# Read path
# ---------------------------------------------------------------------------


def get_history(
    conn: sqlite3.Connection,
    symbol: str,
    start: str,
    end: str,
) -> Optional[pd.Series]:
    """Return the cached adj-close series for ``symbol`` covering ``[start, end]``.

    Returns ``None`` on cache miss, defined as:
    - no row in ``ticker_meta`` for ``symbol`` whose ``period_start`` and
      ``period_end`` exactly match the requested window, OR
    - no rows in ``ticker_prices`` for that symbol.

    On hit, returns a ``pd.Series`` named ``"adj_close"`` indexed by a
    ``DatetimeIndex`` with float64 values, sorted by date.
    """
    meta_row = conn.execute(
        "SELECT period_start, period_end FROM ticker_meta WHERE symbol = ?;",
        (symbol,),
    ).fetchone()
    if meta_row is None:
        return None

    period_start, period_end = meta_row
    if period_start != start or period_end != end:
        return None

    rows = conn.execute(
        "SELECT date, adj_close FROM ticker_prices WHERE symbol = ? ORDER BY date ASC;",
        (symbol,),
    ).fetchall()
    if not rows:
        return None

    dates = pd.to_datetime([r[0] for r in rows])
    values = np.asarray([r[1] for r in rows], dtype=np.float64)
    series = pd.Series(values, index=pd.DatetimeIndex(dates), name="adj_close")
    return series


# ---------------------------------------------------------------------------
# Write path
# ---------------------------------------------------------------------------


def _coerce_iso_date(ts: object) -> str:
    """Coerce a value coming from a DatetimeIndex element to an ISO date string."""
    if isinstance(ts, str):
        # Already a string; trust the caller — but normalize via pandas to be safe.
        return pd.Timestamp(ts).strftime("%Y-%m-%d")
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def put_history(
    conn: sqlite3.Connection,
    symbol: str,
    tier: str,
    start: str,
    end: str,
    prices: pd.Series,
    source: str = "yfinance",
) -> None:
    """Upsert ``symbol``'s meta row and bulk-replace its price rows.

    The series index must be a ``DatetimeIndex``; values are coerced to
    floats and dates to ISO ``YYYY-MM-DD`` strings on write. The whole
    operation runs inside a single transaction so a re-put is atomic and
    leaves no stale rows behind.
    """
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError(
            "put_history expects a pandas Series with a DatetimeIndex; "
            f"got index type {type(prices.index).__name__}"
        )

    # Compute summary stats from the *float-coerced* values so callers
    # passing ints / numpy scalars don't trip up downstream consumers.
    values = np.asarray(prices.to_numpy(), dtype=np.float64)
    if values.size == 0:
        mean_price: Optional[float] = None
        min_price: Optional[float] = None
        max_price: Optional[float] = None
    else:
        mean_price = float(np.mean(values))
        min_price = float(np.min(values))
        max_price = float(np.max(values))

    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    price_rows = [
        (symbol, _coerce_iso_date(idx), float(val))
        for idx, val in zip(prices.index, values)
    ]

    with conn:  # single transaction
        # Upsert meta row keyed by symbol PK.
        conn.execute(
            """
            INSERT INTO ticker_meta
                (symbol, tier, period_start, period_end,
                 mean_price, min_price, max_price, fetched_at, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                tier         = excluded.tier,
                period_start = excluded.period_start,
                period_end   = excluded.period_end,
                mean_price   = excluded.mean_price,
                min_price    = excluded.min_price,
                max_price    = excluded.max_price,
                fetched_at   = excluded.fetched_at,
                source       = excluded.source;
            """,
            (
                symbol,
                tier,
                start,
                end,
                mean_price,
                min_price,
                max_price,
                fetched_at,
                source,
            ),
        )
        # Wipe any stale rows for this symbol so a re-put with a shorter
        # series doesn't leave orphan dates lying around.
        conn.execute("DELETE FROM ticker_prices WHERE symbol = ?;", (symbol,))
        if price_rows:
            conn.executemany(
                "INSERT OR REPLACE INTO ticker_prices (symbol, date, adj_close) "
                "VALUES (?, ?, ?);",
                price_rows,
            )


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------


def list_cached_symbols(
    conn: sqlite3.Connection,
    tier: Optional[str] = None,
) -> List[str]:
    """Return cached symbols, sorted ascending. Optionally tier-filtered."""
    if tier is None:
        rows = conn.execute(
            "SELECT symbol FROM ticker_meta ORDER BY symbol ASC;"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT symbol FROM ticker_meta WHERE tier = ? ORDER BY symbol ASC;",
            (tier,),
        ).fetchall()
    return [r[0] for r in rows]
