"""Tests for ``src.storage.db`` — SQLite-backed ticker price cache.

Each test uses a fresh on-disk DB inside ``tmp_path`` (pytest fixture) so
we exercise the real connection path, including parent-directory creation
and the WAL semantics of a real file. Nothing is mocked.
"""

from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.storage.db import (
    get_history,
    init_schema,
    list_cached_symbols,
    open_db,
    put_history,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_path(tmp_path) -> str:
    """An on-disk DB path inside tmp_path (under a nested dir, to exercise
    the parent-directory-creation branch in ``open_db``).
    """
    return str(tmp_path / "ticker_data" / "cache.db")


@pytest.fixture()
def conn(db_path: str):
    """Opened + schema-initialized connection. Closed after the test."""
    c = open_db(db_path)
    init_schema(c)
    try:
        yield c
    finally:
        c.close()


def _make_series(start: str = "2020-01-01", n: int = 5, base: float = 10.0) -> pd.Series:
    """Build a small business-day Series of floats with a DatetimeIndex."""
    idx = pd.date_range(start=start, periods=n, freq="B")
    values = base + np.arange(n, dtype=np.float64)
    return pd.Series(values, index=idx, name="adj_close")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_round_trip_put_then_get(conn: sqlite3.Connection) -> None:
    """put_history → get_history round-trips values and DatetimeIndex."""
    series = _make_series(start="2020-01-06", n=5, base=12.5)
    start, end = "2020-01-06", "2020-01-10"

    put_history(conn, "AAA", tier="tier1", start=start, end=end, prices=series)

    got = get_history(conn, "AAA", start, end)
    assert got is not None
    assert got.name == "adj_close"
    assert isinstance(got.index, pd.DatetimeIndex)
    assert got.dtype == np.float64
    # Values & index match the original (compare normalized to dates).
    np.testing.assert_array_equal(got.to_numpy(), series.to_numpy())
    assert list(got.index.strftime("%Y-%m-%d")) == list(series.index.strftime("%Y-%m-%d"))


def test_cache_miss_returns_none(conn: sqlite3.Connection) -> None:
    """Both 'never inserted' and 'inserted but wrong period' are misses."""
    # 1) Symbol that was never inserted.
    assert get_history(conn, "NEVER", "2020-01-01", "2020-12-31") is None

    # 2) Symbol present, but for a different (start, end).
    series = _make_series(start="2020-01-06", n=3)
    put_history(conn, "BBB", tier="tier1", start="2020-01-06", end="2020-01-08", prices=series)

    # Same symbol, different window → miss.
    assert get_history(conn, "BBB", "2021-01-01", "2021-12-31") is None
    # Sanity: the matching window IS a hit.
    hit = get_history(conn, "BBB", "2020-01-06", "2020-01-08")
    assert hit is not None
    assert len(hit) == 3


def test_re_put_overwrites_cleanly(conn: sqlite3.Connection) -> None:
    """A second put for the same symbol replaces both meta and prices.

    The new series is *shorter* and has different values, which catches
    bugs where we'd append rather than overwrite.
    """
    first = _make_series(start="2020-01-06", n=5, base=10.0)
    put_history(conn, "CCC", tier="tier1", start="2020-01-06", end="2020-01-10", prices=first)

    second_idx = pd.date_range(start="2020-02-03", periods=3, freq="B")
    second = pd.Series([100.0, 200.0, 300.0], index=second_idx, name="adj_close")
    put_history(conn, "CCC", tier="tier1", start="2020-02-03", end="2020-02-05", prices=second)

    # Old window is no longer a hit.
    assert get_history(conn, "CCC", "2020-01-06", "2020-01-10") is None

    got = get_history(conn, "CCC", "2020-02-03", "2020-02-05")
    assert got is not None
    assert len(got) == 3  # no leftover rows from the first put
    np.testing.assert_array_equal(got.to_numpy(), np.array([100.0, 200.0, 300.0]))

    # And exactly 3 rows live in ticker_prices for this symbol.
    n_rows = conn.execute(
        "SELECT COUNT(*) FROM ticker_prices WHERE symbol = ?;", ("CCC",)
    ).fetchone()[0]
    assert n_rows == 3


def test_index_exists(conn: sqlite3.Connection) -> None:
    """``init_schema`` creates ``idx_prices_symbol`` and it shows up in sqlite_master."""
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'index' AND name = ?;",
        ("idx_prices_symbol",),
    ).fetchone()
    assert row is not None
    assert row[0] == "idx_prices_symbol"

    # And init_schema is idempotent.
    init_schema(conn)
    init_schema(conn)
    row2 = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'index' AND name = ?;",
        ("idx_prices_symbol",),
    ).fetchone()
    assert row2 is not None


def test_list_cached_symbols_tier_filter(conn: sqlite3.Connection) -> None:
    """Inserting 2 tier1 + 1 tier2 → tier filter returns the right slices."""
    s = _make_series(start="2020-01-06", n=2)

    put_history(conn, "TONE", tier="tier1", start="2020-01-06", end="2020-01-07", prices=s)
    put_history(conn, "AONE", tier="tier1", start="2020-01-06", end="2020-01-07", prices=s)
    put_history(conn, "BTWO", tier="tier2", start="2020-01-06", end="2020-01-07", prices=s)

    all_symbols = list_cached_symbols(conn)
    assert all_symbols == ["AONE", "BTWO", "TONE"]  # sorted ascending

    tier1 = list_cached_symbols(conn, tier="tier1")
    assert tier1 == ["AONE", "TONE"]

    tier2 = list_cached_symbols(conn, tier="tier2")
    assert tier2 == ["BTWO"]

    tier3 = list_cached_symbols(conn, tier="tier3")
    assert tier3 == []


def test_open_db_creates_parent_directory(tmp_path) -> None:
    """``open_db`` creates a missing parent dir rather than erroring."""
    nested = tmp_path / "deep" / "nested" / "cache.db"
    assert not nested.parent.exists()
    c = open_db(str(nested))
    try:
        init_schema(c)
        assert nested.parent.is_dir()
        assert os.path.exists(str(nested))
    finally:
        c.close()
