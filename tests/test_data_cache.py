"""Tests for the cache-aware behaviour of ``src.data.load_returns``.

These tests exercise the three-layer fall-through (DB cache hit → yfinance
fetch + cache write → legacy CSV fallback) introduced in Phase C without
making any real network calls — ``download_prices`` is monkey-patched in
every test that exercises a yfinance path.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

import src.data as data_mod
from src.data import load_returns
from src.storage.db import get_history, init_schema, open_db, put_history


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


PERIOD_START = "2020-01-06"
PERIOD_END = "2020-01-17"


@pytest.fixture()
def conn(tmp_path):
    """Fresh on-disk SQLite cache, schema initialized, closed after the test."""
    db_path = str(tmp_path / "ticker_data" / "cache.db")
    c = open_db(db_path)
    init_schema(c)
    try:
        yield c
    finally:
        c.close()


def _make_price_series(
    start: str = PERIOD_START, n: int = 10, base: float = 50.0
) -> pd.Series:
    """Build a small business-day price series with a DatetimeIndex."""
    idx = pd.date_range(start=start, periods=n, freq="B")
    values = base + np.arange(n, dtype=np.float64)
    return pd.Series(values, index=idx, name="adj_close")


def _series_to_download_frame(prices: pd.Series) -> pd.DataFrame:
    """Match the ``download_prices`` return shape (date, adj_close)."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(prices.index),
            "adj_close": prices.to_numpy(dtype=float),
        }
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 1) Cache HIT must NOT call yfinance
# ---------------------------------------------------------------------------


def test_load_returns_db_hit_avoids_yfinance(monkeypatch, conn) -> None:
    """Pre-seed the DB and ensure ``download_prices`` is never invoked."""
    series = _make_price_series(start=PERIOD_START, n=10, base=42.0)
    put_history(
        conn, "TEST", tier="tier2", start=PERIOD_START, end=PERIOD_END, prices=series
    )

    def _boom(*_args, **_kwargs):  # pragma: no cover — must not fire
        raise AssertionError("yfinance was called")

    monkeypatch.setattr(data_mod, "download_prices", _boom)

    out = load_returns(
        "TEST",
        conn=conn,
        period_start=PERIOD_START,
        period_end=PERIOD_END,
    )

    assert not out.empty
    # Legacy columns + ``t`` from concat_trading_days (concat=True default).
    for col in ("date", "adj_close", "log_return", "t"):
        assert col in out.columns, f"missing column: {col}"
    # First NaN-row was dropped by compute_log_returns.
    assert len(out) == len(series) - 1
    # Values match the seeded price series.
    np.testing.assert_allclose(
        out["adj_close"].to_numpy(dtype=float),
        series.to_numpy(dtype=float)[1:],
    )


# ---------------------------------------------------------------------------
# 2) Cache MISS must fetch via yfinance and persist into the cache
# ---------------------------------------------------------------------------


def test_load_returns_db_miss_calls_yfinance_and_persists(monkeypatch, conn) -> None:
    series = _make_price_series(start=PERIOD_START, n=8, base=20.0)
    fake_frame = _series_to_download_frame(series)

    calls: list[tuple] = []

    def _fake_download(ticker, start, end):
        calls.append((ticker, start, end))
        return fake_frame

    monkeypatch.setattr(data_mod, "download_prices", _fake_download)

    out = load_returns(
        "MISS",
        conn=conn,
        period_start=PERIOD_START,
        period_end=PERIOD_END,
    )

    # download_prices was used exactly once with the right args.
    assert calls == [("MISS", PERIOD_START, PERIOD_END)]

    # Returned frame is correctly shaped and log returns are right.
    for col in ("date", "adj_close", "log_return", "t"):
        assert col in out.columns
    assert len(out) == len(series) - 1  # leading NaN dropped
    expected_log_return = np.log(series.to_numpy(dtype=float)[1:]) - np.log(
        series.to_numpy(dtype=float)[:-1]
    )
    np.testing.assert_allclose(
        out["log_return"].to_numpy(dtype=float), expected_log_return, atol=1e-12
    )

    # Cache is now populated for the same window.
    cached = get_history(conn, "MISS", PERIOD_START, PERIOD_END)
    assert cached is not None
    assert len(cached) == len(series)
    np.testing.assert_array_equal(cached.to_numpy(), series.to_numpy())


# ---------------------------------------------------------------------------
# 3) ``refresh=True`` must bypass the cache and overwrite it with new data
# ---------------------------------------------------------------------------


def test_load_returns_refresh_bypasses_cache(monkeypatch, conn) -> None:
    # Pre-populate the cache with a stale series.
    stale = _make_price_series(start=PERIOD_START, n=8, base=10.0)
    put_history(
        conn,
        "REFRESH",
        tier="tier1",
        start=PERIOD_START,
        end=PERIOD_END,
        prices=stale,
    )

    # New (different) values that yfinance will "return".
    fresh = _make_price_series(start=PERIOD_START, n=8, base=200.0)
    fresh_frame = _series_to_download_frame(fresh)

    calls: list[tuple] = []

    def _fake_download(ticker, start, end):
        calls.append((ticker, start, end))
        return fresh_frame

    monkeypatch.setattr(data_mod, "download_prices", _fake_download)

    out = load_returns(
        "REFRESH",
        conn=conn,
        refresh=True,
        period_start=PERIOD_START,
        period_end=PERIOD_END,
    )

    assert calls == [("REFRESH", PERIOD_START, PERIOD_END)]
    # Returned values reflect the FRESH series (not the stale cache).
    np.testing.assert_allclose(
        out["adj_close"].to_numpy(dtype=float),
        fresh.to_numpy(dtype=float)[1:],
    )
    # And the cache itself now holds the fresh series.
    cached = get_history(conn, "REFRESH", PERIOD_START, PERIOD_END)
    assert cached is not None
    np.testing.assert_array_equal(cached.to_numpy(), fresh.to_numpy())


# ---------------------------------------------------------------------------
# 4) Legacy CSV path must still work when no DB connection is supplied
# ---------------------------------------------------------------------------


def test_load_returns_legacy_csv_path_still_works(tmp_path) -> None:
    """No ``conn`` kwarg → behave exactly like the original implementation."""
    ticker = "LEGACY"
    csv_path = tmp_path / f"{ticker}.csv"

    # Synthesize a CSV that matches the existing on-disk format:
    # header = date, adj_close, log_return; date as YYYY-MM-DD strings;
    # log_return already computed and the leading NaN row dropped.
    series = _make_price_series(start=PERIOD_START, n=6, base=15.0)
    log_ret = np.log(series.to_numpy(dtype=float)[1:]) - np.log(
        series.to_numpy(dtype=float)[:-1]
    )
    csv_df = pd.DataFrame(
        {
            "date": pd.to_datetime(series.index[1:]).strftime("%Y-%m-%d"),
            "adj_close": series.to_numpy(dtype=float)[1:],
            "log_return": log_ret,
        }
    )
    csv_df.to_csv(csv_path, index=False)

    out = load_returns(ticker, data_dir=str(tmp_path))

    for col in ("date", "adj_close", "log_return", "t"):
        assert col in out.columns, f"missing column: {col}"
    assert len(out) == len(csv_df)
    np.testing.assert_allclose(
        out["adj_close"].to_numpy(dtype=float),
        series.to_numpy(dtype=float)[1:],
    )
    np.testing.assert_allclose(
        out["log_return"].to_numpy(dtype=float), log_ret, atol=1e-12
    )
    # Trading-day step column is contiguous from 0.
    assert out["t"].tolist() == list(range(len(out)))
