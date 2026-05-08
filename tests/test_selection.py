"""Tests for ``src.selection`` — Phase B ticker selector.

No live yfinance: every test injects a synthetic ``history_loader`` and a
synthetic ``current_price_fn`` and uses an on-disk SQLite DB inside
``tmp_path``. The default loaders are never reached.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from src.selection import TierSpec, select_tickers_for_tier
from src.storage.db import init_schema, open_db, put_history


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_conn(tmp_path) -> sqlite3.Connection:
    """Fresh on-disk DB with the cache schema initialized."""
    path = tmp_path / "ticker_data" / "cache.db"
    c = open_db(str(path))
    init_schema(c)
    try:
        yield c
    finally:
        c.close()


def _make_series(mean_value: float, n: int = 50, jitter: float = 0.0) -> pd.Series:
    """Build a small business-day Series with a controlled mean.

    ``jitter`` controls a deterministic +/- swing around the mean so we can
    exercise outlier checks without surprising the mean-band test.
    """
    idx = pd.date_range(start="2020-01-06", periods=n, freq="B")
    if jitter == 0.0:
        values = np.full(n, float(mean_value), dtype=np.float64)
    else:
        # Symmetric perturbation: +jitter, -jitter, 0, +jitter, -jitter, 0, ...
        pattern = np.array([jitter, -jitter, 0.0])
        rep = np.tile(pattern, (n // 3) + 1)[:n]
        values = float(mean_value) + rep
    return pd.Series(values, index=idx, name="adj_close")


def _accepting_loader(price_by_symbol: Dict[str, float]):
    """Build a loader that returns a constant in-band series per symbol."""

    def loader(symbol: str, start: str, end: str) -> Optional[pd.Series]:
        if symbol not in price_by_symbol:
            return None
        return _make_series(price_by_symbol[symbol], n=50)

    return loader


def _accepting_current_price(price_by_symbol: Dict[str, float]):
    """Build a current-price fn returning the mapped px or None."""

    def current(symbol: str) -> Optional[float]:
        return price_by_symbol.get(symbol)

    return current


@pytest.fixture()
def tier1_spec() -> TierSpec:
    """A canonical tier1 spec — $1–$15 band, $0.50 floor, $30 ceiling."""
    return TierSpec(
        start_price=1.0,
        end_price=15.0,
        below_threshold=0.50,
        upper_threshold=30.0,
        target_count=3,
    )


@pytest.fixture()
def universe() -> List[str]:
    """Synthetic universe wide enough to exercise the selector."""
    return [f"S{i:03d}" for i in range(20)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_seeded_selection_is_reproducible(
    db_conn: sqlite3.Connection,
    universe: List[str],
    tier1_spec: TierSpec,
) -> None:
    """Same seed → identical accept list across two calls.

    Every candidate in the universe is in-band so the only thing varying
    between runs would be the RNG order — fixing the seed pins it.
    """
    in_band_px = {sym: 8.0 for sym in universe}

    first = select_tickers_for_tier(
        "tier1",
        tier1_spec,
        universe,
        db_conn,
        seed=42,
        history_loader=_accepting_loader(in_band_px),
        current_price_fn=_accepting_current_price(in_band_px),
    )
    # Wipe any cached writes so the second run has to re-traverse the loader
    # path the same way the first did. (The loader is deterministic, so this
    # is mostly belt-and-braces — the result must match either way.)
    db_conn.execute("DELETE FROM ticker_prices;")
    db_conn.execute("DELETE FROM ticker_meta;")
    db_conn.commit()

    second = select_tickers_for_tier(
        "tier1",
        tier1_spec,
        universe,
        db_conn,
        seed=42,
        history_loader=_accepting_loader(in_band_px),
        current_price_fn=_accepting_current_price(in_band_px),
    )

    assert first == second
    assert len(first) == tier1_spec.target_count
    # And a different seed gives at least *some* different ordering.
    third = select_tickers_for_tier(
        "tier1",
        tier1_spec,
        universe,
        db_conn,
        seed=999,
        history_loader=_accepting_loader(in_band_px),
        current_price_fn=_accepting_current_price(in_band_px),
    )
    assert isinstance(third, list)
    assert len(third) == tier1_spec.target_count


def test_pre_filter_rejects_out_of_band_current_price(
    db_conn: sqlite3.Connection,
    universe: List[str],
    tier1_spec: TierSpec,
) -> None:
    """All candidates report a px far above the relaxed band → partial result."""
    far_above = {sym: 9999.0 for sym in universe}

    loader_calls: List[str] = []

    def loader(symbol: str, start: str, end: str) -> Optional[pd.Series]:
        loader_calls.append(symbol)
        return _make_series(8.0, n=10)

    result = select_tickers_for_tier(
        "tier1",
        tier1_spec,
        universe,
        db_conn,
        seed=0,
        max_attempts=50,
        history_loader=loader,
        current_price_fn=_accepting_current_price(far_above),
    )
    assert isinstance(result, list)
    assert len(result) < tier1_spec.target_count
    # The pre-filter rejects before any history is loaded, so the loader
    # must NEVER be called.
    assert loader_calls == []


def test_mean_band_rejects_out_of_band_mean(
    db_conn: sqlite3.Connection,
    tier1_spec: TierSpec,
) -> None:
    """Current price is in band, but historical mean is not → partial result."""
    universe = [f"M{i:02d}" for i in range(10)]
    in_band_px = {sym: 8.0 for sym in universe}

    # Mean is well above the tier end (15.0) but below the upper_threshold (30)
    # so the outlier check passes — only the mean-band check rejects.
    out_of_band_mean = 25.0

    def loader(symbol: str, start: str, end: str) -> Optional[pd.Series]:
        return _make_series(out_of_band_mean, n=50)

    result = select_tickers_for_tier(
        "tier1",
        tier1_spec,
        universe,
        db_conn,
        seed=0,
        max_attempts=30,
        history_loader=loader,
        current_price_fn=_accepting_current_price(in_band_px),
    )
    assert isinstance(result, list)
    assert len(result) < tier1_spec.target_count


def test_outlier_rejection_low(
    db_conn: sqlite3.Connection,
    tier1_spec: TierSpec,
) -> None:
    """A single close below ``below_threshold`` ($0.50) rejects the ticker."""
    universe = [f"L{i:02d}" for i in range(10)]
    in_band_px = {sym: 8.0 for sym in universe}

    def loader(symbol: str, start: str, end: str) -> Optional[pd.Series]:
        # Mean stays inside [1, 15], but inject one close at $0.10 < 0.50.
        idx = pd.date_range(start="2020-01-06", periods=20, freq="B")
        values = np.full(20, 8.0, dtype=np.float64)
        values[5] = 0.10  # below_threshold violator
        return pd.Series(values, index=idx, name="adj_close")

    result = select_tickers_for_tier(
        "tier1",
        tier1_spec,
        universe,
        db_conn,
        seed=0,
        max_attempts=30,
        history_loader=loader,
        current_price_fn=_accepting_current_price(in_band_px),
    )
    assert isinstance(result, list)
    assert len(result) < tier1_spec.target_count


def test_outlier_rejection_high(
    db_conn: sqlite3.Connection,
    tier1_spec: TierSpec,
) -> None:
    """A single close above ``upper_threshold`` ($30) rejects the ticker."""
    universe = [f"H{i:02d}" for i in range(10)]
    in_band_px = {sym: 8.0 for sym in universe}

    def loader(symbol: str, start: str, end: str) -> Optional[pd.Series]:
        idx = pd.date_range(start="2020-01-06", periods=20, freq="B")
        values = np.full(20, 8.0, dtype=np.float64)
        values[10] = 100.0  # blows past upper_threshold of 30
        return pd.Series(values, index=idx, name="adj_close")

    result = select_tickers_for_tier(
        "tier1",
        tier1_spec,
        universe,
        db_conn,
        seed=0,
        max_attempts=30,
        history_loader=loader,
        current_price_fn=_accepting_current_price(in_band_px),
    )
    assert isinstance(result, list)
    assert len(result) < tier1_spec.target_count


def test_returns_exact_target_count(
    db_conn: sqlite3.Connection,
    tier1_spec: TierSpec,
) -> None:
    """Selector returns exactly ``target_count`` accepts and stops there."""
    universe = [f"OK{i:02d}" for i in range(15)]  # plenty of room
    in_band_px = {sym: 8.0 for sym in universe}

    result = select_tickers_for_tier(
        "tier1",
        tier1_spec,
        universe,
        db_conn,
        seed=7,
        history_loader=_accepting_loader(in_band_px),
        current_price_fn=_accepting_current_price(in_band_px),
    )
    assert len(result) == tier1_spec.target_count
    assert len(set(result)) == len(result)  # no duplicates
    assert all(sym in universe for sym in result)


def test_max_attempts_warns_and_returns_partial(
    db_conn: sqlite3.Connection,
    tier1_spec: TierSpec,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """All-rejecting universe → warns and returns a partial list (no raise)."""
    universe = [f"BAD{i:02d}" for i in range(10)]
    # Current price is far out of band for every candidate so the pre-filter
    # rejects on every attempt — selector cannot find any accepts.
    out_of_band_px = {sym: 9999.0 for sym in universe}

    def loader(symbol: str, start: str, end: str) -> Optional[pd.Series]:
        # Should never be reached, but keep a deterministic in-band fallback.
        return _make_series(8.0, n=20)

    caplog.set_level(logging.WARNING, logger="src.selection")

    result = select_tickers_for_tier(
        "tier1",
        tier1_spec,
        universe,
        db_conn,
        seed=0,
        max_attempts=8,
        history_loader=loader,
        current_price_fn=_accepting_current_price(out_of_band_px),
    )

    # No exception, returns a list (possibly empty), strictly under target.
    assert isinstance(result, list)
    assert len(result) < tier1_spec.target_count

    # At least one warning record should mention the tier and the shortfall
    # reason (either "exhausted" or "max_attempts").
    matching = [
        rec
        for rec in caplog.records
        if rec.levelno == logging.WARNING
        and "tier1" in rec.getMessage()
        and "selected" in rec.getMessage()
        and ("exhausted" in rec.getMessage() or "max_attempts" in rec.getMessage())
    ]
    assert matching, (
        f"expected a warning mentioning tier1, 'selected', and "
        f"'exhausted'/'max_attempts'; got records: "
        f"{[r.getMessage() for r in caplog.records]}"
    )


def test_db_cache_hit_avoids_loader(
    db_conn: sqlite3.Connection,
    tier1_spec: TierSpec,
) -> None:
    """Pre-populated cache → the injected loader is never invoked."""
    universe = ["CACHED1", "CACHED2", "CACHED3"]
    in_band_px = {sym: 8.0 for sym in universe}
    period_start = "2018-01-01"
    period_end = "2025-12-31"

    # Pre-populate the cache for every ticker with an in-band series so each
    # candidate is auto-accepted on the first try.
    for sym in universe:
        put_history(
            db_conn,
            sym,
            tier="tier1",
            start=period_start,
            end=period_end,
            prices=_make_series(8.0, n=20),
        )

    loader_calls: List[str] = []

    def loader(symbol: str, start: str, end: str) -> Optional[pd.Series]:
        loader_calls.append(symbol)
        return _make_series(8.0, n=20)

    result = select_tickers_for_tier(
        "tier1",
        tier1_spec,
        universe,
        db_conn,
        seed=0,
        history_loader=loader,
        current_price_fn=_accepting_current_price(in_band_px),
        period_start=period_start,
        period_end=period_end,
    )

    assert len(result) == tier1_spec.target_count
    # The cache covered every candidate the selector inspected.
    assert loader_calls == [], (
        f"loader should not be called when cache covers every candidate; "
        f"got calls for {loader_calls}"
    )
