"""Phase B — Step 1: Ticker selection per tier.

Implements the algorithm in ``measurement_pipeline.md`` §4.2:

    1. Naive-first-look: sample without replacement from the candidate
       universe, seeded by ``random.Random(seed)``.
    2. Cheap pre-filter on today's close in the relaxed window
       ``[start*1.1, end*1.1]``.
    3. History fetch — try the SQLite cache first, fall through to an
       injected ``history_loader`` (real callers wire this to yfinance).
       On a real fetch the history is persisted via ``put_history`` so the
       next run is a cache hit.
    4. Mean-band check: ``start <= mean(adj_close) <= end``.
    5. Outlier rejection: any close < ``below_threshold`` or
       > ``upper_threshold`` rejects the ticker outright.

The selector NEVER calls yfinance directly. It only uses the two injected
callables (``history_loader`` / ``current_price_fn``) — both have safe
default implementations that delegate to the existing ``src.data`` and
``yfinance`` modules. Tests inject fakes and never touch the network.
"""

from __future__ import annotations

import logging
import random
import sqlite3
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from src.config import TierSpec
from src.storage.db import get_history, put_history

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclass — re-exported from ``src.config`` so there is exactly one
# definition of ``TierSpec`` in the codebase. Existing imports
# ``from src.selection import TierSpec`` keep working.
# ---------------------------------------------------------------------------


__all__ = ["TierSpec", "select_tickers_for_tier"]


# ---------------------------------------------------------------------------
# Internal helpers (unit-testable in isolation)
# ---------------------------------------------------------------------------


def _naive_first_look(
    universe: List[str],
    attempted: set,
    rng: random.Random,
) -> Optional[str]:
    """Return a fresh candidate from ``universe`` not yet in ``attempted``.

    Uses the supplied ``random.Random`` instance — never the global RNG —
    so seeded selections are reproducible across processes.
    Returns ``None`` when the universe has been exhausted.
    """
    remaining = [s for s in universe if s not in attempted]
    if not remaining:
        return None
    return rng.choice(remaining)


def _passes_pre_filter(spec: TierSpec, px_now: Optional[float]) -> bool:
    """Quick cheap check on today's close.

    The window is ``[start * 1.1, end * 1.1]`` per spec §3 item 2 —
    a 10% relaxation that lets a ticker drifting near the band edge still
    qualify for the (more expensive) full history check.
    """
    if px_now is None:
        return False
    if not np.isfinite(px_now):
        return False
    lo = spec.start_price * 1.1
    hi = spec.end_price * 1.1
    return lo <= float(px_now) <= hi


def _passes_mean_band(spec: TierSpec, prices: pd.Series) -> bool:
    """Mean of adj_close must lie inside ``[start_price, end_price]``."""
    if prices is None or len(prices) == 0:
        return False
    mean_px = float(prices.mean())
    return spec.start_price <= mean_px <= spec.end_price


def _passes_outlier_check(spec: TierSpec, prices: pd.Series) -> bool:
    """Reject if any single close is < below_threshold or > upper_threshold."""
    if prices is None or len(prices) == 0:
        return False
    arr = np.asarray(prices.to_numpy(), dtype=np.float64)
    if (arr < spec.below_threshold).any():
        return False
    if (arr > spec.upper_threshold).any():
        return False
    return True


def _load_or_fetch_history(
    conn: sqlite3.Connection,
    symbol: str,
    tier_name: str,
    period_start: str,
    period_end: str,
    history_loader: Callable[[str, str, str], Optional[pd.Series]],
) -> Optional[pd.Series]:
    """Cache-first history loader.

    1. Hit the DB via ``get_history``. If that returns a series, use it
       directly — no fetch, no write.
    2. On miss, call the injected ``history_loader``. If it returns a
       non-empty series, persist via ``put_history`` and return it.
    3. On loader failure or empty result, return ``None``.
    """
    cached = get_history(conn, symbol, period_start, period_end)
    if cached is not None and len(cached) > 0:
        return cached

    fetched = history_loader(symbol, period_start, period_end)
    if fetched is None or len(fetched) == 0:
        return None

    # Defensive: the loader contract is "Series indexed by date". If a caller
    # gives us something with a non-DatetimeIndex, coerce — put_history will
    # raise otherwise.
    if not isinstance(fetched.index, pd.DatetimeIndex):
        fetched = pd.Series(
            np.asarray(fetched.to_numpy(), dtype=np.float64),
            index=pd.to_datetime(fetched.index),
            name="adj_close",
        )

    put_history(
        conn,
        symbol,
        tier=tier_name,
        start=period_start,
        end=period_end,
        prices=fetched,
    )
    return fetched


# ---------------------------------------------------------------------------
# Default injected callables (real callers wire these in)
# ---------------------------------------------------------------------------


def _default_history_loader(
    symbol: str, period_start: str, period_end: str
) -> Optional[pd.Series]:
    """Production fallback: download via ``src.data.download_prices``.

    Returns a Series indexed by date with adj_close floats, or ``None`` on
    any failure (network error, empty download, etc).
    """
    try:
        from src.data import download_prices  # local import to avoid cycle

        df = download_prices(symbol, period_start, period_end)
        if df is None or df.empty or "adj_close" not in df.columns:
            return None
        idx = pd.to_datetime(df["date"])
        values = np.asarray(df["adj_close"].to_numpy(), dtype=np.float64)
        return pd.Series(values, index=pd.DatetimeIndex(idx), name="adj_close")
    except Exception:
        return None


def _default_current_price(symbol: str) -> Optional[float]:
    """Production fallback: yfinance fast lookup. Catches all exceptions."""
    try:
        import yfinance as yf  # local import — tests never reach this

        ticker = yf.Ticker(symbol)
        # ``fast_info`` is the cheapest call that returns a current price.
        fi = getattr(ticker, "fast_info", None)
        if fi is not None:
            px = getattr(fi, "last_price", None)
            if px is not None and np.isfinite(float(px)):
                return float(px)
        # Fallback: pull last close from a 5-day history.
        hist = ticker.history(period="5d", auto_adjust=False)
        if hist is None or hist.empty or "Close" not in hist.columns:
            return None
        last = float(hist["Close"].iloc[-1])
        if not np.isfinite(last):
            return None
        return last
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def select_tickers_for_tier(
    tier_name: str,
    spec: TierSpec,
    universe: List[str],
    conn: sqlite3.Connection,
    *,
    seed: Optional[int] = None,
    max_attempts: int = 200,
    history_loader: Optional[Callable[[str, str, str], Optional[pd.Series]]] = None,
    current_price_fn: Optional[Callable[[str], Optional[float]]] = None,
    period_start: str = "2022-08-15",
    period_end: str = "2023-12-31",
) -> List[str]:
    """Select ``spec.target_count`` tickers from ``universe`` that satisfy
    the tier's pre-filter, mean-band, and outlier checks.

    Parameters
    ----------
    tier_name
        Logical tier label (``"tier1"`` / ``"tier2"`` / ``"tier3"``). Used
        as the cache row's ``tier`` column on writes.
    spec
        ``TierSpec`` with start/end prices, outlier thresholds, and target
        count.
    universe
        Pool of ticker symbols to draw from (already deduplicated).
    conn
        Open SQLite connection with the cache schema initialized.
    seed
        Reproducibility seed for ``random.Random``. ``None`` means
        non-deterministic (uses Python's default seeding).
    max_attempts
        Hard cap on the number of candidates inspected before giving up.
        On shortfall (cap hit OR universe drained before ``target_count``
        accepts), logs a warning and returns the partial list. Never raises.
    history_loader
        Injected ``(symbol, start, end) -> Optional[Series]``. Defaults to
        :func:`_default_history_loader` which uses ``src.data.download_prices``.
    current_price_fn
        Injected ``(symbol) -> Optional[float]``. Defaults to
        :func:`_default_current_price` (yfinance fast lookup).
    period_start, period_end
        ISO-date window for the historical sample. Must match the cache key.

    Returns
    -------
    list[str]
        Up to ``spec.target_count`` accepted ticker symbols, in selection
        order. May be shorter (or empty) on shortfall — see ``max_attempts``.
    """
    if history_loader is None:
        history_loader = _default_history_loader
    if current_price_fn is None:
        current_price_fn = _default_current_price

    rng = random.Random(seed)
    selected: List[str] = []
    attempted: set = set()
    attempts = 0

    while len(selected) < spec.target_count:
        if attempts >= max_attempts:
            logger.warning(
                "%s: %d / %d selected — %s",
                tier_name,
                len(selected),
                spec.target_count,
                "max_attempts reached",
            )
            return selected

        candidate = _naive_first_look(universe, attempted, rng)
        if candidate is None:
            # Universe drained before target_count — caller should grow the
            # universe or relax thresholds. Warn and return what we have.
            logger.warning(
                "%s: %d / %d selected — %s",
                tier_name,
                len(selected),
                spec.target_count,
                "universe exhausted",
            )
            return selected
        attempted.add(candidate)
        attempts += 1

        # --- step 1: pre-filter on current price -------------------------
        try:
            px_now = current_price_fn(candidate)
        except Exception:
            px_now = None
        if not _passes_pre_filter(spec, px_now):
            continue

        # --- step 2: history (cache or loader) ---------------------------
        history = _load_or_fetch_history(
            conn,
            candidate,
            tier_name,
            period_start,
            period_end,
            history_loader,
        )
        if history is None:
            continue

        # --- step 3: mean-band check -------------------------------------
        if not _passes_mean_band(spec, history):
            continue

        # --- step 4: outlier rejection -----------------------------------
        if not _passes_outlier_check(spec, history):
            continue

        selected.append(candidate)

    return selected
