"""Price download, log returns, and the per-ticker load path.

Functions
---------
download_prices(ticker, start, end) -> DataFrame[date, adj_close]
compute_log_returns(prices) -> DataFrame[date, adj_close, log_return]
concat_trading_days(df) -> DataFrame  (adds the integer trading-day index)
load_returns(ticker, ...) -> DataFrame  (cache-first, used by the runner)

Tier assignment is not here: ``src.selection`` decides it from the price bands
in ``src.config.TIERS``, so there is one definition of where a tier begins.

Conventions
-----------
* Daily business-day frequency (yfinance default).
* Price field: ``Adj Close`` from ``yf.download(auto_adjust=False)``; falls
  back to ``Close`` of ``auto_adjust=True`` if Adj Close is unavailable in the
  installed yfinance version.
* Per-ticker CSV columns: ``date, adj_close, log_return`` — date is ISO
  ``YYYY-MM-DD`` strings, the two value columns are float64. The first row's
  ``log_return`` (the NaN row) is dropped before persisting.
"""

from __future__ import annotations

import os
import sqlite3
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from src.storage.db import get_history, put_history


# ---------------------------------------------------------------------------
# Download + transform
# ---------------------------------------------------------------------------

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance can return a MultiIndex on columns when ``group_by='ticker'``
    or when a single ticker is passed in newer versions. Flatten so we can
    refer to ``Adj Close`` / ``Close`` by simple string names.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # If one level has only a single unique value (the ticker), drop it.
        levels_to_drop = [
            i for i, lvl in enumerate(df.columns.levels) if len(lvl) == 1
        ]
        for i in sorted(levels_to_drop, reverse=True):
            df.columns = df.columns.droplevel(i)
        if isinstance(df.columns, pd.MultiIndex):
            # Still a MultiIndex — join the levels with a space.
            df.columns = [" ".join([str(c) for c in tup]).strip()
                          for tup in df.columns.values]
    return df


def download_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download a single ticker's daily prices from Yahoo Finance.

    Returns a DataFrame with two columns: ``date`` (datetime64[ns]) and
    ``adj_close`` (float64). Returns an empty DataFrame on failure.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
    except Exception as exc:  # pragma: no cover — network errors
        print(f"[download_prices] {ticker}: download error: {exc}")
        return pd.DataFrame(columns=["date", "adj_close"])

    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "adj_close"])

    df = _flatten_columns(df)

    if "Adj Close" in df.columns:
        prices = df["Adj Close"].astype(float)
    elif "Close" in df.columns:
        prices = df["Close"].astype(float)
    else:
        # Fallback path: re-download with auto_adjust=True and use Close.
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df2 = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                )
            df2 = _flatten_columns(df2)
            if df2 is None or df2.empty or "Close" not in df2.columns:
                return pd.DataFrame(columns=["date", "adj_close"])
            prices = df2["Close"].astype(float)
        except Exception as exc:  # pragma: no cover
            print(f"[download_prices] {ticker}: fallback failed: {exc}")
            return pd.DataFrame(columns=["date", "adj_close"])

    out = pd.DataFrame({
        "date": pd.to_datetime(prices.index),
        "adj_close": prices.to_numpy(dtype=float),
    }).reset_index(drop=True)
    out = out.dropna(subset=["adj_close"]).reset_index(drop=True)
    return out


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Append a ``log_return`` column (log P_t − log P_{t−1}) and drop the
    first NaN row. Input must contain ``date`` and ``adj_close``.
    """
    if prices.empty or "adj_close" not in prices.columns:
        return prices.assign(log_return=pd.Series(dtype=float))
    out = prices.copy()
    out["log_return"] = np.log(out["adj_close"]).diff()
    out = out.dropna(subset=["log_return"]).reset_index(drop=True)
    return out


def concat_trading_days(df: pd.DataFrame) -> pd.DataFrame:
    """Treat the trading-day series as a contiguous sequence.

    Stock markets are closed on weekends and US public holidays, so the date
    column already arrives as a sparse "trading-day calendar" — there are no
    rows for non-trading days, and we never forward-fill them (that would
    inject zero-return days that don't exist and bias variance downward).

    This helper makes the convention explicit:
        1. Sort ascending by ``date``.
        2. Drop any row whose ``adj_close`` or ``log_return`` is NaN.
        3. Drop duplicate dates (keep first).
        4. Reset the index and add an integer ``t`` column = 0, 1, 2, …
           — the trading-day step number used by every downstream model.

    A 3-day weekend's worth of price movement is absorbed into the next
    trading day's log return; that is the standard convention in financial
    econometrics. Apply this to every series before fitting any model so
    every phase agrees on what "t = 1 day" means.

    Parameters
    ----------
    df : DataFrame
        Must contain ``date`` and ``adj_close``; ``log_return`` is optional
        (added by :func:`compute_log_returns`).

    Returns
    -------
    DataFrame
        Same columns plus an integer ``t`` column on a fresh RangeIndex.
    """
    if df.empty:
        return df.assign(t=pd.Series(dtype=int))
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"])
        out = out.sort_values("date", kind="mergesort")
        out = out.drop_duplicates(subset=["date"], keep="first")
    drop_cols = [c for c in ("adj_close", "log_return") if c in out.columns]
    if drop_cols:
        out = out.dropna(subset=drop_cols)
    out = out.reset_index(drop=True)
    out["t"] = np.arange(len(out), dtype=int)
    return out


# ---------------------------------------------------------------------------
# Helper used by Phases 2–5
# ---------------------------------------------------------------------------

def _frame_from_prices_series(prices: pd.Series) -> pd.DataFrame:
    """Build a ``date, adj_close, log_return`` DataFrame from a price Series.

    Drops the leading NaN row produced by the diff so the output matches the
    legacy CSV layout exactly.
    """
    if prices is None or len(prices) == 0:
        return pd.DataFrame(
            {
                "date": pd.Series(dtype="datetime64[ns]"),
                "adj_close": pd.Series(dtype=float),
                "log_return": pd.Series(dtype=float),
            }
        )
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(prices.index),
            "adj_close": prices.to_numpy(dtype=float),
        }
    ).reset_index(drop=True)
    return compute_log_returns(frame)


def load_returns(
    ticker: str,
    data_dir: str = "data",
    concat: bool = True,
    *,
    conn: Optional[sqlite3.Connection] = None,
    refresh: bool = False,
    legacy_csv_fallback: bool = True,
    period_start: Optional[str] = None,
    period_end: Optional[str] = None,
) -> pd.DataFrame:
    """Load a per-ticker price/return series.

    Resolution order (first match wins):

    1. **DB cache hit** — when ``conn`` is supplied and ``refresh`` is False,
       look up ``get_history(conn, ticker, period_start, period_end)``. On
       hit, build the legacy frame in memory.
    2. **yfinance + cache write** — when ``conn`` is supplied and either the
       cache missed or ``refresh=True``, call :func:`download_prices`,
       persist via :func:`put_history`, and return the same legacy frame.
    3. **Legacy CSV fallback** — when ``conn is None`` and
       ``legacy_csv_fallback`` is True, read ``<data_dir>/<ticker>.csv``
       exactly as the original implementation did.
    4. Otherwise, raise ``FileNotFoundError``.

    Returns columns ``date`` (datetime64[ns]), ``adj_close`` (float64),
    ``log_return`` (float64), and (when ``concat=True``, the default) an
    integer ``t`` column from :func:`concat_trading_days` so the series is
    treated as a contiguous trading-day sequence with no calendar gaps.
    """
    df: Optional[pd.DataFrame] = None

    if conn is not None:
        # Resolve the requested window from config defaults when missing.
        if period_start is None or period_end is None:
            from src import config as _cfg  # local import — avoid cycles
            if period_start is None:
                period_start = _cfg.START_DATE
            if period_end is None:
                period_end = _cfg.END_DATE

        if not refresh:
            cached = get_history(conn, ticker, period_start, period_end)
            if cached is not None and len(cached) > 0:
                df = _frame_from_prices_series(cached)

        if df is None:
            # Cache miss (or forced refresh) — fetch and persist.
            raw = download_prices(ticker, period_start, period_end)
            if raw is None or raw.empty:
                raise FileNotFoundError(
                    f"load_returns: no data for {ticker!r} in "
                    f"[{period_start}, {period_end}] (yfinance returned empty)"
                )
            prices_series = pd.Series(
                raw["adj_close"].to_numpy(dtype=float),
                index=pd.DatetimeIndex(pd.to_datetime(raw["date"])),
                name="adj_close",
            )
            # Tier is unknown at this layer — selection.py is what tags real
            # tier rows. Write an "unknown" placeholder for now.
            put_history(
                conn,
                ticker,
                tier="unknown",
                start=period_start,
                end=period_end,
                prices=prices_series,
                source="yfinance",
            )
            df = _frame_from_prices_series(prices_series)

    elif legacy_csv_fallback:
        path = os.path.join(data_dir, f"{ticker}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"load_returns: no CSV at {path!r} and no DB connection given"
            )
        df = pd.read_csv(path, parse_dates=["date"])
        df["adj_close"] = df["adj_close"].astype(float)
        df["log_return"] = df["log_return"].astype(float)

    if df is None:
        raise FileNotFoundError(
            f"load_returns: could not resolve data for {ticker!r} "
            f"(no DB connection and legacy CSV fallback disabled)"
        )

    if concat:
        df = concat_trading_days(df)
    return df


# ---------------------------------------------------------------------------
# Legacy CLI placeholder — replaced by `python -m src.runner`.
# ---------------------------------------------------------------------------

def main() -> None:
    print("Legacy CLI removed — use `python -m src.runner` instead.")


if __name__ == "__main__":
    main()
