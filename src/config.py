"""Central configuration for the forecasting pipeline.

Edit these constants to change behavior without touching pipeline code.

Conventions
-----------
* ``t = 1 day`` (business-day frequency). All windows and sample limits are
  expressed in trading days, not calendar days.
* Every downstream module (``src.runner``, ``src.rolling``, ``src.plots``)
  should import its tunables from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Tier specification (mirrors ``src.selection.TierSpec`` byte-for-byte)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TierSpec:
    """Per-tier selection thresholds.

    Source of truth for the three-tier price-band definition consumed by
    :func:`src.selection.select_tickers_for_tier` and the runner.
    """

    start_price: float
    end_price: float
    below_threshold: float
    upper_threshold: float
    target_count: int


# ---------------------------------------------------------------------------
# Data — sample period
# ---------------------------------------------------------------------------

#: ISO start date for ``yfinance`` downloads. Begins early enough that the
#: longest-window models (MA(90)/ARMA(90)) have a full window on test-day 1.
START_DATE: str = "2022-08-15"

#: ISO end date for ``yfinance`` downloads.
END_DATE: str = "2023-12-31"

#: Scored test period: predictions on dates strictly before ``TEST_START_DATE``
#: are used as warm-up only and are NOT included in metrics.
TEST_START_DATE: str = "2023-01-01"
TEST_END_DATE: str = "2023-12-31"

#: Cap each per-ticker series to its last ``MAX_SAMPLE_DAYS`` trading days.
#: ``None`` means use the full downloaded history.
MAX_SAMPLE_DAYS: int | None = None

# ---------------------------------------------------------------------------
# Tier bands & selection (LOCKED — see measurement_pipeline.md §3, §8)
# ---------------------------------------------------------------------------

#: Locked per-tier mean-band ranges, outlier thresholds, and target counts.
#: ``below = start_price / 2``, ``upper = 2 × end_price``. Tier 3 has no
#: upper bound on the mean band, so ``end_price=1e9`` is used as a sentinel.
#: Broadened tier bands with no outlier thresholds: a stock belongs to a
#: tier purely by mean adjusted price falling in the band over the sample.
#: ``below_threshold = 0`` and ``upper_threshold = 1e9`` make the per-day
#: outlier check trivially pass.
TIERS: Dict[str, TierSpec] = {
    "tier1": TierSpec(start_price=0.0,   end_price=30.0,    below_threshold=0.0, upper_threshold=1.0e9, target_count=30),
    "tier2": TierSpec(start_price=30.0,  end_price=100.0,   below_threshold=0.0, upper_threshold=1.0e9, target_count=30),
    "tier3": TierSpec(start_price=100.0, end_price=1.0e9,   below_threshold=0.0, upper_threshold=1.0e9, target_count=30),
}

#: Path to the candidate-universe text file (one symbol per line).
CANDIDATE_UNIVERSE_FILE: str = "data/universe/russell3000.txt"

#: Default per-tier desired count. CLI ``--target`` overrides per-run.
TARGET_PER_TIER: int = 30

#: Reproducibility seed for ticker selection. ``None`` => non-deterministic.
RUN_SEED: Optional[int] = None

#: SQLite cache for ticker price history (Phase A storage layer).
TICKER_DB_PATH: str = "ticker_data/cache.db"

#: Root directory beneath which each test run materialises its
#: ``test_<ISO_TIMESTAMP>/`` bundle.
TEST_RUN_ROOT: str = "results/test_runs"

# ---------------------------------------------------------------------------
# Forecasting protocol — per-model lookbacks
# ---------------------------------------------------------------------------

#: Step size between forecasts in trading days. ``1`` = one-step-ahead.
ROLLING_STEP: int = 1

#: Lookback lengths (trading days) for ``MovingAverageModel``. The MA(s)
#: forecast at step t is mean(y[t-s : t]).
MA_LOOKBACKS: Tuple[int, ...] = (30, 60, 90)

#: Lookback lengths (trading days) for ``ARMAModel``. ARMA fits on the past
#: L returns at every step.
ARMA_LOOKBACKS: Tuple[int, ...] = (60, 90)

# ---------------------------------------------------------------------------
# ARMA model
# ---------------------------------------------------------------------------

#: Inclusive upper bound for the AR order ``p`` searched over via AIC.
#: Reduced from 4 → 2 to keep the 300-ticker run within a reasonable budget;
#: empirically AIC almost never picks p > 2 on log returns anyway.
ARMA_MAX_P: int = 2

#: Inclusive upper bound for the MA order ``q`` searched over via AIC.
ARMA_MAX_Q: int = 2

#: Re-run the AIC grid search every ``ARMA_REFIT_EVERY`` steps. In between,
#: the cached best ``(p, q)`` is reused.
ARMA_REFIT_EVERY: int = 40

# ---------------------------------------------------------------------------
# Output paths (relative to the project root)
# ---------------------------------------------------------------------------

DATA_DIR: str = "data"
RESULTS_DIR: str = "results"
PREDICTIONS_DIR: str = "results/predictions"
FIGURES_DIR: str = "results/figures"
FIGURES_DATA_DIR: str = "results/figures/data"

# ---------------------------------------------------------------------------
# Convenience: validate at import time so a bad edit fails fast.
# ---------------------------------------------------------------------------

assert ROLLING_STEP >= 1, "ROLLING_STEP must be a positive integer."
assert ARMA_MAX_P >= 0 and ARMA_MAX_Q >= 0, "ARMA orders must be non-negative."
assert all(L >= 1 for L in MA_LOOKBACKS), "MA lookbacks must be positive."
assert all(L >= 1 for L in ARMA_LOOKBACKS), "ARMA lookbacks must be positive."
assert TEST_START_DATE >= START_DATE, "TEST_START_DATE must not precede START_DATE."
assert TEST_END_DATE <= END_DATE, "TEST_END_DATE must not exceed END_DATE."
assert all(spec.target_count > 0 for spec in TIERS.values())
assert all(spec.start_price < spec.end_price for spec in TIERS.values())
assert TARGET_PER_TIER > 0
