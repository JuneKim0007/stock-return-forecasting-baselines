"""Unit tests for ``src/data.py`` — log-return computation, tier
classification, and the persisted manifest.
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd
import pytest

from src.data import classify_tier, compute_log_returns


# ---------------------------------------------------------------------------
# compute_log_returns
# ---------------------------------------------------------------------------


def test_compute_log_returns_matches_formula() -> None:
    """Each ``log_return`` row must equal ``log(P_t) - log(P_{t-1})`` to
    1e-12, and the first NaN row must be dropped.
    """
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
            ),
            "adj_close": [100.0, 101.0, 99.0, 102.0],
        }
    )

    out = compute_log_returns(prices)

    # The first row (NaN log_return) must be dropped.
    assert len(out) == 3
    # The earliest retained date must be the SECOND original date — not the
    # first — confirming the NaN row was dropped from the front.
    assert out["date"].iloc[0] == pd.Timestamp("2024-01-03")

    expected: List[float] = [
        float(np.log(101.0) - np.log(100.0)),
        float(np.log(99.0) - np.log(101.0)),
        float(np.log(102.0) - np.log(99.0)),
    ]
    actual = out["log_return"].to_numpy(dtype=float)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


# ---------------------------------------------------------------------------
# classify_tier
# ---------------------------------------------------------------------------


def test_classify_tier() -> None:
    """Verify tier classification at the canonical points.

    The reference (per src/data.py): boundaries ``<= 10`` → tier1,
    ``<= 100`` → tier2, otherwise tier3. Note that 10 → tier1 and
    100 → tier2 here (inclusive upper edges); this is a deliberate Phase-1
    convention and is documented in HANDOFF_PHASE_3.md.
    """
    # Core spec points.
    assert classify_tier(5) == "tier1"
    assert classify_tier(50) == "tier2"
    assert classify_tier(150) == "tier3"

    # Boundary points — match what src/data.py actually does.
    assert classify_tier(10) == "tier1"  # boundary inclusive at upper edge
    assert classify_tier(100) == "tier2"  # boundary inclusive at upper edge
    # Just above the boundaries.
    assert classify_tier(10.01) == "tier2"
    assert classify_tier(100.01) == "tier3"


