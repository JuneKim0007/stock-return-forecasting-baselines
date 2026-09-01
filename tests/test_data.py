"""Unit tests for ``src/data.py`` — log-return computation, tier
classification, and the persisted manifest.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from src.data import compute_log_returns


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


