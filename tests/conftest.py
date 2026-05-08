"""Shared pytest fixtures and path setup for the project test suite.

This file (a) ensures the project root is on ``sys.path`` so ``from src.<mod>
import ...`` works regardless of how pytest is invoked, and (b) provides a
project-wide ``autouse`` fixture that re-seeds NumPy's legacy global RNG
before every test for deterministic behavior.

Tests that need their own RNG should still construct ``np.random.default_rng``
with an explicit seed — this fixture only stabilizes any code that uses the
legacy ``np.random.*`` API.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# sys.path: make the project root importable as the ``src`` package parent.
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Determinism: seed the legacy global RNG before each test.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _seed_numpy_rng() -> None:
    """Re-seed ``np.random`` to 0 before every test for determinism."""
    np.random.seed(0)


# ---------------------------------------------------------------------------
# Convenience constants for tests that need filesystem paths.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def project_root() -> str:
    return _PROJECT_ROOT


@pytest.fixture(scope="session")
def data_dir(project_root: str) -> str:
    return os.path.join(project_root, "data")
