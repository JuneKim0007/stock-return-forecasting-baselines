"""Smoke test for the forecaster lineup.

Generates a synthetic AR(1) series, fits every model on the first 100 obs,
predicts one step, and asserts every prediction is a finite float matching
its closed-form value.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.models import ARMAModel, default_models


def _make_ar1(n: int = 200, phi: float = 0.5, sigma: float = 1.0,
              seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eps = rng.normal(loc=0.0, scale=sigma, size=n)
    y = np.zeros(n, dtype=float)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + eps[t]
    return y


def test_smoke_all_models_predict_finite_floats() -> None:
    y = _make_ar1(n=200, phi=0.5, sigma=1.0, seed=42)
    window = y[:100]

    models = default_models()

    preds = {}
    for m in models:
        m.fit(window)
        p = m.predict_one()
        assert isinstance(p, float), f"{m.name}: not a float ({type(p)})"
        assert math.isfinite(p), f"{m.name}: non-finite prediction ({p})"
        preds[m.name] = p

    # Closed-form sanity checks (new lineup).
    assert math.isclose(preds["naive"], float(window[-1]), abs_tol=1e-12)
    assert math.isclose(preds["global"], float(np.mean(window)), abs_tol=1e-12)
    assert math.isclose(preds["expanding"], float(np.mean(window)), abs_tol=1e-12)
    assert math.isclose(preds["ma30"], float(np.mean(window[-30:])), abs_tol=1e-12)
    assert math.isclose(preds["ma60"], float(np.mean(window[-60:])), abs_tol=1e-12)
    assert math.isclose(preds["ma90"], float(np.mean(window[-90:])), abs_tol=1e-12)

    # ARMA should have selected an order (or fallen back to mean).
    arma = next(m for m in models if m.name.startswith("arma"))
    assert isinstance(arma, ARMAModel)
    assert arma.best_order is not None, "ARMA AIC search returned no order"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v", "-s"])
