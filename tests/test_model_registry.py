"""Tests for the model-metadata registry in ``src.models``.

These pin the property whose absence caused the original defect: the presentation
metadata and the actual model lineup must not be able to drift apart. Before the
registry existed, ``MODEL_COLORS`` was hand-listed in ``src.plots`` and had gone
stale against ``default_models()`` — six of nine models resolved to the single
grey fallback and were indistinguishable in every figure.
"""

from __future__ import annotations


from src.models import (
    ENSEMBLE_NAME,
    MODEL_COLORS,
    MODEL_ORDER,
    UNKNOWN_MODEL_COLOR,
    color_for,
    default_models,
)


def test_model_order_matches_the_lineup_actually_built() -> None:
    """Every model ``default_models()`` emits has a row, and vice versa.

    This is the assertion that would have caught the stale palette: it fails the
    moment a lookback is added to config without the registry following.
    """
    built = tuple(m.name for m in default_models())
    assert MODEL_ORDER == built + (ENSEMBLE_NAME,)


def test_every_live_model_has_its_own_colour() -> None:
    """No live model falls through to the unknown-model fallback, and no two
    models share a colour — either would make them indistinguishable in a figure.
    """
    fallbacks = [m for m in MODEL_ORDER if color_for(m) == UNKNOWN_MODEL_COLOR]
    assert fallbacks == [], f"models with no colour of their own: {fallbacks}"
    assert len(set(MODEL_COLORS.values())) == len(MODEL_ORDER)


