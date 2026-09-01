"""Tests for the model-metadata registry in ``src.models``.

These pin the property whose absence caused the original defect: the presentation
metadata and the actual model lineup must not be able to drift apart. Before the
registry existed, ``MODEL_COLORS`` was hand-listed in ``src.plots`` and had gone
stale against ``default_models()`` — six of nine models resolved to the single
grey fallback and were indistinguishable in every figure.
"""

from __future__ import annotations

import pandas as pd

from src.models import (
    ENSEMBLE_CHILDREN,
    ENSEMBLE_NAME,
    MODEL_COLORS,
    MODEL_ORDER,
    UNKNOWN_MODEL_COLOR,
    color_for,
    default_models,
    linestyle_for,
    ordered_models,
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


def test_unregistered_model_gets_the_fallback() -> None:
    assert color_for("not_a_model") == UNKNOWN_MODEL_COLOR
    assert linestyle_for("not_a_model") == "-"


def test_ensemble_is_visually_distinct() -> None:
    """The ensemble is a meta-model averaged from the others, so it is dashed."""
    assert linestyle_for(ENSEMBLE_NAME) == "--"
    assert all(linestyle_for(m) == "-" for m in MODEL_ORDER if m != ENSEMBLE_NAME)


def test_ensemble_children_exclude_the_two_benchmarks() -> None:
    """``naive`` is the trivial benchmark and ``global`` leaks the future, so
    neither may feed the ensemble; everything else must."""
    assert "naive" not in ENSEMBLE_CHILDREN
    assert "global" not in ENSEMBLE_CHILDREN
    assert ENSEMBLE_NAME not in ENSEMBLE_CHILDREN
    assert set(ENSEMBLE_CHILDREN) == set(MODEL_ORDER) - {"naive", "global", ENSEMBLE_NAME}


def test_ordered_models_accepts_every_shape_its_callers_pass() -> None:
    """Callers pass a dict keyed by model, a list, and a pandas Series."""
    expected = ["naive", "ma30", ENSEMBLE_NAME]
    assert ordered_models({m: None for m in expected}) == expected
    assert ordered_models(list(reversed(expected))) == expected
    assert ordered_models(pd.Series(expected * 2)) == expected


def test_ordered_models_appends_unknown_names_rather_than_dropping_them() -> None:
    """An unrecognised model must still be plotted, after the known ones."""
    assert ordered_models(["zeta", "naive", "alpha"]) == ["naive", "alpha", "zeta"]
