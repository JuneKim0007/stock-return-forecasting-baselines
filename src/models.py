"""One-step-ahead forecasting models.

Every forecaster implements the same two-call interface, which the rolling
engine (``src.rolling``) drives once per test step::

    model.fit(y_window)          # 1-D non-empty float array
    yhat = model.predict_one()   # Python float, possibly nan

Contract every subclass must honour
-----------------------------------
* ``fit`` accepts any 1-D non-empty array and never raises. A model needing
  more observations than the window holds degrades instead — see
  :class:`MovingAverageModel`, which falls back to the mean of what it has.
* ``predict_one`` returns a Python ``float`` and never raises. ``nan`` is a
  legitimate return in exactly two cases: called before any ``fit``, or every
  numerical path inside ``fit`` failed and the fallback is itself non-finite.
  Callers must tolerate it — the post-hoc ensemble averages with ``np.nanmean``
  so one ``nan`` child does not poison the result.
* ``fit`` and ``predict_one`` mutate neither ``y`` nor anything outside
  ``self``. This module performs no I/O.
* Given the same window and prior state, the forecast is deterministic.

Cross-window state is overwritten by each ``fit``, with one deliberate
exception: :class:`ARMAModel` caches its AIC-selected order ``(p*, q*)`` and
re-searches only every ``refit_every`` calls, because the grid search dominates
runtime. The order chosen therefore depends on history; the forecast given that
order does not.

The module also owns the model-metadata registry (:data:`MODEL_ORDER`,
:data:`MODEL_COLORS`, :func:`ordered_models`) that ``src.plots``,
``src.summary`` and ``src.analysis.dotplot`` all render from.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

# statsmodels emits a flurry of convergence / value-error warnings during the
# AIC grid search. Silence those (only) — never silence ImportError/ValueError,
# which we still want to see from the rest of the program.
try:
    from statsmodels.tools.sm_exceptions import (
        ConvergenceWarning,
        ValueWarning,
    )

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=ValueWarning)
except ImportError:  # pragma: no cover — statsmodels missing
    pass

# A handful of statsmodels internals raise RuntimeWarning (overflow in
# log-likelihood, divide-by-zero in score) on poorly-fitted (p, q) pairs.
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"statsmodels.*")
# UserWarning fires for things like "Non-stationary starting AR parameters".
warnings.filterwarnings("ignore", category=UserWarning, module=r"statsmodels.*")


# ---------------------------------------------------------------------------
# Strategy — abstract base class (enforced via ABC)
# ---------------------------------------------------------------------------


@runtime_checkable
class ForecasterProtocol(Protocol):
    """Structural counterpart to :class:`Forecaster`, for runtime checks.

    Anything exposing ``name``, ``fit`` and ``predict_one`` satisfies this,
    inheritance or not. It captures only what ``typing`` can verify; the
    behavioural half of the contract is in the module docstring.
    """

    name: str

    def fit(self, y: np.ndarray) -> None: ...  # noqa: D102

    def predict_one(self) -> float: ...  # noqa: D102


class Forecaster(ABC):
    """Abstract base for a one-step-ahead forecaster.

    Subclasses supply a ``name`` and implement both methods; ``ABC`` turns a
    missing one into a ``TypeError`` at construction rather than a failure
    mid-run. ``src.rolling`` and ``src.evaluate`` depend on this type, never on
    a concrete model. The behavioural contract is in the module docstring.
    """

    #: Short machine-readable label used as dict keys and CSV column names.
    name: str = "base"

    @abstractmethod
    def fit(self, y: np.ndarray) -> None:
        """Fit the model on ``y`` (a 1-D array of in-window observations).

        Parameters
        ----------
        y : np.ndarray
            1-D, non-empty float array of observations from the rolling window.
            Subclasses must not require a minimum length beyond 1 (they may
            gracefully degrade for short windows, but must not raise).

        Postcondition: after this returns, ``predict_one()`` is callable.
        """

    @abstractmethod
    def predict_one(self) -> float:
        """Return the one-step-ahead forecast as a Python ``float``.

        Returns
        -------
        float
            The one-step-ahead point forecast.  May be ``float('nan')`` in
            degenerate cases (e.g. called before ``fit``, or when every
            numerical path inside ``fit`` failed).  Never raises.
        """


# ---------------------------------------------------------------------------
# Trivial baselines
# ---------------------------------------------------------------------------


class NaiveModel(Forecaster):
    """Predict the next value as the most recent observation."""

    name: str = "naive"
    kind: str = "naive"
    lookback: int = 1

    def __init__(self) -> None:
        self._last: float = float("nan")

    def fit(self, y: np.ndarray) -> None:
        self._last = float(y[-1])

    def predict_one(self) -> float:
        return self._last


class GlobalMeanModel(Forecaster):
    """Constant predictor: mean of the entire downloaded series.

    Future-leaking by design: this model is fit ONCE on the full y_full
    (including the test set) before scoring begins, then returns that mean
    for every test step. Included only as a benchmark.
    """

    name: str = "global"
    kind: str = "global"
    lookback: int = -1  # sentinel — uses full series

    def __init__(self) -> None:
        self._mean: float = float("nan")

    def fit(self, y: np.ndarray) -> None:
        self._mean = float(np.mean(y))

    def predict_one(self) -> float:
        return self._mean


class ExpandingMeanModel(Forecaster):
    """Running mean over all past observations (no fixed lookback).

    State semantics: ``set_state(running_sum, count)`` initialises the
    running mean to a known prefix; ``fit(y)`` recomputes from the slice
    given (used by the legacy rolling engine for backwards compatibility).
    The new engine drives ``set_state`` directly per step.
    """

    name: str = "expanding"
    kind: str = "expanding"
    lookback: int = -1  # sentinel — grows with t

    def __init__(self) -> None:
        self._mean: float = float("nan")

    def set_state(self, running_sum: float, count: int) -> None:
        self._mean = float(running_sum / count) if count > 0 else float("nan")

    def fit(self, y: np.ndarray) -> None:
        self._mean = float(np.mean(y)) if y.size else float("nan")

    def predict_one(self) -> float:
        return self._mean


class MovingAverageModel(Forecaster):
    """Predict the next value as the mean of the last ``s`` observations.

    Falls back to the mean of all available data when the window is shorter
    than ``s``, rather than raising — the engine may supply short windows.
    """

    kind: str = "windowed"

    def __init__(self, s: int) -> None:
        if s <= 0:
            raise ValueError(f"MovingAverageModel: s must be positive, got {s}")
        self.s: int = int(s)
        self.lookback: int = int(s)
        self.name: str = f"ma{self.s}"
        self._mean: float = float("nan")

    def fit(self, y: np.ndarray) -> None:
        if len(y) < self.s:
            self._mean = float(np.mean(y))
        else:
            self._mean = float(np.mean(y[-self.s :]))

    def predict_one(self) -> float:
        return self._mean


# ---------------------------------------------------------------------------
# ARMA(p, q) with AIC order selection + caching
# ---------------------------------------------------------------------------


class ARMAModel(Forecaster):
    """ARMA(p, q) forecaster with AIC-selected order.

    At each fit:

    * If the step counter is a multiple of ``refit_every`` (or no order is
      cached yet), do a full AIC grid search over
      ``(p, q) in [0..max_p] x [0..max_q]`` and cache the winner.
    * Otherwise, fit the model with the cached order only.

    Non-convergence or any exception during a single ``(p, q)`` fit causes that
    candidate to be skipped. If *every* candidate fails, the model falls back
    to the window mean for that step.

    Parameters
    ----------
    max_p, max_q : int
        Inclusive upper bounds for the AIC grid.
    refit_every : int
        Re-run the AIC grid search every ``refit_every`` calls to ``fit``.

    Fallbacks: a failing ``(p, q)`` candidate is skipped; if every candidate
    fails and no order is cached, the window mean is used, so ``predict_one``
    still returns a float.
    """

    kind: str = "windowed"

    def __init__(
        self,
        max_p: int = 4,
        max_q: int = 4,
        refit_every: int = 20,
        lookback: Optional[int] = None,
    ) -> None:
        if max_p < 0 or max_q < 0:
            raise ValueError("max_p and max_q must be non-negative")
        if refit_every <= 0:
            raise ValueError("refit_every must be positive")

        self.max_p: int = int(max_p)
        self.max_q: int = int(max_q)
        self.refit_every: int = int(refit_every)
        # Lookback parameterises both the windowed slice the engine feeds in
        # and the model name (``arma60``, ``arma90``).
        self.lookback: int = int(lookback) if lookback is not None else 0
        self.name: str = (
            f"arma{int(lookback)}" if lookback is not None else "arma"
        )

        self._step: int = 0
        self._best_order: Optional[Tuple[int, int]] = None
        self._forecast: float = float("nan")
        # Optional: AverageModel-style fallback when every fit fails.
        self._fallback_mean: float = float("nan")

    # --- internal helpers -------------------------------------------------

    @staticmethod
    def _fit_one(y: np.ndarray, p: int, q: int):
        """Fit a single ARMA(p, 0, q) and return the fitted result, or None
        on failure. Caller is responsible for any warning suppression.
        """
        # statsmodels.tsa.arima.model.ARIMA — order=(p, 0, q) is ARMA(p, q).
        from statsmodels.tsa.arima.model import ARIMA

        try:
            model = ARIMA(y, order=(p, 0, q))
            res = model.fit()
            # AIC may be NaN on degenerate fits; treat that as failure.
            if res is None or not np.isfinite(res.aic):
                return None
            return res
        except (ValueError, np.linalg.LinAlgError, Exception):  # noqa: BLE001
            # statsmodels raises a zoo of exceptions (LinAlgError,
            # ConvergenceWarning-as-error in some versions, IndexError on
            # tiny series, etc.). Any failure → skip this (p, q).
            return None

    def _grid_search(self, y: np.ndarray) -> Tuple[Optional[Tuple[int, int]], object]:
        """Run the AIC grid; return (best_order, best_result)."""
        best_aic = np.inf
        best_order: Optional[Tuple[int, int]] = None
        best_res = None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for p in range(self.max_p + 1):
                for q in range(self.max_q + 1):
                    res = self._fit_one(y, p, q)
                    if res is None:
                        continue
                    aic = float(res.aic)
                    if aic < best_aic:
                        best_aic = aic
                        best_order = (p, q)
                        best_res = res
        return best_order, best_res

    # --- Forecaster API ---------------------------------------------------

    def fit(self, y: np.ndarray) -> None:
        self._fallback_mean = float(np.mean(y))

        do_search = (
            self._best_order is None or (self._step % self.refit_every == 0)
        )

        result = None
        if do_search:
            best_order, best_res = self._grid_search(y)
            if best_order is not None:
                self._best_order = best_order
                result = best_res
            # else: leave any prior cached order intact and try it below
        if result is None and self._best_order is not None:
            # Cached-order refit (or post-search retry of cached order).
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p, q = self._best_order
                result = self._fit_one(y, p, q)

        if result is None:
            # Total failure — fall back to the window mean for this step.
            self._forecast = self._fallback_mean
        else:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fc = result.forecast(steps=1)
                # forecast() returns a pandas Series in current statsmodels;
                # coerce to a plain Python float.
                fc_arr = np.asarray(fc, dtype=float).ravel()
                if fc_arr.size == 0 or not np.isfinite(fc_arr[0]):
                    self._forecast = self._fallback_mean
                else:
                    self._forecast = float(fc_arr[0])
            except Exception:  # noqa: BLE001
                self._forecast = self._fallback_mean

        self._step += 1

    def predict_one(self) -> float:
        return self._forecast

    # --- introspection (handy for Phase 5 ARMA-order heatmap) -----------

    @property
    def best_order(self) -> Optional[Tuple[int, int]]:
        """Last AIC-selected ``(p*, q*)``; ``None`` until a successful fit."""
        return self._best_order


# ---------------------------------------------------------------------------
# Convenience factories
# ---------------------------------------------------------------------------


def default_models() -> List[Forecaster]:
    """Return one fresh instance of each base model in the canonical lineup.

    Reads ``MA_LOOKBACKS``, ``ARMA_LOOKBACKS``, ``ARMA_MAX_P``, ``ARMA_MAX_Q``,
    and ``ARMA_REFIT_EVERY`` from :mod:`src.config`. Lineup:
    ``naive, global, expanding, ma30, ma60, ma90, arma60, arma90``.
    """
    try:
        from src import config as cfg
        ma_lookbacks = tuple(cfg.MA_LOOKBACKS)
        arma_lookbacks = tuple(cfg.ARMA_LOOKBACKS)
        max_p = int(cfg.ARMA_MAX_P)
        max_q = int(cfg.ARMA_MAX_Q)
        refit_every = int(cfg.ARMA_REFIT_EVERY)
    except Exception:  # pragma: no cover — config missing/broken
        ma_lookbacks = (30, 60, 90)
        arma_lookbacks = (60, 90)
        max_p, max_q, refit_every = 4, 4, 20

    models: List[Forecaster] = [
        NaiveModel(),
        GlobalMeanModel(),
        ExpandingMeanModel(),
    ]
    for s in ma_lookbacks:
        models.append(MovingAverageModel(s))
    for L in arma_lookbacks:
        models.append(ARMAModel(
            max_p=max_p, max_q=max_q, refit_every=refit_every, lookback=L,
        ))
    return models


# ---------------------------------------------------------------------------
# Model metadata registry
# ---------------------------------------------------------------------------
#
# One source for "which models exist, in what order, in what colour". Consumed
# by src.plots, src.summary, src.analysis.dotplot and src.evaluate.
#
# Note this is NOT the name -> constructor registry that used to live here: that
# one had no callers and was deleted. This one holds presentation metadata that
# four modules were previously each keeping their own divergent copy of, which
# is how six of nine models ended up sharing one fallback colour.
#
# It is DERIVED from ``default_models()`` rather than hand-listed, so a new
# entry in ``MA_LOOKBACKS`` / ``ARMA_LOOKBACKS`` cannot produce a model that has
# no row here. Hand-listing is what caused the original drift.

#: Name of the post-hoc ensemble. Not a ``Forecaster`` — it is averaged from
#: the others after the fact — so it is appended rather than derived.
ENSEMBLE_NAME: str = "ensemble"

#: Matplotlib ``tab10`` minus its grey, which is reserved as the fallback below
#: so an unregistered model is visibly anomalous rather than quietly plausible.
_PALETTE: Tuple[str, ...] = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d628a7",
    "#9467bd", "#8c564b", "#e377c2", "#bcbd22",
)

#: Returned for any name not in :data:`MODEL_COLORS`.
UNKNOWN_MODEL_COLOR: str = "#444444"


def _derive_model_order() -> Tuple[str, ...]:
    """Canonical model order, read off the lineup ``default_models`` builds."""
    return tuple(m.name for m in default_models()) + (ENSEMBLE_NAME,)


#: Canonical order — drives legend order and colour assignment.
MODEL_ORDER: Tuple[str, ...] = _derive_model_order()

#: Colour per model, assigned by position so the same model is the same colour
#: in every figure.
MODEL_COLORS: Dict[str, str] = {
    name: _PALETTE[i % len(_PALETTE)]
    for i, name in enumerate(MODEL_ORDER)
    if name != ENSEMBLE_NAME
}
MODEL_COLORS[ENSEMBLE_NAME] = "#17becf"

#: Linestyle per model; the ensemble is dashed to set it apart from the
#: individual learners it averages.
MODEL_LINESTYLES: Dict[str, str] = {m: "-" for m in MODEL_ORDER}
MODEL_LINESTYLES[ENSEMBLE_NAME] = "--"

#: Models contributing to the post-hoc ensemble. ``naive`` and ``global`` are
#: excluded: the first is the trivial benchmark, the second is future-leaking.
ENSEMBLE_CHILDREN: Tuple[str, ...] = tuple(
    n for n in MODEL_ORDER if n not in ("naive", "global", ENSEMBLE_NAME)
)


def color_for(model: str) -> str:
    """Colour for ``model``, or :data:`UNKNOWN_MODEL_COLOR` if unregistered."""
    return MODEL_COLORS.get(model, UNKNOWN_MODEL_COLOR)


def linestyle_for(model: str) -> str:
    """Linestyle for ``model``, solid if unregistered."""
    return MODEL_LINESTYLES.get(model, "-")


def ordered_models(names: Iterable[str]) -> List[str]:
    """Return the distinct ``names`` in :data:`MODEL_ORDER` order.

    Names outside the canonical order are appended alphabetically, so an
    unrecognised model is still plotted rather than dropped. Accepts anything
    iterable of names — a list, a ``dict`` keyed by model, a pandas Series.
    """
    seen = set(names)
    known = [m for m in MODEL_ORDER if m in seen]
    return known + sorted(seen - set(MODEL_ORDER))


__all__ = [
    "Forecaster",
    "ENSEMBLE_NAME",
    "ENSEMBLE_CHILDREN",
    "MODEL_ORDER",
    "MODEL_COLORS",
    "MODEL_LINESTYLES",
    "UNKNOWN_MODEL_COLOR",
    "color_for",
    "linestyle_for",
    "ordered_models",
    "ForecasterProtocol",
    "NaiveModel",
    "GlobalMeanModel",
    "ExpandingMeanModel",
    "MovingAverageModel",
    "ARMAModel",
    "default_models",
]
