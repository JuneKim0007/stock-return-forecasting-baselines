"""Phase 2 — Forecasting models.

A small zoo of one-step-ahead forecasters sharing a uniform interface:

    class Forecaster(ABC):
        name: str
        def fit(self, y: np.ndarray) -> None: ...
        def predict_one(self) -> float: ...

`fit(y)` is called every rolling step with the in-window log-return slice;
`predict_one()` returns the one-step-ahead forecast for the next observation.

Behavioural (LSP) contract that all concrete subclasses honour
--------------------------------------------------------------
* **fit precondition (weakened from "any array"):** ``y`` must be a 1-D,
  non-empty ``np.ndarray`` (or array-like).  Subclasses must not require a
  *longer* ``y`` than the window supplied by the rolling engine.  Any length
  requirement is silently handled internally (e.g. MA falls back to the full
  mean when ``len(y) < s``).
* **fit postcondition:** after ``fit(y)`` returns, ``predict_one()`` is ready
  to produce a valid Python ``float`` (possibly ``nan`` — see below).
* **predict_one return type:** always a *Python* ``float``, never
  ``np.float64`` or any other numeric subtype.  The rolling engine wraps
  every call in ``float(...)`` as a belt-and-suspenders guard, so coercion
  at the call site is redundant but harmless.
* **nan is a valid return:** ``predict_one()`` may return ``float('nan')``
  in two sanctioned cases:
    1. It is called before any ``fit()`` has been made (undefined state).
    2. Every numerical path inside ``fit`` failed (e.g. ARMA convergence
       failure on a degenerate series) and the fallback itself is non-finite.
  Callers (``src.rolling``, ``EnsembleModel``) must tolerate ``nan`` without
  raising.  ``EnsembleModel`` explicitly uses ``np.nanmean`` so that a single
  ``nan`` child does not poison the ensemble average.
* **determinism:** given the same ``y`` and the same internal state prior to
  ``fit``, ``predict_one()`` returns the same value every time.  ARMAModel
  deviates slightly: it caches the last AIC-selected order across windows to
  amortise the grid search, so the *order chosen* depends on history — but
  the *forecast* is deterministic given the fitted result at each step.
* **no side-effects beyond self:** ``fit`` and ``predict_one`` must not
  mutate ``y``, write files, emit log messages, or raise for any input that
  satisfies the precondition above.

Models are stateless across windows in the sense that ``fit`` overwrites prior
state, with one deliberate exception: :class:`ARMAModel` caches the last
AIC-selected order ``(p*, q*)`` to amortize the grid search across rolling
windows (re-searched every ``refit_every`` calls).

Design patterns
---------------
* **Strategy** — every concrete class follows the same ``Forecaster`` ABC,
  letting the rolling engine (``src.rolling``) depend only on the abstract
  interface rather than any concrete implementation (Dependency Inversion).
* **Factory / Registry** — :func:`model_factory` and :func:`default_models`
  centralise construction so callers never ``import`` concrete classes to
  build a model by name.  Adding a new model only requires registering it in
  ``_MODEL_REGISTRY``; no changes needed in ``evaluate.py`` or ``rolling.py``.

This module is deliberately I/O-free — Phase 4 owns the rolling loop, metric
computation, and persistence.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Protocol, Tuple, runtime_checkable

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
    """Structural (duck-typing) counterpart to the :class:`Forecaster` ABC.

    Any object that exposes a ``name: str``, a ``fit(y: np.ndarray) -> None``
    method, and a ``predict_one() -> float`` method satisfies this Protocol —
    even if it does not inherit from :class:`Forecaster`.  This is used to
    type-check injected callables and third-party adapters at runtime via
    ``isinstance(obj, ForecasterProtocol)``.

    LSP note: the full behavioural contract (nan semantics, no-raise
    requirement, return-type guarantee) is documented in the module docstring
    and in :class:`Forecaster`.  This Protocol captures only the *structural*
    part that ``typing`` can verify statically.
    """

    name: str

    def fit(self, y: np.ndarray) -> None: ...  # noqa: D102

    def predict_one(self) -> float: ...  # noqa: D102


class Forecaster(ABC):
    """Abstract base class for a one-step-ahead forecaster (Strategy pattern).

    Every concrete forecaster **must** supply a ``name`` class attribute and
    implement both :meth:`fit` and :meth:`predict_one`.  Using ``ABC`` makes
    the contract explicit and raises ``TypeError`` at instantiation time if a
    subclass forgets either abstract method — catching bugs before the rolling
    engine ever runs.

    Dependency-Inversion note: ``src.rolling`` and ``src.evaluate`` depend
    only on this abstract type, never on concrete subclasses.

    Liskov Substitution contract (summary — full details in module docstring):
    - ``fit(y)`` accepts any 1-D non-empty array without raising.
    - ``predict_one()`` always returns a Python ``float`` (may be ``nan``).
    - Subclasses must not strengthen the precondition (e.g. require longer y).
    - Subclasses must not weaken the postcondition (e.g. return non-float).

    Rolling-engine calling convention (``src.rolling``):
        The rolling engine calls ``fit(slice)`` immediately followed by
        ``predict_one()`` exactly once per window step, in strict sequence::

            for t in range(window, n):
                model.fit(y[t - window : t])   # in-window slice (length == window)
                yhat = float(model.predict_one())

        Subclasses **must** honour this protocol:

        1. ``fit`` is always called before ``predict_one`` at each step —
           never the reverse, never ``predict_one`` without a preceding
           ``fit`` (except the unfitted-state ``nan`` edge case documented
           in the module docstring).
        2. The slice length equals the requested rolling window (``W``), so
           ``len(y)`` is always ``>= 1``.  Subclasses that need a minimum
           of ``s > W`` observations must degrade gracefully (see
           :class:`MovingAverageModel` for the canonical fallback pattern).
        3. ``predict_one`` must return the forecast corresponding to the
           *most recent* ``fit`` call.  State from earlier steps may be
           retained only as an optimisation (e.g. :class:`ARMAModel` caches
           the AIC-selected order), but the *forecast value* must reflect
           the current window's fit.
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

    LSP note — weakened precondition (safe):
        The rolling engine may supply windows shorter than ``s`` during early
        steps.  Rather than raising (which would strengthen the precondition
        and violate LSP), ``fit`` silently falls back to the mean of all
        available observations when ``len(y) < s``.  This *weakens* the
        precondition relative to a strict "need exactly s points" reading —
        the subtype is more permissive than a naive reading of the name would
        suggest, which is LSP-safe.
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
        """Fit on the last ``s`` observations of ``y``, or all of ``y`` if shorter.

        When ``len(y) < self.s`` the model degrades gracefully to the full-window
        mean rather than raising, preserving the base-class precondition (any
        1-D non-empty array is accepted without error).
        """
        if len(y) < self.s:
            # Window shorter than s — fall back to the available data.
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

    LSP notes
    ---------
    * **Documented deviation from pure statelesness:** ``_best_order`` and
      ``_step`` persist across ``fit`` calls on successive rolling windows.
      This is an intentional performance optimisation, not an LSP violation —
      the behavioural contract guarantees that ``predict_one()`` always returns
      a finite Python ``float`` after any ``fit``, regardless of the cached
      order.  The only observable cross-window effect is *which* ARMA order is
      fitted; the output type and range (finite float) are invariant.
    * **Fallback semantics:** when every ``(p, q)`` candidate fails AND no
      prior order is cached, the model uses the window mean as a fallback so
      that ``predict_one()`` never raises.  If the window mean itself is
      non-finite (e.g. all-NaN input), the return is ``nan`` — the one
      sanctioned case where a finite result cannot be guaranteed.
    * **predict_one before fit:** returns ``float('nan')`` (initial
      ``_forecast`` value).  The rolling engine always calls ``fit`` before
      ``predict_one``, so this edge case does not arise in production.
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
# Ensemble
# ---------------------------------------------------------------------------


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
# Factory / Registry
# ---------------------------------------------------------------------------

#: Internal registry mapping model names to zero-argument factory callables.
#: Each callable returns a *fresh* :class:`Forecaster` instance with
#: config-driven defaults baked in.  Extend this dict to add a new model
#: without touching any other module (Open/Closed Principle).
_MODEL_REGISTRY: Dict[str, Callable[[], Forecaster]] = {}


def register_model(name: str, factory: Callable[[], Forecaster]) -> None:
    """Register a factory for the model identified by ``name``.

    Parameters
    ----------
    name : str
        The machine-readable label that will appear in CSV outputs and plots.
    factory : () -> Forecaster
        Zero-argument callable that constructs and returns a fresh instance.

    Raises
    ------
    ValueError
        If ``name`` is already registered (prevents silent overwrites).
    """
    if name in _MODEL_REGISTRY:
        raise ValueError(
            f"A model named {name!r} is already registered. "
            "Use a different name or unregister the existing entry first."
        )
    _MODEL_REGISTRY[name] = factory


def model_factory(name: str) -> Forecaster:
    """Construct and return a fresh :class:`Forecaster` by *name*.

    This is the preferred way to build models by string identifier (e.g. when
    re-loading results from CSV and needing a matching fresh instance).

    Parameters
    ----------
    name : str
        Must be a key in :data:`_MODEL_REGISTRY`.

    Raises
    ------
    KeyError
        If ``name`` is not registered.
    """
    if name not in _MODEL_REGISTRY:
        available = sorted(_MODEL_REGISTRY)
        raise KeyError(
            f"Unknown model {name!r}. Available: {available}"
        )
    return _MODEL_REGISTRY[name]()


def _populate_registry() -> None:
    """Seed the registry with the built-in models (called once at module load).

    Reads config for ARMA / MA parameters — same logic as :func:`default_models`.
    This function is idempotent in the sense that it only runs if the registry
    is still empty.

    OCP note: built-in model registrations are expressed as a data-driven list
    of ``(name, factory)`` pairs assembled from config values.  Adding a new
    built-in model requires only appending one entry to ``_builtins`` below —
    the loop body never needs to change.  External callers may also call
    :func:`register_model` directly to add models without touching this file.
    """
    if _MODEL_REGISTRY:
        return  # already populated (e.g. tests importing multiple times)

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

    _builtins: List[Tuple[str, Callable[[], Forecaster]]] = [
        ("naive", NaiveModel),
        ("global", GlobalMeanModel),
        ("expanding", ExpandingMeanModel),
    ]
    for s in ma_lookbacks:
        s_int = int(s)
        _builtins.append((f"ma{s_int}", lambda _s=s_int: MovingAverageModel(_s)))
    for L in arma_lookbacks:
        L_int = int(L)
        _builtins.append((
            f"arma{L_int}",
            lambda _p=max_p, _q=max_q, _r=refit_every, _L=L_int:
                ARMAModel(max_p=_p, max_q=_q, refit_every=_r, lookback=_L),
        ))

    for name, factory in _builtins:
        _MODEL_REGISTRY[name] = factory


_populate_registry()


__all__ = [
    "Forecaster",
    "ForecasterProtocol",
    "NaiveModel",
    "GlobalMeanModel",
    "ExpandingMeanModel",
    "MovingAverageModel",
    "ARMAModel",
    "default_models",
    "register_model",
    "model_factory",
    "_MODEL_REGISTRY",
]
