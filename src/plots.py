"""Phase 5 — Visualization & Persistent Artifacts.

Reads the per-(ticker, window, model) prediction CSVs in
``results/predictions/`` plus the aggregated tables in ``results/`` and
renders a fixed set of figures. Every figure is written as both PNG (300
dpi) and SVG, and the dataframe used to draw it is persisted as CSV next
to the figure under ``results/figures/data/``.

Design choices
--------------
* Pure ``matplotlib`` (no seaborn). ``Agg`` backend so the script works on
  headless boxes (CI, servers).
* Discovery is filename-driven: a regex over
  ``results/predictions/*.csv`` finds every
  ``(ticker, window, model)`` triple. Adding new tickers later only
  requires re-running the backtest — no code change here.
* Filenames are deterministic and include the parameters they describe,
  so re-rendering is idempotent.
* Color palette per model is fixed at module import (``MODEL_COLORS``)
  so the same model has the same color across every figure.

CLI
---
``python -m src.plots`` — render everything end-to-end. Prints a summary
table at the end listing every artifact written.
"""

from __future__ import annotations

import glob
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Protocol, Tuple

import matplotlib

matplotlib.use("Agg")  # noqa: E402 — must precede pyplot import.

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402

# ---------------------------------------------------------------------------
# Constants & palette
# ---------------------------------------------------------------------------

#: Fixed model order (drives legend order and color palette).
MODEL_ORDER: Tuple[str, ...] = (
    "naive",
    "average",
    "ma10",
    "ma20",
    "ma30",
    "arma",
    "ensemble",
)

#: Color per model — Matplotlib's ``tab10`` cycle, frozen so the same
#: model is the same color across every figure in the report.
MODEL_COLORS: Dict[str, str] = {
    "naive": "#1f77b4",
    "average": "#ff7f0e",
    "ma10": "#2ca02c",
    "ma20": "#d62728",
    "ma30": "#9467bd",
    "arma": "#8c564b",
    "ensemble": "#17becf",
}

#: Linestyle per model; ensemble is dashed to set it apart from the
#: individual learners.
MODEL_LINESTYLES: Dict[str, str] = {m: "-" for m in MODEL_ORDER}
MODEL_LINESTYLES["ensemble"] = "--"

#: Rolling-window size (in steps) used by the rolling-RMSE / rolling-MAE
#: figures. The actual window used per figure is
#: ``min(ROLLING_ERROR_WINDOW, rolling_window)``.
ROLLING_ERROR_WINDOW: int = 60

#: Number of trailing steps shown in the actual-vs-predicted plots.
ACTUAL_VS_PRED_TAIL: int = 200

#: Filename regex for ``results/predictions/*.csv``. Capture groups are
#: ``ticker``, ``window``, ``model``.
_PREDICTION_FILENAME = re.compile(
    r"(?P<ticker>[A-Za-z0-9]+)_(?P<window>\d+)_(?P<model>[A-Za-z0-9]+)\.csv$"
)

__all__ = [
    # Constants
    "MODEL_ORDER",
    "MODEL_COLORS",
    # I/O helpers
    "ensure_dirs",
    "save_fig_and_data",
    "figure_dirs",
    # Discovery
    "discover_predictions",
    "ordered_models",
    "color_for",
    # Registry API
    "PerPairRendererProtocol",
    "register_per_pair_figure",
    # Top-level orchestration
    "render_all",
]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def ensure_dirs() -> None:
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    os.makedirs(config.FIGURES_DATA_DIR, exist_ok=True)


def save_fig_and_data(fig: plt.Figure, df: pd.DataFrame, basename: str) -> List[str]:
    """Write ``fig`` (PNG + SVG) and ``df`` (CSV) under the figures dir.

    Returns the list of absolute paths written, in the order
    ``[png, svg, csv]``. Closes ``fig`` afterwards so the caller does
    not need to.
    """
    ensure_dirs()
    png_path = os.path.join(config.FIGURES_DIR, f"{basename}.png")
    svg_path = os.path.join(config.FIGURES_DIR, f"{basename}.svg")
    csv_path = os.path.join(config.FIGURES_DATA_DIR, f"{basename}.csv")

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    df.to_csv(csv_path, index=False)
    plt.close(fig)
    return [png_path, svg_path, csv_path]


@contextmanager
def figure_dirs(figures_dir: str, figures_data_dir: str) -> Iterator[None]:
    """Temporarily override the module-level figure output dirs.

    The per-pair registry (and group_analysis figures) all funnel through
    :func:`save_fig_and_data`, which reads from ``src.config.FIGURES_DIR`` /
    ``FIGURES_DATA_DIR``. This context manager swaps those constants in for
    the duration of the ``with`` block and restores them afterwards (even on
    exception), so callers can route output to a tier-scoped directory
    without touching every helper.

    The pipeline runs tier-by-tier sequentially in a single process, so this
    scoped mutation is safe.
    """
    import src.config as cfg

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(figures_data_dir, exist_ok=True)
    prev_fig = cfg.FIGURES_DIR
    prev_data = cfg.FIGURES_DATA_DIR
    cfg.FIGURES_DIR = figures_dir
    cfg.FIGURES_DATA_DIR = figures_data_dir
    try:
        yield
    finally:
        cfg.FIGURES_DIR = prev_fig
        cfg.FIGURES_DATA_DIR = prev_data


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PredKey:
    ticker: str
    window: int
    model: str


def discover_predictions(predictions_dir: str = config.PREDICTIONS_DIR) -> Dict[Tuple[str, int], Dict[str, pd.DataFrame]]:
    """Return a nested mapping ``(ticker, window) -> {model: df}``.

    ``df`` has columns ``[idx, y_true, y_pred]`` exactly as written by
    Phase 4. Unknown filenames are skipped silently.
    """
    out: Dict[Tuple[str, int], Dict[str, pd.DataFrame]] = {}
    for path in sorted(glob.glob(os.path.join(predictions_dir, "*.csv"))):
        m = _PREDICTION_FILENAME.search(os.path.basename(path))
        if not m:
            continue
        ticker = m.group("ticker")
        window = int(m.group("window"))
        model = m.group("model")
        df = pd.read_csv(path)
        out.setdefault((ticker, window), {})[model] = df
    return out


def ordered_models(model_dict: Dict[str, pd.DataFrame]) -> List[str]:
    """Return models in canonical order, with any unknown names appended."""
    known = [m for m in MODEL_ORDER if m in model_dict]
    extras = [m for m in model_dict if m not in MODEL_ORDER]
    return known + sorted(extras)


def color_for(model: str) -> str:
    return MODEL_COLORS.get(model, "#444444")


def _linestyle_for(model: str) -> str:
    return MODEL_LINESTYLES.get(model, "-")


# ---------------------------------------------------------------------------
# Per-(ticker, window) figures
# ---------------------------------------------------------------------------


def _build_long_predictions(model_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack each model's ``(idx, y_true, y_pred)`` into a long frame.

    Columns: ``model, idx, y_true, y_pred, sq_err, abs_err``.
    """
    rows = []
    for model in ordered_models(model_dict):
        df = model_dict[model].copy()
        df = df[["idx", "y_true", "y_pred"]].copy()
        df["model"] = model
        df["sq_err"] = (df["y_true"] - df["y_pred"]) ** 2
        df["abs_err"] = (df["y_true"] - df["y_pred"]).abs()
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def _cumulative_error_frame(
    model_dict: Dict[str, pd.DataFrame], err_col: str
) -> pd.DataFrame:
    """Return a wide frame: index=idx, columns=models, values=cumulative err."""
    pieces = []
    for model in ordered_models(model_dict):
        df = model_dict[model]
        cum = (df["y_true"] - df["y_pred"])
        cum = cum * cum if err_col == "sq_err" else cum.abs()
        cum = cum.cumsum().rename(model)
        cum.index = df["idx"].values
        pieces.append(cum)
    out = pd.concat(pieces, axis=1)
    out.index.name = "idx"
    return out.reset_index()


def _rolling_error_frame(
    model_dict: Dict[str, pd.DataFrame], window: int, kind: str
) -> Tuple[pd.DataFrame, int]:
    """Wide frame of rolling RMSE/MAE per model.

    ``window`` here is the *rolling backtest* window (60 or 120). The
    actual smoothing window is ``min(ROLLING_ERROR_WINDOW, window)``.
    """
    smooth = min(ROLLING_ERROR_WINDOW, window)
    pieces = []
    for model in ordered_models(model_dict):
        df = model_dict[model]
        diff = df["y_true"] - df["y_pred"]
        if kind == "rmse":
            roll = diff.pow(2).rolling(smooth, center=True, min_periods=1).mean().pow(0.5)
        elif kind == "mae":
            roll = diff.abs().rolling(smooth, center=True, min_periods=1).mean()
        else:  # pragma: no cover — internal call site only.
            raise ValueError(f"unknown kind: {kind!r}")
        roll = roll.rename(model)
        roll.index = df["idx"].values
        pieces.append(roll)
    out = pd.concat(pieces, axis=1)
    out.index.name = "idx"
    return out.reset_index(), smooth


def _actual_vs_pred_frame(model_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Last ``ACTUAL_VS_PRED_TAIL`` steps: idx, y_true, then one col per model."""
    models = ordered_models(model_dict)
    base = model_dict[models[0]][["idx", "y_true"]].copy()
    base = base.tail(ACTUAL_VS_PRED_TAIL).reset_index(drop=True)
    for model in models:
        df = model_dict[model][["idx", "y_pred"]].rename(columns={"y_pred": model})
        df = df.tail(ACTUAL_VS_PRED_TAIL).reset_index(drop=True)
        base = base.merge(df, on="idx", how="left")
    return base


def plot_cumulative_error(
    model_dict: Dict[str, pd.DataFrame],
    ticker: str,
    window: int,
    *,
    kind: str,
) -> List[str]:
    """``kind`` ∈ {``'sq_err'``, ``'abs_err'``}."""
    label = "Squared" if kind == "sq_err" else "Absolute"
    df = _cumulative_error_frame(model_dict, kind)

    fig, ax = plt.subplots(figsize=(10, 6))
    for model in ordered_models(model_dict):
        ax.plot(
            df["idx"],
            df[model],
            color=color_for(model),
            linestyle=_linestyle_for(model),
            linewidth=1.6,
            label=model,
        )
    ax.set_title(f"Cumulative {label} Error — {ticker} (window={window})")
    ax.set_xlabel("Step index")
    ax.set_ylabel(f"cumsum({'(y - y_hat)^2' if kind == 'sq_err' else '|y - y_hat|'})")
    ax.grid(True, alpha=0.3)
    ax.legend(title="model", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()

    base = (
        f"cumulative_sq_error_{ticker}_{window}"
        if kind == "sq_err"
        else f"cumulative_abs_error_{ticker}_{window}"
    )
    return save_fig_and_data(fig, df, base)


def plot_rolling_error(
    model_dict: Dict[str, pd.DataFrame],
    ticker: str,
    window: int,
    *,
    kind: str,
) -> List[str]:
    """``kind`` ∈ {``'rmse'``, ``'mae'``}."""
    df, smooth = _rolling_error_frame(model_dict, window, kind)

    fig, ax = plt.subplots(figsize=(10, 6))
    for model in ordered_models(model_dict):
        ax.plot(
            df["idx"],
            df[model],
            color=color_for(model),
            linestyle=_linestyle_for(model),
            linewidth=1.4,
            label=model,
        )
    label = kind.upper()
    ax.set_title(
        f"Rolling {label} (smooth={smooth}) — {ticker} (window={window})"
    )
    ax.set_xlabel("Step index")
    ax.set_ylabel(f"Rolling {label}")
    ax.grid(True, alpha=0.3)
    ax.legend(title="model", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()

    base = f"rolling_{kind}_{ticker}_{window}"
    return save_fig_and_data(fig, df, base)


def plot_actual_vs_pred(
    model_dict: Dict[str, pd.DataFrame], ticker: str, window: int
) -> List[str]:
    df = _actual_vs_pred_frame(model_dict)

    fig, ax = plt.subplots(figsize=(11, 6))
    # Actuals heavier and dashed, drawn last so it sits on top.
    for model in ordered_models(model_dict):
        ax.plot(
            df["idx"],
            df[model],
            color=color_for(model),
            linestyle=_linestyle_for(model),
            linewidth=1.0,
            alpha=0.85,
            label=model,
        )
    ax.plot(
        df["idx"],
        df["y_true"],
        color="black",
        linestyle="--",
        linewidth=2.2,
        label="actual",
    )

    ax.set_title(
        f"Actual vs predicted (last {len(df)} steps) — {ticker} (window={window})"
    )
    ax.set_xlabel("Step index")
    ax.set_ylabel("Log return")
    ax.grid(True, alpha=0.3)
    ax.legend(title="series", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()

    base = f"actual_vs_pred_{ticker}_{window}"
    return save_fig_and_data(fig, df, base)


# ---------------------------------------------------------------------------
# Aggregate figures (across the metrics table)
# ---------------------------------------------------------------------------


def _load_metrics(path: Optional[str] = None) -> pd.DataFrame:
    """Load the metrics CSV and apply stable model ordering.

    Parameters
    ----------
    path : str, optional
        Explicit path to ``metrics.csv``. Defaults to
        ``config.RESULTS_DIR/metrics.csv`` when omitted. Accepting a path
        argument removes the side-effect dependency on ``config.RESULTS_DIR``
        so callers (tests, alternate pipelines) can point at any file.
    """
    if path is None:
        path = os.path.join(config.RESULTS_DIR, "metrics.csv")
    df = pd.read_csv(path)
    # Stable model ordering for plotting.
    df["model"] = pd.Categorical(df["model"], categories=list(MODEL_ORDER), ordered=True)
    return df.sort_values(["tier", "window", "model"]).reset_index(drop=True)


def plot_metric_by_model_tier(metric: str) -> List[str]:
    """Bar chart of ``metric`` (``rmse`` or ``mae``) faceted by tier × window.

    Layout: rows = window, cols = tier. With only one tier present (the
    current backtest covers ``tier1`` only) the figure is a single
    column — still valid, just sparse.
    """
    metrics = _load_metrics()
    tiers = sorted(metrics["tier"].unique())
    windows = sorted(metrics["window"].unique())

    n_rows = max(1, len(windows))
    n_cols = max(1, len(tiers))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.0 * n_cols + 2, 3.2 * n_rows + 1),
        squeeze=False,
        sharey=True,
    )

    for r, win in enumerate(windows):
        for c, tier in enumerate(tiers):
            ax = axes[r][c]
            sub = metrics[(metrics["window"] == win) & (metrics["tier"] == tier)]
            sub = sub.sort_values("model")
            colors = [color_for(m) for m in sub["model"]]
            ax.bar(sub["model"].astype(str), sub[metric], color=colors)
            ax.set_title(f"{tier} — window={win}")
            ax.set_ylabel(metric.upper() if c == 0 else "")
            ax.grid(True, axis="y", alpha=0.3)
            ax.tick_params(axis="x", rotation=30)

    fig.suptitle(f"{metric.upper()} by model (rows=window, cols=tier)", y=1.02)
    fig.tight_layout()

    base = f"{metric}_by_model_tier"
    return save_fig_and_data(fig, metrics, base)


def plot_ensemble_vs_best(tier: str, window: int, metrics: pd.DataFrame) -> List[str]:
    sub = metrics[(metrics["tier"] == tier) & (metrics["window"] == window)]
    if sub.empty:
        return []

    ensemble_row = sub[sub["model"] == "ensemble"]
    others = sub[sub["model"] != "ensemble"].sort_values("rmse")
    if ensemble_row.empty or others.empty:
        return []

    best_row = others.iloc[[0]]
    plot_df = pd.concat([ensemble_row, best_row], ignore_index=True)[
        ["tier", "window", "model", "rmse", "mae"]
    ]

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = [color_for(m) for m in plot_df["model"]]
    ax.bar(plot_df["model"].astype(str), plot_df["rmse"], color=colors)
    for i, val in enumerate(plot_df["rmse"]):
        ax.text(i, val, f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_title(f"Ensemble vs best single model — {tier} (window={window})")
    ax.set_ylabel("RMSE")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    base = f"ensemble_vs_best_{tier}_{window}"
    return save_fig_and_data(fig, plot_df, base)


# ---------------------------------------------------------------------------
# Figure-type registry
# ---------------------------------------------------------------------------


class PerPairRendererProtocol(Protocol):
    """Structural contract for per-(ticker, window) figure renderers registered
    in :data:`_PER_PAIR_FIGURE_REGISTRY`.

    Any callable that satisfies this signature is a valid entry in the
    registry and will be called correctly by :func:`render_all`:

    Parameters
    ----------
    model_dict : Dict[str, pd.DataFrame]
        Mapping from model name to its prediction DataFrame (columns:
        ``idx``, ``y_true``, ``y_pred``).  May contain any subset of the
        known model names.  The renderer must tolerate an arbitrary set of
        keys — it must not assume a fixed model list.
    ticker : str
        The ticker symbol (used for titles and output filenames).
    window : int
        The rolling backtest window size (used for titles and filenames).

    Returns
    -------
    List[str]
        Absolute paths of all artifacts written (PNG, SVG, CSV).  An empty
        list is valid (e.g. if ``model_dict`` is empty or the renderer
        decides to skip).  Must never raise.

    LSP / substitutability guarantee
    ----------------------------------
    * Any registered renderer is called with exactly these three positional
      arguments — no keyword arguments are guaranteed.  Renderers that need
      additional parameters must use closures or ``functools.partial`` (as
      the built-in lambdas in :data:`_PER_PAIR_FIGURE_REGISTRY` do for
      ``kind``).
    * Renderers must not mutate ``model_dict`` (or its DataFrames) in ways
      visible to other renderers in the same dispatch loop.
    * Renderers may write files to :data:`config.FIGURES_DIR` and
      :data:`config.FIGURES_DATA_DIR`; they must not write elsewhere.
    """

    def __call__(
        self,
        model_dict: Dict[str, pd.DataFrame],
        ticker: str,
        window: int,
    ) -> List[str]: ...


# Per-(ticker, window) figure renderers.
#
# Each entry maps a figure-type key to a callable satisfying
# :class:`PerPairRendererProtocol`.
# Renderers that require an extra ``kind`` keyword argument are wrapped in
# a lambda so the dispatch loop can call every entry with the same three
# positional arguments.
#
# OCP: to add a new per-pair figure type (e.g. a scatter plot), register one
# new entry here — :func:`render_all` iterates the registry and requires no
# change to its body.
#
# The dict is ordered (Python 3.7+) so figures always appear in the same
# sequence in the summary table.
_PER_PAIR_FIGURE_REGISTRY: Dict[str, PerPairRendererProtocol] = {
    "cumulative_sq_err": (
        lambda md, t, w: plot_cumulative_error(md, t, w, kind="sq_err")
    ),
    "cumulative_abs_err": (
        lambda md, t, w: plot_cumulative_error(md, t, w, kind="abs_err")
    ),
    "rolling_rmse": (
        lambda md, t, w: plot_rolling_error(md, t, w, kind="rmse")
    ),
    "rolling_mae": (
        lambda md, t, w: plot_rolling_error(md, t, w, kind="mae")
    ),
    "actual_vs_pred": plot_actual_vs_pred,
}


def register_per_pair_figure(
    key: str,
    renderer: PerPairRendererProtocol,
) -> None:
    """Register a new per-(ticker, window) figure renderer.

    Parameters
    ----------
    key : str
        Unique identifier for the figure type (used as a label in the summary).
    renderer : PerPairRendererProtocol
        Callable satisfying ``(model_dict, ticker, window) -> List[str]``.
        See :class:`PerPairRendererProtocol` for the full LSP contract.
        Must return a list of absolute file paths written (PNG, SVG, CSV).

    Raises
    ------
    ValueError
        If ``key`` is already registered (prevents silent overwrites).
    """
    if key in _PER_PAIR_FIGURE_REGISTRY:
        raise ValueError(
            f"A per-pair figure renderer named {key!r} is already registered."
        )
    _PER_PAIR_FIGURE_REGISTRY[key] = renderer


# Aggregate (cross-tier) figure renderers keyed by metric name.
#
# Each value is a zero-argument callable that returns a list of file paths.
# They are rebuilt inside render_all once the metrics frame is available.
# This dict is not pre-populated at import time because the renderers need
# the live metrics frame; render_all constructs fresh callables per run.
#
# OCP: to add a new aggregate metric figure, extend the metrics list passed
# into render_all (or override _AGGREGATE_METRICS) — the loop requires no
# change.
_AGGREGATE_METRICS: Tuple[str, ...] = ("rmse", "mae")


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def render_all() -> List[Tuple[str, str]]:
    """Render every figure and return a list of (basename, kind) entries.

    Per-pair figures are driven by :data:`_PER_PAIR_FIGURE_REGISTRY` —
    adding a new figure type is a :func:`register_per_pair_figure` call,
    not an edit here (OCP). Aggregate metric figures iterate
    :data:`_AGGREGATE_METRICS`.
    """
    artifacts: List[Tuple[str, str]] = []

    # 1) Per-(ticker, window) figures — dispatch through the registry.
    discovered = discover_predictions()
    for (ticker, window), model_dict in sorted(discovered.items()):
        if not model_dict:
            continue
        for renderer in _PER_PAIR_FIGURE_REGISTRY.values():
            paths = renderer(model_dict, ticker, window)
            for p in paths:
                artifacts.append((os.path.basename(p), "per-pair"))

    # 2) Cross-tier metric figures.
    try:
        metrics = _load_metrics()
    except FileNotFoundError:
        metrics = pd.DataFrame()

    if not metrics.empty:
        for metric in _AGGREGATE_METRICS:
            paths = plot_metric_by_model_tier(metric)
            for p in paths:
                artifacts.append((os.path.basename(p), "aggregate"))

        for tier in sorted(metrics["tier"].unique()):
            for window in sorted(metrics["window"].unique()):
                paths = plot_ensemble_vs_best(tier, window, metrics)
                for p in paths:
                    artifacts.append((os.path.basename(p), "ensemble-vs-best"))

    return artifacts


def _summarize(artifacts: Iterable[Tuple[str, str]]) -> None:
    rows = list(artifacts)
    if not rows:
        print("[plots] no artifacts written.")
        return
    by_kind: Dict[str, int] = {}
    for _, kind in rows:
        by_kind[kind] = by_kind.get(kind, 0) + 1
    png = sum(1 for n, _ in rows if n.endswith(".png"))
    svg = sum(1 for n, _ in rows if n.endswith(".svg"))
    csv = sum(1 for n, _ in rows if n.endswith(".csv"))

    print()
    print("=== plots: summary ===")
    print(f"figures dir : {os.path.abspath(config.FIGURES_DIR)}")
    print(f"data dir    : {os.path.abspath(config.FIGURES_DATA_DIR)}")
    print(f"PNG written : {png}")
    print(f"SVG written : {svg}")
    print(f"CSV written : {csv}")
    print(f"by kind     : {by_kind}")
    print()
    print("artifact".ljust(60), "kind")
    print("-" * 80)
    for name, kind in rows:
        print(name.ljust(60), kind)


def main() -> None:
    ensure_dirs()
    artifacts = render_all()
    _summarize(artifacts)


if __name__ == "__main__":
    main()
