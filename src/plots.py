"""Phase 5 — Visualization & Persistent Artifacts.

Reads the per-(ticker, model) prediction CSVs in ``results/predictions/``
plus the aggregated tables in ``results/`` and renders a fixed set of figures. Every figure is written as both PNG (300
dpi) and SVG, and the dataframe used to draw it is persisted as CSV next
to the figure under ``results/figures/data/``.

Design choices
--------------
* Pure ``matplotlib`` (no seaborn). ``Agg`` backend so the script works on
  headless boxes (CI, servers).
* Discovery is filename-driven: a regex over ``results/predictions/*.csv``
  matching ``<TICKER>_<MODEL>.csv``. Adding tickers needs no code change here.
* Filenames are deterministic and include the parameters they describe,
  so re-rendering is idempotent.
* Model order, color and linestyle come from the registry in ``src.models``
  so the same model looks the same in every figure the project renders.

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
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # noqa: E402 — must precede pyplot import.

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.models import (  # noqa: E402  — one source for model presentation
    MODEL_COLORS,
    MODEL_ORDER,
    color_for,
    linestyle_for,
    ordered_models,
)

# ---------------------------------------------------------------------------
# Constants & palette
# ---------------------------------------------------------------------------

#: Rolling-window size (in steps) used by the rolling-RMSE / rolling-MAE
#: figures. The actual window used per figure is
#: ``min(ROLLING_ERROR_WINDOW, rolling_window)``.
ROLLING_ERROR_WINDOW: int = 60

#: Number of trailing steps shown in the actual-vs-predicted plots.
ACTUAL_VS_PRED_TAIL: int = 200

#: Per-pair renderers still take a ``window`` argument, which they use only for
#: the figure title and filename. Each model carries its own lookback now, so
#: there is no shared window to report; both call sites pass this. Removing the
#: argument would rename every per-pair figure the runner writes, so it is left
#: as a follow-up rather than folded into a bug fix.
_NOMINAL_WINDOW: int = 0

#: Prediction filenames as ``src.evaluate`` writes them: ``<TICKER>_<MODEL>.csv``.
#: The model name is the last underscore-separated field, so the ticker pattern
#: is non-greedy — model names carry digits (``ma30``, ``arma60``) and a greedy
#: split would swallow them.
_PREDICTION_FILENAME = re.compile(
    r"^(?P<ticker>[A-Za-z0-9]+?)_(?P<model>[A-Za-z0-9]+)\.csv$"
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


def discover_predictions(
    predictions_dir: str = config.PREDICTIONS_DIR,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Return a nested mapping ``ticker -> {model: df}``.

    ``df`` has columns ``[idx, y_true, y_pred]`` exactly as the runner writes
    them. Filenames that do not match are skipped silently, so stray files in
    the directory are harmless.
    """
    out: Dict[str, Dict[str, pd.DataFrame]] = {}
    for path in sorted(glob.glob(os.path.join(predictions_dir, "*.csv"))):
        m = _PREDICTION_FILENAME.match(os.path.basename(path))
        if not m:
            continue
        out.setdefault(m.group("ticker"), {})[m.group("model")] = pd.read_csv(path)
    return out


# ---------------------------------------------------------------------------
# Per-(ticker, window) figures
# ---------------------------------------------------------------------------


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
    model_dict: Dict[str, pd.DataFrame], kind: str
) -> Tuple[pd.DataFrame, int]:
    """Wide frame of rolling RMSE/MAE per model, plus the span used.

    The smoothing span is capped by the number of steps available rather than
    by a backtest window. It used to be ``min(ROLLING_ERROR_WINDOW, W)``, which
    stopped meaning anything when the unified window was retired: callers pass
    no real window any more, so the cap collapsed to 0 and ``rolling(0)``
    raised. Capping by the data is what the cap was for.
    """
    n_steps = max((len(df) for df in model_dict.values()), default=0)
    smooth = max(1, min(ROLLING_ERROR_WINDOW, n_steps))
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
            linestyle=linestyle_for(model),
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
    df, smooth = _rolling_error_frame(model_dict, kind)

    fig, ax = plt.subplots(figsize=(10, 6))
    for model in ordered_models(model_dict):
        ax.plot(
            df["idx"],
            df[model],
            color=color_for(model),
            linestyle=linestyle_for(model),
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
            linestyle=linestyle_for(model),
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
    return df.sort_values(["tier", "model"]).reset_index(drop=True)


def plot_metric_by_model_tier(metric: str, metrics: Optional[pd.DataFrame] = None) -> List[str]:
    """Bar chart of ``metric`` (``rmse`` or ``mae``) per model, one panel per tier.

    Each model carries its own lookback now, so there is no window axis to
    facet on: the layout is a single row of tiers. Values are averaged across
    the tickers in a tier, since ``metrics.csv`` is per (tier, ticker, model).
    """
    if metrics is None:
        metrics = _load_metrics()
    tiers = sorted(metrics["tier"].unique())
    n_cols = max(1, len(tiers))
    fig, axes = plt.subplots(
        1, n_cols, figsize=(4.0 * n_cols + 2, 4.2), squeeze=False, sharey=True,
    )

    for c, tier in enumerate(tiers):
        ax = axes[0][c]
        sub = metrics[metrics["tier"] == tier]
        means = sub.groupby("model", observed=True)[metric].mean().reset_index()
        means = means.sort_values("model")
        ax.bar(
            means["model"].astype(str), means[metric],
            color=[color_for(m) for m in means["model"]],
        )
        ax.set_title(str(tier))
        ax.set_ylabel(f"mean {metric.upper()}" if c == 0 else "")
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle(f"Mean {metric.upper()} by model (one panel per tier)", y=1.02)
    fig.tight_layout()

    base = f"{metric}_by_model_tier"
    return save_fig_and_data(fig, metrics, base)


def plot_ensemble_vs_best(tier: str, metrics: pd.DataFrame) -> List[str]:
    """Compare the ensemble against the best single model within one tier.

    Both are tier-level means across tickers, so the comparison answers "does
    averaging the models beat the best one" rather than comparing two rows that
    happen to belong to different tickers.
    """
    sub = metrics[metrics["tier"] == tier]
    if sub.empty:
        return []

    means = (
        sub.groupby("model", observed=True)[["rmse", "mae"]].mean().reset_index()
    )
    ensemble_row = means[means["model"] == "ensemble"]
    others = means[means["model"] != "ensemble"].sort_values("rmse")
    if ensemble_row.empty or others.empty:
        return []

    best_row = others.iloc[[0]]
    plot_df = pd.concat([ensemble_row, best_row], ignore_index=True)
    plot_df.insert(0, "tier", tier)

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = [color_for(m) for m in plot_df["model"]]
    ax.bar(plot_df["model"].astype(str), plot_df["rmse"], color=colors)
    for i, val in enumerate(plot_df["rmse"]):
        ax.text(i, val, f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_title(f"Ensemble vs best single model — {tier}")
    ax.set_ylabel("mean RMSE")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    base = f"ensemble_vs_best_{tier}"
    return save_fig_and_data(fig, plot_df, base)


# ---------------------------------------------------------------------------
# Figure-type registry
# ---------------------------------------------------------------------------


#: Per-(ticker, window) figure renderers, keyed by figure type. Each value is
#: called as ``renderer(model_dict, ticker, window)`` and returns the paths it
#: wrote. Renderers needing an extra ``kind`` argument are wrapped in a lambda
#: so the dispatch loop can call every entry the same way. Insertion-ordered,
#: so figures always appear in the same sequence.
PerPairRenderer = Callable[[Dict[str, pd.DataFrame], str, int], List[str]]

_PER_PAIR_FIGURE_REGISTRY: Dict[str, PerPairRenderer] = {
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


# Aggregate (cross-tier) figure renderers keyed by metric name.
#
# Each value is a zero-argument callable that returns a list of file paths.
# They are rebuilt inside render_all once the metrics frame is available.
# This dict is not pre-populated at import time because the renderers need
# the live metrics frame; render_all constructs fresh callables per run.
_AGGREGATE_METRICS: Tuple[str, ...] = ("rmse", "mae")


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def render_all() -> List[Tuple[str, str]]:
    """Render every figure and return a list of (basename, kind) entries.

    Per-pair figures are driven by :data:`_PER_PAIR_FIGURE_REGISTRY`;
    aggregate metric figures iterate :data:`_AGGREGATE_METRICS`.
    """
    artifacts: List[Tuple[str, str]] = []

    # 1) Per-ticker figures — dispatch through the registry.
    discovered = discover_predictions()
    for ticker, model_dict in sorted(discovered.items()):
        if not model_dict:
            continue
        for renderer in _PER_PAIR_FIGURE_REGISTRY.values():
            for p in renderer(model_dict, ticker, _NOMINAL_WINDOW):
                artifacts.append((os.path.basename(p), "per-pair"))

    # 2) Cross-tier metric figures.
    try:
        metrics = _load_metrics()
    except FileNotFoundError:
        metrics = pd.DataFrame()

    if not metrics.empty:
        for metric in _AGGREGATE_METRICS:
            for p in plot_metric_by_model_tier(metric, metrics):
                artifacts.append((os.path.basename(p), "aggregate"))

        for tier in sorted(metrics["tier"].unique()):
            for p in plot_ensemble_vs_best(tier, metrics):
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
