"""Per-tier and cross-tier summary outputs for the new analysis section.

Reads the per-(ticker, model) prediction CSVs written by the runner under
``<test_root>/<tier>/predictions/<TICKER>_<MODEL>.csv`` and produces:

1. ``summary_<tier>.csv`` and ``summary_<tier>.png`` per tier — model-level
   mean / variance / min / max for both RMSE and MAE.
2. ``cumulative_<tier>.png`` per tier — sum-of-squared-errors over time
   pooled across the tier's tickers, one line per model.
3. ``summary_overall.csv`` / ``summary_overall.png`` and
   ``cumulative_overall.png`` — same outputs flattened across all tiers.
4. ``score_histogram.csv`` / ``score_histogram.png`` — count of how many
   tickers each model "won" (lowest RMSE on that ticker), excluding
   ``naive`` and ``ensemble``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.metrics import mae as _mae, rmse as _rmse
from src.plots import MODEL_COLORS, color_for


_PRED_RE = re.compile(r"^(?P<ticker>[A-Za-z0-9]+)_(?P<model>[A-Za-z0-9]+)\.csv$")

#: Score-histogram exclusions: ``naive`` rarely wins, ``ensemble`` is a
#: meta-model derived from the others, and ``global`` is the future-leaking
#: benchmark that wins every ticker by construction.
_HISTOGRAM_EXCLUDE = {"naive", "ensemble", "global"}


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _scan_predictions(pred_dir: Path) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Return ``{ticker: {model: df}}`` for every <TICKER>_<MODEL>.csv in
    ``pred_dir``. Each df has columns ``idx, y_true, y_pred``."""
    out: Dict[str, Dict[str, pd.DataFrame]] = {}
    if not pred_dir.is_dir():
        return out
    for path in sorted(pred_dir.iterdir()):
        m = _PRED_RE.match(path.name)
        if not m:
            continue
        ticker = m.group("ticker")
        model = m.group("model")
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if {"y_true", "y_pred"}.issubset(df.columns):
            out.setdefault(ticker, {})[model] = df
    return out


# ---------------------------------------------------------------------------
# Per-(ticker, model) RMSE / MAE
# ---------------------------------------------------------------------------


def _per_ticker_metrics(
    by_ticker: Dict[str, Dict[str, pd.DataFrame]],
) -> pd.DataFrame:
    """Return long-form: tier(unset), ticker, model, rmse, mae, n."""
    rows: List[Dict] = []
    for ticker, model_dict in by_ticker.items():
        for model, df in model_dict.items():
            yt = df["y_true"].to_numpy(dtype=float)
            yp = df["y_pred"].to_numpy(dtype=float)
            mask = np.isfinite(yt) & np.isfinite(yp)
            if not mask.any():
                continue
            rows.append({
                "ticker": ticker,
                "model": model,
                "rmse": _rmse(yt[mask], yp[mask]),
                "mae": _mae(yt[mask], yp[mask]),
                "n": int(mask.sum()),
            })
    return pd.DataFrame(rows)


def _model_summary(per_ticker: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-model: mean/var/min/max RMSE & MAE plus n_tickers."""
    if per_ticker.empty:
        return pd.DataFrame(columns=[
            "model", "n_tickers",
            "mean_rmse", "var_rmse", "min_rmse", "max_rmse",
            "mean_mae", "var_mae", "min_mae", "max_mae",
        ])
    agg = (
        per_ticker.groupby("model")
        .agg(
            n_tickers=("ticker", "nunique"),
            mean_rmse=("rmse", "mean"),
            var_rmse=("rmse", lambda s: float(np.var(s, ddof=0))),
            min_rmse=("rmse", "min"),
            max_rmse=("rmse", "max"),
            mean_mae=("mae", "mean"),
            var_mae=("mae", lambda s: float(np.var(s, ddof=0))),
            min_mae=("mae", "min"),
            max_mae=("mae", "max"),
        )
        .reset_index()
    )
    return agg


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _ordered_models(models_present: Sequence[str]) -> List[str]:
    """Stable model order: known model order first, then any extras alphabetically."""
    canonical = ["naive", "global", "expanding", "ma30", "ma60", "ma90",
                 "arma60", "arma90", "ensemble"]
    seen = set(models_present)
    ordered = [m for m in canonical if m in seen]
    extras = sorted(seen - set(canonical))
    return ordered + extras


def _plot_summary_bars(
    summary: pd.DataFrame,
    *,
    title: str,
    out_path: Path,
) -> Path:
    """Bar chart with mean per model and (min, max) capped error bars.

    Two subplots side by side: RMSE on the left, MAE on the right.
    """
    out_path = Path(out_path)
    os.makedirs(out_path.parent, exist_ok=True)

    if summary.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    models = _ordered_models(summary["model"].tolist())
    summary = summary.set_index("model").reindex(models).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    for ax, metric in zip(axes, ("rmse", "mae")):
        means = summary[f"mean_{metric}"].to_numpy(dtype=float)
        mins = summary[f"min_{metric}"].to_numpy(dtype=float)
        maxs = summary[f"max_{metric}"].to_numpy(dtype=float)
        # Asymmetric error bars: mean - min (lower) and max - mean (upper).
        lower = np.clip(means - mins, 0, None)
        upper = np.clip(maxs - means, 0, None)
        colors = [color_for(m) for m in models]
        x = np.arange(len(models))
        ax.bar(x, means, color=colors, edgecolor="black", linewidth=0.4)
        ax.errorbar(
            x, means, yerr=[lower, upper],
            fmt="none", ecolor="black", elinewidth=1.2, capsize=4,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, fontsize=9)
        ax.set_ylabel(metric.upper())
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_cumulative(
    by_ticker: Dict[str, Dict[str, pd.DataFrame]],
    *,
    title: str,
    out_path: Path,
) -> Path:
    """Sum of squared errors across tickers, one line per model."""
    out_path = Path(out_path)
    os.makedirs(out_path.parent, exist_ok=True)

    cum_by_model: Dict[str, np.ndarray] = {}
    for ticker, model_dict in by_ticker.items():
        for model, df in model_dict.items():
            err = df["y_true"].to_numpy(dtype=float) - df["y_pred"].to_numpy(dtype=float)
            sq = err * err
            sq[~np.isfinite(sq)] = 0.0
            if model not in cum_by_model:
                cum_by_model[model] = np.zeros_like(sq, dtype=float)
            n = min(cum_by_model[model].size, sq.size)
            if cum_by_model[model].size < sq.size:
                pad = np.zeros(sq.size, dtype=float)
                pad[: cum_by_model[model].size] = cum_by_model[model]
                cum_by_model[model] = pad
            cum_by_model[model][:n] += sq[:n]

    if not cum_by_model:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path

    fig, ax = plt.subplots(figsize=(9, 5))
    for model in _ordered_models(list(cum_by_model.keys())):
        cum = np.cumsum(cum_by_model[model])
        ax.plot(np.arange(cum.size), cum,
                color=color_for(model), linewidth=1.4, label=model,
                linestyle="--" if model == "ensemble" else "-")
    ax.set_xlabel("Test step")
    ax.set_ylabel("cumulative squared error (summed across tickers)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Score histogram
# ---------------------------------------------------------------------------


def _score_histogram(
    per_ticker: pd.DataFrame,
    *,
    exclude: Sequence[str] = ("naive", "ensemble", "global"),
) -> pd.DataFrame:
    """Return ``model, wins`` ranked by wins descending, after dropping
    excluded models from candidate pool."""
    if per_ticker.empty:
        return pd.DataFrame(columns=["model", "wins"])
    candidates = per_ticker[~per_ticker["model"].isin(exclude)]
    if candidates.empty:
        return pd.DataFrame(columns=["model", "wins"])
    winners = candidates.loc[candidates.groupby("ticker")["rmse"].idxmin()]
    counts = winners["model"].value_counts().reset_index()
    counts.columns = ["model", "wins"]
    return counts.sort_values("wins", ascending=False).reset_index(drop=True)


def _plot_score_histogram(
    hist: pd.DataFrame,
    *,
    out_path: Path,
    title: str = "Score histogram (excluding naive, global, ensemble)",
) -> Path:
    out_path = Path(out_path)
    os.makedirs(out_path.parent, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if hist.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        ax.set_axis_off()
    else:
        models = hist["model"].tolist()
        wins = hist["wins"].to_numpy(dtype=int)
        x = np.arange(len(models))
        colors = [color_for(m) for m in models]
        ax.bar(x, wins, color=colors, edgecolor="black", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, fontsize=9)
        ax.set_ylabel("# tickers won")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Public API — runner integration
# ---------------------------------------------------------------------------


def summarise_tier(
    tier: str,
    test_root: Path,
) -> List[str]:
    """Build per-tier summary outputs under ``<test_root>/<tier>/grouped/``."""
    test_root = Path(test_root)
    pred_dir = test_root / tier / "predictions"
    out_dir = test_root / tier / "grouped"
    os.makedirs(out_dir, exist_ok=True)

    by_ticker = _scan_predictions(pred_dir)
    per_ticker = _per_ticker_metrics(by_ticker)
    summary = _model_summary(per_ticker)

    csv_path = out_dir / f"summary_{tier}.csv"
    summary.to_csv(csv_path, index=False)

    written: List[str] = [str(csv_path)]
    written.append(str(_plot_cumulative(
        by_ticker,
        title=f"Cumulative squared error pooled across {tier} tickers",
        out_path=out_dir / f"cumulative_{tier}.png",
    )))
    written.append(str(_plot_summary_bars(
        summary,
        title=f"{tier} model summary (mean, min, max)",
        out_path=out_dir / f"summary_{tier}.png",
    )))
    return written


def summarise_overall(test_root: Path, tiers: Sequence[str]) -> List[str]:
    """Cross-tier summary at ``<test_root>/analysis/``."""
    test_root = Path(test_root)
    out_dir = test_root / "analysis"
    os.makedirs(out_dir, exist_ok=True)

    merged_by_ticker: Dict[str, Dict[str, pd.DataFrame]] = {}
    for tier in tiers:
        pred_dir = test_root / tier / "predictions"
        by_ticker = _scan_predictions(pred_dir)
        for ticker, model_dict in by_ticker.items():
            merged_by_ticker[f"{tier}/{ticker}"] = model_dict

    per_ticker = _per_ticker_metrics(merged_by_ticker)
    summary = _model_summary(per_ticker)

    csv_path = out_dir / "summary_overall.csv"
    summary.to_csv(csv_path, index=False)

    written: List[str] = [str(csv_path)]
    written.append(str(_plot_cumulative(
        merged_by_ticker,
        title="Cumulative squared error pooled across all tickers",
        out_path=out_dir / "cumulative_overall.png",
    )))
    written.append(str(_plot_summary_bars(
        summary,
        title="Overall model summary (mean, min, max)",
        out_path=out_dir / "summary_overall.png",
    )))

    hist = _score_histogram(per_ticker, exclude=("naive", "ensemble", "global"))
    hist_csv = out_dir / "score_histogram.csv"
    hist.to_csv(hist_csv, index=False)
    written.append(str(hist_csv))
    written.append(str(_plot_score_histogram(
        hist, out_path=out_dir / "score_histogram.png",
    )))
    return written


__all__ = ["summarise_tier", "summarise_overall"]
