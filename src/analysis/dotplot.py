"""Unlabeled vertical dot plots over per-stock RMSE / MAE."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.plots import MODEL_ORDER, color_for


def _models_in_order(models_present: pd.Series) -> list:
    seen = set(models_present.unique().tolist())
    ordered = [m for m in MODEL_ORDER if m in seen]
    extras = sorted(seen - set(MODEL_ORDER))
    return ordered + extras


def _scatter_axis(
    ax: plt.Axes,
    summaries: pd.DataFrame,
    metric: str,
    jitter: float = 0.12,
) -> list:
    """Strip-plot one tier's summary frame onto ``ax`` with a mean marker
    and a ±1 standard-deviation range bar overlaid per model. Returns the
    model order used for axis ticks.
    """
    models = _models_in_order(summaries["model"])
    rng = random.Random(0)
    for mi, model in enumerate(models):
        sub = summaries[summaries["model"] == model]
        if sub.empty:
            continue
        ys = sub[metric].to_numpy(dtype=float)
        xs = np.array(
            [mi + (rng.random() - 0.5) * 2 * jitter for _ in range(len(ys))],
            dtype=float,
        )
        ax.scatter(xs, ys, color=color_for(model), s=22, alpha=0.55,
                   edgecolor="black", linewidth=0.4, zorder=2)

        if ys.size >= 1:
            mean = float(np.mean(ys))
            std = float(np.std(ys, ddof=0)) if ys.size > 1 else 0.0
            # ±1 std vertical range bar (volatility) + mean tick.
            ax.errorbar(
                mi, mean, yerr=std,
                fmt="_", color="black", ecolor="black",
                elinewidth=2.0, capsize=8, capthick=1.6,
                markersize=22, markeredgewidth=2.5, zorder=4,
            )
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=20, fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    return models


def render_dotplot(
    summaries: pd.DataFrame,
    *,
    metric: str,
    title: str,
    out_path: Path,
    jitter: float = 0.12,
) -> Path:
    """One PNG + sidecar CSV. ``summaries`` columns: ticker, window, model, rmse, mae.

    The figure shows one dot per stock plus a per-model ±1 std range bar
    centred on the mean so the cross-stock volatility is visible at a
    glance. The sidecar CSV has both the raw rows and a per-model
    aggregate (``model, n, mean, std, min, max``) appended below.
    """
    out_path = Path(out_path)
    os.makedirs(out_path.parent, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    _scatter_axis(ax, summaries, metric, jitter=jitter)
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{title}  (mean ± 1 std)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    csv_path = out_path.with_suffix(".csv")
    agg = (
        summaries.groupby("model")[metric]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
        .rename(columns={"count": "n"})
    )
    agg["std"] = agg["std"].fillna(0.0)
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("# raw per-stock rows\n")
        summaries.to_csv(fh, index=False)
        fh.write(f"\n# per-model aggregate ({metric})\n")
        agg.to_csv(fh, index=False)
    return out_path


def render_combined_dotplot(
    summaries_by_tier: Dict[str, pd.DataFrame],
    *,
    metric: str,
    out_path: Path,
) -> Path:
    """Side-by-side subplots — one per tier — sharing the y-axis."""
    out_path = Path(out_path)
    os.makedirs(out_path.parent, exist_ok=True)
    tiers = list(summaries_by_tier.keys())
    n = max(1, len(tiers))
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n + 1.5, 5), sharey=True,
                             squeeze=False)
    for ax, tier in zip(axes[0], tiers):
        df = summaries_by_tier[tier]
        if df.empty:
            ax.set_visible(False)
            continue
        _scatter_axis(ax, df, metric)
        ax.set_title(tier)
    axes[0][0].set_ylabel(metric.upper())
    fig.suptitle(f"All tiers — {metric.upper()} per stock")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


__all__ = ["render_dotplot", "render_combined_dotplot"]
