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
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.evaluate import parse_prediction_filename
from src.metrics import mae as _mae, n_scored as _n_scored, rmse as _rmse
from src.models import color_for, ordered_models
from src.plots import save_figure, save_placeholder


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
        parsed = parse_prediction_filename(path.name)
        if parsed is None:
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if {"y_true", "y_pred"}.issubset(df.columns):
            ticker, model = parsed
            out.setdefault(ticker, {})[model] = df
    return out


# ---------------------------------------------------------------------------
# Per-(ticker, model) RMSE / MAE
# ---------------------------------------------------------------------------


def _per_ticker_metrics(
    by_ticker: Dict[str, Dict[str, pd.DataFrame]],
) -> pd.DataFrame:
    """Return long-form: tier(unset), ticker, model, rmse, mae, n.

    The nan policy lives in :mod:`src.metrics`; a model with nothing scoreable
    raises there and is skipped rather than contributing an empty row.
    """
    rows: List[Dict] = []
    for ticker, model_dict in by_ticker.items():
        for model, df in model_dict.items():
            yt = df["y_true"].to_numpy(dtype=float)
            yp = df["y_pred"].to_numpy(dtype=float)
            try:
                row_rmse, row_mae = _rmse(yt, yp), _mae(yt, yp)
            except ValueError:
                continue  # nothing finite to score for this model
            rows.append({
                "ticker": ticker,
                "model": model,
                "rmse": row_rmse,
                "mae": row_mae,
                "n": _n_scored(yt, yp),
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


def _plot_summary_bars(
    summary: pd.DataFrame,
    *,
    title: str,
    out_path: Path,
) -> Path:
    """Bar chart with mean per model and (min, max) capped error bars.

    Two subplots side by side: RMSE on the left, MAE on the right.
    """
    if summary.empty:
        return save_placeholder(out_path)

    models = ordered_models(summary["model"].tolist())
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
    return save_figure(fig, out_path, dpi=150)


def _plot_cumulative(
    by_ticker: Dict[str, Dict[str, pd.DataFrame]],
    *,
    title: str,
    out_path: Path,
) -> Path:
    """Sum of squared errors across tickers, one line per model."""
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
        return save_placeholder(out_path)

    fig, ax = plt.subplots(figsize=(9, 5))
    for model in ordered_models(list(cum_by_model.keys())):
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
    return save_figure(fig, out_path, dpi=150)


# ---------------------------------------------------------------------------
# Score histogram
# ---------------------------------------------------------------------------


def _score_histogram(
    per_ticker: pd.DataFrame,
    *,
    exclude: Iterable[str] = _HISTOGRAM_EXCLUDE,
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
    if hist.empty:
        return save_placeholder(out_path)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    models = hist["model"].tolist()
    x = np.arange(len(models))
    ax.bar(x, hist["wins"].to_numpy(dtype=int),
           color=[color_for(m) for m in models],
           edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, fontsize=9)
    ax.set_ylabel("# tickers won")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return save_figure(fig, out_path, dpi=150)


# ---------------------------------------------------------------------------
# Best-model error vs. price level
# ---------------------------------------------------------------------------


def _best_causal_rmse(per_ticker: pd.DataFrame) -> pd.DataFrame:
    """Lowest RMSE per ticker among the causal candidates.

    Uses the same exclusions as the score histogram: ``naive`` is the trivial
    benchmark, ``global`` peeks at the test set, and ``ensemble`` is derived
    from the others — none is a candidate for "the best forecaster available".
    """
    if per_ticker.empty:
        return pd.DataFrame(columns=["ticker", "model", "rmse"])
    candidates = per_ticker[~per_ticker["model"].isin(_HISTOGRAM_EXCLUDE)]
    if candidates.empty:
        return pd.DataFrame(columns=["ticker", "model", "rmse"])
    best = candidates.loc[candidates.groupby("ticker")["rmse"].idxmin()]
    return best[["ticker", "model", "rmse"]].reset_index(drop=True)


def _loglog_slope(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """OLS slope of ``log10(y)`` on ``log10(x)``, or ``None`` if undefined.

    A slope here is a power-law exponent: ``rmse ~ price ** slope``.
    """
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if ok.sum() < 2:
        return None
    return float(np.polyfit(np.log10(x[ok]), np.log10(y[ok]), 1)[0])


def _plot_error_vs_price(
    best: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    out_path: Path,
) -> Path:
    """Best-model RMSE against mean adjusted close, log-log, coloured by tier.

    One panel, deliberately. An earlier hand-made version of this figure drew
    RMSE and realised volatility side by side, but for a central-tendency
    forecaster those are the same quantity: predicting the mean makes the
    root-mean-squared error the sample standard deviation, so the second panel
    restated the first. The identity is stated in the axis label instead.
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))
    best = best.assign(ticker=best["ticker"].astype(str))
    prices = prices.assign(ticker=prices["ticker"].astype(str))
    merged = best.merge(prices, on="ticker", how="inner")
    merged = merged[np.isfinite(merged["mean_price"]) & (merged["mean_price"] > 0)]

    if merged.empty:
        plt.close(fig)
        return save_placeholder(out_path, dpi=200)

    tier_palette = ["#2ca02c", "#ff7f0e", "#1f77b4"]
    for i, tier in enumerate(sorted(merged["tier"].unique())):
        sub = merged[merged["tier"] == tier]
        ax.scatter(
            sub["mean_price"], sub["rmse"],
            color=tier_palette[i % len(tier_palette)], s=34, alpha=0.85,
            edgecolor="black", linewidth=0.4, label=str(tier), zorder=2,
        )

    x = merged["mean_price"].to_numpy(dtype=float)
    y = merged["rmse"].to_numpy(dtype=float)
    slope = _loglog_slope(x, y)
    if slope is not None:
        intercept = float(
            np.mean(np.log10(y)) - slope * np.mean(np.log10(x))
        )
        xs = np.linspace(np.log10(x.min()), np.log10(x.max()), 50)
        ax.plot(
            10 ** xs, 10 ** (intercept + slope * xs),
            color="black", linestyle="--", linewidth=1.6,
            label=f"fit: slope = {slope:.2f}", zorder=3,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("mean Adj Close ($, log scale)")
    ax.set_ylabel("best causal model RMSE = realised volatility (log scale)")
    title = "Best-predictor error vs. price level (log-log)"
    if slope is not None:
        title += f" — slope {slope:.2f}"
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=True)
    fig.tight_layout()
    return save_figure(fig, out_path, dpi=200)


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


def _read_ticker_prices(test_root: Path) -> pd.DataFrame:
    """Return ``tier, ticker, mean_price`` from a run's ``ticker_tested.csv``.

    Returns an empty frame when the file is missing or predates the
    ``mean_price`` column, so a caller can skip the price figure rather than
    fail the whole summary over a derived artifact.
    """
    path = Path(test_root) / "ticker_tested.csv"
    if not path.is_file():
        return pd.DataFrame(columns=["tier", "ticker", "mean_price"])
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["tier", "ticker", "mean_price"])
    if not {"tier", "ticker", "mean_price"}.issubset(df.columns):
        return pd.DataFrame(columns=["tier", "ticker", "mean_price"])
    out = df[["tier", "ticker", "mean_price"]].dropna(subset=["mean_price"]).copy()
    # Symbols come back from CSV as int64 when they happen to look numeric, and
    # as str everywhere they are parsed out of a filename. Normalise here so the
    # join below cannot fail on dtype.
    out["ticker"] = out["ticker"].astype(str)
    return out


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

    hist = _score_histogram(per_ticker)
    hist_csv = out_dir / "score_histogram.csv"
    hist.to_csv(hist_csv, index=False)
    written.append(str(hist_csv))
    written.append(str(_plot_score_histogram(
        hist, out_path=out_dir / "score_histogram.png",
    )))

    # Error against price level. Needs the per-ticker mean price the runner
    # recorded in ticker_tested.csv; if that file is absent (an older run tree,
    # or summarise_overall called directly), the figure is simply skipped —
    # every other output above is already written by this point.
    prices = _read_ticker_prices(test_root)
    if not prices.empty:
        best = _best_causal_rmse(per_ticker)
        # per_ticker keys tickers as "<tier>/<TICKER>" across tiers; split it
        # back so the rows can be matched against ticker_tested.csv.
        if not best.empty:
            split = best["ticker"].str.split("/", n=1, expand=True)
            best = best.assign(ticker=split[1].fillna(split[0]))
        written.append(str(_plot_error_vs_price(
            best, prices, out_path=out_dir / "best_predictor_vs_price.png",
        )))
    return written


__all__ = ["summarise_tier", "summarise_overall"]
