"""End-to-end orchestrator for the analysis stage.

Sweeps ``<test_root>/<tier>/predictions/*.csv``, computes per-step + summary
statistics, persists into a shared ``results/analysis.db``, and renders the
unlabeled vertical dot plots specified in ``measurement_pipeline.md`` §12.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.analysis.compute import compute_per_step, compute_summary
from src.analysis.db import init_analysis_schema, open_analysis_db
from src.analysis.dotplot import render_combined_dotplot, render_dotplot
from src.analysis.persist import persist_per_step, persist_summary

logger = logging.getLogger(__name__)

_PRED_RE = re.compile(r"^(?P<ticker>[A-Za-z0-9]+)_(?P<window>\d+)_(?P<model>[A-Za-z0-9]+)\.csv$")


def _scan_tier_predictions(tier_dir: Path) -> List[Dict[str, Any]]:
    """Yield per-file metadata for the predictions in one tier."""
    out: List[Dict[str, Any]] = []
    pred_dir = tier_dir / "predictions"
    if not pred_dir.is_dir():
        return out
    for path in sorted(pred_dir.iterdir()):
        m = _PRED_RE.match(path.name)
        if not m:
            continue
        out.append({
            "path": path,
            "ticker": m.group("ticker"),
            "window": int(m.group("window")),
            "model": m.group("model"),
        })
    return out


def analyse_test_run(test_root: Path, *, db_path: Optional[Path] = None) -> Path:
    """Walk the per-tier predictions tree, persist analysis rows, and render
    dotplots. Returns the analysis DB path. Anchors the DB at
    ``test_root.parent / "analysis.db"`` by default so multiple runs share it
    (the ``test_run`` PK column distinguishes them).
    """
    test_root = Path(test_root)
    if db_path is None:
        db_path = test_root.parent / "analysis.db"
    db_path = Path(db_path)
    test_run = test_root.name

    conn = open_analysis_db(str(db_path))
    init_analysis_schema(conn)

    tier_dirs = sorted(
        d for d in test_root.iterdir()
        if d.is_dir() and d.name.startswith("tier")
    )

    summaries_all: List[Dict[str, Any]] = []
    summaries_by_tier: Dict[str, pd.DataFrame] = {}

    for tier_dir in tier_dirs:
        tier = tier_dir.name
        rows = _scan_tier_predictions(tier_dir)
        per_tier_summaries: List[Dict[str, Any]] = []
        for r in rows:
            try:
                per_step = compute_per_step(r["path"])
            except Exception as exc:
                logger.warning("compute_per_step failed for %s: %s", r["path"], exc)
                continue
            stats = compute_summary(per_step)
            persist_per_step(conn, test_run, tier, r["ticker"],
                             r["window"], r["model"], per_step)
            entry = {
                "tier": tier, "ticker": r["ticker"],
                "window": r["window"], "model": r["model"],
                **stats,
            }
            per_tier_summaries.append(entry)
            summaries_all.append(entry)

        persist_summary(conn, test_run, per_tier_summaries)

        if per_tier_summaries:
            df = pd.DataFrame(per_tier_summaries)[
                ["ticker", "window", "model", "rmse", "mae"]
            ]
            summaries_by_tier[tier] = df
            analysis_dir = tier_dir / "analysis"
            os.makedirs(analysis_dir, exist_ok=True)
            for metric in ("rmse", "mae"):
                render_dotplot(
                    df,
                    metric=metric,
                    title=f"{tier} — {metric.upper()} per stock",
                    out_path=analysis_dir / f"dotplot_{metric}.png",
                )

    if summaries_by_tier:
        cross_dir = test_root / "analysis"
        os.makedirs(cross_dir, exist_ok=True)
        for metric in ("rmse", "mae"):
            render_combined_dotplot(
                summaries_by_tier,
                metric=metric,
                out_path=cross_dir / f"all_tiers_dotplot_{metric}.png",
            )

    conn.close()
    return db_path


__all__ = ["analyse_test_run"]
