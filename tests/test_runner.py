"""Tests for ``src.runner`` — Phase D smoke coverage.

These tests must run fast (< 30s wall time): they use a tiny synthetic
universe of 4 tickers, a single rolling window of 60, and the NaiveModel
only.  No network calls — the SQLite cache is pre-populated via
``put_history`` so the selector accepts every candidate without invoking
the default yfinance loaders.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

from src.models import NaiveModel
from src.runner import main as runner_main, run_test
from src.selection import TierSpec
from src.storage.db import init_schema, open_db, put_history


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_cache(db_path: Path, tickers: List[str], price: float = 12.0) -> None:
    """Populate the cache with a series spanning the new pipeline's dates.

    Burn-in: 2022-08-15 → 2022-12-30. Test: 2023-01-01 → 2023-12-31.
    The synthetic series covers both with light noise so log-returns are
    well-defined.
    """
    conn = open_db(str(db_path))
    init_schema(conn)
    idx = pd.date_range(start="2022-08-15", end="2023-12-31", freq="B")
    rng = np.random.default_rng(0)
    noise = rng.normal(loc=0.0, scale=0.02, size=len(idx))
    values = price * np.exp(noise.cumsum() / 50.0)
    series = pd.Series(values, index=idx, name="adj_close")
    for sym in tickers:
        put_history(
            conn,
            sym,
            tier="tier1",
            start="2022-08-15",
            end="2023-12-31",
            prices=series,
        )
    conn.close()


def _write_universe(path: Path, tickers: List[str]) -> None:
    path.write_text("\n".join(tickers) + "\n", encoding="utf-8")


@pytest.fixture()
def tiny_setup(tmp_path: Path):
    """Build a 4-ticker universe + cache + 1-tier spec, return paths."""
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    universe_file = tmp_path / "universe.txt"
    _write_universe(universe_file, tickers)
    db_path = tmp_path / "ticker_data" / "cache.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _seed_cache(db_path, tickers, price=12.0)
    out_root = tmp_path / "results"
    out_root.mkdir(parents=True, exist_ok=True)
    spec = TierSpec(
        start_price=10.0, end_price=20.0,
        below_threshold=5.0, upper_threshold=40.0,
        target_count=2,
    )
    return {
        "tickers": tickers,
        "universe_file": universe_file,
        "db_path": db_path,
        "out_root": out_root,
        "spec": spec,
    }


def _stub_current_price(tickers, price=12.0):
    return lambda sym: price if sym in tickers else None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_run_test_smoke_tier1_only(tiny_setup, monkeypatch):
    """End-to-end smoke: run_test produces the expected directory tree."""
    # Stub current-price so the selector's pre-filter doesn't try yfinance.
    from src import selection as _sel
    monkeypatch.setattr(
        _sel, "_default_current_price",
        _stub_current_price(tiny_setup["tickers"]),
    )

    test_root = run_test(
        tier_specs={"tier1": tiny_setup["spec"]},
        universe_file=tiny_setup["universe_file"],
        db_path=tiny_setup["db_path"],
        out_root=tiny_setup["out_root"],
        seed=0,
        tiers_subset=["tier1"],
        windows=(60,),
        models_factory=lambda: [NaiveModel()],
    )

    # Returned path exists and matches the expected naming pattern.
    assert isinstance(test_root, Path)
    assert test_root.exists()
    assert test_root.name.startswith("test_")

    # Tier1 dirs.
    tier_dir = test_root / "tier1"
    for sub in ("individual", "grouped", "predictions", "analysis"):
        assert (tier_dir / sub).is_dir(), f"missing dir: {sub}"

    # New schema: <TICKER>_<MODEL>.csv. With NaiveModel only and no ensemble
    # (ensemble requires its specific children), 2 tickers × 1 model = 2 CSVs.
    pred_csvs = sorted((tier_dir / "predictions").glob("*.csv"))
    assert len(pred_csvs) == 2, f"expected 2 prediction CSVs; got {len(pred_csvs)}"
    naive_csvs = [p for p in pred_csvs if p.name.endswith("_naive.csv")]
    assert len(naive_csvs) == 2

    # Per-pair figures from the registry land under individual/.
    indiv_dir = tier_dir / "individual"
    indiv_pngs = sorted(indiv_dir.glob("*.png"))
    assert len(indiv_pngs) >= 2, (
        f"expected at least 2 individual PNGs; got {len(indiv_pngs)}"
    )
    # Registry sidecar CSVs land under individual/data/.
    indiv_data_dir = indiv_dir / "data"
    assert indiv_data_dir.is_dir()

    # metrics.csv schema (no window column).
    metrics_path = test_root / "metrics.csv"
    assert metrics_path.exists()
    metrics = pd.read_csv(metrics_path)
    assert list(metrics.columns) == ["tier", "ticker", "model", "rmse", "mae", "n"]
    assert len(metrics) >= 2
    assert set(metrics["tier"].unique()) == {"tier1"}
    assert set(metrics["model"].unique()).issubset({"naive"})

    # ticker_tested.csv.
    tt_path = test_root / "ticker_tested.csv"
    assert tt_path.exists()
    tt = pd.read_csv(tt_path)
    assert list(tt.columns) == ["tier", "ticker"]
    assert len(tt) == 2

    # manifest.json — keys (new schema, no windows column).
    mf_path = test_root / "manifest.json"
    assert mf_path.exists()
    mf = json.loads(mf_path.read_text())
    expected_keys = {
        "created_at", "seed", "tiers", "target_per_tier_effective",
        "universe_file", "n_tickers_total", "test_start_date",
        "test_end_date", "runtime_seconds", "python_version",
    }
    assert expected_keys.issubset(set(mf.keys()))
    assert mf["seed"] == 0
    assert mf["tiers"] == ["tier1"]
    assert mf["n_tickers_total"] == 2


def test_cli_main_smoke(tiny_setup, monkeypatch, capsys):
    """``main([...])`` direct invocation drives ``run_test`` via the CLI parser.

    We monkeypatch the runner's default-resolution helpers so the CLI uses
    the tmp universe / db / out_root and skips any yfinance access.
    """
    from src import runner as runner_mod
    from src import selection as _sel

    monkeypatch.setattr(
        _sel, "_default_current_price",
        _stub_current_price(tiny_setup["tickers"]),
    )
    monkeypatch.setattr(
        runner_mod, "_resolve_default_tiers",
        lambda: {"tier1": tiny_setup["spec"]},
    )
    monkeypatch.setattr(
        runner_mod, "_resolve_default_paths",
        lambda: (
            tiny_setup["universe_file"],
            tiny_setup["db_path"],
            tiny_setup["out_root"],
        ),
    )
    # Force the CLI's window default to a single small window for speed.
    monkeypatch.setattr(runner_mod.cfg, "ROLLING_WINDOWS", (60,), raising=False)
    # Stub default_models to NaiveModel only — keeps the smoke run < 1s and
    # avoids the ARMA budget interaction.
    monkeypatch.setattr(
        runner_mod._models, "default_models", lambda: [NaiveModel()],
    )

    rc = runner_main([
        "--target", "2",
        "--tiers", "tier1",
        "--seed", "0",
    ])
    assert rc == 0
    out = capsys.readouterr().out.strip().splitlines()[-1]
    test_root = Path(out)
    assert test_root.exists()
    assert (test_root / "tier1" / "predictions").is_dir()
    assert (test_root / "metrics.csv").exists()
