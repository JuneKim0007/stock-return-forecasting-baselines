"""Tests for ``src.plots.figure_dirs`` — Phase G context-manager primitive.

The per-pair figure registry and group-analysis renderers all funnel
through ``save_fig_and_data``, which reads ``config.FIGURES_DIR`` /
``config.FIGURES_DATA_DIR`` from the module-level constants on
``src.config``. ``figure_dirs`` swaps those constants for the duration of
a ``with`` block so callers can route output to a tier-scoped directory
without touching every helper. These tests pin down the swap-and-restore
contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src import config as cfg
from src.plots import figure_dirs


def test_figure_dirs_swaps_and_restores(tmp_path: Path) -> None:
    """Inside the block the constants point at the new dirs; outside they're back."""
    prev_fig = cfg.FIGURES_DIR
    prev_data = cfg.FIGURES_DATA_DIR

    new_fig = tmp_path / "a"
    new_data = tmp_path / "b"

    with figure_dirs(str(new_fig), str(new_data)):
        assert cfg.FIGURES_DIR == str(new_fig)
        assert cfg.FIGURES_DATA_DIR == str(new_data)
        # The CM should also create the directories so callers can write
        # immediately without an extra os.makedirs call.
        assert new_fig.is_dir()
        assert new_data.is_dir()

    # Restored on normal exit.
    assert cfg.FIGURES_DIR == prev_fig
    assert cfg.FIGURES_DATA_DIR == prev_data


def test_figure_dirs_restores_on_exception(tmp_path: Path) -> None:
    """An exception inside the block must still restore the previous values."""
    prev_fig = cfg.FIGURES_DIR
    prev_data = cfg.FIGURES_DATA_DIR

    with pytest.raises(RuntimeError, match="boom"):
        with figure_dirs(str(tmp_path / "x"), str(tmp_path / "y")):
            assert cfg.FIGURES_DIR == str(tmp_path / "x")
            assert cfg.FIGURES_DATA_DIR == str(tmp_path / "y")
            raise RuntimeError("boom")

    assert cfg.FIGURES_DIR == prev_fig
    assert cfg.FIGURES_DATA_DIR == prev_data
