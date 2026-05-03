from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from macrocast.core.figures import (
    US_STATE_GRID,
    render_bar_global,
    render_heatmap,
    render_us_state_choropleth,
)


def test_us_state_grid_covers_50_states_plus_dc():
    assert len(US_STATE_GRID) >= 51
    assert "DC" in US_STATE_GRID
    assert "CA" in US_STATE_GRID and "FL" in US_STATE_GRID and "AK" in US_STATE_GRID


def test_render_us_state_choropleth_writes_pdf(tmp_path: Path):
    importance = {
        "CA": 1.0, "TX": 0.7, "NY": 0.5, "FL": 0.3, "WA": 0.2, "MA": 0.4,
    }
    out = tmp_path / "state_importance.pdf"
    rendered = render_us_state_choropleth(importance, output_path=out, title="Importance by state")
    assert rendered == out and out.exists() and out.stat().st_size > 1024


def test_render_bar_global_writes_image(tmp_path: Path):
    table = pd.DataFrame({"feature": [f"f_{i}" for i in range(10)], "importance": list(range(1, 11))})
    out = tmp_path / "bar.png"
    rendered = render_bar_global(table, output_path=out, top_k=8, title="Top features")
    assert rendered.exists() and rendered.stat().st_size > 512


def test_render_heatmap(tmp_path: Path):
    table = pd.DataFrame(
        {"col_a": [1.0, 2.0], "col_b": [0.5, 0.1]},
        index=pd.Index(["row_1", "row_2"], name="feature"),
    )
    out = tmp_path / "heatmap.png"
    render_heatmap(table, output_path=out, title="Test")
    assert out.exists()
