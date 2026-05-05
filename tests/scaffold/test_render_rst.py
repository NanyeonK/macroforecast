"""Pin the sphinx auto-emit produces stable RST + handles missing
OptionDoc entries gracefully."""
from __future__ import annotations

from pathlib import Path

import pytest

from macroforecast.scaffold import render_rst


def test_render_index_lists_every_layer():
    rst = render_rst.render_index()
    assert "Reference" in rst
    for layer_id in ("l0", "l1", "l4", "l8"):
        assert f"   {layer_id}" in rst


def test_render_layer_l0_includes_documented_option():
    """L0 ``fail_fast`` has a Tier-1 OptionDoc; the rendered RST must
    surface its summary and references."""
    rst = render_rst.render_layer("l0")
    assert "``fail_fast``" in rst
    assert "Stop the entire study on the first cell" in rst
    assert "When to use" in rst
    assert "fail_fast vs continue_on_failure" in rst  # reference text


def test_render_layer_handles_missing_optiondoc_with_warning(monkeypatch):
    """Layers without registered docs surface the schema description and
    a sphinx ``warning`` directive. v1.0 has 100% Tier-1 coverage, so we
    simulate a missing entry by popping a known key from the registry
    inside the test (restored automatically by ``monkeypatch``)."""
    from macroforecast.scaffold.option_docs import OPTION_DOCS
    key = ("l0", "l0_a", "failure_policy", "fail_fast")
    assert key in OPTION_DOCS, "fixture target missing from registry"
    monkeypatch.delitem(OPTION_DOCS, key)
    rst = render_rst.render_layer("l0")
    assert ".. warning:: Detailed OptionDoc not yet registered" in rst


def test_write_all_creates_one_file_per_layer(tmp_path):
    written = render_rst.write_all(tmp_path)
    # 13 layers + 1 index.
    assert len(written) == 14
    for path in written:
        assert path.exists()
        assert path.suffix == ".rst"
        # Each file has at least one section header.
        content = path.read_text()
        assert "=" in content


def test_render_l4_handles_many_axes_without_crash():
    rst = render_rst.render_layer("l4")
    assert "L4" in rst
    # Every fit_model family option should at least surface as a section
    # (whether documented yet or placeholder).
    assert "ridge" in rst
    assert "random_forest" in rst
