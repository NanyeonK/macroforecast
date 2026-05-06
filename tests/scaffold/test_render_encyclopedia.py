"""Pin the encyclopedia renderer:

* ``write_all`` produces enough files to be a real encyclopedia.
* Every layer has at least an ``index.md`` and one axis page.
* ``browse_by_option`` surfaces the L4 model families (>= 30 entries).
* The CLI smoke-test runs end-to-end on a temp directory.
* Missing OptionDoc entries fall through to the explicit "TBD" placeholder
  rather than crashing.
"""
from __future__ import annotations

from pathlib import Path

from macroforecast.scaffold import introspect, render_encyclopedia


def test_write_all_produces_at_least_150_files(tmp_path: Path) -> None:
    written = render_encyclopedia.write_all(tmp_path)
    assert len(written) >= 150
    for path in written:
        assert path.exists(), path


def test_every_layer_has_index_and_at_least_one_axis_page(tmp_path: Path) -> None:
    render_encyclopedia.write_all(tmp_path)
    for layer_id in introspect.list_layers():
        layer_dir = tmp_path / layer_id
        assert (layer_dir / "index.md").exists(), layer_id
        axes_dir = layer_dir / "axes"
        # Every layer in the canonical 13 has at least one axis after the
        # introspect fallback (L3 ``op`` axis, L7 ``op`` axis, ...).
        axis_files = list(axes_dir.glob("*.md"))
        assert axis_files, f"no axis pages emitted under {axes_dir}"


def test_browse_by_option_lists_at_least_30_l4_families(tmp_path: Path) -> None:
    render_encyclopedia.write_all(tmp_path)
    body = (tmp_path / "browse_by_option.md").read_text(encoding="utf-8")
    # Spot-check several canonical L4 model families.
    for family in ("ridge", "lasso", "elastic_net", "random_forest", "xgboost"):
        assert f"`{family}`" in body, family
    # Crude lower bound on row count: count pipe-prefixed table rows.
    table_rows = [line for line in body.splitlines() if line.startswith("| [`")]
    assert len(table_rows) >= 30


def test_top_index_links_to_browse_pages(tmp_path: Path) -> None:
    render_encyclopedia.write_all(tmp_path)
    index = (tmp_path / "index.md").read_text(encoding="utf-8")
    assert "browse_by_layer.md" in index
    assert "browse_by_axis.md" in index
    assert "browse_by_option.md" in index
    assert "public_api.md" in index


def test_public_api_page_preserved(tmp_path: Path) -> None:
    render_encyclopedia.write_all(tmp_path)
    body = (tmp_path / "public_api.md").read_text(encoding="utf-8")
    assert "macroforecast.run" in body
    assert "macroforecast.replicate" in body


def test_l4_family_axis_includes_documented_option(tmp_path: Path) -> None:
    render_encyclopedia.write_all(tmp_path)
    family_md = (tmp_path / "l4" / "axes" / "family.md").read_text(encoding="utf-8")
    # ar_p has a Tier-1 OptionDoc registered; full prose must surface.
    assert "### `ar_p`" in family_md
    assert "Autoregressive AR(p)" in family_md
    # Future options would carry the "future" badge -- just check shape.
    assert "## Operational status summary" in family_md


def test_missing_optiondoc_yields_tbd_placeholder(tmp_path: Path, monkeypatch) -> None:
    """Pop a known OptionDoc and verify the encyclopedia falls back to
    the explicit TBD placeholder instead of crashing."""

    from macroforecast.scaffold.option_docs import OPTION_DOCS

    key = ("l0", "l0_a", "failure_policy", "fail_fast")
    assert key in OPTION_DOCS, "fixture missing"
    monkeypatch.delitem(OPTION_DOCS, key)

    render_encyclopedia.write_all(tmp_path)
    body = (tmp_path / "l0" / "axes" / "failure_policy.md").read_text(encoding="utf-8")
    assert "TBD: option doc not yet authored" in body


def test_cli_smoke(tmp_path: Path) -> None:
    """``macroforecast scaffold`` CLI exposes the ``encyclopedia``
    subcommand as required by the v0.7.0 CI sync gate."""

    from macroforecast.scaffold import cli

    rc = cli.main(["encyclopedia", str(tmp_path)])
    assert rc == 0
    # At minimum, the top-level index + browse pages must land.
    for name in ("index.md", "browse_by_layer.md", "browse_by_axis.md", "browse_by_option.md", "public_api.md"):
        assert (tmp_path / name).exists(), name


def test_module_main_smoke(tmp_path: Path) -> None:
    """``python -m macroforecast.scaffold encyclopedia <out>`` is the form
    used by the CI sync gate; exercise it through the same dispatcher."""

    from macroforecast.scaffold import __main__ as scaffold_main

    rc = scaffold_main._main(["encyclopedia", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "index.md").exists()


def test_browse_by_axis_lists_every_axis(tmp_path: Path) -> None:
    render_encyclopedia.write_all(tmp_path)
    body = (tmp_path / "browse_by_axis.md").read_text(encoding="utf-8")
    # Every layer's axes must show up. Grab a representative axis from
    # each layer and check the link text appears.
    for axis_name in ("failure_policy", "dataset", "family", "search_algorithm"):
        assert f"`{axis_name}`" in body, axis_name


def test_diagnostic_layers_have_pages(tmp_path: Path) -> None:
    render_encyclopedia.write_all(tmp_path)
    for layer_id in ("l1_5", "l2_5", "l3_5", "l4_5"):
        assert (tmp_path / layer_id / "index.md").exists(), layer_id
        # Each diagnostic layer carries at least one axis (selection / scope).
        axis_files = list((tmp_path / layer_id / "axes").glob("*.md"))
        assert axis_files, layer_id
