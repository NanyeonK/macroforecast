"""Tests for L1.A ParameterDoc population (Cycle 17 O-2).

Verifies that custom_source_policy options have ParameterDoc entries
for their conditional leaf_config keys.
"""
from __future__ import annotations

import pytest

from macroforecast.scaffold.option_docs import OPTION_DOCS
from macroforecast.scaffold.option_docs.types import ParameterDoc, REQUIRED


# ---------------------------------------------------------------------------
# custom_panel_only — three mutually-exclusive leaf_config keys
# ---------------------------------------------------------------------------


def test_l1a_custom_panel_only_has_parameters():
    """custom_panel_only OptionDoc has 3 ParameterDoc entries."""
    doc = OPTION_DOCS[("l1", "l1_a", "custom_source_policy", "custom_panel_only")]
    assert isinstance(doc.parameters, tuple)
    assert len(doc.parameters) == 3
    names = {p.name for p in doc.parameters}
    assert {"custom_source_path", "custom_panel_inline", "custom_panel_records"}.issubset(names)


def test_l1a_custom_panel_only_param_types():
    """custom_panel_only parameter types are correctly set."""
    doc = OPTION_DOCS[("l1", "l1_a", "custom_source_policy", "custom_panel_only")]
    type_map = {p.name: p.type for p in doc.parameters}
    assert "str" in type_map["custom_source_path"]
    assert type_map["custom_panel_inline"] == "dict"
    assert "list" in type_map["custom_panel_records"]


def test_l1a_custom_panel_only_all_params_default_none():
    """All custom_panel_only parameters default to None (they are required conditionally)."""
    doc = OPTION_DOCS[("l1", "l1_a", "custom_source_policy", "custom_panel_only")]
    for p in doc.parameters:
        assert p.default is REQUIRED, f"Expected default=REQUIRED for {p.name}, got {p.default}"


def test_l1a_custom_panel_only_params_have_constraint():
    """All custom_panel_only parameters have a mutual-exclusion constraint string."""
    doc = OPTION_DOCS[("l1", "l1_a", "custom_source_policy", "custom_panel_only")]
    for p in doc.parameters:
        assert p.constraint is not None, f"Missing constraint on {p.name}"
        assert "one of" in p.constraint.lower() or "exactly" in p.constraint.lower(), (
            f"Constraint for {p.name} should mention mutual exclusion: {p.constraint!r}"
        )


# ---------------------------------------------------------------------------
# official_plus_custom — custom_source_path + custom_merge_rule
# ---------------------------------------------------------------------------


def test_l1a_official_plus_custom_has_parameters():
    """official_plus_custom OptionDoc has 2 ParameterDoc entries."""
    doc = OPTION_DOCS[("l1", "l1_a", "custom_source_policy", "official_plus_custom")]
    assert isinstance(doc.parameters, tuple)
    assert len(doc.parameters) == 2
    names = {p.name for p in doc.parameters}
    assert {"custom_source_path", "custom_merge_rule"}.issubset(names)


def test_l1a_official_plus_custom_merge_rule_type():
    """custom_merge_rule ParameterDoc has str type and valid join enum in constraint."""
    doc = OPTION_DOCS[("l1", "l1_a", "custom_source_policy", "official_plus_custom")]
    mr = next(p for p in doc.parameters if p.name == "custom_merge_rule")
    assert mr.type == "str"
    assert mr.default is REQUIRED  # REQUIRED sentinel (C26)
    assert mr.constraint is not None
    # Constraint should enumerate the allowed join values
    assert "left_join" in mr.constraint
    assert "inner_join" in mr.constraint
    assert "outer_join" in mr.constraint


def test_l1a_official_plus_custom_source_path_type():
    """custom_source_path ParameterDoc has str | Path type."""
    doc = OPTION_DOCS[("l1", "l1_a", "custom_source_policy", "official_plus_custom")]
    sp = next(p for p in doc.parameters if p.name == "custom_source_path")
    assert "str" in sp.type
    assert sp.default is REQUIRED  # REQUIRED sentinel (C26)


# ---------------------------------------------------------------------------
# official_only — no parameters (no conditional leaf_config keys)
# ---------------------------------------------------------------------------


def test_l1a_official_only_has_no_parameters():
    """official_only has no conditional leaf_config parameters."""
    doc = OPTION_DOCS[("l1", "l1_a", "custom_source_policy", "official_only")]
    assert doc.parameters == ()


# ---------------------------------------------------------------------------
# encyclopedia rendering: Parameters table present where applicable
# ---------------------------------------------------------------------------


def test_l1a_custom_source_policy_page_contains_parameters_table():
    """Encyclopedia page for custom_source_policy renders **Parameters** table."""
    from macroforecast.scaffold.render_encyclopedia import _render_axis_page
    from macroforecast.scaffold.introspect import axes

    l1_axes = {ax.name: ax for ax in axes("l1")}
    page = _render_axis_page("l1", l1_axes["custom_source_policy"])
    assert "**Parameters**" in page
    assert "custom_source_path" in page
    assert "custom_panel_inline" in page
    assert "custom_panel_records" in page
    assert "custom_merge_rule" in page


def test_l1a_official_only_section_no_parameters_table():
    """official_only section in the rendered page has no Parameters table immediately after it."""
    from macroforecast.scaffold.render_encyclopedia import _render_axis_page
    from macroforecast.scaffold.introspect import axes

    l1_axes = {ax.name: ax for ax in axes("l1")}
    page = _render_axis_page("l1", l1_axes["custom_source_policy"])
    # official_only block should not contain a Parameters table
    # (split at official_only section heading and check next 200 chars)
    idx = page.find("`official_only`")
    assert idx != -1, "official_only heading not found in rendered page"
    snippet = page[idx : idx + 300]
    assert "**Parameters**" not in snippet, (
        "official_only section should not have a Parameters table"
    )
