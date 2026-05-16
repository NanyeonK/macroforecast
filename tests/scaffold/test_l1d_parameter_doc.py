"""Tests for L1.D ParameterDoc population (Cycle 20).

Verifies that L1.D Geography axis options have ParameterDoc entries for
their conditional leaf_config keys:
- target_geography_scope/selected_states  -> target_states
- predictor_geography_scope/selected_states -> predictor_states
- state_selection/selected_states         -> sd_states
- sd_variable_selection/selected_sd_variables -> sd_variables
- fred_sd_state_group/custom_state_group  -> sd_state_group_members OR sd_state_groups
- fred_sd_variable_group/custom_sd_variable_group -> sd_variable_group_members OR sd_variable_groups
"""
from __future__ import annotations

import pytest

from macroforecast.scaffold.option_docs import OPTION_DOCS
from macroforecast.scaffold.option_docs.types import ParameterDoc, REQUIRED


# ---------------------------------------------------------------------------
# target_geography_scope
# ---------------------------------------------------------------------------


def test_l1d_target_geography_selected_states_has_target_states():
    """selected_states has 1 ParameterDoc for target_states."""
    doc = OPTION_DOCS[("l1", "l1_d", "target_geography_scope", "selected_states")]
    assert isinstance(doc.parameters, tuple)
    assert len(doc.parameters) == 1
    p = doc.parameters[0]
    assert p.name == "target_states"
    assert "list" in p.type
    assert p.default is REQUIRED
    assert p.constraint is not None
    assert "non-empty" in p.constraint.lower() or "required" in p.constraint.lower()


def test_l1d_target_geography_other_options_no_params():
    """all_states has no conditional parameters."""
    doc = OPTION_DOCS[("l1", "l1_d", "target_geography_scope", "all_states")]
    assert doc.parameters == (), "all_states should have parameters=()"


def test_l1d_target_geography_single_state_has_target_state():
    """single_state has 1 ParameterDoc for target_state (singular).

    C20 follow-up: pre-existing gap surfaced by reviewer cross-model grep.
    """
    doc = OPTION_DOCS[("l1", "l1_d", "target_geography_scope", "single_state")]
    assert isinstance(doc.parameters, tuple)
    assert len(doc.parameters) == 1
    p = doc.parameters[0]
    assert p.name == "target_state"
    assert p.type == "str"
    assert p.default is REQUIRED
    assert p.constraint is not None
    assert "required" in p.constraint.lower() or "single_state" in p.constraint.lower()
    assert "CA" in p.description or "TX" in p.description


# ---------------------------------------------------------------------------
# predictor_geography_scope
# ---------------------------------------------------------------------------


def test_l1d_predictor_geography_selected_states_has_predictor_states():
    """selected_states has 1 ParameterDoc for predictor_states."""
    doc = OPTION_DOCS[("l1", "l1_d", "predictor_geography_scope", "selected_states")]
    assert isinstance(doc.parameters, tuple)
    assert len(doc.parameters) == 1
    p = doc.parameters[0]
    assert p.name == "predictor_states"
    assert "list" in p.type
    assert p.default is REQUIRED
    assert p.constraint is not None
    # Description must mention independence from target_states
    assert "independent" in p.description.lower() or "cross-state" in p.description.lower()


def test_l1d_predictor_geography_other_options_no_params():
    """match_target, all_states, national_only have no conditional parameters."""
    for option in ("match_target", "all_states", "national_only"):
        doc = OPTION_DOCS[("l1", "l1_d", "predictor_geography_scope", option)]
        assert doc.parameters == (), f"{option} should have parameters=()"


# ---------------------------------------------------------------------------
# state_selection
# ---------------------------------------------------------------------------


def test_l1d_state_selection_selected_states_has_sd_states():
    """selected_states has 1 ParameterDoc for sd_states."""
    doc = OPTION_DOCS[("l1", "l1_d", "state_selection", "selected_states")]
    assert isinstance(doc.parameters, tuple)
    assert len(doc.parameters) == 1
    p = doc.parameters[0]
    assert p.name == "sd_states"
    assert "list" in p.type
    assert p.default is REQUIRED
    assert p.constraint is not None
    # Description must mention AFTER / intersect semantics
    assert "after" in p.description.lower() or "intersect" in p.description.lower()


def test_l1d_state_selection_all_states_no_params():
    """all_states has no conditional parameters."""
    doc = OPTION_DOCS[("l1", "l1_d", "state_selection", "all_states")]
    assert doc.parameters == ()


# ---------------------------------------------------------------------------
# sd_variable_selection
# ---------------------------------------------------------------------------


def test_l1d_sd_variable_selection_selected_has_sd_variables():
    """selected_sd_variables has 1 ParameterDoc for sd_variables."""
    doc = OPTION_DOCS[("l1", "l1_d", "sd_variable_selection", "selected_sd_variables")]
    assert isinstance(doc.parameters, tuple)
    assert len(doc.parameters) == 1
    p = doc.parameters[0]
    assert p.name == "sd_variables"
    assert "list" in p.type
    assert p.default is REQUIRED
    assert p.constraint is not None


def test_l1d_sd_variable_selection_all_no_params():
    """all_sd_variables has no conditional parameters."""
    doc = OPTION_DOCS[("l1", "l1_d", "sd_variable_selection", "all_sd_variables")]
    assert doc.parameters == ()


# ---------------------------------------------------------------------------
# fred_sd_state_group
# ---------------------------------------------------------------------------


def test_l1d_fred_sd_state_group_custom_has_two_params():
    """custom_state_group has 2 ParameterDoc entries (OR relationship)."""
    doc = OPTION_DOCS[("l1", "l1_d", "fred_sd_state_group", "custom_state_group")]
    assert isinstance(doc.parameters, tuple)
    assert len(doc.parameters) == 2
    names = {p.name for p in doc.parameters}
    assert "sd_state_group_members" in names
    assert "sd_state_groups" in names


def test_l1d_fred_sd_state_group_custom_members_type():
    """sd_state_group_members has type list[str]."""
    doc = OPTION_DOCS[("l1", "l1_d", "fred_sd_state_group", "custom_state_group")]
    p = next(p for p in doc.parameters if p.name == "sd_state_group_members")
    assert "list" in p.type
    assert "exactly one" in p.constraint.lower() or "required" in p.constraint.lower()


def test_l1d_fred_sd_state_group_custom_groups_type():
    """sd_state_groups has type dict[str, list[str]]."""
    doc = OPTION_DOCS[("l1", "l1_d", "fred_sd_state_group", "custom_state_group")]
    p = next(p for p in doc.parameters if p.name == "sd_state_groups")
    assert "dict" in p.type
    assert "exactly one" in p.constraint.lower() or "required" in p.constraint.lower()


def test_l1d_fred_sd_state_group_categorical_options_no_params():
    """Categorical (non-custom) state group options have no conditional parameters."""
    for option in ("all_states", "census_region_northeast", "census_region_south"):
        doc = OPTION_DOCS[("l1", "l1_d", "fred_sd_state_group", option)]
        assert doc.parameters == (), f"{option} should have parameters=()"


# ---------------------------------------------------------------------------
# fred_sd_variable_group
# ---------------------------------------------------------------------------


def test_l1d_fred_sd_variable_group_custom_has_two_params():
    """custom_sd_variable_group has 2 ParameterDoc entries (OR relationship)."""
    doc = OPTION_DOCS[("l1", "l1_d", "fred_sd_variable_group", "custom_sd_variable_group")]
    assert isinstance(doc.parameters, tuple)
    assert len(doc.parameters) == 2
    names = {p.name for p in doc.parameters}
    assert "sd_variable_group_members" in names
    assert "sd_variable_groups" in names


def test_l1d_fred_sd_variable_group_custom_members_type():
    """sd_variable_group_members has type list[str]."""
    doc = OPTION_DOCS[("l1", "l1_d", "fred_sd_variable_group", "custom_sd_variable_group")]
    p = next(p for p in doc.parameters if p.name == "sd_variable_group_members")
    assert "list" in p.type
    assert "exactly one" in p.constraint.lower() or "required" in p.constraint.lower()


def test_l1d_fred_sd_variable_group_custom_groups_type():
    """sd_variable_groups has type dict[str, list[str]]."""
    doc = OPTION_DOCS[("l1", "l1_d", "fred_sd_variable_group", "custom_sd_variable_group")]
    p = next(p for p in doc.parameters if p.name == "sd_variable_groups")
    assert "dict" in p.type
    assert "exactly one" in p.constraint.lower() or "required" in p.constraint.lower()


def test_l1d_fred_sd_variable_group_categorical_options_no_params():
    """Categorical (non-custom) variable group options have no conditional parameters."""
    for option in ("all_sd_variables", "labor_market_core", "housing"):
        doc = OPTION_DOCS[("l1", "l1_d", "fred_sd_variable_group", option)]
        assert doc.parameters == (), f"{option} should have parameters=()"


# ---------------------------------------------------------------------------
# _KNOWN_LEAF_CONFIG_KEYS extension (Cycle 20)
# ---------------------------------------------------------------------------


def test_known_leaf_config_keys_extended_with_l1d_geography_keys():
    """_KNOWN_LEAF_CONFIG_KEYS['1_data'] includes all 8 L1.D Geography conditional keys."""
    from macroforecast.core.execution import _KNOWN_LEAF_CONFIG_KEYS
    keys = _KNOWN_LEAF_CONFIG_KEYS["1_data"]
    expected = {
        "target_state",
        "target_states",
        "predictor_states",
        "sd_states",
        "sd_variables",
        "sd_state_group_members",
        "sd_state_groups",
        "sd_variable_group_members",
        "sd_variable_groups",
    }
    missing = expected - keys
    assert not missing, f"Missing keys in _KNOWN_LEAF_CONFIG_KEYS['1_data']: {missing}"
