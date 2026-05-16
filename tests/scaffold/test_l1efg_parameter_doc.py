"""Tests for L1.E + L1.F + L1.G ParameterDoc population (Cycle 21).

Verifies that:
- L1.E sample window options have ParameterDoc for conditional leaf_config keys:
  * sample_start_rule=fixed_date -> sample_start_date
  * sample_end_rule=fixed_date   -> sample_end_date
  * other options (max_balanced, earliest_available, latest_available) -> no params

- L1.F horizon_set options have ParameterDoc for conditional leaf_config keys:
  * horizon_set=single           -> target_horizons (optional, length 1)
  * horizon_set=custom_list      -> target_horizons (required, non-empty list)
  * horizon_set=range_up_to_h    -> max_horizon (required int)
  * horizon_set=standard_md      -> no params
  * horizon_set=standard_qd      -> no params

- L1.G regime_definition options have ParameterDoc for conditional leaf_config keys:
  * external_user_provided       -> regime_indicator_path, regime_dates_list, n_regimes
  * estimated_markov_switching   -> n_regimes
  * estimated_threshold          -> threshold_variable, n_thresholds
  * estimated_structural_break   -> max_breaks, break_ic_criterion
  * none / external_nber         -> no params

- _KNOWN_LEAF_CONFIG_KEYS['1_data'] extended with all new regime + horizon keys

- L1 audit is noted as complete in the module docstring

All entries are Tier-1 complete (summary + description + when_to_use + reference
+ last_reviewed non-empty).
"""
from __future__ import annotations

import pytest

from macroforecast.scaffold.option_docs import OPTION_DOCS
from macroforecast.scaffold.option_docs.types import ParameterDoc


# ---------------------------------------------------------------------------
# L1.E sample_start_rule
# ---------------------------------------------------------------------------


def test_l1e_start_fixed_date_has_sample_start_date():
    """sample_start_rule=fixed_date has 1 ParameterDoc for sample_start_date."""
    doc = OPTION_DOCS[("l1", "l1_e", "sample_start_rule", "fixed_date")]
    assert isinstance(doc.parameters, tuple)
    assert len(doc.parameters) == 1
    p = doc.parameters[0]
    assert p.name == "sample_start_date"
    assert p.type == "str"
    assert p.default is None
    assert p.constraint is not None
    assert "required" in p.constraint.lower() or "fixed_date" in p.constraint.lower()
    assert "ISO" in p.constraint or "YYYY" in p.constraint


def test_l1e_start_fixed_date_constraint_mentions_partial_iso():
    """sample_start_date constraint documents partial-ISO normalization."""
    doc = OPTION_DOCS[("l1", "l1_e", "sample_start_rule", "fixed_date")]
    p = doc.parameters[0]
    # C12 F-P0-1 partial-ISO acceptance: YYYY-MM and YYYY forms are valid
    assert "YYYY-MM" in p.constraint or "partial" in p.constraint.lower()


def test_l1e_start_other_options_no_params():
    """max_balanced and earliest_available have no conditional parameters."""
    for option in ("max_balanced", "earliest_available"):
        doc = OPTION_DOCS[("l1", "l1_e", "sample_start_rule", option)]
        assert doc.parameters == (), f"sample_start_rule={option} should have parameters=()"


# ---------------------------------------------------------------------------
# L1.E sample_end_rule
# ---------------------------------------------------------------------------


def test_l1e_end_fixed_date_has_sample_end_date():
    """sample_end_rule=fixed_date has 1 ParameterDoc for sample_end_date."""
    doc = OPTION_DOCS[("l1", "l1_e", "sample_end_rule", "fixed_date")]
    assert isinstance(doc.parameters, tuple)
    assert len(doc.parameters) == 1
    p = doc.parameters[0]
    assert p.name == "sample_end_date"
    assert p.type == "str"
    assert p.default is None
    assert p.constraint is not None
    assert "required" in p.constraint.lower() or "fixed_date" in p.constraint.lower()


def test_l1e_end_fixed_date_constraint_mentions_ordering():
    """sample_end_date constraint mentions must-be-gte-start_date relationship."""
    doc = OPTION_DOCS[("l1", "l1_e", "sample_end_rule", "fixed_date")]
    p = doc.parameters[0]
    assert ">=" in p.constraint or "start" in p.constraint.lower()


def test_l1e_end_latest_available_no_params():
    """latest_available has no conditional parameters."""
    doc = OPTION_DOCS[("l1", "l1_e", "sample_end_rule", "latest_available")]
    assert doc.parameters == ()


# ---------------------------------------------------------------------------
# L1.F horizon_set
# ---------------------------------------------------------------------------


def test_l1f_single_has_target_horizons():
    """horizon_set=single has 1 ParameterDoc for target_horizons."""
    doc = OPTION_DOCS[("l1", "l1_f", "horizon_set", "single")]
    assert isinstance(doc.parameters, tuple)
    assert len(doc.parameters) == 1
    p = doc.parameters[0]
    assert p.name == "target_horizons"
    assert "list" in p.type
    assert p.constraint is not None
    # single implies optional (runtime defaults to [1])
    assert "optional" in p.constraint.lower() or "default" in p.constraint.lower() or "[1]" in p.constraint


def test_l1f_custom_list_has_target_horizons_required():
    """horizon_set=custom_list has 1 ParameterDoc for target_horizons marked required."""
    doc = OPTION_DOCS[("l1", "l1_f", "horizon_set", "custom_list")]
    assert isinstance(doc.parameters, tuple)
    assert len(doc.parameters) == 1
    p = doc.parameters[0]
    assert p.name == "target_horizons"
    assert "list" in p.type
    assert p.constraint is not None
    assert "required" in p.constraint.lower()


def test_l1f_range_up_to_h_has_max_horizon():
    """horizon_set=range_up_to_h has 1 ParameterDoc for max_horizon."""
    doc = OPTION_DOCS[("l1", "l1_f", "horizon_set", "range_up_to_h")]
    assert isinstance(doc.parameters, tuple)
    assert len(doc.parameters) == 1
    p = doc.parameters[0]
    assert p.name == "max_horizon"
    assert "int" in p.type
    assert p.default is None
    assert p.constraint is not None
    assert "required" in p.constraint.lower()
    assert "positive" in p.constraint.lower()


def test_l1f_standard_options_no_params():
    """standard_md and standard_qd have no conditional parameters."""
    for option in ("standard_md", "standard_qd"):
        doc = OPTION_DOCS[("l1", "l1_f", "horizon_set", option)]
        assert doc.parameters == (), f"horizon_set={option} should have parameters=()"


# ---------------------------------------------------------------------------
# L1.G regime_definition
# ---------------------------------------------------------------------------


def test_l1g_external_user_provided_has_three_params():
    """external_user_provided has 3 ParameterDoc entries."""
    doc = OPTION_DOCS[("l1", "l1_g", "regime_definition", "external_user_provided")]
    assert isinstance(doc.parameters, tuple)
    assert len(doc.parameters) == 3
    names = {p.name for p in doc.parameters}
    assert "regime_indicator_path" in names
    assert "regime_dates_list" in names
    assert "n_regimes" in names


def test_l1g_external_user_provided_xor_constraint():
    """regime_indicator_path and regime_dates_list have XOR constraint."""
    doc = OPTION_DOCS[("l1", "l1_g", "regime_definition", "external_user_provided")]
    path_p = next(p for p in doc.parameters if p.name == "regime_indicator_path")
    dates_p = next(p for p in doc.parameters if p.name == "regime_dates_list")
    # Both must mention the mutual exclusivity
    for p in (path_p, dates_p):
        assert (
            "exactly one" in p.constraint.lower()
            or "one of" in p.constraint.lower()
            or "xor" in p.constraint.lower()
        ), f"{p.name} constraint should mention XOR/exactly-one: {p.constraint!r}"


def test_l1g_external_user_n_regimes_has_default():
    """external_user_provided n_regimes has a non-None default (2)."""
    doc = OPTION_DOCS[("l1", "l1_g", "regime_definition", "external_user_provided")]
    n = next(p for p in doc.parameters if p.name == "n_regimes")
    assert n.default == 2


def test_l1g_estimated_markov_switching_has_n_regimes():
    """estimated_markov_switching has 1 ParameterDoc for n_regimes."""
    doc = OPTION_DOCS[("l1", "l1_g", "regime_definition", "estimated_markov_switching")]
    assert isinstance(doc.parameters, tuple)
    assert len(doc.parameters) == 1
    p = doc.parameters[0]
    assert p.name == "n_regimes"
    assert "int" in p.type
    assert p.default == 2


def test_l1g_estimated_threshold_has_threshold_variable_and_n_thresholds():
    """estimated_threshold has 2 ParameterDoc entries."""
    doc = OPTION_DOCS[("l1", "l1_g", "regime_definition", "estimated_threshold")]
    assert isinstance(doc.parameters, tuple)
    assert len(doc.parameters) == 2
    names = [p.name for p in doc.parameters]
    assert names[0] == "threshold_variable"
    assert names[1] == "n_thresholds"


def test_l1g_estimated_threshold_variable_required():
    """threshold_variable is required (default=None, required mention in constraint)."""
    doc = OPTION_DOCS[("l1", "l1_g", "regime_definition", "estimated_threshold")]
    p = next(p for p in doc.parameters if p.name == "threshold_variable")
    assert p.default is None
    assert "required" in p.constraint.lower()


def test_l1g_estimated_structural_break_has_max_breaks_and_ic():
    """estimated_structural_break has 2 ParameterDoc entries."""
    doc = OPTION_DOCS[("l1", "l1_g", "regime_definition", "estimated_structural_break")]
    assert isinstance(doc.parameters, tuple)
    assert len(doc.parameters) == 2
    names = [p.name for p in doc.parameters]
    assert names[0] == "max_breaks"
    assert names[1] == "break_ic_criterion"


def test_l1g_structural_break_ic_has_default_bic():
    """break_ic_criterion defaults to 'bic'."""
    doc = OPTION_DOCS[("l1", "l1_g", "regime_definition", "estimated_structural_break")]
    ic = next(p for p in doc.parameters if p.name == "break_ic_criterion")
    assert ic.default == "bic"
    assert "aic" in ic.constraint.lower() or "aic" in ic.description.lower()


def test_l1g_none_and_nber_have_no_params():
    """none and external_nber have no conditional parameters."""
    for option in ("none", "external_nber"):
        doc = OPTION_DOCS[("l1", "l1_g", "regime_definition", option)]
        assert doc.parameters == (), f"regime_definition={option} should have parameters=()"


# ---------------------------------------------------------------------------
# Tier-1 completeness for all L1.E / L1.F / L1.G entries
# ---------------------------------------------------------------------------


def test_l1efg_all_entries_tier1_complete():
    """All L1.E, L1.F, L1.G entries are Tier-1 complete."""
    efg_axes = {
        "sample_start_rule",
        "sample_end_rule",
        "horizon_set",
        "regime_definition",
        "regime_estimation_temporal_rule",
    }
    efg_entries = [
        doc for key, doc in OPTION_DOCS.items()
        if key[0] == "l1" and key[2] in efg_axes
    ]
    assert efg_entries, "No L1.E/F/G entries found in OPTION_DOCS"
    failing = [e.key() for e in efg_entries if not e.is_tier1_complete()]
    assert not failing, f"Not Tier-1 complete: {failing}"


# ---------------------------------------------------------------------------
# _KNOWN_LEAF_CONFIG_KEYS extension (Cycle 21)
# ---------------------------------------------------------------------------


def test_known_leaf_config_keys_extended_with_l1efg_keys():
    """_KNOWN_LEAF_CONFIG_KEYS['1_data'] includes all new L1.E/F/G conditional keys."""
    from macroforecast.core.execution import _KNOWN_LEAF_CONFIG_KEYS
    keys = _KNOWN_LEAF_CONFIG_KEYS["1_data"]
    # L1.F new key
    assert "max_horizon" in keys, "max_horizon missing from _KNOWN_LEAF_CONFIG_KEYS"
    # L1.G regime keys
    regime_keys = {
        "regime_indicator_path",
        "regime_dates_list",
        "n_regimes",
        "threshold_variable",
        "n_thresholds",
        "max_breaks",
        "break_ic_criterion",
        "regime_rolling_window_size",
        "block_recompute_interval",
    }
    missing = regime_keys - keys
    assert not missing, f"Missing regime keys in _KNOWN_LEAF_CONFIG_KEYS['1_data']: {missing}"


# ---------------------------------------------------------------------------
# Docstring sanity check: L1 audit complete
# ---------------------------------------------------------------------------


def test_l1_option_docs_docstring_marks_audit_complete():
    """The option_docs/l1.py module docstring marks L1 audit as complete."""
    import macroforecast.scaffold.option_docs.l1 as l1_module
    doc = l1_module.__doc__ or ""
    assert "L1 audit complete" in doc or "audit complete" in doc.lower(), (
        "option_docs/l1.py docstring should state 'L1 audit complete'"
    )
