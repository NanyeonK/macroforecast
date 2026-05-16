"""Tests for L1.C ParameterDoc population (Cycle 19 P-2).

Verifies that L1.C Predictor Universe axis options have ParameterDoc
entries for their conditional leaf_config keys:
- variable_universe/explicit_variable_list -> variable_universe_columns
- raw_outlier_policy/winsorize_raw -> winsorize_quantiles
- raw_outlier_policy/iqr_clip_raw -> outlier_iqr_threshold
- raw_outlier_policy/zscore_clip_raw -> zscore_threshold_value
- release_lag_rule/fixed_lag_all_series -> fixed_lag_periods
- release_lag_rule/series_specific_lag -> release_lag_per_series
"""
from __future__ import annotations

import pytest

from macroforecast.scaffold.option_docs import OPTION_DOCS
from macroforecast.scaffold.option_docs.types import ParameterDoc


# ---------------------------------------------------------------------------
# variable_universe
# ---------------------------------------------------------------------------


def test_l1c_explicit_variable_list_has_columns_param():
    """explicit_variable_list has 1 ParameterDoc for variable_universe_columns."""
    doc = OPTION_DOCS[("l1", "l1_c", "variable_universe", "explicit_variable_list")]
    assert isinstance(doc.parameters, tuple)
    assert len(doc.parameters) == 1
    p = doc.parameters[0]
    assert p.name == "variable_universe_columns"
    assert "list" in p.type
    assert p.default is None  # required
    assert p.constraint is not None
    assert "required" in p.constraint.lower() or "non-empty" in p.constraint.lower()


def test_l1c_all_variables_has_no_params():
    """all_variables has no conditional parameters."""
    doc = OPTION_DOCS[("l1", "l1_c", "variable_universe", "all_variables")]
    assert doc.parameters == ()


def test_l1c_core_variables_has_no_params():
    """core_variables has no conditional parameters."""
    doc = OPTION_DOCS[("l1", "l1_c", "variable_universe", "core_variables")]
    assert doc.parameters == ()


# ---------------------------------------------------------------------------
# raw_outlier_policy
# ---------------------------------------------------------------------------


def test_l1c_winsorize_raw_has_quantiles_param():
    """winsorize_raw has 1 ParameterDoc for winsorize_quantiles."""
    doc = OPTION_DOCS[("l1", "l1_c", "raw_outlier_policy", "winsorize_raw")]
    assert isinstance(doc.parameters, tuple)
    assert len(doc.parameters) == 1
    p = doc.parameters[0]
    assert p.name == "winsorize_quantiles"
    assert "list" in p.type or "float" in p.type
    assert p.default == "[0.01, 0.99]"
    assert p.constraint is not None


def test_l1c_iqr_clip_raw_has_threshold_param():
    """iqr_clip_raw has 1 ParameterDoc for outlier_iqr_threshold."""
    doc = OPTION_DOCS[("l1", "l1_c", "raw_outlier_policy", "iqr_clip_raw")]
    assert isinstance(doc.parameters, tuple)
    assert len(doc.parameters) == 1
    p = doc.parameters[0]
    assert p.name == "outlier_iqr_threshold"
    assert p.type == "float"
    assert p.default == "10.0"
    assert p.constraint is not None
    assert ">0" in p.constraint or "> 0" in p.constraint


def test_l1c_zscore_clip_raw_has_threshold_param():
    """zscore_clip_raw has 1 ParameterDoc for zscore_threshold_value."""
    doc = OPTION_DOCS[("l1", "l1_c", "raw_outlier_policy", "zscore_clip_raw")]
    assert isinstance(doc.parameters, tuple)
    assert len(doc.parameters) == 1
    p = doc.parameters[0]
    assert p.name == "zscore_threshold_value"
    assert p.type == "float"
    assert p.default == "3.0"
    assert p.constraint is not None


def test_l1c_preserve_raw_outliers_has_no_params():
    """preserve_raw_outliers has no conditional parameters."""
    doc = OPTION_DOCS[("l1", "l1_c", "raw_outlier_policy", "preserve_raw_outliers")]
    assert doc.parameters == ()


def test_l1c_mad_clip_raw_has_no_params():
    """mad_clip_raw has no conditional leaf_config parameters (runtime not yet parametrized)."""
    doc = OPTION_DOCS[("l1", "l1_c", "raw_outlier_policy", "mad_clip_raw")]
    assert doc.parameters == ()


def test_l1c_set_raw_outliers_to_missing_has_no_params():
    """set_raw_outliers_to_missing has no conditional parameters."""
    doc = OPTION_DOCS[("l1", "l1_c", "raw_outlier_policy", "set_raw_outliers_to_missing")]
    assert doc.parameters == ()


# ---------------------------------------------------------------------------
# release_lag_rule
# ---------------------------------------------------------------------------


def test_l1c_fixed_lag_all_series_has_periods_param():
    """fixed_lag_all_series has 1 ParameterDoc for fixed_lag_periods."""
    doc = OPTION_DOCS[("l1", "l1_c", "release_lag_rule", "fixed_lag_all_series")]
    assert isinstance(doc.parameters, tuple)
    assert len(doc.parameters) == 1
    p = doc.parameters[0]
    assert p.name == "fixed_lag_periods"
    assert p.type == "int"
    assert p.default is None  # required
    assert p.constraint is not None
    assert "required" in p.constraint.lower() or ">=0" in p.constraint


def test_l1c_series_specific_lag_has_map_param():
    """series_specific_lag has 1 ParameterDoc for release_lag_per_series."""
    doc = OPTION_DOCS[("l1", "l1_c", "release_lag_rule", "series_specific_lag")]
    assert isinstance(doc.parameters, tuple)
    assert len(doc.parameters) == 1
    p = doc.parameters[0]
    assert p.name == "release_lag_per_series"
    assert "dict" in p.type
    assert p.default is None  # required
    assert p.constraint is not None
    assert "required" in p.constraint.lower() or "non-empty" in p.constraint.lower()


def test_l1c_ignore_release_lag_has_no_params():
    """ignore_release_lag has no conditional parameters."""
    doc = OPTION_DOCS[("l1", "l1_c", "release_lag_rule", "ignore_release_lag")]
    assert doc.parameters == ()


# ---------------------------------------------------------------------------
# Other L1.C axes — verify no spurious parameters
# ---------------------------------------------------------------------------


def test_l1c_missing_availability_no_params():
    """missing_availability options have no conditional leaf_config parameters."""
    for option in ("require_complete_rows", "keep_available_rows", "impute_predictors_only",
                   "zero_fill_leading_predictor_gaps"):
        doc = OPTION_DOCS[("l1", "l1_c", "missing_availability", option)]
        assert doc.parameters == (), f"{option} should have parameters=()"


def test_l1c_contemporaneous_x_rule_no_params():
    """contemporaneous_x_rule options have no conditional leaf_config parameters."""
    for option in ("allow_same_period_predictors", "forbid_same_period_predictors"):
        doc = OPTION_DOCS[("l1", "l1_c", "contemporaneous_x_rule", option)]
        assert doc.parameters == (), f"{option} should have parameters=()"


def test_l1c_official_transform_scope_no_params():
    """official_transform_scope options have no conditional leaf_config parameters."""
    for option in ("target_only", "predictors_only", "target_and_predictors", "none"):
        doc = OPTION_DOCS[("l1", "l1_c", "official_transform_scope", option)]
        assert doc.parameters == (), f"{option} should have parameters=()"


# ---------------------------------------------------------------------------
# _KNOWN_LEAF_CONFIG_KEYS extension (P-3)
# ---------------------------------------------------------------------------


def test_known_leaf_config_keys_extended():
    """_KNOWN_LEAF_CONFIG_KEYS['1_data'] includes all 6 L1.C conditional keys."""
    from macroforecast.core.execution import _KNOWN_LEAF_CONFIG_KEYS
    keys = _KNOWN_LEAF_CONFIG_KEYS["1_data"]
    expected = {
        "variable_universe_columns",
        "fixed_lag_periods",
        "release_lag_per_series",
        "outlier_iqr_threshold",
        "zscore_threshold_value",
        "winsorize_quantiles",
    }
    missing = expected - keys
    assert not missing, f"Missing keys in _KNOWN_LEAF_CONFIG_KEYS['1_data']: {missing}"


# ---------------------------------------------------------------------------
# Encyclopedia rendering: Parameters tables present where applicable
# ---------------------------------------------------------------------------


def test_l1c_encyclopedia_page_contains_parameters_tables():
    """Encyclopedia pages for L1.C raw_outlier_policy / release_lag_rule render Parameters tables."""
    from macroforecast.scaffold.render_encyclopedia import _render_axis_page
    from macroforecast.scaffold.introspect import axes

    l1_axes = {ax.name: ax for ax in axes("l1")}

    outlier_page = _render_axis_page("l1", l1_axes["raw_outlier_policy"])
    assert "**Parameters**" in outlier_page
    assert "winsorize_quantiles" in outlier_page
    assert "outlier_iqr_threshold" in outlier_page
    assert "zscore_threshold_value" in outlier_page

    lag_page = _render_axis_page("l1", l1_axes["release_lag_rule"])
    assert "**Parameters**" in lag_page
    assert "fixed_lag_periods" in lag_page
    assert "release_lag_per_series" in lag_page
