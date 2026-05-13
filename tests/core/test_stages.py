"""Tests for macroforecast.core.stages — Stage Label System.

Covers all 73 scenarios from test-spec.md:
  T-01..T-19  STAGE_BY_LAYER constant verification
  T-20..T-32  stage_of(layer_id=...) — 13 layer IDs
  T-33..T-51  stage_of(sink_name=...) — 18 known sinks + 1 parametric
  T-52..T-62  Edge cases and errors
  T-63..T-67  Property-based invariants
  T-68..T-71  Public API imports
  T-72..T-73  Regression verification (collection-count + no-prior-failure)
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# T-01 to T-19: STAGE_BY_LAYER constant verification
# ---------------------------------------------------------------------------


def test_t01_import_and_type():
    """T-01: STAGE_BY_LAYER is importable and is a dict."""
    from macroforecast.core.stages import STAGE_BY_LAYER
    assert isinstance(STAGE_BY_LAYER, dict)


def test_t02_exact_key_count():
    """T-02: STAGE_BY_LAYER has exactly 13 keys."""
    from macroforecast.core.stages import STAGE_BY_LAYER
    assert len(STAGE_BY_LAYER) == 13


def test_t03_all_13_keys_present():
    """T-03: All 13 canonical layer IDs are present as keys."""
    from macroforecast.core.stages import STAGE_BY_LAYER
    expected_keys = {
        "l0", "l1", "l2", "l3", "l4", "l5", "l6", "l7", "l8",
        "l1_5", "l2_5", "l3_5", "l4_5",
    }
    assert set(STAGE_BY_LAYER.keys()) == expected_keys


def test_t04_l0_value():
    """T-04: STAGE_BY_LAYER['l0'] == 'meta'."""
    from macroforecast.core.stages import STAGE_BY_LAYER
    assert STAGE_BY_LAYER["l0"] == "meta"


def test_t05_l1_value():
    """T-05: STAGE_BY_LAYER['l1'] == 'data'."""
    from macroforecast.core.stages import STAGE_BY_LAYER
    assert STAGE_BY_LAYER["l1"] == "data"


def test_t06_l2_value():
    """T-06: STAGE_BY_LAYER['l2'] == 'clean'."""
    from macroforecast.core.stages import STAGE_BY_LAYER
    assert STAGE_BY_LAYER["l2"] == "clean"


def test_t07_l3_value():
    """T-07: STAGE_BY_LAYER['l3'] == 'features'."""
    from macroforecast.core.stages import STAGE_BY_LAYER
    assert STAGE_BY_LAYER["l3"] == "features"


def test_t08_l4_value():
    """T-08: STAGE_BY_LAYER['l4'] == 'forecasts'."""
    from macroforecast.core.stages import STAGE_BY_LAYER
    assert STAGE_BY_LAYER["l4"] == "forecasts"


def test_t09_l5_value():
    """T-09: STAGE_BY_LAYER['l5'] == 'evaluation'."""
    from macroforecast.core.stages import STAGE_BY_LAYER
    assert STAGE_BY_LAYER["l5"] == "evaluation"


def test_t10_l6_value():
    """T-10: STAGE_BY_LAYER['l6'] == 'tests'."""
    from macroforecast.core.stages import STAGE_BY_LAYER
    assert STAGE_BY_LAYER["l6"] == "tests"


def test_t11_l7_value():
    """T-11: STAGE_BY_LAYER['l7'] == 'interpretation'."""
    from macroforecast.core.stages import STAGE_BY_LAYER
    assert STAGE_BY_LAYER["l7"] == "interpretation"


def test_t12_l8_value():
    """T-12: STAGE_BY_LAYER['l8'] == 'artifacts'."""
    from macroforecast.core.stages import STAGE_BY_LAYER
    assert STAGE_BY_LAYER["l8"] == "artifacts"


def test_t13_l1_5_value():
    """T-13: STAGE_BY_LAYER['l1_5'] == 'data_diagnostic'."""
    from macroforecast.core.stages import STAGE_BY_LAYER
    assert STAGE_BY_LAYER["l1_5"] == "data_diagnostic"


def test_t14_l2_5_value():
    """T-14: STAGE_BY_LAYER['l2_5'] == 'clean_diagnostic'."""
    from macroforecast.core.stages import STAGE_BY_LAYER
    assert STAGE_BY_LAYER["l2_5"] == "clean_diagnostic"


def test_t15_l3_5_value():
    """T-15: STAGE_BY_LAYER['l3_5'] == 'features_diagnostic'."""
    from macroforecast.core.stages import STAGE_BY_LAYER
    assert STAGE_BY_LAYER["l3_5"] == "features_diagnostic"


def test_t16_l4_5_value():
    """T-16: STAGE_BY_LAYER['l4_5'] == 'model_diagnostic'."""
    from macroforecast.core.stages import STAGE_BY_LAYER
    assert STAGE_BY_LAYER["l4_5"] == "model_diagnostic"


def test_t17_all_values_nonempty_strings():
    """T-17: All values in STAGE_BY_LAYER are non-empty strings."""
    from macroforecast.core.stages import STAGE_BY_LAYER
    for k, v in STAGE_BY_LAYER.items():
        assert isinstance(v, str) and len(v) > 0, f"Empty or non-str value for {k!r}"


def test_t18_all_values_distinct():
    """T-18: All 13 stage label values are distinct (bijection property)."""
    from macroforecast.core.stages import STAGE_BY_LAYER
    assert len(set(STAGE_BY_LAYER.values())) == 13


def test_t19_import_from_public_api_same_object():
    """T-19: STAGE_BY_LAYER from macroforecast.core is the same object as from .stages."""
    from macroforecast.core import STAGE_BY_LAYER as SBL2
    from macroforecast.core.stages import STAGE_BY_LAYER as SBL1
    assert SBL1 is SBL2


# ---------------------------------------------------------------------------
# T-20 to T-32: stage_of(layer_id=...) — 13 layer IDs
# ---------------------------------------------------------------------------


def test_t20_stage_of_l0():
    """T-20: stage_of(layer_id='l0') == 'meta'."""
    from macroforecast.core.stages import stage_of
    assert stage_of(layer_id="l0") == "meta"


def test_t21_stage_of_l1():
    """T-21: stage_of(layer_id='l1') == 'data'."""
    from macroforecast.core.stages import stage_of
    assert stage_of(layer_id="l1") == "data"


def test_t22_stage_of_l2():
    """T-22: stage_of(layer_id='l2') == 'clean'."""
    from macroforecast.core.stages import stage_of
    assert stage_of(layer_id="l2") == "clean"


def test_t23_stage_of_l3():
    """T-23: stage_of(layer_id='l3') == 'features'."""
    from macroforecast.core.stages import stage_of
    assert stage_of(layer_id="l3") == "features"


def test_t24_stage_of_l4():
    """T-24: stage_of(layer_id='l4') == 'forecasts'."""
    from macroforecast.core.stages import stage_of
    assert stage_of(layer_id="l4") == "forecasts"


def test_t25_stage_of_l5():
    """T-25: stage_of(layer_id='l5') == 'evaluation'."""
    from macroforecast.core.stages import stage_of
    assert stage_of(layer_id="l5") == "evaluation"


def test_t26_stage_of_l6():
    """T-26: stage_of(layer_id='l6') == 'tests'."""
    from macroforecast.core.stages import stage_of
    assert stage_of(layer_id="l6") == "tests"


def test_t27_stage_of_l7():
    """T-27: stage_of(layer_id='l7') == 'interpretation'."""
    from macroforecast.core.stages import stage_of
    assert stage_of(layer_id="l7") == "interpretation"


def test_t28_stage_of_l8():
    """T-28: stage_of(layer_id='l8') == 'artifacts'."""
    from macroforecast.core.stages import stage_of
    assert stage_of(layer_id="l8") == "artifacts"


def test_t29_stage_of_l1_5():
    """T-29: stage_of(layer_id='l1_5') == 'data_diagnostic'."""
    from macroforecast.core.stages import stage_of
    assert stage_of(layer_id="l1_5") == "data_diagnostic"


def test_t30_stage_of_l2_5():
    """T-30: stage_of(layer_id='l2_5') == 'clean_diagnostic'."""
    from macroforecast.core.stages import stage_of
    assert stage_of(layer_id="l2_5") == "clean_diagnostic"


def test_t31_stage_of_l3_5():
    """T-31: stage_of(layer_id='l3_5') == 'features_diagnostic'."""
    from macroforecast.core.stages import stage_of
    assert stage_of(layer_id="l3_5") == "features_diagnostic"


def test_t32_stage_of_l4_5():
    """T-32: stage_of(layer_id='l4_5') == 'model_diagnostic'."""
    from macroforecast.core.stages import stage_of
    assert stage_of(layer_id="l4_5") == "model_diagnostic"


# ---------------------------------------------------------------------------
# T-33 to T-51: stage_of(sink_name=...) — 18 known sinks + 1 parametric
# ---------------------------------------------------------------------------


def test_t33_sink_l0_meta_v1():
    """T-33: stage_of(sink_name='l0_meta_v1') == 'meta'."""
    from macroforecast.core.stages import stage_of
    assert stage_of(sink_name="l0_meta_v1") == "meta"


def test_t34_sink_l1_data_definition_v1():
    """T-34: stage_of(sink_name='l1_data_definition_v1') == 'data'."""
    from macroforecast.core.stages import stage_of
    assert stage_of(sink_name="l1_data_definition_v1") == "data"


def test_t35_sink_l1_regime_metadata_v1():
    """T-35: stage_of(sink_name='l1_regime_metadata_v1') == 'data'."""
    from macroforecast.core.stages import stage_of
    assert stage_of(sink_name="l1_regime_metadata_v1") == "data"


def test_t36_sink_l2_clean_panel_v1():
    """T-36: stage_of(sink_name='l2_clean_panel_v1') == 'clean'."""
    from macroforecast.core.stages import stage_of
    assert stage_of(sink_name="l2_clean_panel_v1") == "clean"


def test_t37_sink_l3_features_v1():
    """T-37: stage_of(sink_name='l3_features_v1') == 'features'."""
    from macroforecast.core.stages import stage_of
    assert stage_of(sink_name="l3_features_v1") == "features"


def test_t38_sink_l3_metadata_v1():
    """T-38: stage_of(sink_name='l3_metadata_v1') == 'features'."""
    from macroforecast.core.stages import stage_of
    assert stage_of(sink_name="l3_metadata_v1") == "features"


def test_t39_sink_l4_forecasts_v1():
    """T-39: stage_of(sink_name='l4_forecasts_v1') == 'forecasts'."""
    from macroforecast.core.stages import stage_of
    assert stage_of(sink_name="l4_forecasts_v1") == "forecasts"


def test_t40_sink_l4_model_artifacts_v1():
    """T-40: stage_of(sink_name='l4_model_artifacts_v1') == 'forecasts'."""
    from macroforecast.core.stages import stage_of
    assert stage_of(sink_name="l4_model_artifacts_v1") == "forecasts"


def test_t41_sink_l4_training_metadata_v1():
    """T-41: stage_of(sink_name='l4_training_metadata_v1') == 'forecasts'."""
    from macroforecast.core.stages import stage_of
    assert stage_of(sink_name="l4_training_metadata_v1") == "forecasts"


def test_t42_sink_l5_evaluation_v1():
    """T-42: stage_of(sink_name='l5_evaluation_v1') == 'evaluation'."""
    from macroforecast.core.stages import stage_of
    assert stage_of(sink_name="l5_evaluation_v1") == "evaluation"


def test_t43_sink_l6_tests_v1():
    """T-43: stage_of(sink_name='l6_tests_v1') == 'tests'."""
    from macroforecast.core.stages import stage_of
    assert stage_of(sink_name="l6_tests_v1") == "tests"


def test_t44_sink_l7_importance_v1():
    """T-44: stage_of(sink_name='l7_importance_v1') == 'interpretation'."""
    from macroforecast.core.stages import stage_of
    assert stage_of(sink_name="l7_importance_v1") == "interpretation"


def test_t45_sink_l7_transformation_attribution_v1():
    """T-45: stage_of(sink_name='l7_transformation_attribution_v1') == 'interpretation'."""
    from macroforecast.core.stages import stage_of
    assert stage_of(sink_name="l7_transformation_attribution_v1") == "interpretation"


def test_t46_sink_l8_artifacts_v1():
    """T-46: stage_of(sink_name='l8_artifacts_v1') == 'artifacts'."""
    from macroforecast.core.stages import stage_of
    assert stage_of(sink_name="l8_artifacts_v1") == "artifacts"


def test_t47_sink_l1_5_diagnostic_v1():
    """T-47: stage_of(sink_name='l1_5_diagnostic_v1') == 'data_diagnostic'."""
    from macroforecast.core.stages import stage_of
    assert stage_of(sink_name="l1_5_diagnostic_v1") == "data_diagnostic"


def test_t48_sink_l2_5_diagnostic_v1():
    """T-48: stage_of(sink_name='l2_5_diagnostic_v1') == 'clean_diagnostic'."""
    from macroforecast.core.stages import stage_of
    assert stage_of(sink_name="l2_5_diagnostic_v1") == "clean_diagnostic"


def test_t49_sink_l3_5_diagnostic_v1():
    """T-49: stage_of(sink_name='l3_5_diagnostic_v1') == 'features_diagnostic'."""
    from macroforecast.core.stages import stage_of
    assert stage_of(sink_name="l3_5_diagnostic_v1") == "features_diagnostic"


def test_t50_sink_l4_5_diagnostic_v1():
    """T-50: stage_of(sink_name='l4_5_diagnostic_v1') == 'model_diagnostic'."""
    from macroforecast.core.stages import stage_of
    assert stage_of(sink_name="l4_5_diagnostic_v1") == "model_diagnostic"


def test_t51_parametric_consistency_stage_of_matches_layer_sinks():
    """T-51: For every (layer_id, sinks) in LAYER_SINKS, stage_of(sink_name=key) == STAGE_BY_LAYER[layer_id]."""
    from macroforecast.core.layers import LAYER_SINKS
    from macroforecast.core.stages import STAGE_BY_LAYER, stage_of
    for layer_id, sinks in LAYER_SINKS.items():
        expected = STAGE_BY_LAYER[layer_id]
        for sink_key in sinks:
            assert stage_of(sink_name=sink_key) == expected, (
                f"sink {sink_key!r} under layer {layer_id!r}: "
                f"expected {expected!r}, got {stage_of(sink_name=sink_key)!r}"
            )


# ---------------------------------------------------------------------------
# T-52 to T-62: Edge cases and errors
# ---------------------------------------------------------------------------


def test_t52_unknown_layer_id_raises_value_error():
    """T-52: Unknown layer_id raises ValueError."""
    from macroforecast.core.stages import stage_of
    with pytest.raises(ValueError):
        stage_of(layer_id="l99")


def test_t53_unknown_layer_id_error_message_contains_bad_value():
    """T-53: ValueError for unknown layer_id includes the bad value in the message."""
    from macroforecast.core.stages import stage_of
    with pytest.raises(ValueError, match="l99"):
        stage_of(layer_id="l99")


def test_t54_empty_string_layer_id_raises_value_error():
    """T-54: Empty string layer_id raises ValueError."""
    from macroforecast.core.stages import stage_of
    with pytest.raises(ValueError):
        stage_of(layer_id="")


def test_t55_non_l_prefix_layer_id_raises_value_error():
    """T-55: A non-l-prefixed layer_id raises ValueError."""
    from macroforecast.core.stages import stage_of
    with pytest.raises(ValueError):
        stage_of(layer_id="features")


def test_t56_unknown_sink_name_raises_value_error():
    """T-56: Unknown sink_name raises ValueError."""
    from macroforecast.core.stages import stage_of
    with pytest.raises(ValueError):
        stage_of(sink_name="l99_unknown_v1")


def test_t57_unparseable_sink_name_raises_value_error():
    """T-57: Sink name with no layer prefix raises ValueError."""
    from macroforecast.core.stages import stage_of
    with pytest.raises(ValueError):
        stage_of(sink_name="bad_name")


def test_t58_both_args_raises_value_error():
    """T-58: Providing both layer_id and sink_name raises ValueError."""
    from macroforecast.core.stages import stage_of
    with pytest.raises(ValueError):
        stage_of(layer_id="l3", sink_name="l3_features_v1")


def test_t59_neither_arg_raises_value_error():
    """T-59: Providing neither argument raises ValueError."""
    from macroforecast.core.stages import stage_of
    with pytest.raises(ValueError):
        stage_of()


def test_t60_positional_arg_raises_type_error():
    """T-60: stage_of cannot be called with positional arguments (keyword-only)."""
    from macroforecast.core.stages import stage_of
    with pytest.raises(TypeError):
        stage_of("l3")  # type: ignore[call-arg]


def test_t61_dotted_form_L1_5_raises_value_error():
    """T-61: 'L1.5' (wrong case and separator) raises ValueError."""
    from macroforecast.core.stages import stage_of
    with pytest.raises(ValueError):
        stage_of(layer_id="L1.5")


def test_t62_uppercase_L3_raises_value_error():
    """T-62: 'L3' (uppercase) raises ValueError."""
    from macroforecast.core.stages import stage_of
    with pytest.raises(ValueError):
        stage_of(layer_id="L3")


# ---------------------------------------------------------------------------
# T-63 to T-67: Property-based invariants
# ---------------------------------------------------------------------------


def test_t63_stage_of_layer_id_matches_constant_for_all_keys():
    """T-63: stage_of(layer_id=k) == STAGE_BY_LAYER[k] for every k."""
    from macroforecast.core.stages import STAGE_BY_LAYER, stage_of
    for k in STAGE_BY_LAYER:
        assert stage_of(layer_id=k) == STAGE_BY_LAYER[k]


def test_t64_round_trip_no_accidental_collision():
    """T-64: All 13 stage label strings are distinct (no key maps to another key's value)."""
    from macroforecast.core.stages import STAGE_BY_LAYER
    values = list(STAGE_BY_LAYER.values())
    assert len(values) == len(set(values))


def test_t65_diagnostic_layers_have_diagnostic_suffix():
    """T-65: Each diagnostic layer's stage label ends with '_diagnostic'."""
    from macroforecast.core.stages import STAGE_BY_LAYER
    diagnostic_layers = {"l1_5", "l2_5", "l3_5", "l4_5"}
    for dk in diagnostic_layers:
        assert STAGE_BY_LAYER[dk].endswith("_diagnostic"), (
            f"Diagnostic layer {dk!r} stage label must end with '_diagnostic'"
        )


def test_t66_non_diagnostic_layers_no_diagnostic_suffix():
    """T-66: Core (non-diagnostic) layers do not have '_diagnostic' suffix."""
    from macroforecast.core.stages import STAGE_BY_LAYER
    non_diagnostic_layers = {"l0", "l1", "l2", "l3", "l4", "l5", "l6", "l7", "l8"}
    for k in non_diagnostic_layers:
        assert not STAGE_BY_LAYER[k].endswith("_diagnostic"), (
            f"Core layer {k!r} must not have '_diagnostic' suffix"
        )


def test_t67_stage_by_layer_keys_subset_of_layer_id_literal():
    """T-67: STAGE_BY_LAYER keys are a subset of LayerId Literal values."""
    import typing
    from macroforecast.core.dag import LayerId
    from macroforecast.core.stages import STAGE_BY_LAYER
    layer_id_values = set(typing.get_args(LayerId))
    assert set(STAGE_BY_LAYER.keys()).issubset(layer_id_values)


# ---------------------------------------------------------------------------
# T-68 to T-71: Public API imports from macroforecast.core
# ---------------------------------------------------------------------------


def test_t68_core_exports_stage_of():
    """T-68: macroforecast.core exports callable stage_of."""
    from macroforecast.core import stage_of
    assert callable(stage_of)


def test_t69_core_exports_stage_by_layer():
    """T-69: macroforecast.core exports STAGE_BY_LAYER as a dict."""
    from macroforecast.core import STAGE_BY_LAYER
    assert isinstance(STAGE_BY_LAYER, dict)


def test_t70_core_exports_stage_label():
    """T-70: macroforecast.core exports StageLabel (importable without error)."""
    from macroforecast.core import StageLabel  # noqa: F401


def test_t71_names_in_core_all():
    """T-71: stage_of, STAGE_BY_LAYER, and StageLabel appear in macroforecast.core.__all__."""
    import macroforecast.core as core
    assert "stage_of" in core.__all__
    assert "STAGE_BY_LAYER" in core.__all__
    assert "StageLabel" in core.__all__


# ---------------------------------------------------------------------------
# T-72 to T-73: Regression verification
# ---------------------------------------------------------------------------


def test_t72_existing_test_count_not_reduced():
    """T-72: At least 1432 tests can be collected (no test was deleted)."""
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "--collect-only", "-q",
         "--ignore=tests/core/test_stages.py"],
        capture_output=True,
        text=True,
        cwd="/Users/nanyeon/code/macroforecast",
    )
    # Extract collected count from last informational line, e.g. "1432 tests collected"
    output = result.stdout + result.stderr
    import re
    m = re.search(r"(\d+) tests? collected", output)
    assert m is not None, f"Could not find test count in output:\n{output}"
    count = int(m.group(1))
    assert count >= 1432, f"Expected at least 1432 prior tests, found {count}"


def test_t73_no_existing_test_failures():
    """T-73: All pre-existing tests pass; zero failures and zero errors."""
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-x", "-q",
         "--ignore=tests/core/test_stages.py",
         "--tb=no"],
        capture_output=True,
        text=True,
        cwd="/Users/nanyeon/code/macroforecast",
        timeout=600,
    )
    output = result.stdout + result.stderr
    # Must exit 0 (all pass) — any nonzero exit means a failure
    assert result.returncode == 0, (
        f"Pre-existing tests had failures (returncode={result.returncode}).\n"
        f"Output tail:\n{output[-2000:]}"
    )
