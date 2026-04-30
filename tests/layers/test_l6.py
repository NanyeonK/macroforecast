from pathlib import Path

from macrocast.core.layers.l6 import (
    make_recipe_with_density_forecast,
    make_recipe_without_benchmark,
    normalize_to_dag_form,
    parse_layer_yaml,
    parse_recipe_yaml,
    resolve_axes,
    validate_layer,
    validate_recipe,
)
from macrocast.core.ops import get_op


ROOT = Path(__file__).resolve().parents[2]


def _example(name: str) -> str:
    return (ROOT / "examples" / "recipes" / name).read_text()


def test_l6_disabled_by_default():
    layer = parse_layer_yaml("6_statistical_tests:\n  enabled: false", "l6")
    resolved = resolve_axes(normalize_to_dag_form(layer, "l6"))
    assert resolved["enabled"] is False


def test_l6_default_enabled_is_false():
    layer = parse_layer_yaml("6_statistical_tests: {}", "l6")
    resolved = resolve_axes(normalize_to_dag_form(layer, "l6"))
    assert resolved["enabled"] is False


def test_l6_enabled_but_no_sub_layer():
    layer = parse_layer_yaml("6_statistical_tests:\n  enabled: true\n", "l6")
    assert validate_layer(layer).has_hard_errors is False


def test_l6_a_minimal():
    layer = parse_layer_yaml("6_statistical_tests:\n  enabled: true\n  sub_layers:\n    L6_A_equal_predictive:\n      enabled: true\n", "l6")
    resolved = resolve_axes(normalize_to_dag_form(layer, "l6"))
    assert resolved["L6_A_equal_predictive"]["equal_predictive_test"] == "dm_diebold_mariano"
    assert resolved["L6_A_equal_predictive"]["loss_function"] == "squared"


def test_l6_a_vs_benchmark_only_requires_benchmark():
    root = make_recipe_without_benchmark()
    root["6_statistical_tests"] = {"enabled": True, "sub_layers": {"L6_A_equal_predictive": {"enabled": True, "fixed_axes": {"model_pair_strategy": "vs_benchmark_only"}}}}
    assert validate_recipe(parse_recipe_yaml(root)).has_hard_errors


def test_l6_a_user_list_invalid_model_id():
    layer = parse_layer_yaml(
        """
6_statistical_tests:
  enabled: true
  sub_layers:
    L6_A_equal_predictive:
      enabled: true
      fixed_axes:
        model_pair_strategy: user_list
  leaf_config:
    pair_user_list: [["nonexistent_model", "another_nonexistent"]]
""",
        "l6",
    )
    assert validate_layer(layer).has_hard_errors


def test_l6_b_clark_west_with_no_adjustment_hard_error():
    layer = parse_layer_yaml("6_statistical_tests:\n  enabled: true\n  sub_layers:\n    L6_B_nested:\n      enabled: true\n      fixed_axes:\n        nested_test: clark_west\n        cw_adjustment: false\n", "l6")
    assert validate_layer(layer).has_hard_errors


def test_l6_c_regime_conditioning_requires_regime_active():
    recipe = parse_recipe_yaml("1_data:\n  fixed_axes:\n    regime_definition: none\n6_statistical_tests:\n  enabled: true\n  sub_layers:\n    L6_C_cpa:\n      enabled: true\n      fixed_axes:\n        cpa_conditioning_info: regime\n")
    assert validate_recipe(recipe).has_hard_errors


def test_l6_d_mcs_alpha_default_0_10():
    layer = parse_layer_yaml("6_statistical_tests:\n  enabled: true\n  sub_layers:\n    L6_D_multiple_model:\n      enabled: true\n", "l6")
    resolved = resolve_axes(normalize_to_dag_form(layer, "l6"))
    assert resolved["L6_D_multiple_model"]["mcs_alpha"] == 0.10


def test_l6_d_bootstrap_n_replications_default_1000():
    layer = parse_layer_yaml("6_statistical_tests:\n  enabled: true\n  sub_layers:\n    L6_D_multiple_model:\n      enabled: true\n", "l6")
    resolved = resolve_axes(normalize_to_dag_form(layer, "l6"))
    assert resolved["L6_D_multiple_model"]["bootstrap_n_replications"] == 1000


def test_l6_d_bootstrap_n_too_low_hard_error():
    layer = parse_layer_yaml("6_statistical_tests:\n  enabled: true\n  sub_layers:\n    L6_D_multiple_model:\n      enabled: true\n      fixed_axes:\n        bootstrap_n_replications: 50\n", "l6")
    assert validate_layer(layer).has_hard_errors


def test_l6_d_mcs_t_statistic_only_for_mcs_test():
    layer = parse_layer_yaml("6_statistical_tests:\n  enabled: true\n  sub_layers:\n    L6_D_multiple_model:\n      enabled: true\n      fixed_axes:\n        multiple_model_test: spa_hansen\n        mcs_t_statistic: t_max\n", "l6")
    resolved = resolve_axes(normalize_to_dag_form(layer, "l6"))
    assert resolved["L6_D_multiple_model"].get_active("mcs_t_statistic") is False


def test_l6_d_stepm_alpha_only_for_stepm_test():
    layer = parse_layer_yaml("6_statistical_tests:\n  enabled: true\n  sub_layers:\n    L6_D_multiple_model:\n      enabled: true\n      fixed_axes:\n        multiple_model_test: mcs_hansen\n        stepm_alpha: 0.10\n", "l6")
    resolved = resolve_axes(normalize_to_dag_form(layer, "l6"))
    assert resolved["L6_D_multiple_model"].get_active("stepm_alpha") is False


def test_l6_e_inactive_for_point_forecast():
    root = make_recipe_without_benchmark()
    root["6_statistical_tests"] = {"enabled": True, "sub_layers": {"L6_E_density_interval": {"enabled": True, "fixed_axes": {"density_test": "pit_berkowitz"}}}}
    assert validate_recipe(parse_recipe_yaml(root)).has_hard_errors


def test_l6_e_density_test_default():
    root = make_recipe_with_density_forecast()
    root["6_statistical_tests"] = {"enabled": True, "sub_layers": {"L6_E_density_interval": {"enabled": True}}}
    resolved = resolve_axes(parse_recipe_yaml(root).layers["l6"].dag)
    assert resolved["L6_E_density_interval"]["density_test"] == "pit_berkowitz"


def test_l6_e_coverage_levels_constraint():
    layer = parse_layer_yaml("6_statistical_tests:\n  enabled: true\n  sub_layers:\n    L6_E_density_interval:\n      enabled: true\n      fixed_axes:\n        coverage_levels: [0.5, 1.5]\n", "l6")
    assert validate_layer(layer).has_hard_errors


def test_l6_f_direction_test_default():
    layer = parse_layer_yaml("6_statistical_tests:\n  enabled: true\n  sub_layers:\n    L6_F_direction:\n      enabled: true\n", "l6")
    resolved = resolve_axes(normalize_to_dag_form(layer, "l6"))
    assert resolved["L6_F_direction"]["direction_test"] == "pesaran_timmermann_1992"


def test_l6_f_user_defined_threshold_requires_value():
    layer = parse_layer_yaml("6_statistical_tests:\n  enabled: true\n  sub_layers:\n    L6_F_direction:\n      enabled: true\n      fixed_axes:\n        direction_threshold: user_defined\n", "l6")
    assert validate_layer(layer).has_hard_errors


def test_l6_g_residual_test_default():
    layer = parse_layer_yaml("6_statistical_tests:\n  enabled: true\n  sub_layers:\n    L6_G_residual:\n      enabled: true\n", "l6")
    resolved = resolve_axes(normalize_to_dag_form(layer, "l6"))
    assert resolved["L6_G_residual"]["residual_test"] == ["ljung_box_q", "arch_lm", "jarque_bera_normality"]


def test_l6_g_residual_lag_count_derived_default():
    recipe = parse_recipe_yaml("1_data:\n  fixed_axes:\n    frequency: quarterly\n6_statistical_tests:\n  enabled: true\n  sub_layers:\n    L6_G_residual:\n      enabled: true\n")
    resolved = resolve_axes(recipe.layers["l6"].dag)
    assert resolved["L6_G_residual"]["residual_lag_count"] == 4


def test_l6_global_test_scope_default():
    layer = parse_layer_yaml("6_statistical_tests:\n  enabled: true\n", "l6")
    resolved = resolve_axes(normalize_to_dag_form(layer, "l6"))
    assert resolved["test_scope"] == "per_target_horizon"


def test_l6_global_dependence_correction_default():
    layer = parse_layer_yaml("6_statistical_tests:\n  enabled: true\n", "l6")
    resolved = resolve_axes(normalize_to_dag_form(layer, "l6"))
    assert resolved["dependence_correction"] == "newey_west"


def test_l6_overlap_handling_none_with_h_gt_1_hard_error():
    recipe = parse_recipe_yaml("1_data:\n  fixed_axes:\n    horizon_set: custom_list\n  leaf_config:\n    target_horizons: [1, 6]\n6_statistical_tests:\n  enabled: true\n  overlap_handling: none\n")
    assert validate_recipe(recipe).has_hard_errors


def test_l6_axes_not_sweepable():
    layer = parse_layer_yaml("6_statistical_tests:\n  enabled: true\n  sub_layers:\n    L6_A_equal_predictive:\n      enabled: true\n      fixed_axes:\n        equal_predictive_test: {sweep: [dm_diebold_mariano, gw_giacomini_white]}\n", "l6")
    assert validate_layer(layer).has_hard_errors


def test_l6_full_replication_yaml_parses():
    assert validate_recipe(parse_recipe_yaml(_example("l6_full_replication.yaml"))).has_hard_errors is False


def test_l6_standard_yaml_parses():
    assert validate_recipe(parse_recipe_yaml(_example("l6_standard.yaml"))).has_hard_errors is False


def test_l6_step_m_operational_not_future():
    assert get_op("multiple_model_test_step_m_romano_wolf").status == "operational"


def test_l6_registered_with_spec_correct_class():
    from macrocast.core.layers.registry import get_layer
    from macrocast.core.layers.l6 import L6StatisticalTests

    spec = get_layer("l6")
    assert spec.cls is L6StatisticalTests
    assert spec.produces == ("l6_tests_v1",)
    assert spec.ui_mode == "list"
    assert spec.category == "consumption"


def test_l6_sink_in_layer_sinks():
    from macrocast.core.types import LAYER_SINKS

    assert "l6" in LAYER_SINKS
    assert "l6_tests_v1" in LAYER_SINKS["l6"]


def test_l6_layer_sinks_exact_contract():
    from macrocast.core.types import L6TestsArtifact, LAYER_SINKS

    assert LAYER_SINKS["l6"] == {"l6_tests_v1": L6TestsArtifact}


def test_l6_tests_artifact_has_required_fields():
    from macrocast.core.types import L6TestsArtifact

    fields = L6TestsArtifact.__dataclass_fields__
    for field in [
        "equal_predictive_results",
        "nested_results",
        "cpa_results",
        "multiple_model_results",
        "density_results",
        "direction_results",
        "residual_results",
        "test_metadata",
        "upstream_hashes",
        "l6_axis_resolved",
    ]:
        assert field in fields


def test_l6_sub_layer_count():
    from macrocast.core.layers.registry import get_layer

    assert len(get_layer("l6").cls.sub_layers) == 7


def test_l6_sub_layer_names():
    from macrocast.core.layers.registry import get_layer

    expected = {"L6_A_equal_predictive", "L6_B_nested", "L6_C_cpa", "L6_D_multiple_model", "L6_E_density_interval", "L6_F_direction", "L6_G_residual"}
    assert set(get_layer("l6").cls.sub_layers.keys()) == expected


def test_l6_axis_count_per_sub_layer():
    from macrocast.core.layers.registry import get_layer

    counts = {
        "L6_A_equal_predictive": 4,
        "L6_B_nested": 4,
        "L6_C_cpa": 4,
        "L6_D_multiple_model": 9,
        "L6_E_density_interval": 5,
        "L6_F_direction": 3,
        "L6_G_residual": 4,
    }
    layer_class = get_layer("l6").cls
    for sub_name, expected_count in counts.items():
        assert len(layer_class.sub_layers[sub_name].axes) == expected_count


def test_l6_globals_count():
    from macrocast.core.layers.registry import get_layer

    assert len(get_layer("l6").cls.layer_globals) == 3


def test_l6_inverse_msfe_alias_works_in_l4_combine_check_unchanged():
    op_spec = get_op("weighted_average_forecast")
    weights_methods = op_spec.params_schema["weights_method"]["options"]
    assert "dmsfe" in weights_methods
    assert "inverse_msfe" in weights_methods


def test_l6_regime_conditioning_inactive_for_pooled_test():
    recipe = parse_recipe_yaml("1_data:\n  fixed_axes:\n    regime_definition: external_nber\n6_statistical_tests:\n  enabled: true\n  test_scope: pooled\n  sub_layers:\n    L6_C_cpa:\n      enabled: true\n      fixed_axes:\n        cpa_conditioning_info: regime\n")
    assert validate_recipe(recipe).has_hard_errors is False


def test_l6_inactive_when_l5_disabled():
    recipe = parse_recipe_yaml("5_evaluation:\n  fixed_axes: {}\n6_statistical_tests:\n  enabled: true\n")
    assert validate_recipe(recipe).has_hard_errors is False
