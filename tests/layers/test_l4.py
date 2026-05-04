from pathlib import Path

import pytest

from macrocast.core.layers.l4 import (
    execute_layer,
    make_l4_yaml,
    make_l4_yaml_no_benchmark,
    make_l4_yaml_refit,
    make_l4_yaml_search,
    make_l4_yaml_training_window,
    make_l4_yaml_with_combine,
    make_l4_yaml_with_combine_method,
    make_l4_yaml_with_combine_op,
    make_l4_yaml_with_combine_temporal,
    make_l4_yaml_with_cv_path,
    make_l4_yaml_with_strategy,
    make_l4_yaml_with_validation_method,
    normalize_to_dag_form,
    parse_dag_form,
    parse_layer_yaml,
    parse_recipe_yaml,
    resolve_axes,
    resolve_combine_node,
    validate_layer,
    validate_recipe,
)
from macrocast.core.ops import get_op, list_ops
from macrocast.core.ops.l4_ops import FUTURE_MODEL_FAMILIES, OPERATIONAL_MODEL_FAMILIES, get_family_status
from macrocast.core.validator import validate_dag


ROOT = Path(__file__).resolve().parents[2]


def _example(name: str) -> str:
    return (ROOT / "examples" / "recipes" / name).read_text()


def test_l4_minimal_ridge_parses():
    layer = parse_layer_yaml(_example("l4_minimal_ridge.yaml"), "l4")
    dag = parse_dag_form(layer)
    assert "l4_forecasts_v1" in layer["sinks"]
    assert validate_dag(dag).valid


def test_l4_ensemble_with_benchmark_parses():
    layer = parse_layer_yaml(_example("l4_ensemble_ridge_xgb_vs_ar1.yaml"), "l4")
    dag = parse_dag_form(layer)
    assert validate_dag(dag).valid


def test_l4_benchmark_detected_via_is_benchmark_flag():
    layer = parse_layer_yaml(_example("l4_ensemble_ridge_xgb_vs_ar1.yaml"), "l4")
    benchmark_nodes = [node for node in layer["nodes"] if node.get("is_benchmark")]
    assert len(benchmark_nodes) == 1
    assert benchmark_nodes[0]["params"]["family"] == "ar_p"


def test_l4_two_benchmarks_hard_error():
    yaml_text = """
4_forecasting_model:
  nodes:
    - {id: src_X, type: source, selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: X_final}}}
    - {id: src_y, type: source, selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: y_final}}}
    - {id: fit_a, type: step, op: fit_model, params: {family: ar_p, n_lag: 1}, is_benchmark: true, inputs: [src_y]}
    - {id: fit_b, type: step, op: fit_model, params: {family: ar_p, n_lag: 4}, is_benchmark: true, inputs: [src_y]}
    - {id: predict_a, type: step, op: predict, inputs: [fit_a]}
    - {id: predict_b, type: step, op: predict, inputs: [fit_b]}
  sinks:
    l4_forecasts_v1: predict_a
    l4_model_artifacts_v1: [fit_a, fit_b]
    l4_training_metadata_v1: auto
"""
    assert validate_layer(parse_layer_yaml(yaml_text)).has_hard_errors


def test_l4_no_benchmark_means_no_relative_metrics():
    layer = parse_layer_yaml(make_l4_yaml_no_benchmark())
    assert all(not node.get("is_benchmark") for node in layer["nodes"] if node["type"] == "step")


def test_l4_ar_p_family():
    assert not validate_layer(parse_layer_yaml(make_l4_yaml(family="ar_p", n_lag=4))).has_hard_errors


def test_l4_xgboost_family():
    assert not validate_layer(parse_layer_yaml(make_l4_yaml(family="xgboost", n_estimators=100))).has_hard_errors


def test_l4_macroeconomic_random_forest_rejected_as_future():
    # PR-B (v0.1 honesty pass): MRF was promoted to operational in v0.1 even
    # though the runtime wrapper is a plain RandomForest + time_trend (not
    # the Coulombe 2024 GTVP local-linear forest). Demoted to ``future`` so
    # the validator hard-rejects until the real implementation lands.
    layer = parse_layer_yaml(_example("l4_mrf_placeholder.yaml"))
    report = validate_layer(layer)
    assert report.has_hard_errors
    assert any("future or unknown" in issue.message.lower() for issue in report.hard_errors)


def test_l4_dfm_mixed_mariano_murasawa_rejected_as_future():
    layer = parse_layer_yaml(make_l4_yaml(family="dfm_mixed_mariano_murasawa", n_factors=2))
    report = validate_layer(layer)
    assert report.has_hard_errors
    assert any("future or unknown" in issue.message.lower() for issue in report.hard_errors)


def test_l4_midas_almon_future_rejected():
    report = validate_layer(parse_layer_yaml(make_l4_yaml(family="midas_almon", n_lag=12, polynomial_degree=3)))
    assert report.has_hard_errors
    assert any("future" in issue.message.lower() for issue in report.hard_errors)


def test_l4_all_midas_future_families_rejected():
    for family in FUTURE_MODEL_FAMILIES:
        assert validate_layer(parse_layer_yaml(make_l4_yaml(family=family, n_lag=12))).has_hard_errors


def test_l4_oracle_strategy_does_not_exist():
    assert validate_layer(parse_layer_yaml(make_l4_yaml_with_strategy("oracle"))).has_hard_errors


def test_l4_path_average_requires_cumulative_average_target():
    yaml_text = """
3_feature_engineering:
  nodes:
    - {id: y_h, type: step, op: target_construction, params: {mode: point_forecast, method: direct, horizon: 6}, inputs: [src_y]}
  sinks: {}
""" + make_l4_yaml_with_strategy("path_average")
    assert validate_recipe(parse_recipe_yaml(yaml_text)).has_hard_errors


def test_l4_full_sample_once_rejected_for_combination_temporal():
    assert validate_layer(parse_layer_yaml(make_l4_yaml_with_combine_temporal("full_sample_once"))).has_hard_errors


def test_l4_dmsfe_default_theta():
    assert get_op("weighted_average_forecast").params_schema["dmsfe_theta"]["default"] == 0.95


def test_l4_inverse_msfe_aliases_dmsfe_theta_one():
    resolved = resolve_combine_node(parse_layer_yaml(make_l4_yaml_with_combine_method("inverse_msfe")))
    assert resolved["weights_method"] == "dmsfe"
    assert resolved["dmsfe_theta"] == 1.0


def test_l4_cv_path_only_for_lasso_ridge_elastic_net():
    assert validate_layer(parse_layer_yaml(make_l4_yaml_with_cv_path(family="xgboost"))).has_hard_errors


def test_l4_cv_path_valid_for_ridge():
    yaml_text = make_l4_yaml_with_cv_path(family="ridge") + "\n  leaf_config:\n    cv_path_alphas: [0.001, 0.01, 0.1]\n"
    assert not validate_layer(parse_layer_yaml(yaml_text)).has_hard_errors


def test_l4_kfold_rejected_for_time_series():
    assert validate_layer(parse_layer_yaml(make_l4_yaml_with_validation_method("kfold"))).has_hard_errors


def test_l4_training_window_fixed_requires_fixed_end_date():
    assert validate_layer(parse_layer_yaml(make_l4_yaml_training_window("fixed"))).has_hard_errors


def test_l4_refit_every_n_origins_requires_interval():
    assert validate_layer(parse_layer_yaml(make_l4_yaml_refit("every_n_origins"))).has_hard_errors


def test_l4_search_algorithm_none_disables_tuning_objective():
    layer = parse_layer_yaml(make_l4_yaml(search_algorithm="none", tuning_objective="cv_mse"))
    resolved = resolve_axes(normalize_to_dag_form(layer, "l4"))
    assert resolved.get_active("tuning_objective") is False


def test_l4_grid_search_requires_tuning_grid():
    assert validate_layer(parse_layer_yaml(make_l4_yaml_search("grid_search"))).has_hard_errors


def test_l4_bayesian_requires_tuning_distributions_and_budget():
    assert validate_layer(parse_layer_yaml(make_l4_yaml_search("bayesian_optimization"))).has_hard_errors


def test_l4_genetic_requires_population_and_generations():
    assert validate_layer(parse_layer_yaml(make_l4_yaml_search("genetic_algorithm"))).has_hard_errors


def test_l4_combine_op_requires_2_or_more_inputs():
    assert validate_layer(parse_layer_yaml(make_l4_yaml_with_combine(n_inputs=1))).has_hard_errors


def test_l4_bivariate_ardl_requires_exactly_2():
    assert validate_layer(parse_layer_yaml(make_l4_yaml_with_combine_op("bivariate_ardl_combination", n_inputs=3))).has_hard_errors


def test_l4_regime_wrapper_requires_l1_regime_active():
    yaml_text = """
1_data:
  fixed_axes:
    regime_definition: none
4_forecasting_model:
  nodes:
    - {id: src_X, type: source, selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: X_final}}}
    - {id: src_y, type: source, selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: y_final}}}
    - id: fit_ridge
      type: step
      op: fit_model
      params: {family: ridge, regime_wrapper: separate_fit_per_regime}
      inputs: [src_X, src_y]
    - {id: predict_ridge, type: step, op: predict, inputs: [fit_ridge, src_X]}
  sinks:
    l4_forecasts_v1: predict_ridge
    l4_model_artifacts_v1: fit_ridge
    l4_training_metadata_v1: auto
"""
    assert validate_recipe(parse_recipe_yaml(yaml_text)).has_hard_errors


def test_l4_three_sinks_in_layer_sinks():
    from macrocast.core.types import LAYER_SINKS

    assert "l4_forecasts_v1" in LAYER_SINKS["l4"]
    assert "l4_model_artifacts_v1" in LAYER_SINKS["l4"]
    assert "l4_training_metadata_v1" in LAYER_SINKS["l4"]


def test_l4_registered_with_spec_correct_class():
    from macrocast.core.layers.registry import get_layer
    from macrocast.core.layers.l4 import L4ForecastingModel

    spec = get_layer("l4")
    assert spec.cls is L4ForecastingModel
    assert spec.produces == ("l4_forecasts_v1", "l4_model_artifacts_v1", "l4_training_metadata_v1")
    assert spec.ui_mode == "graph"
    assert spec.category == "construction"


def test_l4_operational_model_families_registered():
    # v0.1 expanded the design's 30 operational families with two extra linear
    # baselines (bayesian_ridge, huber). The PR-B honesty pass then demoted
    # 5 misleading-implementation families (factor_augmented_var, BVAR x2,
    # MRF, DFM-MM) so the operational tuple is now ~27.
    assert len(OPERATIONAL_MODEL_FAMILIES) >= 25
    assert all(get_family_status(family) == "operational" for family in OPERATIONAL_MODEL_FAMILIES)


def test_l4_future_model_families_includes_midas_and_v0_1_demotions():
    # PR-B honesty pass demoted 5 families; v0.2 has since re-promoted
    # bvar_minnesota / bvar_normal_inverse_wishart with proper closed-form
    # Minnesota / NIW posterior mean estimators (#185 / #186). The other
    # PR-B demotions remain future until their tracking issue lands.
    expected_future = {
        "midas_almon", "midas_beta", "midas_step", "dfm_unrestricted_midas",
        "factor_augmented_var",
        "macroeconomic_random_forest",
        "dfm_mixed_mariano_murasawa",
    }
    assert expected_future <= set(FUTURE_MODEL_FAMILIES)
    assert all(get_family_status(family) == "future" for family in FUTURE_MODEL_FAMILIES)
    # The two BVAR families must NOT be in FUTURE anymore.
    assert "bvar_minnesota" not in FUTURE_MODEL_FAMILIES
    assert "bvar_normal_inverse_wishart" not in FUTURE_MODEL_FAMILIES


def test_l4_5_forecast_combine_ops_registered():
    expected = {"weighted_average_forecast", "median_forecast", "trimmed_mean_forecast", "bma_forecast", "bivariate_ardl_combination"}
    assert expected.issubset(list_ops())


def test_l4_does_not_register_l3_combine_ops():
    for op_name in ["concat", "interact", "hierarchical_pca", "weighted_concat", "simple_average"]:
        op = list_ops().get(op_name)
        assert op is None or op.layer_scope == "universal" or "l4" not in op.layer_scope


def test_l4_forecasts_v1_has_forecast_object_field():
    from macrocast.core.types import L4ForecastsArtifact

    assert "forecast_object" in L4ForecastsArtifact.__dataclass_fields__


def test_l4_training_metadata_records_per_origin_runtime():
    from macrocast.core.types import L4TrainingMetadataArtifact

    assert "runtime_per_origin" in L4TrainingMetadataArtifact.__dataclass_fields__


def test_l4_cv_optimized_default_window():
    assert "cv_optimized_window" in get_op("weighted_average_forecast").params_schema


def test_l4_horizon_in_l1_horizon_set():
    yaml_text = """
1_data:
  fixed_axes:
    horizon_set: custom_list
  leaf_config:
    target_horizons: [1, 3, 6]
""" + make_l4_yaml("xgboost")
    assert not validate_recipe(parse_recipe_yaml(yaml_text)).has_hard_errors
