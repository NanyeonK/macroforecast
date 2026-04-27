from __future__ import annotations

from macrocast.registry import axis_governance_table, get_axis_registry, get_axis_registry_entry
from macrocast.registry.base import AxisDefinition, BaseRegistryEntry, EnumRegistryEntry
from macrocast.registry.types import AxisRegistryEntry


EXPECTED_AXIS_COUNT = 147


def test_registry_loader_discovers_existing_axes() -> None:
    registry = get_axis_registry()
    assert len(registry) == EXPECTED_AXIS_COUNT
    assert {
        "research_design",
        "dataset",
        "information_set_type",
        "target_structure",
        "model_family",
        "importance_method",
        "source_adapter",
        "state_selection",
        "sd_variable_selection",
        "fred_sd_frequency_policy",
        "fred_sd_state_group",
        "fred_sd_variable_group",
        "fred_sd_mixed_frequency_representation",
        "relative_metrics",
        "direction_metrics",
        "regime_definition",
        "custom_preprocessor",
        "target_transformer",
        "feature_selection_semantics",
    }.issubset(registry)
    assert "task" not in registry
    assert "dataset_source" not in registry


def test_registry_loader_preserves_legacy_entry_contract() -> None:
    entry = get_axis_registry_entry("model_family")
    assert isinstance(entry, AxisRegistryEntry)
    assert entry.layer == "3_training"
    assert entry.allowed_values[:5] == ("ar", "ols", "ridge", "lasso", "elasticnet")
    assert "xgboost" in entry.allowed_values
    assert "midas_almon" in entry.allowed_values
    assert "midasr" in entry.allowed_values
    assert "midasr_nealmon" in entry.allowed_values
    assert entry.current_status["xgboost"] == "operational"
    assert entry.current_status["randomforest"] == "operational"
    assert entry.current_status["midas_almon"] == "operational_narrow"
    assert entry.current_status["midasr"] == "operational_narrow"
    assert entry.current_status["midasr_nealmon"] == "operational_narrow"
    assert entry.default_policy == "sweep"

    weight_entry = get_axis_registry_entry("midasr_weight_family")
    assert weight_entry.layer == "3_training"
    assert weight_entry.allowed_values[:2] == ("nealmon", "almonp")
    assert weight_entry.current_status["nealmon"] == "operational_narrow"
    assert weight_entry.current_status["almonp"] == "operational_narrow"
    assert weight_entry.current_status["nbeta"] == "operational_narrow"
    assert weight_entry.current_status["genexp"] == "operational_narrow"
    assert weight_entry.current_status["harstep"] == "operational_narrow"


def test_axis_governance_table_matches_discovered_registry() -> None:
    table = axis_governance_table()
    by_name = {row["axis_name"]: row for row in table}
    assert by_name["importance_method"]["current_status"]["minimal_importance"] == "operational"
    assert by_name["feature_builder"]["current_status"]["pca_factor_features"] == "operational"
    assert (
        by_name["fred_sd_mixed_frequency_representation"]["current_status"][
            "native_frequency_block_payload"
        ]
        == "operational_narrow"
    )
    assert (
        by_name["fred_sd_mixed_frequency_representation"]["current_status"][
            "mixed_frequency_model_adapter"
        ]
        == "operational_narrow"
    )


def test_base_registry_types_available() -> None:
    entry = EnumRegistryEntry(
        id="demo",
        description="demo entry",
        status="planned",
        priority="A",
    )
    definition = AxisDefinition(
        axis_name="demo_axis",
        layer="0_meta",
        axis_type="enum",
        default_policy="fixed",
        entries=(entry,),
        compatible_with={},
        incompatible_with={},
    )
    assert isinstance(entry, BaseRegistryEntry)
    assert definition.entries[0].id == "demo"



def test_registry_loader_discovers_axis_type_meta_axis() -> None:
    registry = get_axis_registry()
    assert len(registry) == EXPECTED_AXIS_COUNT
    assert "axis_type" in registry
    entry = get_axis_registry_entry("axis_type")
    assert entry.allowed_values == (
        "fixed",
        "sweep",
        "nested_sweep",
        "conditional",
        "derived",
    )



def test_registry_loader_discovers_reproducibility_mode_meta_axis() -> None:
    registry = get_axis_registry()
    assert len(registry) == EXPECTED_AXIS_COUNT
    assert "reproducibility_mode" in registry
    entry = get_axis_registry_entry("reproducibility_mode")
    assert entry.allowed_values == (
        "strict_reproducible",
        "seeded_reproducible",
        "best_effort",
        "exploratory",
    )



def test_registry_loader_discovers_failure_policy_meta_axis() -> None:
    registry = get_axis_registry()
    assert len(registry) == EXPECTED_AXIS_COUNT
    assert "failure_policy" in registry
    entry = get_axis_registry_entry("failure_policy")
    assert entry.allowed_values == (
        "fail_fast",
        "skip_failed_cell",
        "skip_failed_model",
        "retry_then_skip",
        "fallback_to_default_hp",
        "save_partial_results",
        "warn_only",
    )



def test_registry_loader_discovers_compute_mode_meta_axis() -> None:
    registry = get_axis_registry()
    assert len(registry) == EXPECTED_AXIS_COUNT
    assert "compute_mode" in registry
    entry = get_axis_registry_entry("compute_mode")
    assert entry.allowed_values == (
        "serial",
        "parallel_by_model",
        "parallel_by_horizon",
        "parallel_by_target",
        "parallel_by_oos_date",
        "parallel_by_trial",
        "distributed_cluster",
    )



def test_registry_loader_discovers_stage1_data_task_axes() -> None:
    registry = get_axis_registry()
    expected = {
        "source_adapter",
        "frequency",
        "information_set_type",
        "official_transform_policy",
        "official_transform_scope",
        "contemporaneous_x_rule",
        "missing_availability",
        "raw_missing_policy",
        "raw_outlier_policy",
        "release_lag_rule",
        "target_structure",
        "variable_universe",
        "fred_sd_state_group",
        "fred_sd_variable_group",
    }
    assert expected.issubset(registry)
    assert all(registry[axis].layer == "1_data_task" for axis in expected)


def test_registry_loader_discovers_stage4_evaluation_axes() -> None:
    registry = get_axis_registry()
    expected = {
        "oos_period",
        "point_metrics",
        "relative_metrics",
        "direction_metrics",
        "density_metrics",
        "economic_metrics",
        "benchmark_window",
        "benchmark_scope",
        "agg_time",
        "agg_horizon",
        "agg_target",
        "ranking",
        "report_style",
        "regime_definition",
        "regime_use",
        "regime_metrics",
        "decomposition_target",
        "decomposition_order",
    }
    assert expected.issubset(registry)


def test_registry_loader_tracks_migrated_axis_layers() -> None:
    registry = get_axis_registry()
    assert registry["benchmark_family"].layer == "3_training"
    assert registry["forecast_type"].layer == "3_training"
    assert registry["forecast_object"].layer == "3_training"
    assert registry["predictor_family"].layer == "2_preprocessing"
    assert registry["feature_builder"].layer == "2_preprocessing"
    assert registry["data_richness_mode"].layer == "2_preprocessing"
    assert registry["factor_count"].layer == "2_preprocessing"
    assert registry["horizon_target_construction"].layer == "2_preprocessing"
    assert registry["deterministic_components"].layer == "2_preprocessing"
    assert registry["structural_break_segmentation"].layer == "2_preprocessing"
    assert registry["oos_period"].layer == "4_evaluation"
    assert registry["overlap_handling"].layer == "6_stat_tests"
    assert registry["evaluation_scale"].layer == "2_preprocessing"  # 1.5 cleanup: re-homed to Layer 2 where the PreprocessContract field lives


def test_registry_loader_discovers_information_set_type_axis() -> None:
    entry = get_axis_registry_entry("information_set_type")
    assert entry.allowed_values == (
        "final_revised_data",
        "pseudo_oos_on_revised_data",
    )


def test_registry_loader_preserves_stage1_operational_values() -> None:
    variable_universe = get_axis_registry_entry("variable_universe")
    assert variable_universe.current_status["all_variables"] == "operational"
    assert variable_universe.current_status["explicit_variable_list"] == "operational"

    state_selection = get_axis_registry_entry("state_selection")
    assert state_selection.layer == "1_data_task"
    assert state_selection.current_status["all_states"] == "operational"
    assert state_selection.current_status["selected_states"] == "operational"

    sd_variable_selection = get_axis_registry_entry("sd_variable_selection")
    assert sd_variable_selection.layer == "1_data_task"
    assert sd_variable_selection.current_status["all_sd_variables"] == "operational"
    assert sd_variable_selection.current_status["selected_sd_variables"] == "operational"

    fred_sd_frequency_policy = get_axis_registry_entry("fred_sd_frequency_policy")
    assert fred_sd_frequency_policy.layer == "1_data_task"
    assert fred_sd_frequency_policy.allowed_values == (
        "report_only",
        "allow_mixed_frequency",
        "reject_mixed_known_frequency",
        "require_single_known_frequency",
    )
    assert fred_sd_frequency_policy.current_status["require_single_known_frequency"] == "operational"

    fred_sd_state_group = get_axis_registry_entry("fred_sd_state_group")
    assert fred_sd_state_group.layer == "1_data_task"
    assert fred_sd_state_group.current_status["all_states"] == "operational"
    assert fred_sd_state_group.current_status["census_region_west"] == "operational"
    assert fred_sd_state_group.current_status["custom_state_group"] == "operational"

    fred_sd_variable_group = get_axis_registry_entry("fred_sd_variable_group")
    assert fred_sd_variable_group.layer == "1_data_task"
    assert fred_sd_variable_group.current_status["all_sd_variables"] == "operational"
    assert fred_sd_variable_group.current_status["labor_market_core"] == "operational"
    assert fred_sd_variable_group.current_status["custom_sd_variable_group"] == "operational"



def test_registry_loader_discovers_stage2_governance_axes() -> None:
    registry = get_axis_registry()
    expected = {
        "horizon_target_construction",
        "deterministic_components",
        "structural_break_segmentation",
        "representation_policy",
        "tcode_application_scope",
        "target_transform",
        "target_normalization",
        "target_domain",
        "scaling_scope",
        "additional_preprocessing",
        "x_lag_creation",
        "feature_grouping",
        "feature_selection_semantics",
        "feature_builder",
        "predictor_family",
        "data_richness_mode",
        "factor_count",
        "feature_block_set",
        "target_lag_block",
        "target_lag_selection",
        "x_lag_feature_block",
        "factor_feature_block",
        "factor_rotation_order",
        "level_feature_block",
        "rotation_feature_block",
        "temporal_feature_block",
        "feature_block_combination",
    }
    assert expected.issubset(registry)


def test_registry_loader_marks_target_scale_runtime_values_operational() -> None:
    evaluation_scale = get_axis_registry_entry("evaluation_scale")
    target_normalization = get_axis_registry_entry("target_normalization")
    inverse_transform = get_axis_registry_entry("inverse_transform_policy")

    assert evaluation_scale.current_status["transformed_scale"] == "operational"
    assert evaluation_scale.current_status["both"] == "operational"
    assert target_normalization.current_status["zscore_train_only"] == "operational"
    assert target_normalization.current_status["robust_zscore"] == "operational"
    assert target_normalization.current_status["minmax"] == "operational"
    assert target_normalization.current_status["unit_variance"] == "operational"
    assert inverse_transform.current_status["target_only"] == "operational"
    assert inverse_transform.current_status["forecast_scale_only"] == "operational"


def test_registry_loader_defines_layer2_feature_block_grammar() -> None:
    registry = get_axis_registry()
    block_axes = {
        "feature_block_set",
        "target_lag_block",
        "target_lag_selection",
        "x_lag_feature_block",
        "factor_feature_block",
        "level_feature_block",
        "rotation_feature_block",
        "temporal_feature_block",
        "feature_block_combination",
    }
    assert all(registry[axis].layer == "2_preprocessing" for axis in block_axes)
    assert registry["feature_block_set"].current_status["target_lags_only"] == "operational"
    assert registry["feature_block_set"].current_status["transformed_predictors"] == "operational"
    assert registry["feature_block_set"].current_status["factor_blocks_only"] == "operational"
    assert registry["feature_block_set"].current_status["mixed_feature_blocks"] == "operational_narrow"
    assert registry["feature_block_set"].current_status["feature_builder_compatibility_bridge"] == "registry_only"
    assert registry["rotation_feature_block"].current_status["none"] == "operational"
    assert registry["rotation_feature_block"].current_status["moving_average_rotation"] == "operational"
    assert registry["rotation_feature_block"].current_status["marx_rotation"] == "operational"
    assert registry["rotation_feature_block"].current_status["maf_rotation"] == "operational"
    assert registry["rotation_feature_block"].current_status["custom_rotation"] == "registry_only"
    assert registry["factor_rotation_order"].current_status["rotation_then_factor"] == "operational"
    assert registry["factor_rotation_order"].current_status["factor_then_rotation"] == "operational"
    assert registry["factor_feature_block"].current_status["pca_static_factors"] == "operational"
    assert registry["level_feature_block"].current_status["none"] == "operational"
    assert registry["level_feature_block"].current_status["target_level_addback"] == "operational"
    assert registry["level_feature_block"].current_status["x_level_addback"] == "operational"
    assert registry["level_feature_block"].current_status["selected_level_addbacks"] == "operational"
    assert registry["level_feature_block"].current_status["level_growth_pairs"] == "operational"
    assert registry["temporal_feature_block"].current_status["none"] == "operational"
    assert registry["temporal_feature_block"].current_status["moving_average_features"] == "operational"
    assert registry["temporal_feature_block"].current_status["rolling_moments"] == "operational"
    assert registry["temporal_feature_block"].current_status["volatility_features"] == "operational"
    assert registry["temporal_feature_block"].current_status["local_temporal_factors"] == "operational"
    assert registry["temporal_feature_block"].current_status["custom_temporal_features"] == "registry_only"
    assert registry["feature_block_combination"].current_status["replace_with_selected_blocks"] == "operational"
    assert registry["feature_block_combination"].current_status["append_to_base_predictors"] == "operational"
    assert registry["feature_block_combination"].current_status["append_to_target_lags"] == "operational"
    assert registry["feature_block_combination"].current_status["concatenate_named_blocks"] == "operational"
    assert registry["feature_block_combination"].current_status["custom_feature_combiner"] == "registry_only"
    assert registry["target_lag_selection"].current_status["ic_select"] == "registry_only"


def test_registry_loader_expands_stage2_operational_values() -> None:
    x_missing = get_axis_registry_entry("x_missing_policy")
    x_outlier = get_axis_registry_entry("x_outlier_policy")
    scaling = get_axis_registry_entry("scaling_policy")
    dimred = get_axis_registry_entry("dimensionality_reduction_policy")
    feature_selection = get_axis_registry_entry("feature_selection_policy")
    feature_selection_semantics = get_axis_registry_entry("feature_selection_semantics")
    tcode_policy = get_axis_registry_entry("tcode_policy")
    preprocess_order = get_axis_registry_entry("preprocess_order")
    assert x_missing.current_status["mean_impute"] == "operational"
    assert x_missing.current_status["median_impute"] == "operational"
    assert x_outlier.current_status["winsorize"] == "operational"
    assert x_outlier.current_status["iqr_clip"] == "operational"
    assert scaling.current_status["minmax"] == "operational"
    assert dimred.current_status["pca"] == "operational"
    assert dimred.current_status["static_factor"] == "operational"
    assert feature_selection.current_status["correlation_filter"] == "operational"
    assert feature_selection.current_status["lasso_selection"] == "operational"
    assert feature_selection_semantics.current_status["select_before_factor"] == "operational"
    assert feature_selection_semantics.current_status["select_after_factor"] == "operational"
    assert feature_selection_semantics.current_status["select_after_custom_feature_blocks"] == "operational"
    assert tcode_policy.current_status["official_tcode_then_extra_preprocess"] == "operational"
    assert preprocess_order.current_status["official_tcode_then_extra"] == "operational"
    target_construction = get_axis_registry_entry("horizon_target_construction")
    assert target_construction.current_status["future_target_level_t_plus_h"] == "operational"
    assert target_construction.current_status["future_target_level_t_plus_h"] == "operational"
    assert target_construction.current_status["average_growth_1_to_h"] == "operational"
    assert target_construction.current_status["average_difference_1_to_h"] == "operational"
    assert target_construction.current_status["average_log_growth_1_to_h"] == "operational"
    target_lag_block = get_axis_registry_entry("target_lag_block")
    target_lag_selection = get_axis_registry_entry("target_lag_selection")
    x_lag_feature_block = get_axis_registry_entry("x_lag_feature_block")
    factor_feature_block = get_axis_registry_entry("factor_feature_block")
    assert target_lag_block.current_status["none"] == "operational"
    assert target_lag_block.current_status["fixed_target_lags"] == "operational"
    assert target_lag_selection.current_status["none"] == "operational"
    assert target_lag_selection.current_status["fixed"] == "operational"
    assert x_lag_feature_block.current_status["none"] == "operational"
    assert x_lag_feature_block.current_status["fixed_predictor_lags"] == "operational"
    assert factor_feature_block.current_status["none"] == "operational"
    assert factor_feature_block.current_status["pca_static_factors"] == "operational"
    assert factor_feature_block.current_status["pca_factor_lags"] == "operational"
    assert factor_feature_block.current_status["supervised_factors"] == "operational"


def test_registry_loader_marks_remaining_stage2_non_executable_values() -> None:
    target_missing = get_axis_registry_entry("target_missing_policy")
    dimred = get_axis_registry_entry("dimensionality_reduction_policy")
    x_lag = get_axis_registry_entry("x_lag_creation")
    feature_grouping = get_axis_registry_entry("feature_grouping")
    separation_rule = get_axis_registry_entry("separation_rule")
    target_construction = get_axis_registry_entry("horizon_target_construction")
    target_lag_block = get_axis_registry_entry("target_lag_block")
    target_lag_selection = get_axis_registry_entry("target_lag_selection")
    x_lag_feature_block = get_axis_registry_entry("x_lag_feature_block")
    factor_feature_block = get_axis_registry_entry("factor_feature_block")

    assert target_missing.current_status["em_impute"] == "registry_only"
    assert set(dimred.current_status) == {"none", "pca", "static_factor", "custom"}
    assert x_lag.current_status["cv_selected_predictor_lags"] == "registry_only"
    assert feature_grouping.current_status["fred_category_group"] == "registry_only"
    assert feature_grouping.current_status["lag_group"] == "registry_only"
    assert separation_rule.current_status["shared_transform_then_split"] == "registry_only"
    assert separation_rule.current_status["X_only_transform"] == "registry_only"
    assert separation_rule.current_status["target_only_transform"] == "registry_only"
    assert target_construction.current_status["path_average_growth_1_to_h"] == "operational"
    assert target_construction.current_status["path_average_difference_1_to_h"] == "operational"
    assert target_construction.current_status["path_average_log_growth_1_to_h"] == "operational"
    assert target_lag_block.current_status["ic_selected_target_lags"] == "registry_only"
    assert target_lag_block.current_status["custom_target_lags"] == "registry_only"
    assert target_lag_selection.current_status["ic_select"] == "registry_only"
    assert x_lag_feature_block.current_status["cv_selected_predictor_lags"] == "registry_only"



def test_registry_loader_discovers_stage3_training_axes() -> None:
    registry = get_axis_registry()
    expected = {
        "benchmark_family",
        "forecast_type",
        "forecast_object",
        "min_train_size",
        "training_start_rule",
        "outer_window",
        "refit_policy",
        "data_richness_mode",
        "sequence_framework",
        "horizon_modelization",
        "validation_size_rule",
        "validation_location",
        "embargo_gap",
        "split_family",
        "shuffle_rule",
        "alignment_fairness",
        "search_algorithm",
        "tuning_objective",
        "tuning_budget",
        "hp_space_style",
        "seed_policy",
        "early_stopping",
        "convergence_handling",
        "y_lag_count",
        "lookback",
        "logging_level",
        "checkpointing",
        "cache_policy",
        "execution_backend",
    }
    assert expected.issubset(registry)


def test_registry_loader_expands_stage3_model_family_axis() -> None:
    entry = get_axis_registry_entry("model_family")
    for value in ("ols", "bayesianridge", "huber", "adaptivelasso", "svr_linear", "svr_rbf", "componentwise_boosting", "boosting_ridge", "boosting_lasso", "pcr", "pls", "factor_augmented_linear", "extratrees", "gbm", "xgboost", "lightgbm", "catboost", "mlp"):
        assert value in entry.allowed_values
        assert entry.current_status[value] == "operational"



def test_registry_loader_discovers_stage5_output_axes() -> None:
    registry = get_axis_registry()
    expected = {"saved_objects", "provenance_fields", "export_format", "artifact_granularity"}
    assert expected.issubset(registry)
    assert registry["saved_objects"].layer == "5_output_provenance"
    assert registry["saved_objects"].current_status["full_bundle"] == "operational"
    assert registry["saved_objects"].current_status["predictions_and_metrics"] == "operational"
    assert registry["saved_objects"].current_status["predictions_only"] == "operational"
    assert registry["saved_objects"].current_status["none"] == "registry_only"
    assert registry["saved_objects"].current_status["models_only"] == "future"
    assert registry["saved_objects"].current_status["data_only"] == "future"
    assert registry["artifact_granularity"].current_status["aggregated"] == "operational"
    assert registry["artifact_granularity"].current_status["per_target"] == "registry_only"
    assert registry["artifact_granularity"].current_status["per_target_horizon"] == "future"
    assert registry["artifact_granularity"].current_status["hierarchical"] == "future"
    assert registry["export_format"].current_status["parquet"] == "operational"



def test_registry_loader_discovers_stage6_test_axes() -> None:
    registry = get_axis_registry()
    assert registry["overlap_handling"].layer == "6_stat_tests"
    assert registry["stat_test"].current_status["mcs"] == "operational"
    assert registry["stat_test"].current_status["spa"] == "operational"
    assert registry["stat_test"].current_status["diagnostics_full"] == "operational"
    assert registry["dependence_correction"].current_status["block_bootstrap"] == "operational"
    assert registry["test_scope"].current_status["per_target"] == "operational"
    assert registry["test_scope"].current_status["per_horizon"] == "registry_only"
    assert registry["test_scope"].current_status["full_grid_pairwise"] == "future"



def test_registry_loader_discovers_stage7_importance_axes() -> None:
    registry = get_axis_registry()
    expected = {
        "importance_method",
        "importance_scope",
        "importance_model_native",
        "importance_model_agnostic",
        "importance_shap",
        "importance_local_surrogate",
        "importance_partial_dependence",
        "importance_grouped",
        "importance_stability",
        "importance_aggregation",
        "importance_output_style",
        "importance_temporal",
        "importance_gradient_path",
    }
    assert expected.issubset(registry)
    assert registry["importance_method"].current_status["tree_shap"] == "operational"
    assert registry["importance_model_native"].current_status["none"] == "operational"
    assert registry["importance_shap"].current_status["none"] == "operational"
    assert registry["importance_grouped"].current_status["grouped_permutation"] == "operational"
    assert registry["importance_grouped"].current_status["variable_root_groups"] == "registry_only"
    assert registry["importance_stability"].current_status["importance_stability"] == "operational"
    assert registry["importance_stability"].current_status["seed_stability"] == "registry_only"
