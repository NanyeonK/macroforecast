from __future__ import annotations

from macrocast.registry import axis_governance_table, get_axis_registry, get_axis_registry_entry
from macrocast.registry.base import AxisDefinition, BaseRegistryEntry, EnumRegistryEntry
from macrocast.registry.types import AxisRegistryEntry


def test_registry_loader_discovers_existing_axes() -> None:
    registry = get_axis_registry()
    assert len(registry) == 125
    assert {"research_design", "dataset", "information_set_type", "task", "model_family", "importance_method", "dataset_source", "relative_metrics", "direction_metrics", "regime_definition"}.issubset(registry)


def test_registry_loader_preserves_legacy_entry_contract() -> None:
    entry = get_axis_registry_entry("model_family")
    assert isinstance(entry, AxisRegistryEntry)
    assert entry.layer == "3_training"
    assert entry.allowed_values[:5] == ("ar", "ols", "ridge", "lasso", "elasticnet")
    assert "xgboost" in entry.allowed_values
    assert entry.current_status["xgboost"] == "operational"
    assert entry.current_status["randomforest"] == "operational"
    assert entry.default_policy == "sweep"


def test_axis_governance_table_matches_discovered_registry() -> None:
    table = axis_governance_table()
    by_name = {row["axis_name"]: row for row in table}
    assert by_name["importance_method"]["current_status"]["minimal_importance"] == "operational"
    assert by_name["feature_builder"]["current_status"]["factor_pca"] == "operational"


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
    assert len(registry) == 125
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
    assert len(registry) == 125
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
    assert len(registry) == 125
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
    assert len(registry) == 125
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
        "dataset_source",
        "frequency",
        "information_set_type",
        "forecast_type",
        "forecast_object",
        "horizon_target_construction",
        "overlap_handling",
        "predictor_family",
        "training_start_rule",
        "oos_period",
        "min_train_size",
        "structural_break_segmentation",
        "contemporaneous_x_rule",
        "deterministic_components",
    }
    assert expected.issubset(registry)


def test_registry_loader_discovers_stage4_evaluation_axes() -> None:
    registry = get_axis_registry()
    expected = {
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


def test_registry_loader_moves_benchmark_family_and_evaluation_scale_to_data_task() -> None:
    registry = get_axis_registry()
    assert registry["benchmark_family"].layer == "1_data_task"
    assert registry["evaluation_scale"].layer == "2_preprocessing"  # 1.5 cleanup: re-homed to Layer 2 where the PreprocessContract field lives


def test_registry_loader_discovers_information_set_type_axis() -> None:
    entry = get_axis_registry_entry("information_set_type")
    assert entry.allowed_values == (
        "revised",
        "pseudo_oos_revised",
    )


def test_registry_loader_preserves_stage1_operational_values() -> None:
    predictor_family = get_axis_registry_entry("predictor_family")
    assert predictor_family.current_status["target_lags_only"] == "operational"
    assert predictor_family.current_status["all_macro_vars"] == "operational"



def test_registry_loader_discovers_stage2_governance_axes() -> None:
    registry = get_axis_registry()
    expected = {
        "representation_policy",
        "preprocessing_axis_role",
        "tcode_application_scope",
        "target_transform",
        "target_normalization",
        "target_domain",
        "scaling_scope",
        "additional_preprocessing",
        "x_lag_creation",
        "feature_grouping",
        "recipe_mode",
    }
    assert expected.issubset(registry)


def test_registry_loader_expands_stage2_operational_values() -> None:
    x_missing = get_axis_registry_entry("x_missing_policy")
    x_outlier = get_axis_registry_entry("x_outlier_policy")
    scaling = get_axis_registry_entry("scaling_policy")
    dimred = get_axis_registry_entry("dimensionality_reduction_policy")
    feature_selection = get_axis_registry_entry("feature_selection_policy")
    assert x_missing.current_status["mean_impute"] == "operational"
    assert x_missing.current_status["median_impute"] == "operational"
    assert x_outlier.current_status["winsorize"] == "operational"
    assert x_outlier.current_status["iqr_clip"] == "operational"
    assert scaling.current_status["minmax"] == "operational"
    assert dimred.current_status["pca"] == "operational"
    assert dimred.current_status["static_factor"] == "operational"
    assert feature_selection.current_status["correlation_filter"] == "operational"
    assert feature_selection.current_status["lasso_select"] == "operational"



def test_registry_loader_discovers_stage3_training_axes() -> None:
    registry = get_axis_registry()
    expected = {
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
        "factor_count",
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
    assert registry["artifact_granularity"].current_status["aggregated"] == "operational"
    assert registry["export_format"].current_status["parquet"] == "operational"



def test_registry_loader_discovers_stage6_test_axes() -> None:
    registry = get_axis_registry()
    assert registry["stat_test"].current_status["mcs"] == "operational"
    assert registry["stat_test"].current_status["spa"] == "operational"
    assert registry["stat_test"].current_status["diagnostics_full"] == "operational"
    assert registry["dependence_correction"].current_status["block_bootstrap"] == "operational"



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
    assert registry["importance_grouped"].current_status["grouped_permutation"] == "operational"
    assert registry["importance_stability"].current_status["importance_stability"] == "operational"
