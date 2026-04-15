from __future__ import annotations

from macrocast.registry import axis_governance_table, get_axis_registry, get_axis_registry_entry
from macrocast.registry.base import AxisDefinition, BaseRegistryEntry, EnumRegistryEntry
from macrocast.registry.types import AxisRegistryEntry


def test_registry_loader_discovers_existing_axes() -> None:
    registry = get_axis_registry()
    assert len(registry) == 55
    assert {"study_mode", "dataset", "information_set_type", "task", "model_family", "importance_method", "data_domain", "dataset_source"}.issubset(registry)


def test_registry_loader_preserves_legacy_entry_contract() -> None:
    entry = get_axis_registry_entry("model_family")
    assert isinstance(entry, AxisRegistryEntry)
    assert entry.layer == "3_training"
    assert entry.allowed_values == ("ar", "ridge", "lasso", "elasticnet", "randomforest")
    assert entry.current_status["randomforest"] == "operational"
    assert entry.default_policy == "sweep"


def test_axis_governance_table_matches_discovered_registry() -> None:
    table = axis_governance_table()
    by_name = {row["axis_name"]: row for row in table}
    assert by_name["importance_method"]["current_status"]["minimal_importance"] == "operational"
    assert by_name["feature_builder"]["current_status"]["factor_pca"] == "planned"


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
    assert len(registry) == 55
    assert "axis_type" in registry
    entry = get_axis_registry_entry("axis_type")
    assert entry.allowed_values == (
        "fixed",
        "sweep",
        "nested_sweep",
        "conditional",
        "derived",
        "eval_only",
        "report_only",
    )



def test_registry_loader_discovers_registry_type_meta_axis() -> None:
    registry = get_axis_registry()
    assert len(registry) == 55
    assert "registry_type" in registry
    entry = get_axis_registry_entry("registry_type")
    assert entry.allowed_values == (
        "enum_registry",
        "numeric_registry",
        "callable_registry",
        "custom_plugin",
        "user_defined_yaml",
        "external_adapter",
    )



def test_axis_definition_defaults_registry_type_to_enum_registry() -> None:
    entry = EnumRegistryEntry(
        id="demo_two",
        description="demo two",
        status="operational",
        priority="A",
    )
    definition = AxisDefinition(
        axis_name="demo_axis_two",
        layer="0_meta",
        axis_type="enum",
        entries=(entry,),
        compatible_with={},
        incompatible_with={},
    )
    assert definition.registry_type == "enum_registry"



def test_registry_loader_discovers_reproducibility_mode_meta_axis() -> None:
    registry = get_axis_registry()
    assert len(registry) == 55
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
    assert len(registry) == 55
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
        "hard_error",
    )



def test_registry_loader_discovers_compute_mode_meta_axis() -> None:
    registry = get_axis_registry()
    assert len(registry) == 55
    assert "compute_mode" in registry
    entry = get_axis_registry_entry("compute_mode")
    assert entry.allowed_values == (
        "serial",
        "parallel_by_model",
        "parallel_by_horizon",
        "parallel_by_oos_date",
        "parallel_by_trial",
        "gpu_single",
        "gpu_multi",
        "distributed_cluster",
    )



def test_registry_loader_discovers_stage1_data_task_axes() -> None:
    registry = get_axis_registry()
    expected = {
        "data_domain",
        "dataset_source",
        "frequency",
        "information_set_type",
        "vintage_policy",
        "alignment_rule",
        "forecast_type",
        "forecast_object",
        "horizon_target_construction",
        "overlap_handling",
        "target_family",
        "predictor_family",
        "training_start_rule",
        "oos_period",
        "min_train_size",
        "warmup_rule",
        "structural_break_segmentation",
        "contemporaneous_x_rule",
        "own_target_lags",
        "deterministic_components",
        "exogenous_block",
        "x_map_policy",
        "target_to_target_inclusion",
        "multi_target_architecture",
        "regime_task",
    }
    assert expected.issubset(registry)


def test_registry_loader_moves_benchmark_family_and_evaluation_scale_to_data_task() -> None:
    registry = get_axis_registry()
    assert registry["benchmark_family"].layer == "1_data_task"
    assert registry["evaluation_scale"].layer == "1_data_task"


def test_registry_loader_discovers_information_set_type_axis() -> None:
    entry = get_axis_registry_entry("information_set_type")
    assert entry.allowed_values == (
        "revised",
        "real_time_vintage",
        "pseudo_oos_revised",
        "pseudo_oos_vintage_aware",
        "release_calendar_aware",
        "publication_lag_aware",
    )


def test_registry_loader_preserves_stage1_operational_values() -> None:
    data_domain = get_axis_registry_entry("data_domain")
    predictor_family = get_axis_registry_entry("predictor_family")
    assert data_domain.current_status["macro"] == "operational"
    assert predictor_family.current_status["target_lags_only"] == "operational"
    assert predictor_family.current_status["all_macro_vars"] == "operational"
