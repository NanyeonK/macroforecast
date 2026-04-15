from __future__ import annotations

from macrocast.registry import axis_governance_table, get_axis_registry, get_axis_registry_entry
from macrocast.registry.base import AxisDefinition, BaseRegistryEntry, EnumRegistryEntry
from macrocast.registry.types import AxisRegistryEntry


def test_registry_loader_discovers_existing_axes() -> None:
    registry = get_axis_registry()
    assert len(registry) == 31
    assert {"study_mode", "dataset", "info_set", "task", "model_family", "importance_method"}.issubset(registry)


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
    assert len(registry) == 31
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
    assert len(registry) == 31
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
    assert len(registry) == 31
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
    assert len(registry) == 31
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
    assert len(registry) == 31
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
