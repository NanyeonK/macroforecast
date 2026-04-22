"""Tests for migrate_legacy_stat_test (Phase 2 sub-task 02.4 / 02.5)."""

from __future__ import annotations

import warnings

import pytest

from macrocast.compiler.migrations import migrate_legacy_stat_test
from macrocast.execution.stat_tests import LEGACY_TO_NEW


def _recipe_with_stat_test(value: str) -> dict:
    return {
        "recipe_id": "test",
        "path": {
            "6_stat_tests": {"fixed_axes": {"stat_test": value}},
        },
    }


def test_dm_rewrites_to_equal_predictive_dm_with_warning() -> None:
    recipe = _recipe_with_stat_test("dm")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        migrated = migrate_legacy_stat_test(recipe)
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    fixed = migrated["path"]["6_stat_tests"]["fixed_axes"]
    assert fixed["equal_predictive"] == "dm"
    assert fixed.get("stat_test") == "dm"


@pytest.mark.parametrize("legacy_value", sorted(LEGACY_TO_NEW.keys()))
def test_every_legacy_value_migrates(legacy_value: str) -> None:
    recipe = _recipe_with_stat_test(legacy_value)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        migrated = migrate_legacy_stat_test(recipe)
    axis, value = LEGACY_TO_NEW[legacy_value]
    fixed = migrated["path"]["6_stat_tests"]["fixed_axes"]
    assert fixed.get("stat_test") == legacy_value
    assert fixed[axis] == value


def test_none_value_drops_stat_test_without_warning() -> None:
    recipe = _recipe_with_stat_test("none")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        migrated = migrate_legacy_stat_test(recipe)
    assert not any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert migrated["path"]["6_stat_tests"]["fixed_axes"] == {}


def test_unknown_legacy_value_raises() -> None:
    with pytest.raises(ValueError, match="unknown legacy stat_test"):
        migrate_legacy_stat_test(_recipe_with_stat_test("not_a_real_test"))


def test_already_migrated_recipe_is_idempotent() -> None:
    recipe = {
        "recipe_id": "rt",
        "path": {"6_stat_tests": {"fixed_axes": {"equal_predictive": "dm"}}},
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = migrate_legacy_stat_test(recipe)
    assert not any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert result == recipe


def test_recipe_without_layer_6_passes_through() -> None:
    recipe = {"recipe_id": "rt", "path": {"0_meta": {"fixed_axes": {"research_design": "single_path_benchmark"}}}}
    result = migrate_legacy_stat_test(recipe)
    assert result == recipe


def test_compile_hook_runs_migration_via_compile_recipe_dict(tmp_path) -> None:
    """End-to-end: compile_recipe_dict entry point runs the shim automatically."""
    from pathlib import Path

    from macrocast import compile_recipe_dict

    recipe = {
        "recipe_id": "migration-hook-check",
        "path": {
            "0_meta": {"fixed_axes": {"research_design": "single_path_benchmark"}},
            "1_data_task": {
                "fixed_axes": {"dataset": "fred_md", "info_set": "revised", "task": "single_target_point_forecast"},
                "leaf_config": {"target": "INDPRO", "horizons": [1]},
            },
            "2_preprocessing": {"fixed_axes": {
                "target_transform_policy": "raw_level", "x_transform_policy": "raw_level", "tcode_policy": "raw_only",
                "target_missing_policy": "none", "x_missing_policy": "none", "target_outlier_policy": "none", "x_outlier_policy": "none",
                "scaling_policy": "none", "dimensionality_reduction_policy": "none", "feature_selection_policy": "none",
                "preprocess_order": "none", "preprocess_fit_scope": "not_applicable", "inverse_transform_policy": "none", "evaluation_scale": "raw_level",
            }},
            "3_training": {"fixed_axes": {
                "framework": "rolling", "benchmark_family": "zero_change", "feature_builder": "raw_feature_panel", "model_family": "ridge",
            }},
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"manifest_mode": "full", "benchmark_config": {"minimum_train_size": 5, "rolling_window_size": 5}}},
            "6_stat_tests": {"fixed_axes": {"stat_test": "dm"}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = compile_recipe_dict(recipe)

    assert any(issubclass(w.category, DeprecationWarning) for w in caught), (
        "compile_recipe_dict should emit DeprecationWarning via the migration shim"
    )
    assert result.compiled.execution_status == "executable"
