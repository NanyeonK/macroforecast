"""Tests for macrocast.studies.multi_target.execute_separate_runs."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from macrocast import execute_separate_runs, SeparateRunsResult, SEPARATE_RUNS_MANIFEST_SCHEMA_VERSION
from macrocast.studies.multi_target import _build_single_target_recipe_dict


FIXTURE_RAW = Path("tests/fixtures/fred_md_ar_sample.csv")


def _multi_target_recipe() -> dict:
    return {
        "recipe_id": "separate-runs-test",
        "path": {
            "0_meta": {"fixed_axes": {
                "research_design": "single_forecast_run",
                "experiment_unit": "multi_target_separate_runs",
            }},
            "1_data_task": {
                "fixed_axes": {
                    "dataset": "fred_md",
                    "information_set_type": "final_revised_data",
                    "target_structure": "multi_target",
                },
                "leaf_config": {"targets": ["INDPRO", "RPI"], "horizons": [1]},
            },
            "2_preprocessing": {"fixed_axes": {
                "target_transform_policy": "raw_level", "x_transform_policy": "raw_level",
                "tcode_policy": "raw_only", "target_missing_policy": "none",
                "x_missing_policy": "none", "target_outlier_policy": "none",
                "x_outlier_policy": "none", "scaling_policy": "none",
                "dimensionality_reduction_policy": "none", "feature_selection_policy": "none",
                "preprocess_order": "none", "preprocess_fit_scope": "not_applicable",
                "inverse_transform_policy": "none", "evaluation_scale": "raw_level",
            }},
            "3_training": {"fixed_axes": {
                "framework": "rolling", "benchmark_family": "zero_change",
                "feature_builder": "target_lag_features", "model_family": "ar",
            }},
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"manifest_mode": "full", "benchmark_config": {"minimum_train_size": 5, "rolling_window_size": 5}}},
            "6_stat_tests": {"fixed_axes": {}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }


def test_build_single_target_recipe_dict_sets_target_structure_and_target() -> None:
    src = _multi_target_recipe()
    variant = _build_single_target_recipe_dict(src, "RPI")

    assert variant["path"]["1_data_task"]["fixed_axes"]["target_structure"] == "single_target"
    assert "task" not in variant["path"]["1_data_task"]["fixed_axes"]
    assert variant["path"]["1_data_task"]["leaf_config"]["target"] == "RPI"
    assert "targets" not in variant["path"]["1_data_task"]["leaf_config"]
    # experiment_unit=multi_target_separate_runs must be cleared on the child
    assert "experiment_unit" not in variant["path"]["0_meta"].get("fixed_axes", {})
    assert variant["recipe_id"].endswith("__target__RPI")


def test_execute_separate_runs_fans_out_per_target(tmp_path: Path) -> None:
    result = execute_separate_runs(
        source_recipe_dict=_multi_target_recipe(),
        output_root=tmp_path,
        local_raw_source=FIXTURE_RAW,
    )
    assert isinstance(result, SeparateRunsResult)
    assert set(result.successful_targets) == {"INDPRO", "RPI"}
    assert result.failed_targets == ()

    # Per-target artifact directories exist
    assert (tmp_path / "targets" / "INDPRO").is_dir()
    assert (tmp_path / "targets" / "RPI").is_dir()

    # Top-level aggregate manifest
    manifest = json.loads(Path(result.manifest_path).read_text())
    assert manifest["schema_version"] == SEPARATE_RUNS_MANIFEST_SCHEMA_VERSION
    assert manifest["experiment_unit"] == "multi_target_separate_runs"
    assert manifest["targets"] == ["INDPRO", "RPI"]
    assert manifest["summary"]["total"] == 2
    assert manifest["summary"]["successful"] == 2
    assert manifest["summary"]["failed"] == 0


def test_execute_separate_runs_requires_at_least_two_targets(tmp_path: Path) -> None:
    recipe = _multi_target_recipe()
    recipe["path"]["1_data_task"]["leaf_config"]["targets"] = ["INDPRO"]  # only 1
    with pytest.raises(ValueError, match="at least two target names"):
        execute_separate_runs(
            source_recipe_dict=recipe,
            output_root=tmp_path,
            local_raw_source=FIXTURE_RAW,
        )


def test_multi_target_separate_runs_registry_entry_is_operational() -> None:
    from macrocast.registry.stage0.experiment_unit import get_experiment_unit_entry

    entry = get_experiment_unit_entry("multi_target_separate_runs")
    assert entry.status == "operational"
    assert entry.runner == "macrocast.studies.multi_target:execute_separate_runs"


def test_multi_target_separate_runs_exported_at_top_level() -> None:
    import macrocast

    assert hasattr(macrocast, "execute_separate_runs")
    assert hasattr(macrocast, "SeparateRunsResult")
    assert hasattr(macrocast, "SEPARATE_RUNS_MANIFEST_SCHEMA_VERSION")


def test_multi_output_joint_model_is_dropped_from_registry() -> None:
    """Regression test: multi_output_joint_model was dropped in PR #27."""
    from macrocast.registry.stage0.experiment_unit import get_experiment_unit_entry

    with pytest.raises(KeyError):
        get_experiment_unit_entry("multi_output_joint_model")

    from macrocast.registry import get_axis_registry_entry
    entry = get_axis_registry_entry("experiment_unit")
    assert "multi_output_joint_model" not in entry.allowed_values
