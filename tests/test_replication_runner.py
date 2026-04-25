"""Tests for macrocast.studies.replication.execute_replication."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from macrocast.studies.replication import (
    REPLICATION_DIFF_SCHEMA_VERSION,
    execute_replication,
)


FIXTURE_RAW = Path("tests/fixtures/fred_md_ar_sample.csv")


def _baseline_recipe() -> dict:
    return {
        "recipe_id": "replication-src",
        "path": {
            "0_meta": {"fixed_axes": {"research_design": "single_path_benchmark"}},
            "1_data_task": {
                "fixed_axes": {
                    "dataset": "fred_md",
                    "info_set": "revised",
                    "task": "single_target_point_forecast",
                },
                "leaf_config": {"target": "INDPRO", "horizons": [1]},
            },
            "2_preprocessing": {
                "fixed_axes": {
                    "target_transform_policy": "raw_level",
                    "x_transform_policy": "raw_level",
                    "tcode_policy": "raw_only",
                    "target_missing_policy": "none",
                    "x_missing_policy": "none",
                    "target_outlier_policy": "none",
                    "x_outlier_policy": "none",
                    "scaling_policy": "none",
                    "dimensionality_reduction_policy": "none",
                    "feature_selection_policy": "none",
                    "preprocess_order": "none",
                    "preprocess_fit_scope": "not_applicable",
                    "inverse_transform_policy": "none",
                    "evaluation_scale": "raw_level",
                }
            },
            "3_training": {
                "fixed_axes": {
                    "framework": "rolling",
                    "benchmark_family": "zero_change",
                    "feature_builder": "raw_feature_panel",
                    "model_family": "ridge",
                }
            },
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {
                "leaf_config": {
                    "manifest_mode": "full",
                    "benchmark_config": {
                        "minimum_train_size": 5,
                        "rolling_window_size": 5,
                    },
                }
            },
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "minimal_importance"}},
        },
    }


def _execute_source(tmp_path: Path):
    from macrocast.compiler.build import compile_recipe_dict
    from macrocast.execution.build import execute_recipe

    recipe = _baseline_recipe()
    compile_result = compile_recipe_dict(recipe)
    return execute_recipe(
        recipe=compile_result.compiled.recipe_spec,
        preprocess=compile_result.compiled.preprocess_contract,
        output_root=tmp_path / "source",
        local_raw_source=FIXTURE_RAW,
    )


def test_replication_with_override_writes_diff_report(tmp_path: Path) -> None:
    src_exec = _execute_source(tmp_path)
    src_recipe = _baseline_recipe()

    result = execute_replication(
        source_recipe_dict=src_recipe,
        overrides={"path.3_training.fixed_axes.model_family": "lasso"},
        source_artifact_dir=src_exec.artifact_dir,
        output_root=tmp_path / "replay",
        local_raw_source=FIXTURE_RAW,
    )
    assert result.byte_identical_predictions is False
    assert result.overrides_applied == {"path.3_training.fixed_axes.model_family": "lasso"}

    diff = json.loads(Path(result.diff_report_path).read_text())
    assert diff["schema_version"] == REPLICATION_DIFF_SCHEMA_VERSION
    assert diff["overrides_applied"] == {"path.3_training.fixed_axes.model_family": "lasso"}
    assert len(diff["override_diff_entries"]) == 1
    assert diff["override_diff_entries"][0]["old"] == "ridge"
    assert diff["override_diff_entries"][0]["new"] == "lasso"
    assert diff["byte_identical_predictions"] is False
    assert isinstance(diff["metrics_delta"], dict)


def test_replication_no_overrides_no_source_artifact(tmp_path: Path) -> None:
    """Without source_artifact_dir, byte-identical is False and metrics_delta is empty."""
    src_recipe = _baseline_recipe()

    result = execute_replication(
        source_recipe_dict=src_recipe,
        overrides={},
        source_artifact_dir=None,
        output_root=tmp_path / "replay-standalone",
        local_raw_source=FIXTURE_RAW,
    )
    assert result.byte_identical_predictions is False
    diff = json.loads(Path(result.diff_report_path).read_text())
    assert diff["metrics_delta"] == {}
    assert diff["source_artifact_dir"] is None
    assert diff["override_diff_entries"] == []


def test_replication_override_mode_compiles_as_replication_handoff(tmp_path: Path) -> None:
    """replication_override is represented for execute_replication, not direct execution."""
    from macrocast.compiler.build import compile_recipe_dict

    src_recipe = _baseline_recipe()
    src_recipe["path"]["0_meta"]["fixed_axes"]["research_design"] = "replication_override"

    compile_result = compile_recipe_dict(src_recipe)
    assert compile_result.compiled.execution_status == "ready_for_replication_runner"
    assert compile_result.manifest["tree_context"]["route_owner"] == "replication"
    assert compile_result.manifest["tree_context"]["route_contract"] == "replication_handoff"
    # No research_design wrapper-route warning in the manifest.
    manifest_warnings = compile_result.manifest.get("warnings", [])
    assert not any(
        "replication_override" in w and "wrapper/orchestrator route" in w
        for w in manifest_warnings
    ), f"unexpected wrapper-route warning: {manifest_warnings}"


def test_replication_with_study_mode_replication_override_runs_end_to_end(tmp_path: Path) -> None:
    """End-to-end: source recipe has research_design=replication_override;
    execute_replication accepts it and produces a replay artifact."""
    from macrocast.compiler.build import compile_recipe_dict
    from macrocast.execution.build import execute_recipe

    recipe = _baseline_recipe()
    recipe["path"]["0_meta"]["fixed_axes"]["research_design"] = "replication_override"
    compile_result = compile_recipe_dict(recipe)
    src_exec = execute_recipe(
        recipe=compile_result.compiled.recipe_spec,
        preprocess=compile_result.compiled.preprocess_contract,
        output_root=tmp_path / "source",
        local_raw_source=FIXTURE_RAW,
    )

    result = execute_replication(
        source_recipe_dict=recipe,
        overrides={"path.3_training.fixed_axes.model_family": "lasso"},
        source_artifact_dir=src_exec.artifact_dir,
        output_root=tmp_path / "replay",
        local_raw_source=FIXTURE_RAW,
    )
    assert result.overrides_applied == {"path.3_training.fixed_axes.model_family": "lasso"}
    assert Path(result.diff_report_path).exists()
