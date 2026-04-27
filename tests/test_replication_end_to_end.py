"""Synthetic round-trip test — byte-identical predictions under overrides={}.

This is Phase 6's Acceptance Gate P0 test: execute a recipe, run
execute_replication with no overrides over the source recipe + source
artifact dir, and verify predictions.csv is byte-identical across both runs.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from macrocast.studies.replication import execute_replication


FIXTURE_RAW = Path("tests/fixtures/fred_md_ar_sample.csv")

REPRODUCIBLE_PROVENANCE = {
    "compiler": {
        "reproducibility_spec": {
            "reproducibility_mode": "seeded_reproducible",
            "seed": 42,
        }
    }
}


def _recipe() -> dict:
    return {
        "recipe_id": "roundtrip-src",
        "path": {
            "0_meta": {"fixed_axes": {"research_design": "single_forecast_run"}},
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_synthetic_replication_roundtrip_byte_identical(tmp_path: Path) -> None:
    from macrocast.compiler.build import compile_recipe_dict
    from macrocast.execution.build import execute_recipe

    recipe = _recipe()
    compile_result = compile_recipe_dict(recipe)

    src = execute_recipe(
        recipe=compile_result.compiled.recipe_spec,
        preprocess=compile_result.compiled.preprocess_contract,
        output_root=tmp_path / "src",
        local_raw_source=FIXTURE_RAW,
        provenance_payload=REPRODUCIBLE_PROVENANCE,
    )

    rep = execute_replication(
        source_recipe_dict=recipe,
        overrides={},
        source_artifact_dir=src.artifact_dir,
        output_root=tmp_path / "replay",
        local_raw_source=FIXTURE_RAW,
        provenance_payload=REPRODUCIBLE_PROVENANCE,
    )

    src_pred = Path(src.artifact_dir) / "predictions.csv"
    rep_pred = Path(rep.execution_result.artifact_dir) / "predictions.csv"
    assert src_pred.is_file()
    assert rep_pred.is_file()
    assert _sha256(src_pred) == _sha256(rep_pred), (
        "round-trip predictions diverged under seeded_reproducible mode"
    )
    assert rep.byte_identical_predictions is True
