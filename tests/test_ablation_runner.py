"""Tests for macrocast.studies.ablation.execute_ablation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from macrocast.studies.ablation import (
    ABLATION_REPORT_SCHEMA_VERSION,
    AblationSpec,
    execute_ablation,
)


FIXTURE_RAW = Path("tests/fixtures/fred_md_ar_sample.csv")


def _baseline_single_path_recipe() -> dict:
    return {
        "recipe_id": "ablation-baseline",
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


def test_ablation_produces_baseline_plus_components(tmp_path: Path) -> None:
    baseline = _baseline_single_path_recipe()
    spec = AblationSpec(
        baseline_recipe_dict=baseline,
        components_to_ablate=(
            ("path.3_training.fixed_axes.model_family", "lasso"),
        ),
    )

    result = execute_ablation(
        spec=spec,
        output_root=tmp_path,
        local_raw_source=FIXTURE_RAW,
    )
    assert result.size == 2  # baseline + one ablation
    assert result.failed_count == 0

    baseline_v = next(v for v in result.per_variant_results if v.variant_id == "v-baseline")
    assert baseline_v.status == "success"
    ablated = [v for v in result.per_variant_results if v.variant_id != "v-baseline"]
    assert len(ablated) == 1
    assert ablated[0].status == "success"

    report_path = Path(result.output_root) / "ablation_report.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text())
    assert report["schema_version"] == ABLATION_REPORT_SCHEMA_VERSION
    assert report["baseline_variant_id"] == "v-baseline"
    assert len(report["components"]) == 1
    comp = report["components"][0]
    assert comp["axis_name"] == "path.3_training.fixed_axes.model_family"
    assert comp["original_value"] == "ridge"
    assert comp["neutral_value"] == "lasso"
    assert "metrics" in comp
    assert "delta_vs_baseline" in comp


def test_ablation_study_id_is_deterministic(tmp_path: Path) -> None:
    baseline = _baseline_single_path_recipe()
    spec_a = AblationSpec(
        baseline_recipe_dict=baseline,
        components_to_ablate=(
            ("path.3_training.fixed_axes.model_family", "lasso"),
        ),
    )
    spec_b = AblationSpec(
        baseline_recipe_dict=baseline,
        components_to_ablate=(
            ("path.3_training.fixed_axes.model_family", "lasso"),
        ),
    )

    from macrocast.studies.ablation import _hash_ablation_id
    assert _hash_ablation_id(spec_a) == _hash_ablation_id(spec_b)
    assert _hash_ablation_id(spec_a).startswith("abl-")
    assert len(_hash_ablation_id(spec_a)) == 4 + 12  # "abl-" + 12 hex


def test_ablation_with_explicit_study_id(tmp_path: Path) -> None:
    baseline = _baseline_single_path_recipe()
    spec = AblationSpec(
        baseline_recipe_dict=baseline,
        components_to_ablate=(
            ("path.3_training.fixed_axes.model_family", "lasso"),
        ),
        ablation_study_id="abl-my-custom-id",
    )
    result = execute_ablation(
        spec=spec,
        output_root=tmp_path,
        local_raw_source=FIXTURE_RAW,
    )
    assert result.study_id == "abl-my-custom-id"
    report = json.loads((Path(result.output_root) / "ablation_report.json").read_text())
    assert report["ablation_study_id"] == "abl-my-custom-id"
