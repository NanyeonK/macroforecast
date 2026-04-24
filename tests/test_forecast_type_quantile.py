"""End-to-end tests for forecast_type (1.2.2) + forecast_object=quantile (1.2.3).

v1.0 semantics:
- `forecast_type` default is feature-builder dynamic:
  autoreg_lagged_target -> "iterated" (matches the existing recursive path),
  raw_feature_panel     -> "direct"   (matches the existing h-step path).
- `forecast_type=iterated` + feature_builder=autoreg_lagged_target   : executable.
- `forecast_type=iterated` + feature_builder=raw_feature_panel       : blocked_by_incompatibility.
- `forecast_type=direct`   + feature_builder=autoreg_lagged_target   : blocked_by_incompatibility.
- `forecast_type=direct`   + feature_builder=raw_feature_panel       : executable.

- `forecast_object=quantile` + `model_family=quantile_linear`         : executable (quantile level via training_spec.hp.quantile, default 0.5).
- `forecast_object in {point_median, quantile}` + any other model_family: blocked_by_incompatibility.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from macrocast.compiler.build import compile_recipe_dict, run_compiled_recipe


def _recipe(*, feature_builder: str = "autoreg_lagged_target", model_family: str = "ar",
            forecast_type: str | None = None, forecast_object: str | None = None) -> dict:
    axes_1 = {
        "dataset": "fred_md",
        "info_set": "revised",
        "task": "single_target_point_forecast",
    }
    if forecast_type is not None:
        axes_1["forecast_type"] = forecast_type
    if forecast_object is not None:
        axes_1["forecast_object"] = forecast_object

    training = {
        "framework": "expanding",
        "benchmark_family": "historical_mean",
        "feature_builder": feature_builder,
        "model_family": model_family,
    }

    return {
        "recipe_id": "ft-q-test",
        "path": {
            "0_meta": {"fixed_axes": {"research_design": "single_path_benchmark"}},
            "1_data_task": {
                "fixed_axes": axes_1,
                "leaf_config": {"target": "INDPRO", "horizons": [1]},
            },
            "2_preprocessing": {"fixed_axes": {
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
            }},
            "3_training": {"fixed_axes": training},
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {
                "manifest_mode": "full",
                "benchmark_config": {"minimum_train_size": 5, "rolling_window_size": 5},
            }},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }


def test_forecast_type_default_is_iterated_for_autoreg() -> None:
    r = compile_recipe_dict(_recipe())
    assert r.manifest["training_spec"]["forecast_type"] == "iterated"
    assert "forecast_type" not in r.manifest["data_task_spec"]
    assert r.compiled.execution_status == "executable"


def test_forecast_type_default_is_direct_for_raw_panel() -> None:
    r = compile_recipe_dict(_recipe(feature_builder="raw_feature_panel", model_family="ridge"))
    assert r.manifest["training_spec"]["forecast_type"] == "direct"
    assert "forecast_type" not in r.manifest["data_task_spec"]
    assert r.compiled.execution_status == "executable"
    matrix = r.manifest["layer3_capability_matrix"]
    assert matrix["schema_version"] == "layer3_capability_matrix_v1"
    assert matrix["active_cell"] == {
        "model_family": "ridge",
        "feature_builder": "raw_feature_panel",
        "feature_runtime": "raw_feature_panel",
        "forecast_type": "direct",
        "forecast_object": "point_mean",
        "runtime_status": "operational",
        "blocked_reasons": [],
    }


def test_layer3_capability_matrix_records_model_runtime_block() -> None:
    r = compile_recipe_dict(_recipe(feature_builder="raw_feature_panel", model_family="ar"))
    assert r.compiled.execution_status == "blocked_by_incompatibility"
    cell = r.manifest["layer3_capability_matrix"]["active_cell"]
    assert cell["model_family"] == "ar"
    assert cell["feature_runtime"] == "raw_feature_panel"
    assert cell["runtime_status"] == "blocked_by_incompatibility"
    assert cell["blocked_reasons"] == r.manifest["blocked_reasons"]


def test_layer3_capability_matrix_records_future_status_catalog() -> None:
    r = compile_recipe_dict(_recipe(feature_builder="raw_feature_panel", model_family="ridge"))
    matrix = r.manifest["layer3_capability_matrix"]

    assert matrix["schema_version"] == "layer3_capability_matrix_v1"
    assert matrix["schema_revision"] == 3
    assert "not_supported_yet" in matrix["status_catalog"]
    future_cells = {cell["cell_id"]: cell for cell in matrix["future_cells"]}
    assert future_cells["forecast_object.interval"]["runtime_status"] == "not_supported_yet"
    assert future_cells["forecast_object.interval"]["payload_contract"] == "interval_forecast_payload_v1"
    assert future_cells["forecast_object.density"]["dimension"] == "forecast_object"
    assert future_cells["forecast_object.density"]["payload_contract"] == "density_forecast_payload_v1"
    assert future_cells["feature_runtime.sequence_tensor"]["owner_layer"] == "2_preprocessing"
    assert future_cells["feature_runtime.sequence_tensor"]["upstream_contract"] == "sequence_representation_contract_v1"
    assert future_cells["forecast_type.raw_panel_iterated"]["scenario_contract"] == "exogenous_x_path_contract_v1"
    assert matrix["rules"]["forecast_object"]["interval"]["runtime_status"] == "not_supported_yet"


def test_forecast_type_iterated_autoreg_executes(tmp_path: Path) -> None:
    recipe = _recipe(forecast_type="iterated")
    r = compile_recipe_dict(recipe)
    assert r.compiled.execution_status == "executable"
    execution = run_compiled_recipe(
        r.compiled,
        output_root=tmp_path,
        local_raw_source=Path("tests/fixtures/fred_md_ar_sample.csv"),
    )
    manifest = json.loads((Path(execution.artifact_dir) / "manifest.json").read_text())
    assert manifest["training_spec"]["forecast_type"] == "iterated"
    assert "forecast_type" not in manifest["data_task_spec"]


def test_forecast_type_iterated_raw_panel_blocked() -> None:
    r = compile_recipe_dict(_recipe(
        feature_builder="raw_feature_panel",
        model_family="ridge",
        forecast_type="iterated",
    ))
    assert r.compiled.execution_status == "blocked_by_incompatibility"
    assert any(
        "forecast_type='iterated' is not implemented for the raw-panel feature runtime" in r_msg
        for r_msg in r.manifest.get("blocked_reasons", [])
    )
    assert r.manifest["layer3_capability_matrix"]["active_cell"]["runtime_status"] == "blocked_by_incompatibility"
    assert r.manifest["layer3_capability_matrix"]["active_cell"]["blocked_reasons"] == r.manifest["blocked_reasons"]


def test_forecast_type_direct_autoreg_blocked() -> None:
    r = compile_recipe_dict(_recipe(forecast_type="direct"))
    assert r.compiled.execution_status == "blocked_by_incompatibility"
    assert any(
        "forecast_type='direct' is not implemented for the target-lag-only feature runtime" in r_msg
        for r_msg in r.manifest.get("blocked_reasons", [])
    )


def test_forecast_type_direct_raw_panel_executes() -> None:
    r = compile_recipe_dict(_recipe(
        feature_builder="raw_feature_panel",
        model_family="ridge",
        forecast_type="direct",
    ))
    assert r.compiled.execution_status == "executable"
    assert r.manifest["training_spec"]["forecast_type"] == "direct"
    assert "forecast_type" not in r.manifest["data_task_spec"]


def test_forecast_object_quantile_with_quantile_linear_executes(tmp_path: Path) -> None:
    recipe = _recipe(
        model_family="quantile_linear",
        forecast_object="quantile",
    )
    r = compile_recipe_dict(recipe)
    assert r.compiled.execution_status == "executable", (
        f"expected executable, got {r.compiled.execution_status}; "
        f"blocked={r.manifest.get('blocked_reasons', [])}"
    )
    execution = run_compiled_recipe(
        r.compiled,
        output_root=tmp_path,
        local_raw_source=Path("tests/fixtures/fred_md_ar_sample.csv"),
    )
    manifest = json.loads((Path(execution.artifact_dir) / "manifest.json").read_text())
    assert manifest["training_spec"]["forecast_object"] == "quantile"
    assert "forecast_object" not in manifest["data_task_spec"]


def test_forecast_object_point_median_still_allowed_with_quantile_linear() -> None:
    """Backward compat: the pre-existing recipe form stays executable."""
    r = compile_recipe_dict(_recipe(
        model_family="quantile_linear",
        forecast_object="point_median",
    ))
    assert r.compiled.execution_status == "executable"


def test_forecast_object_point_mean_rejected_with_quantile_linear() -> None:
    r = compile_recipe_dict(_recipe(
        model_family="quantile_linear",
        forecast_object="point_mean",
    ))
    assert r.compiled.execution_status == "blocked_by_incompatibility"
    assert any(
        "quantile_linear" in r_msg and "point_median" in r_msg
        for r_msg in r.manifest.get("blocked_reasons", [])
    )


@pytest.mark.parametrize("forecast_object", ["point_median", "quantile"])
def test_forecast_object_distributional_values_require_quantile_linear(forecast_object: str) -> None:
    r = compile_recipe_dict(_recipe(
        model_family="ar",
        forecast_object=forecast_object,
    ))
    assert r.compiled.execution_status == "blocked_by_incompatibility"
    assert any(
        forecast_object in r_msg and "quantile_linear" in r_msg
        for r_msg in r.manifest.get("blocked_reasons", [])
    )
