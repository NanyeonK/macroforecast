"""End-to-end tests for forecast_type (1.2.2) + forecast_object=quantile (1.2.3).

v1.0 semantics:
- `forecast_type` default is feature-builder dynamic:
  target_lag_features -> "iterated" (matches the existing recursive path),
  raw_feature_panel     -> "direct"   (matches the existing h-step path).
- `forecast_type=iterated` + feature_builder=target_lag_features   : executable.
- `forecast_type=iterated` + feature_builder=raw_feature_panel       : executable only for the
  explicit hold-last-observed narrow slice; otherwise blocked_by_incompatibility.
- `forecast_type=direct`   + feature_builder=target_lag_features   : blocked_by_incompatibility.
- `forecast_type=direct`   + feature_builder=raw_feature_panel       : executable.

- `forecast_object=quantile` + `model_family=quantile_linear`         : executable (quantile level via training_spec.hp.quantile, default 0.5).
- `forecast_object in {point_median, quantile}` + any other model_family: blocked_by_incompatibility.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from macrocast import clear_custom_extensions, custom_model
from macrocast.compiler.build import compile_recipe_dict, run_compiled_recipe


def _recipe(
    *,
    feature_builder: str = "target_lag_features",
    model_family: str = "ar",
    forecast_type: str | None = None,
    forecast_object: str | None = None,
    target_lag_block: str | None = None,
    exogenous_x_path_policy: str | None = None,
    scheduled_known_future_x_columns: list[str] | None = None,
    recursive_x_model_family: str | None = None,
) -> dict:
    axes_1 = {
        "dataset": "fred_md",
        "information_set_type": "final_revised_data",
        "target_structure": "single_target",
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

    layer2_axes = {
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
    if target_lag_block is not None:
        layer2_axes["target_lag_block"] = target_lag_block

    data_leaf = {"target": "INDPRO", "horizons": [1]}
    if exogenous_x_path_policy is not None:
        data_leaf["exogenous_x_path_policy"] = exogenous_x_path_policy
    if scheduled_known_future_x_columns is not None:
        data_leaf["scheduled_known_future_x_columns"] = scheduled_known_future_x_columns
    if recursive_x_model_family is not None:
        data_leaf["recursive_x_model_family"] = recursive_x_model_family

    return {
        "recipe_id": "ft-q-test",
        "path": {
            "0_meta": {"fixed_axes": {"study_scope": "one_target_one_method"}},
            "1_data_task": {
                "fixed_axes": axes_1,
                "leaf_config": data_leaf,
            },
            "2_preprocessing": {"fixed_axes": layer2_axes},
            "3_training": {"fixed_axes": training},
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {
                "manifest_mode": "full",
                "benchmark_config": {"minimum_train_size": 5, "rolling_window_size": 5},
            }},
            "6_stat_tests": {"fixed_axes": {}},
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
        "payload_contract": "forecast_payload_v1",
        "runtime_status": "operational",
        "blocked_reasons": [],
    }
    assert matrix["canonical_active_cell"] == {
        "forecast_generator_family": "ridge",
        "representation_runtime": "raw_feature_panel",
        "forecast_protocol": "direct",
        "forecast_object": "point_mean",
        "payload_contract": "forecast_payload_v1",
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
    assert matrix["schema_revision"] == 6
    assert matrix["canonical_dimensions"] == [
        "forecast_generator_family",
        "representation_runtime",
        "forecast_protocol",
        "forecast_object",
    ]
    assert matrix["dimension_aliases"]["model_family"] == "forecast_generator_family"
    assert matrix["dimension_aliases"]["benchmark_family"] == "baseline_forecast_generator_role"
    assert matrix["dimension_aliases"]["feature_runtime"] == "representation_runtime"
    assert matrix["dimension_aliases"]["forecast_type"] == "forecast_protocol"
    assert "not_supported_yet" in matrix["status_catalog"]
    future_cells = {cell["cell_id"]: cell for cell in matrix["future_cells"]}
    assert "forecast_object.interval" not in future_cells
    assert matrix["rules"]["forecast_object"]["direction"]["payload_contract"] == "direction_forecast_payload_v1"
    assert matrix["rules"]["forecast_object"]["interval"]["runtime_status"] == "operational"
    assert matrix["rules"]["forecast_object"]["interval"]["payload_contract"] == "interval_forecast_payload_v1"
    assert matrix["rules"]["forecast_object"]["density"]["payload_contract"] == "density_forecast_payload_v1"
    assert future_cells["feature_runtime.sequence_tensor"]["owner_layer"] == "2_preprocessing"
    assert future_cells["feature_runtime.sequence_tensor"]["upstream_contract"] == "sequence_representation_contract_v1"
    assert future_cells["feature_runtime.sequence_tensor"]["required_contracts"] == [
        "sequence_representation_contract_v1",
        "sequence_forecast_payload_v1",
    ]
    sequence_requirements = future_cells["feature_runtime.sequence_tensor"]["contract_requirements"]
    assert "channel_names" in sequence_requirements["sequence_representation_contract_v1"]["required_fields"]
    assert "path_or_vector_payload" in sequence_requirements["sequence_forecast_payload_v1"]["required_fields"]
    assert future_cells["forecast_type.raw_panel_iterated"]["scenario_contract"] == "exogenous_x_path_contract_v1"
    assert future_cells["forecast_type.raw_panel_iterated"]["runtime_status"] == "operational_narrow"
    assert future_cells["forecast_type.raw_panel_iterated"]["required_contracts"] == [
        "exogenous_x_path_contract_v1",
        "multi_step_raw_panel_payload_v1",
    ]
    raw_iterated_requirements = future_cells["forecast_type.raw_panel_iterated"]["contract_requirements"]
    assert raw_iterated_requirements["exogenous_x_path_contract_v1"]["path_kinds"] == [
        "observed_future_x",
        "scheduled_known_future_x",
        "hold_last_observed",
        "recursive_x_model",
        "unavailable",
    ]
    assert "step_predictions" in raw_iterated_requirements["multi_step_raw_panel_payload_v1"]["required_fields"]
    assert "hold_last_observed, observed_future_x, scheduled_known_future_x, and recursive_x_model(ar1) are operational" in future_cells[
        "forecast_type.raw_panel_iterated"
    ]["opening_rule"]
    assert matrix["rules"]["forecast_type"]["raw_feature_panel"]["conditional_operational"]["iterated"][
        "runtime_contract"
    ] == "raw_panel_iterated_future_x_path_v1"


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
        "requires leaf_config.exogenous_x_path_policy in" in r_msg
        for r_msg in r.manifest.get("blocked_reasons", [])
    )
    assert r.manifest["layer3_capability_matrix"]["active_cell"]["runtime_status"] == "blocked_by_incompatibility"
    assert r.manifest["layer3_capability_matrix"]["active_cell"]["blocked_reasons"] == r.manifest["blocked_reasons"]


def test_forecast_type_iterated_raw_panel_hold_last_observed_compiles() -> None:
    r = compile_recipe_dict(
        _recipe(
            feature_builder="raw_feature_panel",
            model_family="ridge",
            forecast_type="iterated",
            target_lag_block="fixed_target_lags",
            exogenous_x_path_policy="hold_last_observed",
        )
    )
    assert r.compiled.execution_status == "executable"
    assert r.manifest["data_task_spec"]["exogenous_x_path_policy"] == "hold_last_observed"
    assert r.manifest["training_spec"]["forecast_type"] == "iterated"
    assert r.manifest["layer3_capability_matrix"]["active_cell"]["runtime_status"] == "operational"


def test_forecast_type_iterated_raw_panel_observed_future_x_compiles() -> None:
    r = compile_recipe_dict(
        _recipe(
            feature_builder="raw_feature_panel",
            model_family="ridge",
            forecast_type="iterated",
            target_lag_block="fixed_target_lags",
            exogenous_x_path_policy="observed_future_x",
        )
    )
    assert r.compiled.execution_status == "executable"
    assert r.manifest["data_task_spec"]["exogenous_x_path_policy"] == "observed_future_x"
    assert r.manifest["training_spec"]["forecast_type"] == "iterated"
    assert r.manifest["layer3_capability_matrix"]["active_cell"]["runtime_status"] == "operational"


def test_forecast_type_iterated_raw_panel_scheduled_known_future_x_compiles() -> None:
    r = compile_recipe_dict(
        _recipe(
            feature_builder="raw_feature_panel",
            model_family="ridge",
            forecast_type="iterated",
            target_lag_block="fixed_target_lags",
            exogenous_x_path_policy="scheduled_known_future_x",
            scheduled_known_future_x_columns=["CPIAUCSL"],
        )
    )
    assert r.compiled.execution_status == "executable"
    assert r.manifest["data_task_spec"]["exogenous_x_path_policy"] == "scheduled_known_future_x"
    assert r.manifest["data_task_spec"]["scheduled_known_future_x_columns"] == ["CPIAUCSL"]
    assert r.manifest["layer3_capability_matrix"]["active_cell"]["runtime_status"] == "operational"


def test_forecast_type_iterated_raw_panel_scheduled_known_future_x_requires_columns() -> None:
    r = compile_recipe_dict(
        _recipe(
            feature_builder="raw_feature_panel",
            model_family="ridge",
            forecast_type="iterated",
            target_lag_block="fixed_target_lags",
            exogenous_x_path_policy="scheduled_known_future_x",
        )
    )
    assert r.compiled.execution_status == "blocked_by_incompatibility"
    assert any(
        "requires leaf_config.scheduled_known_future_x_columns" in r_msg
        for r_msg in r.manifest.get("blocked_reasons", [])
    )


def test_forecast_type_iterated_raw_panel_recursive_x_model_compiles() -> None:
    r = compile_recipe_dict(
        _recipe(
            feature_builder="raw_feature_panel",
            model_family="ridge",
            forecast_type="iterated",
            target_lag_block="fixed_target_lags",
            exogenous_x_path_policy="recursive_x_model",
            recursive_x_model_family="ar1",
        )
    )
    assert r.compiled.execution_status == "executable"
    assert r.manifest["data_task_spec"]["exogenous_x_path_policy"] == "recursive_x_model"
    assert r.manifest["data_task_spec"]["recursive_x_model_family"] == "ar1"
    assert r.manifest["layer3_capability_matrix"]["active_cell"]["runtime_status"] == "operational"


def test_forecast_type_iterated_raw_panel_recursive_x_model_requires_ar1_family() -> None:
    r = compile_recipe_dict(
        _recipe(
            feature_builder="raw_feature_panel",
            model_family="ridge",
            forecast_type="iterated",
            target_lag_block="fixed_target_lags",
            exogenous_x_path_policy="recursive_x_model",
        )
    )
    assert r.compiled.execution_status == "blocked_by_incompatibility"
    assert any(
        "requires leaf_config.recursive_x_model_family='ar1'" in r_msg
        for r_msg in r.manifest.get("blocked_reasons", [])
    )


def test_forecast_type_iterated_raw_panel_custom_model_compiles() -> None:
    clear_custom_extensions()

    @custom_model("iterated_custom_model")
    def _iterated_custom_model(X_train, y_train, X_test, context):
        return float(y_train[-1])

    try:
        r = compile_recipe_dict(
            _recipe(
                feature_builder="raw_feature_panel",
                model_family="iterated_custom_model",
                forecast_type="iterated",
                target_lag_block="fixed_target_lags",
                exogenous_x_path_policy="hold_last_observed",
            )
        )
        assert r.compiled.execution_status == "executable"
        assert r.manifest["model_spec"]["model_family"] == "iterated_custom_model"
        assert r.manifest["layer3_capability_matrix"]["active_cell"]["runtime_status"] == "operational"
    finally:
        clear_custom_extensions()


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


@pytest.mark.parametrize(
    ("forecast_object", "contract", "payload_column"),
    [
        ("direction", "direction_forecast_payload_v1", "direction_hit"),
        ("interval", "interval_forecast_payload_v1", "interval_covered"),
        ("density", "density_forecast_payload_v1", "density_log_score"),
    ],
)
def test_forecast_payload_family_executes(
    tmp_path: Path,
    forecast_object: str,
    contract: str,
    payload_column: str,
) -> None:
    recipe = _recipe(
        model_family="ridge",
        feature_builder="raw_feature_panel",
        forecast_object=forecast_object,
    )
    r = compile_recipe_dict(recipe)
    assert r.compiled.execution_status == "executable"
    assert r.manifest["layer3_capability_matrix"]["active_cell"]["payload_contract"] == contract
    execution = run_compiled_recipe(
        r.compiled,
        output_root=tmp_path,
        local_raw_source=Path("tests/fixtures/fred_md_ar_sample.csv"),
    )
    artifact_dir = Path(execution.artifact_dir)
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    predictions = __import__("pandas").read_csv(artifact_dir / "predictions.csv")
    metrics = json.loads((artifact_dir / "metrics.json").read_text())

    assert manifest["forecast_object"] == forecast_object
    assert manifest["forecast_payload_contract"] == contract
    assert manifest["forecast_payloads_file"] == "forecast_payloads.jsonl"
    assert payload_column in predictions.columns
    assert predictions["forecast_payload_contract"].eq(contract).all()
    assert metrics["metrics_by_horizon"]["h1"]["payload_metrics"]["payload_family"] == forecast_object


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
