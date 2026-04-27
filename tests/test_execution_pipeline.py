from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from macrocast import (
    build_execution_spec,
    build_preprocess_contract,
    build_recipe_spec,
    build_run_spec,
    build_design_frame,
    clear_custom_extensions,
    custom_feature_block,
    custom_feature_combiner,
    custom_model,
    execute_recipe,
    ExecutionError,
    ForecastPayload,
    LAYER1_OFFICIAL_FRAME_CONTRACT_VERSION,
    LAYER2_REPRESENTATION_CONTRACT_VERSION,
    Layer2Representation,
    PREDICTION_ROW_SCHEMA_VERSION,
)
from macrocast.preprocessing import FeatureBlockCallableResult, FeatureCombinerCallableResult
import macrocast.execution.build as execution_build
from macrocast.execution.build import _coerce_forecast_payload


def _stage0(
    model_family: str = "ridge",
    feature_builder: str = "raw_feature_panel",
    benchmark: str = "zero_change",
    framework: str = "rolling",
    info_set: str = "revised_monthly",
):
    sample_split = {
        "expanding": "expanding_window_oos",
        "rolling": "rolling_window_oos",
    }[framework]
    return build_design_frame(
        research_design="single_forecast_run",
        fixed_design={
            "dataset_adapter": "fred_md",
            "information_set": info_set,
            "sample_split": sample_split,
            "benchmark": benchmark,
            "evaluation_protocol": "point_forecast_core",
            "forecast_task": "single_target",
        },
        comparison_contract={
            "information_set_policy": "identical",
            "sample_split_policy": "identical",
            "benchmark_policy": "identical",
            "evaluation_policy": "identical",
        },
        varying_design={"model_families": (model_family,), "feature_recipes": (feature_builder,), "horizons": ("h1",)},
    )


def _recipe(
    model_family: str = "ridge",
    feature_builder: str = "raw_feature_panel",
    benchmark: str = "zero_change",
    framework: str = "rolling",
    benchmark_config: dict | None = None,
    info_set: str = "revised_monthly",
    data_vintage: str | None = None,
    forecast_object: str = "point_mean",
    forecast_type: str | None = None,
    exogenous_x_path_policy: str | None = None,
    scheduled_known_future_x_columns: list[str] | None = None,
    recursive_x_model_family: str | None = None,
    layer2_representation_spec: dict | None = None,
    quantile_level: float | None = None,
):
    data_task_spec = {"forecast_object": forecast_object}
    training_spec = {}
    if forecast_type is not None:
        training_spec["forecast_type"] = forecast_type
    if exogenous_x_path_policy is not None:
        data_task_spec["exogenous_x_path_policy"] = exogenous_x_path_policy
    if scheduled_known_future_x_columns is not None:
        data_task_spec["scheduled_known_future_x_columns"] = scheduled_known_future_x_columns
    if recursive_x_model_family is not None:
        data_task_spec["recursive_x_model_family"] = recursive_x_model_family
    if quantile_level is not None:
        data_task_spec["quantile_level"] = quantile_level
        training_spec["quantile_level"] = quantile_level
    return build_recipe_spec(
        recipe_id=f"fred_md_{framework}_{model_family}_{feature_builder}",
        stage0=_stage0(model_family=model_family, feature_builder=feature_builder, benchmark=benchmark, framework=framework, info_set=info_set),
        target="INDPRO",
        horizons=(1, 3),
        raw_dataset="fred_md",
        benchmark_config=benchmark_config or {},
        data_task_spec=data_task_spec,
        training_spec=training_spec,
        layer2_representation_spec=layer2_representation_spec or {},
        data_vintage=data_vintage,
    )


def _preprocess_raw_only():
    return build_preprocess_contract(
        target_transform_policy="raw_level",
        x_transform_policy="raw_level",
        tcode_policy="raw_only",
        target_missing_policy="none",
        x_missing_policy="none",
        target_outlier_policy="none",
        x_outlier_policy="none",
        scaling_policy="none",
        dimensionality_reduction_policy="none",
        feature_selection_policy="none",
        preprocess_order="none",
        preprocess_fit_scope="not_applicable",
        inverse_transform_policy="none",
        evaluation_scale="raw_level",
    )


def _preprocess_train_only_robust():
    return build_preprocess_contract(
        target_transform_policy="raw_level",
        x_transform_policy="raw_level",
        tcode_policy="extra_preprocess_only",
        target_missing_policy="none",
        x_missing_policy="em_impute",
        target_outlier_policy="none",
        x_outlier_policy="none",
        scaling_policy="robust",
        dimensionality_reduction_policy="none",
        feature_selection_policy="none",
        preprocess_order="extra_only",
        preprocess_fit_scope="train_only",
        inverse_transform_policy="none",
        evaluation_scale="raw_level",
    )


def _preprocess_target_zscore_both_scales():
    return build_preprocess_contract(
        target_transform_policy="raw_level",
        x_transform_policy="raw_level",
        tcode_policy="extra_preprocess_only",
        target_missing_policy="none",
        x_missing_policy="mean_impute",
        target_outlier_policy="none",
        x_outlier_policy="none",
        scaling_policy="standard",
        dimensionality_reduction_policy="none",
        feature_selection_policy="none",
        preprocess_order="extra_only",
        preprocess_fit_scope="train_only",
        inverse_transform_policy="target_only",
        evaluation_scale="both",
        target_normalization="zscore_train_only",
    )


def test_build_execution_spec_with_importance_context() -> None:
    recipe = _recipe()
    run = build_run_spec(recipe)
    preprocess = _preprocess_raw_only()

    execution = build_execution_spec(recipe=recipe, run=run, preprocess=preprocess)

    assert execution.recipe.recipe_id == "fred_md_rolling_ridge_raw_feature_panel"
    assert execution.run.run_id == run.run_id
    assert execution.recipe.stage0.fixed_design.sample_split == "rolling_window_oos"


def test_execute_recipe_records_layer2_representation_contract(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    recipe = _recipe(benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5})

    result = execute_recipe(
        recipe=recipe,
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=fixture,
    )

    run_dir = tmp_path / result.run.artifact_subdir
    manifest = json.loads((run_dir / "manifest.json").read_text())
    tuning_result = json.loads((run_dir / "tuning_result.json").read_text())

    assert manifest["layer2_representation_contract"] == LAYER2_REPRESENTATION_CONTRACT_VERSION
    metadata = manifest["layer2_representation_contract_metadata"]
    assert metadata["schema_version"] == LAYER2_REPRESENTATION_CONTRACT_VERSION
    assert metadata["feature_runtime_builder"] == "raw_feature_panel"
    assert metadata["matrix_shapes"]["Z_train"][1] == metadata["feature_count"]
    assert tuning_result["layer2_representation_contract"] == LAYER2_REPRESENTATION_CONTRACT_VERSION


def test_execute_recipe_records_layer1_official_frame_contract(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    recipe = _recipe(benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5})

    result = execute_recipe(
        recipe=recipe,
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=fixture,
    )

    run_dir = tmp_path / result.run.artifact_subdir
    manifest = json.loads((run_dir / "manifest.json").read_text())
    contract = json.loads((run_dir / "layer1_official_frame.json").read_text())
    artifact_manifest = json.loads((run_dir / "artifact_manifest.json").read_text())

    assert manifest["layer1_official_frame_contract"] == LAYER1_OFFICIAL_FRAME_CONTRACT_VERSION
    assert manifest["layer1_official_frame_file"] == "layer1_official_frame.json"
    assert manifest["layer1_official_frame_summary"]["schema_version"] == LAYER1_OFFICIAL_FRAME_CONTRACT_VERSION
    assert contract["schema_version"] == LAYER1_OFFICIAL_FRAME_CONTRACT_VERSION
    assert contract["owner_layer"] == "1_data_task"
    assert contract["consumer_layer"] == "2_preprocessing"
    assert contract["raw_dataset"] == "fred_md"
    assert contract["dataset_adapter"] == "fred_md"
    assert contract["target"] == "INDPRO"
    assert contract["targets"] == ["INDPRO"]
    assert contract["horizons"] == [1, 3]
    assert contract["target_columns_available"] == ["INDPRO"]
    assert "INDPRO" not in contract["predictor_columns"]
    assert contract["frame_shape"][0] > 0
    assert contract["frame_shape"][1] == contract["column_count"] == len(contract["columns"])
    assert contract["official_transform_policy"] == "keep_official_raw_scale"
    assert contract["raw_missing_policy"] == "preserve_raw_missing"
    assert contract["raw_outlier_policy"] == "preserve_raw_outliers"
    assert contract["missing_availability"] == "require_complete_rows"
    assert contract["release_lag_rule"] == "ignore_release_lag"
    assert contract["variable_universe"] == "all_variables"
    assert contract["dataset_metadata"]["dataset"] == "fred_md"
    assert contract["raw_artifact"]["local_path"]
    source_contract = contract["source_availability_contract"]
    assert source_contract["contract_version"] == "source_availability_contract_v1"
    assert source_contract["raw_dataset"] == "fred_md"
    assert source_contract["dataset"] == "fred_md"
    assert source_contract["source_adapter"] == "fred_md"
    assert source_contract["version_mode"] == "current"
    assert source_contract["vintage"] is None
    assert source_contract["data_vintage_requested"] is None
    assert source_contract["data_through"] == contract["data_through"]
    assert source_contract["observed_data_window"] == {
        "index_start": contract["index_start"],
        "index_end": contract["index_end"],
        "data_through": contract["data_through"],
    }
    assert source_contract["source_url_kind"] == "local_path"
    assert source_contract["uses_local_source"] is True
    assert source_contract["uses_remote_source"] is False
    assert source_contract["artifact_file_sha256"] == contract["raw_artifact"]["file_sha256"]
    assert source_contract["artifact_file_size_bytes"] == contract["raw_artifact"]["file_size_bytes"]
    assert source_contract["component_count"] == 1
    assert source_contract["component_source_contracts"] == []
    assert manifest["layer1_official_frame_summary"]["source_availability_contract"] == "source_availability_contract_v1"
    assert manifest["layer1_official_frame_summary"]["source_url_kind"] == "local_path"
    assert "layer1_official_frame.json" in {
        item["path"] for item in artifact_manifest["artifacts"] if item["artifact_type"] == "layer1_official_frame"
    }


def test_layer1_official_frame_contract_records_local_vintage_evidence(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    recipe = _recipe(
        benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5},
        data_vintage="2020-01",
    )

    result = execute_recipe(
        recipe=recipe,
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=fixture,
    )

    run_dir = tmp_path / result.run.artifact_subdir
    contract = json.loads((run_dir / "layer1_official_frame.json").read_text())
    manifest = json.loads((run_dir / "manifest.json").read_text())

    assert contract["schema_version"] == LAYER1_OFFICIAL_FRAME_CONTRACT_VERSION
    assert contract["version_mode"] == "vintage"
    assert contract["vintage"] == "2020-01"
    assert contract["dataset_metadata"]["version_mode"] == "vintage"
    assert contract["dataset_metadata"]["vintage"] == "2020-01"
    assert contract["raw_artifact"]["version_mode"] == "vintage"
    assert contract["raw_artifact"]["vintage"] == "2020-01"
    assert contract["raw_artifact"]["file_sha256"]
    assert contract["raw_artifact"]["file_size_bytes"] > 0
    assert contract["information_set_contract"] == {
        "version_mode": "vintage",
        "vintage": "2020-01",
        "data_through": contract["data_through"],
        "information_set": "revised_monthly",
        "release_lag_rule": "ignore_release_lag",
        "missing_availability": "require_complete_rows",
        "data_vintage_requested": "2020-01",
        "uses_vintage_source": True,
        "raw_artifact_sha256": contract["raw_artifact"]["file_sha256"],
        "source_availability_contract": "source_availability_contract_v1",
    }
    source_contract = contract["source_availability_contract"]
    assert source_contract["contract_version"] == "source_availability_contract_v1"
    assert source_contract["version_mode"] == "vintage"
    assert source_contract["vintage"] == "2020-01"
    assert source_contract["data_vintage_requested"] == "2020-01"
    assert source_contract["uses_local_source"] is True
    assert source_contract["source_url_kind"] == "local_path"
    assert source_contract["artifact_file_sha256"] == contract["raw_artifact"]["file_sha256"]
    assert source_contract["observed_data_window"]["index_end"] == contract["index_end"]
    coverage = contract["transform_code_coverage"]
    assert coverage["data_column_count"] == contract["column_count"]
    assert coverage["transform_code_column_count"] == 4
    assert coverage["covered_column_count"] == 4
    assert coverage["missing_transform_code_columns"] == []
    assert coverage["coverage_ratio"] == 1.0
    assert manifest["layer1_official_frame_summary"]["index_end"] == contract["index_end"]


def test_layer1_official_frame_contract_records_cached_remote_source_simulation(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    cache_root = tmp_path / "raw_cache"
    recipe = _recipe(benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5})

    seed_result = execute_recipe(
        recipe=recipe,
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path / "seed",
        local_raw_source=fixture,
        cache_root=cache_root,
    )
    cached_result = execute_recipe(
        recipe=recipe,
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path / "cached",
        cache_root=cache_root,
    )

    seed_contract = json.loads(
        ((tmp_path / "seed") / seed_result.run.artifact_subdir / "layer1_official_frame.json").read_text()
    )
    cached_run_dir = (tmp_path / "cached") / cached_result.run.artifact_subdir
    cached_contract = json.loads((cached_run_dir / "layer1_official_frame.json").read_text())
    cached_manifest = json.loads((cached_run_dir / "manifest.json").read_text())
    source_contract = cached_contract["source_availability_contract"]

    assert seed_contract["source_availability_contract"]["source_url_kind"] == "local_path"
    assert seed_contract["source_availability_contract"]["artifact_cache_hit"] is False
    assert source_contract["contract_version"] == "source_availability_contract_v1"
    assert source_contract["source_url_kind"] == "remote_url"
    assert source_contract["uses_local_source"] is False
    assert source_contract["uses_remote_source"] is True
    assert source_contract["artifact_cache_hit"] is True
    assert source_contract["version_mode"] == "current"
    assert source_contract["data_vintage_requested"] is None
    assert source_contract["artifact_file_sha256"] == seed_contract["raw_artifact"]["file_sha256"]
    assert source_contract["observed_data_window"]["index_end"] == cached_contract["index_end"]
    assert cached_manifest["layer1_official_frame_summary"]["source_url_kind"] == "remote_url"
    assert cached_manifest["layer1_official_frame_summary"]["artifact_cache_hit"] is True


def test_layer1_official_frame_contract_records_cached_vintage_remote_source_simulation(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    cache_root = tmp_path / "raw_cache"
    recipe = _recipe(
        benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5},
        data_vintage="2020-01",
    )

    execute_recipe(
        recipe=recipe,
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path / "seed",
        local_raw_source=fixture,
        cache_root=cache_root,
    )
    cached_result = execute_recipe(
        recipe=recipe,
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path / "cached",
        cache_root=cache_root,
    )

    cached_run_dir = (tmp_path / "cached") / cached_result.run.artifact_subdir
    cached_contract = json.loads((cached_run_dir / "layer1_official_frame.json").read_text())
    source_contract = cached_contract["source_availability_contract"]

    assert source_contract["source_url_kind"] == "remote_url"
    assert source_contract["uses_remote_source"] is True
    assert source_contract["artifact_cache_hit"] is True
    assert source_contract["version_mode"] == "vintage"
    assert source_contract["vintage"] == "2020-01"
    assert source_contract["data_vintage_requested"] == "2020-01"
    assert source_contract["data_through"] == cached_contract["data_through"]


def test_layer1_official_frame_contract_records_release_lag_report(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    recipe = _recipe(benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5})
    recipe = replace(
        recipe,
        data_task_spec={
            **recipe.data_task_spec,
            "release_lag_rule": "fixed_lag_all_series",
        },
    )

    result = execute_recipe(
        recipe=recipe,
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=fixture,
    )

    run_dir = tmp_path / result.run.artifact_subdir
    contract = json.loads((run_dir / "layer1_official_frame.json").read_text())
    report = contract["data_reports"]["release_lag"]

    assert contract["release_lag_rule"] == "fixed_lag_all_series"
    assert contract["information_set_contract"]["release_lag_rule"] == "fixed_lag_all_series"
    assert report["rule"] == "fixed_lag_all_series"
    assert report["lag_unit"] == "periods"
    assert report["default_lag"] == 1
    assert report["columns_shifted"] == contract["columns"]
    assert report["lag_by_column"] == {name: 1 for name in contract["columns"]}
    assert report["max_lag"] == 1


def test_execute_recipe_records_prediction_row_schema_contract(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    recipe = _recipe(benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5})

    result = execute_recipe(
        recipe=recipe,
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=fixture,
    )

    run_dir = tmp_path / result.run.artifact_subdir
    manifest = json.loads((run_dir / "manifest.json").read_text())
    schema = json.loads((run_dir / "prediction_row_schema.json").read_text())
    artifact_manifest = json.loads((run_dir / "artifact_manifest.json").read_text())
    predictions = __import__("pandas").read_csv(run_dir / "predictions.csv")

    assert manifest["prediction_row_schema_contract"] == PREDICTION_ROW_SCHEMA_VERSION
    assert manifest["prediction_row_schema_file"] == "prediction_row_schema.json"
    assert schema["schema_version"] == PREDICTION_ROW_SCHEMA_VERSION
    assert schema["row_count"] == manifest["prediction_rows"] == len(predictions)
    assert schema["observed_columns"] == list(predictions.columns)
    assert set(schema["required_columns"]).issubset(schema["observed_columns"])
    assert schema["missing_required_columns"] == []
    assert schema["payload_families"] == ["point"]
    assert "prediction_row_schema.json" in {
        item["path"] for item in artifact_manifest["artifacts"] if item["artifact_type"] == "prediction_row_schema"
    }


def test_forecast_payload_contract_coerces_executor_mapping() -> None:
    payload = _coerce_forecast_payload(
        {
            "y_pred": "1.25",
            "selected_lag": "2",
            "selected_bic": "3.5",
            "tuning_payload": {"source": "unit"},
        },
        executor_name="unit_executor",
    )

    assert isinstance(payload, ForecastPayload)
    assert payload.y_pred == 1.25
    assert payload.selected_lag == 2
    assert payload.selected_bic == 3.5
    assert payload.tuning_payload["source"] == "unit"
    assert payload.tuning_payload["forecast_payload_contract"] == "forecast_payload_v1"
    assert payload.to_dict()["contract_version"] == "forecast_payload_v1"


def test_layer2_representation_public_contract_metadata() -> None:
    representation = Layer2Representation(
        Z_train=np.zeros((3, 2)),
        y_train=np.zeros(3),
        Z_pred=np.zeros((1, 2)),
        feature_names=("x1", "x2"),
        block_order=("base_x",),
        block_roles={"x1": "base_x", "x2": "base_x"},
        alignment={"representation_runtime": "raw_feature_panel"},
        leakage_contract="forecast_origin_only",
        feature_builder="raw_feature_panel",
        feature_runtime_builder="raw_feature_panel",
        legacy_feature_builder="raw_feature_panel",
    )

    metadata = representation.contract_metadata()
    context = representation.runtime_context(mode="fit")

    assert representation.contract_version == LAYER2_REPRESENTATION_CONTRACT_VERSION
    assert metadata["schema_version"] == LAYER2_REPRESENTATION_CONTRACT_VERSION
    assert metadata["matrix_shapes"] == {"Z_train": [3, 2], "y_train": [3], "Z_pred": [1, 2]}
    assert metadata["feature_names"] == ["x1", "x2"]
    assert context["layer2_representation_contract"] == LAYER2_REPRESENTATION_CONTRACT_VERSION
    assert context["mode"] == "fit"


def test_forecast_payload_contract_rejects_missing_fields() -> None:
    with __import__("pytest").raises(ExecutionError, match="missing required fields"):
        _coerce_forecast_payload({"y_pred": 1.0, "selected_lag": 1}, executor_name="bad_executor")


def test_execute_recipe_writes_minimal_importance_artifact_for_ridge(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    recipe = _recipe(benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5})
    preprocess = _preprocess_raw_only()

    result = execute_recipe(
        recipe=recipe,
        preprocess=preprocess,
        output_root=tmp_path,
        local_raw_source=fixture,
        provenance_payload={"compiler": {"importance_spec": {"importance_method": "minimal_importance"}}},
    )

    run_dir = tmp_path / result.run.artifact_subdir
    manifest = json.loads((run_dir / "manifest.json").read_text())
    importance = json.loads((run_dir / "importance_minimal.json").read_text())

    assert manifest["importance_spec"]["importance_method"] == "minimal_importance"
    assert manifest["forecast_payload_contract"] == "forecast_payload_v1"
    assert manifest["importance_file"] == "importance_minimal.json"
    assert importance["importance_method"] == "minimal_importance"
    assert importance["model_family"] == "ridge"
    assert len(importance["feature_importance"]) > 0


def test_execute_recipe_runs_raw_panel_iterated_hold_last_observed(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    recipe = _recipe(
        model_family="ridge",
        feature_builder="raw_feature_panel",
        forecast_type="iterated",
        exogenous_x_path_policy="hold_last_observed",
        benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5},
        layer2_representation_spec={
            "feature_blocks": {
                "feature_block_set": {"value": "mixed_feature_blocks"},
                "target_lag_block": {"value": "fixed_target_lags", "lag_orders": [1, 2]},
                "x_lag_feature_block": {"value": "none"},
                "factor_feature_block": {"value": "none"},
                "level_feature_block": {"value": "none"},
                "rotation_feature_block": {"value": "none"},
                "temporal_feature_block": {"value": "none"},
            }
        },
    )
    result = execute_recipe(
        recipe=recipe,
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=fixture,
    )

    run_dir = tmp_path / result.run.artifact_subdir
    manifest = json.loads((run_dir / "manifest.json").read_text())
    schema = json.loads((run_dir / "prediction_row_schema.json").read_text())
    predictions = __import__("pandas").read_csv(run_dir / "predictions.csv")
    steps = __import__("pandas").read_csv(run_dir / "raw_panel_iterated_steps.csv")
    payloads = (run_dir / "forecast_payloads.jsonl").read_text().strip().splitlines()

    assert manifest["forecast_type"] == "iterated"
    assert manifest["forecast_payload_contract"] == "multi_step_raw_panel_payload_v1"
    assert manifest["forecast_payload_family"] == "raw_panel_iterated"
    assert schema["payload_families"] == ["raw_panel_iterated"]
    assert "raw_panel_iterated" in schema["optional_column_groups"]
    assert manifest["raw_panel_iterated_steps_file"] == "raw_panel_iterated_steps.csv"
    assert manifest["raw_panel_iterated_runtime_contract"] == "raw_panel_iterated_hold_last_observed_v1"
    assert manifest["exogenous_x_path_policy"] == "hold_last_observed"
    assert predictions["payload_family"].eq("raw_panel_iterated").all()
    assert predictions["raw_panel_iterated_payload_contract"].eq("multi_step_raw_panel_payload_v1").all()
    assert predictions["raw_panel_iterated_x_path_policy"].eq("hold_last_observed").all()
    assert predictions["raw_panel_iterated_model_target_scale"].eq("transformed_target_scale").all()
    assert predictions["raw_panel_iterated_forecast_scale"].eq("original_target_scale").all()
    assert predictions["raw_panel_iterated_evaluation_scale"].eq("raw_level").all()
    assert predictions["raw_panel_iterated_target_normalization"].eq("none").all()
    assert predictions["raw_panel_iterated_step_predictions"].map(json.loads).map(len).eq(
        predictions["raw_panel_iterated_step_count"]
    ).all()
    assert (
        predictions["raw_panel_iterated_final_step_prediction"].astype(float)
        == predictions["y_pred_model_scale"].astype(float)
    ).all()
    assert steps["payload_contract"].eq("multi_step_raw_panel_payload_v1").all()
    assert steps["x_path_policy"].eq("hold_last_observed").all()
    assert steps["x_source_date"].eq(steps["origin_date"]).all()
    assert int(steps["step"].max()) == max(recipe.horizons)
    assert payloads
    payload_record = json.loads(payloads[0])
    assert payload_record["raw_panel_iterated_payload_contract"] == "multi_step_raw_panel_payload_v1"
    assert payload_record["raw_panel_iterated_forecast_scale"] == "original_target_scale"


def test_execute_recipe_runs_raw_panel_iterated_observed_future_x(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    recipe = _recipe(
        model_family="ridge",
        feature_builder="raw_feature_panel",
        forecast_type="iterated",
        exogenous_x_path_policy="observed_future_x",
        benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5},
        layer2_representation_spec={
            "feature_blocks": {
                "feature_block_set": {"value": "mixed_feature_blocks"},
                "target_lag_block": {"value": "fixed_target_lags", "lag_orders": [1, 2]},
                "x_lag_feature_block": {"value": "none"},
                "factor_feature_block": {"value": "none"},
                "level_feature_block": {"value": "none"},
                "rotation_feature_block": {"value": "none"},
                "temporal_feature_block": {"value": "none"},
            }
        },
    )
    result = execute_recipe(
        recipe=recipe,
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=fixture,
    )

    run_dir = tmp_path / result.run.artifact_subdir
    manifest = json.loads((run_dir / "manifest.json").read_text())
    predictions = __import__("pandas").read_csv(run_dir / "predictions.csv")
    steps = __import__("pandas").read_csv(run_dir / "raw_panel_iterated_steps.csv")

    assert manifest["forecast_type"] == "iterated"
    assert manifest["raw_panel_iterated_runtime_contract"] == "raw_panel_iterated_observed_future_x_v1"
    assert manifest["exogenous_x_path_policy"] == "observed_future_x"
    assert predictions["raw_panel_iterated_x_path_policy"].eq("observed_future_x").all()
    assert predictions["raw_panel_iterated_runtime"].eq("raw_panel_iterated_observed_future_x_v1").all()
    assert steps["x_path_policy"].eq("observed_future_x").all()
    assert steps.loc[steps["step"] == 1, "x_source_date"].eq(steps.loc[steps["step"] == 1, "origin_date"]).all()
    assert steps.loc[steps["step"] > 1, "x_source_date"].ne(steps.loc[steps["step"] > 1, "origin_date"]).any()
    assert int(steps["step"].max()) == max(recipe.horizons)


def test_execute_recipe_runs_raw_panel_iterated_scheduled_known_future_x(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    recipe = _recipe(
        model_family="ridge",
        feature_builder="raw_feature_panel",
        forecast_type="iterated",
        exogenous_x_path_policy="scheduled_known_future_x",
        scheduled_known_future_x_columns=["CPIAUCSL"],
        benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5},
        layer2_representation_spec={
            "feature_blocks": {
                "feature_block_set": {"value": "mixed_feature_blocks"},
                "target_lag_block": {"value": "fixed_target_lags", "lag_orders": [1, 2]},
                "x_lag_feature_block": {"value": "none"},
                "factor_feature_block": {"value": "none"},
                "level_feature_block": {"value": "none"},
                "rotation_feature_block": {"value": "none"},
                "temporal_feature_block": {"value": "none"},
            }
        },
    )
    result = execute_recipe(
        recipe=recipe,
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=fixture,
    )

    run_dir = tmp_path / result.run.artifact_subdir
    manifest = json.loads((run_dir / "manifest.json").read_text())
    predictions = __import__("pandas").read_csv(run_dir / "predictions.csv")
    steps = __import__("pandas").read_csv(run_dir / "raw_panel_iterated_steps.csv")

    assert manifest["raw_panel_iterated_runtime_contract"] == "raw_panel_iterated_scheduled_known_future_x_v1"
    assert manifest["exogenous_x_path_policy"] == "scheduled_known_future_x"
    assert manifest["scheduled_known_future_x_columns"] == ["CPIAUCSL"]
    assert predictions["raw_panel_iterated_x_path_policy"].eq("scheduled_known_future_x").all()
    assert predictions["raw_panel_iterated_runtime"].eq("raw_panel_iterated_scheduled_known_future_x_v1").all()
    assert steps["x_path_policy"].eq("scheduled_known_future_x").all()
    assert steps["scheduled_known_future_x_columns"].eq('["CPIAUCSL"]').all()
    assert steps.loc[steps["step"] == 1, "x_source_date"].eq(steps.loc[steps["step"] == 1, "origin_date"]).all()
    assert steps.loc[steps["step"] > 1, "x_source_date"].ne(steps.loc[steps["step"] > 1, "origin_date"]).any()


def test_execute_recipe_runs_raw_panel_iterated_recursive_x_model(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    recipe = _recipe(
        model_family="ridge",
        feature_builder="raw_feature_panel",
        forecast_type="iterated",
        exogenous_x_path_policy="recursive_x_model",
        recursive_x_model_family="ar1",
        benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5},
        layer2_representation_spec={
            "feature_blocks": {
                "feature_block_set": {"value": "mixed_feature_blocks"},
                "target_lag_block": {"value": "fixed_target_lags", "lag_orders": [1, 2]},
                "x_lag_feature_block": {"value": "none"},
                "factor_feature_block": {"value": "none"},
                "level_feature_block": {"value": "none"},
                "rotation_feature_block": {"value": "none"},
                "temporal_feature_block": {"value": "none"},
            }
        },
    )
    result = execute_recipe(
        recipe=recipe,
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=fixture,
    )

    run_dir = tmp_path / result.run.artifact_subdir
    manifest = json.loads((run_dir / "manifest.json").read_text())
    predictions = __import__("pandas").read_csv(run_dir / "predictions.csv")
    steps = __import__("pandas").read_csv(run_dir / "raw_panel_iterated_steps.csv")

    assert manifest["raw_panel_iterated_runtime_contract"] == "raw_panel_iterated_recursive_x_model_ar1_v1"
    assert manifest["exogenous_x_path_policy"] == "recursive_x_model"
    assert manifest["recursive_x_model_family"] == "ar1"
    assert isinstance(manifest["recursive_x_model_fallback_columns"], list)
    assert predictions["raw_panel_iterated_x_path_policy"].eq("recursive_x_model").all()
    assert predictions["raw_panel_iterated_runtime"].eq("raw_panel_iterated_recursive_x_model_ar1_v1").all()
    assert steps["x_path_policy"].eq("recursive_x_model").all()
    assert steps["recursive_x_model_family"].eq("ar1").all()
    assert steps.loc[steps["step"] == 1, "x_source_date"].eq(steps.loc[steps["step"] == 1, "origin_date"]).all()
    assert steps.loc[steps["step"] > 1, "x_source_date"].ne(steps.loc[steps["step"] > 1, "origin_date"]).any()


def test_execute_recipe_runs_custom_raw_panel_iterated_hold_last_observed(tmp_path: Path) -> None:
    clear_custom_extensions()
    calls: list[int] = []

    @custom_model("iterated_raw_panel_custom")
    def _iterated_raw_panel_custom(X_train, y_train, X_test, context):
        assert context["contract_version"] == "custom_model_v1"
        assert context["feature_runtime_builder"] == "raw_feature_panel"
        assert context["forecast_type"] == "iterated"
        assert context["x_path_policy"] == "hold_last_observed"
        assert context["raw_panel_iterated_runtime_contract"] == "raw_panel_iterated_hold_last_observed_v1"
        assert context["raw_panel_iterated_payload_contract"] == "multi_step_raw_panel_payload_v1"
        assert context["block_order"] == ["base_x", "target_lag"]
        assert context["alignment"]["target_lag_timing"] == "recursive_target_history_updated_each_step"
        assert X_test.shape == (1, X_train.shape[1])
        calls.append(int(context["raw_panel_iterated_step"]))
        return float(y_train[-1])

    try:
        fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
        recipe = _recipe(
            model_family="iterated_raw_panel_custom",
            feature_builder="raw_feature_panel",
            forecast_type="iterated",
            exogenous_x_path_policy="hold_last_observed",
            benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5},
            layer2_representation_spec={
                "feature_blocks": {
                    "feature_block_set": {"value": "mixed_feature_blocks"},
                    "target_lag_block": {"value": "fixed_target_lags", "lag_orders": [1, 2]},
                    "x_lag_feature_block": {"value": "none"},
                    "factor_feature_block": {"value": "none"},
                    "level_feature_block": {"value": "none"},
                    "rotation_feature_block": {"value": "none"},
                    "temporal_feature_block": {"value": "none"},
                }
            },
        )
        result = execute_recipe(
            recipe=recipe,
            preprocess=_preprocess_raw_only(),
            output_root=tmp_path,
            local_raw_source=fixture,
        )

        run_dir = tmp_path / result.run.artifact_subdir
        manifest = json.loads((run_dir / "manifest.json").read_text())
        steps = __import__("pandas").read_csv(run_dir / "raw_panel_iterated_steps.csv")

        assert manifest["model_spec"]["custom_model"] is True
        assert manifest["forecast_type"] == "iterated"
        assert manifest["forecast_payload_family"] == "raw_panel_iterated"
        assert manifest["raw_panel_iterated_runtime_contract"] == "raw_panel_iterated_hold_last_observed_v1"
        assert steps["payload_contract"].eq("multi_step_raw_panel_payload_v1").all()
        assert calls
        assert max(calls) == max(recipe.horizons)
    finally:
        clear_custom_extensions()


def test_execute_recipe_writes_minimal_importance_artifact_for_random_forest(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    result = execute_recipe(
        recipe=_recipe(model_family="random_forest", benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5}),
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=fixture,
        provenance_payload={"compiler": {"importance_spec": {"importance_method": "minimal_importance"}}},
    )

    run_dir = tmp_path / result.run.artifact_subdir
    importance = json.loads((run_dir / "importance_minimal.json").read_text())
    assert importance["model_family"] == "random_forest"
    assert len(importance["feature_importance"]) > 0


def test_execute_recipe_expanding_still_supported_without_importance(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    result = execute_recipe(
        recipe=_recipe(framework="expanding", benchmark_config={"minimum_train_size": 5}),
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=fixture,
    )

    assert result.raw_result.dataset_metadata.dataset == "fred_md"
    assert result.run.route_owner == "single_run"
    assert result.run.artifact_subdir.startswith("runs/")


def test_execute_recipe_writes_cw_artifact(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    recipe = _recipe(benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5})
    preprocess = _preprocess_raw_only()

    result = execute_recipe(
        recipe=recipe,
        preprocess=preprocess,
        output_root=tmp_path,
        local_raw_source=fixture,
        provenance_payload={"compiler": {"stat_test_spec": {"stat_test": "cw"}}},
    )

    run_dir = tmp_path / result.run.artifact_subdir
    manifest = json.loads((run_dir / "manifest.json").read_text())
    cw_payload = json.loads((run_dir / "stat_test_cw.json").read_text())

    assert manifest["stat_test_spec"]["stat_test"] == "cw"
    assert manifest["stat_test_file"] == "stat_test_cw.json"
    assert cw_payload["stat_test"] == "cw"
    assert "forecast_adjustment_mean" in cw_payload
    assert cw_payload["n"] >= 2


def test_execute_recipe_runs_custom_benchmark_plugin(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    plugin_path = tmp_path / "custom_benchmark_plugin.py"
    plugin_path.write_text(
        "def custom_benchmark(train, horizon, benchmark_config):\n"
        "    offset = float(benchmark_config.get('offset', 0.0))\n"
        "    return float(train.iloc[-1]) + offset + float(horizon)\n",
        encoding="utf-8",
    )
    recipe = _recipe(
        model_family="ar",
        feature_builder="target_lag_features",
        benchmark="custom_benchmark",
        framework="expanding",
        benchmark_config={
            "minimum_train_size": 5,
            "plugin_path": str(plugin_path),
            "callable_name": "custom_benchmark",
            "offset": 2.5,
        },
    )
    preprocess = _preprocess_raw_only()

    result = execute_recipe(
        recipe=recipe,
        preprocess=preprocess,
        output_root=tmp_path,
        local_raw_source=fixture,
    )

    run_dir = tmp_path / result.run.artifact_subdir
    manifest = json.loads((run_dir / "manifest.json").read_text())
    predictions = json.loads((run_dir / "metrics.json").read_text())

    assert manifest["benchmark_name"] == "custom_benchmark"
    assert manifest["benchmark_spec"]["plugin_path"] == str(plugin_path)
    assert manifest["benchmark_spec"]["callable_name"] == "custom_benchmark"
    assert predictions["benchmark_name"] == "custom_benchmark"


def test_execute_recipe_rejects_custom_benchmark_without_plugin_contract(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    recipe = _recipe(
        model_family="ar",
        feature_builder="target_lag_features",
        benchmark="custom_benchmark",
        framework="expanding",
        benchmark_config={"minimum_train_size": 5},
    )

    with __import__("pytest").raises(Exception):
        execute_recipe(
            recipe=recipe,
            preprocess=_preprocess_raw_only(),
            output_root=tmp_path,
            local_raw_source=fixture,
        )


def test_execute_recipe_writes_comparison_summary_artifact(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    result = execute_recipe(
        recipe=_recipe(benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5}),
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=fixture,
    )

    run_dir = tmp_path / result.run.artifact_subdir
    manifest = json.loads((run_dir / "manifest.json").read_text())
    comparison = json.loads((run_dir / "comparison_summary.json").read_text())

    assert manifest["comparison_file"] == "comparison_summary.json"
    assert comparison["benchmark_name"] == manifest["benchmark_name"]
    assert comparison["model_name"] == manifest["forecast_engine"]
    assert "h1" in comparison["comparison_by_horizon"]
    assert comparison["comparison_by_horizon"]["h1"]["n_predictions"] >= 1
    assert "mean_loss_diff" in comparison["comparison_by_horizon"]["h1"]
    assert "win_rate" in comparison["comparison_by_horizon"]["h1"]
    assert "model_msfe" in comparison["comparison_by_horizon"]["h1"]
    assert "benchmark_msfe" in comparison["comparison_by_horizon"]["h1"]


def test_execute_recipe_stage4_metrics_include_relative_and_direction_fields(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    result = execute_recipe(
        recipe=_recipe(benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5}),
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=fixture,
        provenance_payload={
            "compiler": {
                "evaluation_spec": {
                    "primary_metric": "msfe",
                    "point_metrics": "rmse",
                    "relative_metrics": "relative_rmse",
                    "direction_metrics": "directional_accuracy",
                    "regime_definition": "none",
                    "regime_use": "evaluation_only",
                    "regime_metrics": "all_main_metrics_by_regime",
                }
            }
        },
    )
    run_dir = tmp_path / result.run.artifact_subdir
    manifest = json.loads((run_dir / "manifest.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    h1 = metrics["metrics_by_horizon"]["h1"]
    assert manifest["evaluation_spec"]["relative_metrics"] == "relative_rmse"
    assert "relative_rmse" in h1
    assert "relative_mae" in h1
    assert "benchmark_win_rate" in h1
    assert "directional_accuracy" in h1
    assert "sign_accuracy" in h1


def test_execute_recipe_writes_layer4_evaluation_summary_and_report(tmp_path: Path) -> None:
    result = execute_recipe(
        recipe=_recipe(benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5}),
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=Path("tests/fixtures/fred_md_ar_sample.csv"),
        provenance_payload={
            "compiler": {
                "evaluation_spec": {
                    "primary_metric": "rmse",
                    "point_metrics": "rmse",
                    "relative_metrics": "relative_rmse",
                    "direction_metrics": "directional_accuracy",
                    "density_metrics": "pinball_loss",
                    "economic_metrics": "utility_gain",
                    "benchmark_window": "expanding",
                    "benchmark_scope": "same_for_all",
                    "agg_time": "full_out_of_sample_average",
                    "agg_horizon": "equal_weight",
                    "agg_target": "report_separately_only",
                    "ranking": "mean_metric_rank",
                    "report_style": "markdown_table",
                    "regime_definition": "none",
                    "regime_use": "evaluation_only",
                    "regime_metrics": "all_main_metrics_by_regime",
                    "decomposition_target": "preprocessing_effect",
                    "decomposition_order": "marginal_effect_only",
                    "oos_period": "all_oos_data",
                }
            }
        },
    )
    run_dir = tmp_path / result.run.artifact_subdir
    manifest = json.loads((run_dir / "manifest.json").read_text())
    summary = json.loads((run_dir / "evaluation_summary.json").read_text())
    report = (run_dir / "evaluation_report.md").read_text()

    assert manifest["evaluation_summary_file"] == "evaluation_summary.json"
    assert manifest["evaluation_report_file"] == "evaluation_report.md"
    assert manifest["evaluation_summary_contract"] == "layer4_evaluation_summary_v1"
    assert summary["contract_version"] == "layer4_evaluation_summary_v1"
    assert summary["target_mode"] == "single_target"
    assert summary["summary"]["primary_metric"] == "rmse"
    assert summary["summary"]["overall_equal_weight"]["available"] is True
    assert summary["summary"]["selected_metric_availability"]["point_metrics"]["metric_key"] == "rmse"
    assert summary["summary"]["selected_metric_availability"]["density_metrics"]["available"] is False
    assert "Primary metric" in report


def test_execute_recipe_reads_oos_period_from_evaluation_spec(tmp_path: Path, monkeypatch) -> None:
    seen: list[str] = []

    def _capture_filter(origin_plan, *, index, regime):
        seen.append(regime)
        return origin_plan

    monkeypatch.setattr(execution_build, "_filter_origins_by_regime", _capture_filter)
    execute_recipe(
        recipe=_recipe(benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5}),
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=Path("tests/fixtures/fred_md_ar_sample.csv"),
        provenance_payload={
            "compiler": {
                "evaluation_spec": {
                    "primary_metric": "msfe",
                    "oos_period": "recession_only_oos",
                    "regime_definition": "none",
                    "regime_use": "evaluation_only",
                    "regime_metrics": "all_main_metrics_by_regime",
                }
            }
        },
    )

    assert seen == ["recession_only_oos", "recession_only_oos"]


def test_execute_recipe_writes_regime_summary_for_nber_slice(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_sample.csv")
    result = execute_recipe(
        recipe=_recipe(framework="expanding", benchmark_config={"minimum_train_size": 5}),
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=fixture,
        provenance_payload={
            "compiler": {
                "evaluation_spec": {
                    "primary_metric": "msfe",
                    "point_metrics": "msfe",
                    "relative_metrics": "relative_msfe",
                    "direction_metrics": "directional_accuracy",
                    "regime_definition": "nber_recession",
                    "regime_use": "evaluation_only",
                    "regime_metrics": "all_main_metrics_by_regime",
                }
            }
        },
    )
    run_dir = tmp_path / result.run.artifact_subdir
    manifest = json.loads((run_dir / "manifest.json").read_text())
    regime = json.loads((run_dir / "regime_summary.json").read_text())
    assert manifest["regime_file"] == "regime_summary.json"
    assert regime["regime_definition"] == "nber_recession"
    assert "h1" in regime["by_horizon"]


def test_execute_recipe_writes_regime_summary_for_user_defined_slice(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_sample.csv")
    result = execute_recipe(
        recipe=_recipe(framework="expanding", benchmark_config={"minimum_train_size": 5}),
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=fixture,
        provenance_payload={
            "compiler": {
                "evaluation_spec": {
                    "primary_metric": "msfe",
                    "point_metrics": "msfe",
                    "relative_metrics": "relative_msfe",
                    "direction_metrics": "directional_accuracy",
                    "regime_definition": "user_defined_regime",
                    "regime_use": "evaluation_only",
                    "regime_metrics": "all_main_metrics_by_regime",
                    "regime_start": "1900-01-01",
                    "regime_end": "2100-12-31",
                }
            }
        },
    )
    run_dir = tmp_path / result.run.artifact_subdir
    regime = json.loads((run_dir / "regime_summary.json").read_text())
    assert regime["regime_definition"] == "user_defined_regime"
    assert regime["by_horizon"]["h1"]["n_regime"] >= 1


def test_execute_recipe_runs_robust_scaling_preprocess_path(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_raw_panel_missing.csv")
    result = execute_recipe(
        recipe=_recipe(model_family="ridge", feature_builder="raw_feature_panel", benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5}),
        preprocess=_preprocess_train_only_robust(),
        output_root=tmp_path,
        local_raw_source=fixture,
    )

    run_dir = tmp_path / result.run.artifact_subdir
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["preprocess_contract"]["scaling_policy"] == "robust"
    assert manifest["preprocess_contract"]["x_missing_policy"] == "em_impute"
    assert (run_dir / "predictions.csv").exists()
    assert (run_dir / "comparison_summary.json").exists()


def test_execute_recipe_runs_target_normalization_with_dual_scale_artifacts(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    result = execute_recipe(
        recipe=_recipe(model_family="ridge", feature_builder="raw_feature_panel", benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5}),
        preprocess=_preprocess_target_zscore_both_scales(),
        output_root=tmp_path,
        local_raw_source=fixture,
    )

    run_dir = tmp_path / result.run.artifact_subdir
    manifest = json.loads((run_dir / "manifest.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    predictions = __import__("pandas").read_csv(run_dir / "predictions.csv")

    assert manifest["preprocess_contract"]["target_normalization"] == "zscore_train_only"
    assert manifest["preprocess_contract"]["evaluation_scale"] == "both"
    assert set(predictions["target_normalization"]) == {"zscore_train_only"}
    assert "y_pred_model_scale" in predictions
    assert "y_pred_transformed_scale" in predictions
    assert "y_pred_original_scale" in predictions
    assert metrics["metrics_by_horizon"]["h1"]["scale_metrics"]["original_target_scale"]["msfe"] >= 0.0
    assert metrics["metrics_by_horizon"]["h1"]["scale_metrics"]["transformed_target_scale"]["msfe"] >= 0.0


def test_execute_recipe_runs_registered_custom_temporal_feature_block(tmp_path: Path) -> None:
    clear_custom_extensions()

    @custom_feature_block("temporal_spread", block_kind="temporal")
    def _temporal_spread(context):
        train = context.X_train.max(axis=1) - context.X_train.min(axis=1)
        pred = context.X_pred.max(axis=1) - context.X_pred.min(axis=1)
        return FeatureBlockCallableResult(
            train_features=train.to_frame("custom__spread"),
            pred_features=pred.to_frame("custom__spread"),
            feature_names=("custom_spread",),
            runtime_feature_names=("custom__spread",),
            fit_state={"source_columns": list(context.X_train.columns)},
            leakage_metadata={"lookahead": "forbidden"},
            provenance={"composition": "append", "test": "temporal_spread"},
        )

    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    recipe = _recipe(
        model_family="ridge",
        feature_builder="raw_feature_panel",
        benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5},
    )
    recipe = replace(
        recipe,
        data_task_spec={**recipe.data_task_spec, "custom_temporal_feature_block": "temporal_spread"},
        layer2_representation_spec={
            "feature_blocks": {
                "feature_block_set": {"value": "custom_feature_blocks"},
                "x_lag_feature_block": {"value": "none"},
                "factor_feature_block": {"value": "none"},
                "level_feature_block": {"value": "none"},
                "rotation_feature_block": {"value": "none"},
                "temporal_feature_block": {"value": "custom_temporal_features"},
            }
        },
    )

    result = execute_recipe(
        recipe=recipe,
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=fixture,
    )

    run_dir = tmp_path / result.run.artifact_subdir
    manifest = json.loads((run_dir / "manifest.json").read_text())
    fit_state = json.loads((run_dir / "feature_representation_fit_state.json").read_text())

    assert manifest["prediction_rows"] > 0
    assert fit_state["block"] == "custom_temporal_feature_block"
    assert fit_state["name"] == "temporal_spread"
    assert fit_state["feature_names"] == ["custom_spread"]
    clear_custom_extensions()


def test_execute_recipe_runs_custom_l2_block_with_custom_l3_model(tmp_path: Path) -> None:
    clear_custom_extensions()

    @custom_feature_block("temporal_spread", block_kind="temporal")
    def _temporal_spread(context):
        train = context.X_train.max(axis=1) - context.X_train.min(axis=1)
        pred = context.X_pred.max(axis=1) - context.X_pred.min(axis=1)
        return FeatureBlockCallableResult(
            train_features=train.to_frame("custom__spread"),
            pred_features=pred.to_frame("custom__spread"),
            feature_names=("custom_spread",),
            runtime_feature_names=("custom__spread",),
            fit_state={"source_columns": list(context.X_train.columns)},
            leakage_metadata={"lookahead": "forbidden"},
            provenance={"composition": "append", "test": "custom_l2_custom_l3"},
        )

    @custom_model("mean_plus_spread")
    def _mean_plus_spread(X_train, y_train, X_test, context):
        assert context["contract_version"] == "custom_model_v1"
        assert context["feature_runtime_builder"] == "raw_feature_panel"
        assert context["feature_dispatch_source"] == "layer2_feature_blocks"
        assert "temporal" in context["block_order"]
        assert "custom" in context["block_order"]
        assert "custom_spread" in context["feature_names"]
        assert X_train.shape[1] == len(context["feature_names"])
        assert X_test.shape == (1, X_train.shape[1])
        return float(y_train.mean())

    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    recipe = _recipe(
        model_family="mean_plus_spread",
        feature_builder="raw_feature_panel",
        benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5},
    )
    recipe = replace(
        recipe,
        data_task_spec={**recipe.data_task_spec, "custom_temporal_feature_block": "temporal_spread"},
        layer2_representation_spec={
            "feature_blocks": {
                "feature_block_set": {"value": "custom_feature_blocks"},
                "x_lag_feature_block": {"value": "none"},
                "factor_feature_block": {"value": "none"},
                "level_feature_block": {"value": "none"},
                "rotation_feature_block": {"value": "none"},
                "temporal_feature_block": {"value": "custom_temporal_features"},
            }
        },
    )

    result = execute_recipe(
        recipe=recipe,
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=fixture,
    )

    run_dir = tmp_path / result.run.artifact_subdir
    manifest = json.loads((run_dir / "manifest.json").read_text())
    fit_state = json.loads((run_dir / "feature_representation_fit_state.json").read_text())
    predictions = __import__("pandas").read_csv(run_dir / "predictions.csv")

    assert manifest["prediction_rows"] > 0
    assert manifest["model_spec"]["model_family"] == "mean_plus_spread"
    assert manifest["model_spec"]["custom_model"] is True
    assert manifest["forecast_engine"] == "custom_model:mean_plus_spread:raw_feature_panel_v0"
    assert fit_state["block"] == "custom_temporal_feature_block"
    assert fit_state["name"] == "temporal_spread"
    assert fit_state["feature_names"] == ["custom_spread"]
    assert len(predictions) > 0
    clear_custom_extensions()


def test_execute_recipe_runs_registered_custom_feature_combiner(tmp_path: Path) -> None:
    clear_custom_extensions()

    @custom_feature_combiner("sum_first_two")
    def _sum_first_two(context):
        train = context.blocks_train["candidate_z"].iloc[:, :2].sum(axis=1).to_frame("custom_combo")
        pred = context.blocks_pred["candidate_z"].iloc[:, :2].sum(axis=1).to_frame("custom_combo")
        return FeatureCombinerCallableResult(
            Z_train=train,
            Z_pred=pred,
            feature_names=("custom_combo",),
            block_roles={"custom_combo": "custom"},
            fit_state={"source_feature_names": list(context.feature_names[:2])},
            leakage_metadata={"lookahead": "forbidden"},
            provenance={"test": "sum_first_two"},
        )

    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    recipe = _recipe(
        model_family="ridge",
        feature_builder="raw_feature_panel",
        benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5},
    )
    recipe = replace(
        recipe,
        data_task_spec={**recipe.data_task_spec, "custom_feature_combiner": "sum_first_two"},
        layer2_representation_spec={
            "feature_blocks": {
                "feature_block_set": {"value": "custom_feature_blocks"},
                "x_lag_feature_block": {"value": "none"},
                "factor_feature_block": {"value": "none"},
                "level_feature_block": {"value": "none"},
                "rotation_feature_block": {"value": "none"},
                "temporal_feature_block": {"value": "none"},
                "feature_block_combination": {"value": "custom_feature_combiner"},
            }
        },
    )

    result = execute_recipe(
        recipe=recipe,
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=fixture,
    )

    run_dir = tmp_path / result.run.artifact_subdir
    manifest = json.loads((run_dir / "manifest.json").read_text())
    fit_state = json.loads((run_dir / "feature_representation_fit_state.json").read_text())

    assert manifest["prediction_rows"] > 0
    assert fit_state["block"] == "custom_feature_combiner"
    assert fit_state["name"] == "sum_first_two"
    assert fit_state["contract_version"] == "custom_feature_combiner_v1"
    assert fit_state["feature_names"] == ["custom_combo"]
    clear_custom_extensions()


def test_execute_recipe_selects_after_custom_feature_blocks(tmp_path: Path) -> None:
    clear_custom_extensions()

    @custom_feature_block("temporal_spread", block_kind="temporal")
    def _temporal_spread(context):
        train = context.X_train.max(axis=1) - context.X_train.min(axis=1)
        pred = context.X_pred.max(axis=1) - context.X_pred.min(axis=1)
        return FeatureBlockCallableResult(
            train_features=train.to_frame("custom__spread"),
            pred_features=pred.to_frame("custom__spread"),
            feature_names=("custom_spread",),
            runtime_feature_names=("custom__spread",),
            fit_state={"source_columns": list(context.X_train.columns)},
            leakage_metadata={"lookahead": "forbidden"},
            provenance={"composition": "append", "test": "select_after_custom"},
        )

    preprocess = build_preprocess_contract(
        target_transform_policy="raw_level",
        x_transform_policy="raw_level",
        tcode_policy="extra_preprocess_only",
        target_missing_policy="none",
        x_missing_policy="none",
        target_outlier_policy="none",
        x_outlier_policy="none",
        scaling_policy="none",
        dimensionality_reduction_policy="none",
        feature_selection_policy="correlation_filter",
        feature_selection_semantics="select_after_custom_feature_blocks",
        preprocess_order="extra_only",
        preprocess_fit_scope="train_only",
        inverse_transform_policy="none",
        evaluation_scale="raw_level",
    )
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    recipe = _recipe(
        model_family="ridge",
        feature_builder="raw_feature_panel",
        benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5},
    )
    recipe = replace(
        recipe,
        data_task_spec={**recipe.data_task_spec, "custom_temporal_feature_block": "temporal_spread"},
        layer2_representation_spec={
            "feature_blocks": {
                "feature_block_set": {"value": "custom_feature_blocks"},
                "x_lag_feature_block": {"value": "none"},
                "factor_feature_block": {"value": "none"},
                "level_feature_block": {"value": "none"},
                "rotation_feature_block": {"value": "none"},
                "temporal_feature_block": {"value": "custom_temporal_features"},
            }
        },
    )

    result = execute_recipe(
        recipe=recipe,
        preprocess=preprocess,
        output_root=tmp_path,
        local_raw_source=fixture,
    )

    run_dir = tmp_path / result.run.artifact_subdir
    manifest = json.loads((run_dir / "manifest.json").read_text())
    fit_state = json.loads((run_dir / "feature_representation_fit_state.json").read_text())

    assert manifest["prediction_rows"] > 0
    assert manifest["preprocess_contract"]["feature_selection_semantics"] == "select_after_custom_feature_blocks"
    assert fit_state["block"] == "custom_final_z_selection"
    assert fit_state["contract_version"] == "custom_final_z_selection_v1"
    assert "custom_spread" in fit_state["candidate_feature_names"]
    assert fit_state["selected_final_feature_count"] > 0
    clear_custom_extensions()


def test_execute_recipe_writes_minimal_importance_artifact_for_lasso(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    result = execute_recipe(
        recipe=_recipe(model_family="lasso", benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5}),
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=fixture,
        provenance_payload={"compiler": {"importance_spec": {"importance_method": "minimal_importance"}}},
    )

    run_dir = tmp_path / result.run.artifact_subdir
    manifest = json.loads((run_dir / "manifest.json").read_text())
    importance = json.loads((run_dir / "importance_minimal.json").read_text())
    assert manifest["importance_spec"]["importance_method"] == "minimal_importance"
    assert manifest["importance_file"] == "importance_minimal.json"
    assert importance["model_family"] == "lasso"
    assert len(importance["feature_importance"]) > 0



def test_execute_recipe_runs_multi_target_slice(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    stage0 = build_design_frame(
        research_design="single_forecast_run",
        fixed_design={
            "dataset_adapter": "fred_md",
            "information_set": "revised_monthly",
            "sample_split": "expanding_window_oos",
            "benchmark": "zero_change",
            "evaluation_protocol": "point_forecast_core",
            "forecast_task": "multi_target",
        },
        comparison_contract={
            "information_set_policy": "identical",
            "sample_split_policy": "identical",
            "benchmark_policy": "identical",
            "evaluation_policy": "identical",
        },
        varying_design={"model_families": ("ar",), "feature_recipes": ("target_lag_features",), "horizons": ("h1",)},
    )
    recipe = build_recipe_spec(
        recipe_id="fred_md_multi_ar",
        stage0=stage0,
        target="",
        targets=("INDPRO", "RPI"),
        horizons=(1, 3),
        raw_dataset="fred_md",
        benchmark_config={"minimum_train_size": 5},
    )
    result = execute_recipe(
        recipe=recipe,
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=fixture,
    )

    run_dir = tmp_path / result.run.artifact_subdir
    manifest = json.loads((run_dir / "manifest.json").read_text())
    predictions = __import__("pandas").read_csv(run_dir / "predictions.csv")
    metrics = json.loads((run_dir / "metrics.json").read_text())
    comparison = json.loads((run_dir / "comparison_summary.json").read_text())
    assert manifest["targets"] == ["INDPRO", "RPI"]
    assert set(predictions["target"].unique()) == {"INDPRO", "RPI"}
    assert set(metrics["metrics_by_target"].keys()) == {"INDPRO", "RPI"}
    assert set(comparison["comparison_by_target"].keys()) == {"INDPRO", "RPI"}

def test_execute_recipe_manifest_preserves_tree_context_payload(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    stage0 = build_design_frame(
        research_design="single_forecast_run",
        fixed_design={
            "dataset_adapter": "fred_md",
            "information_set": "revised_monthly",
            "sample_split": "expanding_window_oos",
            "benchmark": "zero_change",
            "evaluation_protocol": "point_forecast_core",
            "forecast_task": "single_target",
        },
        comparison_contract={
            "information_set_policy": "identical",
            "sample_split_policy": "identical",
            "benchmark_policy": "identical",
            "evaluation_policy": "identical",
        },
        varying_design={"model_families": ("ar",), "feature_recipes": ("target_lag_features",), "horizons": ("h1",)},
    )
    recipe = build_recipe_spec(
        recipe_id="fred_md_tree_context",
        stage0=stage0,
        target="INDPRO",
        horizons=(1, 3),
        raw_dataset="fred_md",
        benchmark_config={"minimum_train_size": 5},
    )
    provenance_payload = {
        "tree_context": {
            "research_design": "single_forecast_run",
            "design_shape": "one_fixed_env_one_tool_surface",
            "execution_posture": "single_run_recipe",
            "experiment_unit": "single_model_path",
            "route_owner": "single_run",
            "fixed_design": {
                "dataset_adapter": "fred_md",
                "information_set": "revised_monthly",
                "sample_split": "expanding_window_oos",
                "benchmark": "zero_change",
                "evaluation_protocol": "point_forecast_core",
                "forecast_task": "single_target",
            },
            "varying_design": {
                "model_families": ["ar"],
                "feature_recipes": ["target_lag_features"],
                "preprocess_variants": [],
                "tuning_variants": [],
                "horizons": ["h1", "h3"],
            },
            "comparison_contract": {
                "information_set_policy": "identical",
                "sample_split_policy": "identical",
                "benchmark_policy": "identical",
                "evaluation_policy": "identical",
            },
            "fixed_axes": {"dataset": "fred_md", "framework": "expanding", "model_family": "ar"},
            "sweep_axes": {},
            "conditional_axes": {},
            "axis_layers": {"dataset": "1_data_task", "framework": "3_training", "model_family": "3_training"},
            "leaf_config": {"target": "INDPRO", "horizons": [1, 3]},
        },
    }
    result = execute_recipe(
        recipe=recipe,
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=fixture,
        provenance_payload=provenance_payload,
    )

    run_dir = tmp_path / result.run.artifact_subdir
    manifest = json.loads((run_dir / "manifest.json").read_text())
    summary = (run_dir / "summary.txt").read_text()
    assert manifest["tree_context"]["route_owner"] == "single_run"
    assert manifest["tree_context"]["fixed_design"]["dataset_adapter"] == "fred_md"
    assert manifest["tree_context"]["leaf_config"]["target"] == "INDPRO"
    assert "tree_context=route_owner=single_run" in summary




def test_execute_recipe_skip_failed_model_records_partial_manifest(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    stage0 = build_design_frame(
        research_design="single_forecast_run",
        fixed_design={
            "dataset_adapter": "fred_md",
            "information_set": "revised_monthly",
            "sample_split": "expanding_window_oos",
            "benchmark": "zero_change",
            "evaluation_protocol": "point_forecast_core",
            "forecast_task": "multi_target",
        },
        comparison_contract={
            "information_set_policy": "identical",
            "sample_split_policy": "identical",
            "benchmark_policy": "identical",
            "evaluation_policy": "identical",
        },
        varying_design={"model_families": ("ar",), "feature_recipes": ("target_lag_features",), "horizons": ("h1",)},
    )
    recipe = build_recipe_spec(
        recipe_id="fred_md_multi_skip_failure",
        stage0=stage0,
        target="",
        targets=("INDPRO", "MISSING_TARGET"),
        horizons=(1, 3),
        raw_dataset="fred_md",
        benchmark_config={"minimum_train_size": 5},
    )
    result = execute_recipe(
        recipe=recipe,
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=fixture,
        provenance_payload={"compiler": {"failure_policy_spec": {"failure_policy": "skip_failed_model"}}},
    )
    run_dir = tmp_path / result.run.artifact_subdir
    manifest = json.loads((run_dir / "manifest.json").read_text())
    failures = json.loads((run_dir / "failures.json").read_text())
    predictions = __import__("pandas").read_csv(run_dir / "predictions.csv")
    assert manifest["partial_run"] is True
    assert manifest["successful_targets"] == ["INDPRO"]
    assert manifest["failure_log_file"] == "failures.json"
    assert any(item["target"] == "MISSING_TARGET" for item in failures)
    assert set(predictions["target"].unique()) == {"INDPRO"}


def test_execute_recipe_save_partial_results_keeps_metrics_when_optional_artifact_fails(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    result = execute_recipe(
        recipe=_recipe(model_family="ar", feature_builder="target_lag_features", benchmark_config={"minimum_train_size": 5}),
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=fixture,
        provenance_payload={
            "compiler": {
                "failure_policy_spec": {"failure_policy": "save_partial_results"},
                "importance_spec": {"importance_method": "minimal_importance"},
            }
        },
    )
    run_dir = tmp_path / result.run.artifact_subdir
    manifest = json.loads((run_dir / "manifest.json").read_text())
    failures = json.loads((run_dir / "failures.json").read_text())
    assert (run_dir / "metrics.json").exists()
    assert manifest["partial_run"] is True
    assert manifest["failure_log_file"] == "failures.json"
    assert manifest.get("importance_file") is None
    assert any(item["stage"] == "importance_artifact" for item in failures)



def test_execute_recipe_parallel_by_horizon_writes_manifest(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    result = execute_recipe(
        recipe=_recipe(framework="expanding", benchmark_config={"minimum_train_size": 5}),
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=fixture,
        provenance_payload={"compiler": {"compute_mode_spec": {"compute_mode": "parallel_by_horizon"}}},
    )
    run_dir = tmp_path / result.run.artifact_subdir
    manifest = json.loads((run_dir / "manifest.json").read_text())
    predictions = __import__("pandas").read_csv(run_dir / "predictions.csv")
    assert manifest["compute_mode_spec"]["compute_mode"] == "parallel_by_horizon"
    assert set(predictions["horizon"].unique()) == {1, 3}


def test_execute_recipe_parallel_by_target_runs_multi_target_slice(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    stage0 = build_design_frame(
        research_design="single_forecast_run",
        fixed_design={
            "dataset_adapter": "fred_md",
            "information_set": "revised_monthly",
            "sample_split": "expanding_window_oos",
            "benchmark": "zero_change",
            "evaluation_protocol": "point_forecast_core",
            "forecast_task": "multi_target",
        },
        comparison_contract={
            "information_set_policy": "identical",
            "sample_split_policy": "identical",
            "benchmark_policy": "identical",
            "evaluation_policy": "identical",
        },
        varying_design={"model_families": ("ar",), "feature_recipes": ("target_lag_features",), "horizons": ("h1",)},
    )
    recipe = build_recipe_spec(
        recipe_id="fred_md_multi_parallel_target",
        stage0=stage0,
        target="",
        targets=("INDPRO", "RPI"),
        horizons=(1, 3),
        raw_dataset="fred_md",
        benchmark_config={"minimum_train_size": 5},
    )
    result = execute_recipe(
        recipe=recipe,
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=fixture,
        provenance_payload={"compiler": {"compute_mode_spec": {"compute_mode": "parallel_by_target"}}},
    )
    run_dir = tmp_path / result.run.artifact_subdir
    manifest = json.loads((run_dir / "manifest.json").read_text())
    predictions = __import__("pandas").read_csv(run_dir / "predictions.csv")
    assert manifest["compute_mode_spec"]["compute_mode"] == "parallel_by_target"
    assert set(predictions["target"].unique()) == {"INDPRO", "RPI"}



def _preprocess_mean_impute_minmax_winsor() -> PreprocessContract:
    return build_preprocess_contract(
        target_transform_policy="raw_level",
        x_transform_policy="raw_level",
        tcode_policy="extra_preprocess_only",
        target_missing_policy="none",
        x_missing_policy="mean_impute",
        target_outlier_policy="none",
        x_outlier_policy="winsorize",
        scaling_policy="minmax",
        dimensionality_reduction_policy="none",
        feature_selection_policy="none",
        preprocess_order="extra_only",
        preprocess_fit_scope="train_only",
        inverse_transform_policy="none",
        evaluation_scale="raw_level",
    )


def _preprocess_pca_contract() -> PreprocessContract:
    return build_preprocess_contract(
        target_transform_policy="raw_level",
        x_transform_policy="raw_level",
        tcode_policy="extra_preprocess_only",
        target_missing_policy="none",
        x_missing_policy="median_impute",
        target_outlier_policy="none",
        x_outlier_policy="iqr_clip",
        scaling_policy="standard",
        dimensionality_reduction_policy="pca",
        feature_selection_policy="none",
        preprocess_order="extra_only",
        preprocess_fit_scope="train_only",
        inverse_transform_policy="none",
        evaluation_scale="raw_level",
    )


def _preprocess_lasso_selection_contract() -> PreprocessContract:
    return build_preprocess_contract(
        target_transform_policy="raw_level",
        x_transform_policy="raw_level",
        tcode_policy="extra_preprocess_only",
        target_missing_policy="none",
        x_missing_policy="mean_impute",
        target_outlier_policy="none",
        x_outlier_policy="zscore_clip",
        scaling_policy="standard",
        dimensionality_reduction_policy="none",
        feature_selection_policy="lasso_selection",
        preprocess_order="extra_only",
        preprocess_fit_scope="train_only",
        inverse_transform_policy="none",
        evaluation_scale="raw_level",
    )


def _preprocess_pca_lasso_selection_contract() -> PreprocessContract:
    return build_preprocess_contract(
        target_transform_policy="raw_level",
        x_transform_policy="raw_level",
        tcode_policy="extra_preprocess_only",
        target_missing_policy="none",
        x_missing_policy="mean_impute",
        target_outlier_policy="none",
        x_outlier_policy="zscore_clip",
        scaling_policy="standard",
        dimensionality_reduction_policy="pca",
        feature_selection_policy="lasso_selection",
        preprocess_order="extra_only",
        preprocess_fit_scope="train_only",
        inverse_transform_policy="none",
        evaluation_scale="raw_level",
    )


def _preprocess_pca_lasso_selection_after_factor_contract() -> PreprocessContract:
    return build_preprocess_contract(
        target_transform_policy="raw_level",
        x_transform_policy="raw_level",
        tcode_policy="extra_preprocess_only",
        target_missing_policy="none",
        x_missing_policy="mean_impute",
        target_outlier_policy="none",
        x_outlier_policy="zscore_clip",
        scaling_policy="standard",
        dimensionality_reduction_policy="pca",
        feature_selection_policy="lasso_selection",
        preprocess_order="extra_only",
        preprocess_fit_scope="train_only",
        inverse_transform_policy="none",
        evaluation_scale="raw_level",
        feature_selection_semantics="select_after_factor",
    )


def test_execute_recipe_supports_mean_impute_minmax_winsor_path(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    result = execute_recipe(
        recipe=_recipe(benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5}),
        preprocess=_preprocess_mean_impute_minmax_winsor(),
        output_root=tmp_path,
        local_raw_source=fixture,
    )
    run_dir = tmp_path / result.run.artifact_subdir
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["preprocess_contract"]["x_missing_policy"] == "mean_impute"
    assert manifest["preprocess_contract"]["scaling_policy"] == "minmax"
    assert manifest["prediction_rows"] > 0


def test_execute_recipe_supports_pca_preprocessing_path(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    result = execute_recipe(
        recipe=_recipe(benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5}),
        preprocess=_preprocess_pca_contract(),
        output_root=tmp_path,
        local_raw_source=fixture,
    )
    run_dir = tmp_path / result.run.artifact_subdir
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["preprocess_contract"]["dimensionality_reduction_policy"] == "pca"
    assert manifest["feature_representation_fit_state_file"] == "feature_representation_fit_state.json"
    fit_state = json.loads((run_dir / "feature_representation_fit_state.json").read_text())
    assert fit_state["block"] == "pca_static_factors"
    assert fit_state["runtime_policy"] == "pca"
    assert fit_state["feature_names"] == [f"factor_{idx}" for idx in range(1, fit_state["n_components"] + 1)]
    assert set(fit_state["loadings"]) == set(fit_state["feature_names"])
    assert manifest["prediction_rows"] > 0


def test_execute_recipe_supports_lasso_feature_selection_path(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    result = execute_recipe(
        recipe=_recipe(benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5}),
        preprocess=_preprocess_lasso_selection_contract(),
        output_root=tmp_path,
        local_raw_source=fixture,
    )
    run_dir = tmp_path / result.run.artifact_subdir
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["preprocess_contract"]["feature_selection_policy"] == "lasso_selection"
    assert manifest["prediction_rows"] > 0


def test_execute_recipe_supports_select_before_factor_path(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    result = execute_recipe(
        recipe=_recipe(benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5}),
        preprocess=_preprocess_pca_lasso_selection_contract(),
        output_root=tmp_path,
        local_raw_source=fixture,
    )
    run_dir = tmp_path / result.run.artifact_subdir
    manifest = json.loads((run_dir / "manifest.json").read_text())
    fit_state = json.loads((run_dir / "feature_representation_fit_state.json").read_text())
    assert manifest["preprocess_contract"]["dimensionality_reduction_policy"] == "pca"
    assert manifest["preprocess_contract"]["feature_selection_policy"] == "lasso_selection"
    assert fit_state["block"] == "pca_static_factors"
    assert fit_state["feature_selection_policy"] == "lasso_selection"
    assert fit_state["feature_selection_semantics"] == "select_before_factor"
    assert fit_state["selected_source_feature_count"] == len(fit_state["selected_source_feature_names"])
    assert set(fit_state["selected_source_feature_names"]) == set(fit_state["loadings"][fit_state["feature_names"][0]].keys())
    assert manifest["prediction_rows"] > 0


def test_execute_recipe_supports_select_after_factor_path(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    result = execute_recipe(
        recipe=_recipe(benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5}),
        preprocess=_preprocess_pca_lasso_selection_after_factor_contract(),
        output_root=tmp_path,
        local_raw_source=fixture,
    )
    run_dir = tmp_path / result.run.artifact_subdir
    manifest = json.loads((run_dir / "manifest.json").read_text())
    fit_state = json.loads((run_dir / "feature_representation_fit_state.json").read_text())
    assert manifest["preprocess_contract"]["dimensionality_reduction_policy"] == "pca"
    assert manifest["preprocess_contract"]["feature_selection_policy"] == "lasso_selection"
    assert manifest["preprocess_contract"]["feature_selection_semantics"] == "select_after_factor"
    assert fit_state["block"] == "pca_static_factors"
    assert fit_state["feature_selection_semantics"] == "select_after_factor"
    assert fit_state["selected_final_feature_count"] == len(fit_state["selected_final_feature_names"])
    assert set(fit_state["selected_final_feature_names"]).issubset(set(fit_state["post_factor_candidate_feature_names"]))
    assert manifest["prediction_rows"] > 0


def test_execute_recipe_supports_select_after_factor_with_deterministic_append(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    recipe = _recipe(benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5})
    recipe.data_task_spec["deterministic_components"] = "linear_trend"
    result = execute_recipe(
        recipe=recipe,
        preprocess=_preprocess_pca_lasso_selection_after_factor_contract(),
        output_root=tmp_path,
        local_raw_source=fixture,
    )
    run_dir = tmp_path / result.run.artifact_subdir
    manifest = json.loads((run_dir / "manifest.json").read_text())
    fit_state = json.loads((run_dir / "feature_representation_fit_state.json").read_text())
    assert manifest["preprocess_contract"]["feature_selection_semantics"] == "select_after_factor"
    assert manifest["data_task_spec"]["deterministic_components"] == "linear_trend"
    assert fit_state["feature_selection_semantics"] == "select_after_factor"
    assert "_dc_trend" in fit_state["post_factor_candidate_feature_names"]
    assert set(fit_state["selected_final_feature_names"]).issubset(set(fit_state["post_factor_candidate_feature_names"]))
    assert manifest["prediction_rows"] > 0



def test_execute_recipe_supports_ols_autoreg_model(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    result = execute_recipe(
        recipe=_recipe(model_family="ols", feature_builder="target_lag_features", benchmark_config={"minimum_train_size": 5}),
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=fixture,
    )
    manifest = json.loads((tmp_path / result.run.artifact_subdir / "manifest.json").read_text())
    assert manifest["model_spec"]["model_family"] == "ols"
    assert manifest["prediction_rows"] > 0


def test_execute_recipe_supports_xgboost_raw_panel_model(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    result = execute_recipe(
        recipe=_recipe(model_family="xgboost", feature_builder="raw_feature_panel", benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5}),
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=fixture,
    )
    manifest = json.loads((tmp_path / result.run.artifact_subdir / "manifest.json").read_text())
    assert manifest["model_spec"]["model_family"] == "xgboost"
    assert manifest["prediction_rows"] > 0


def test_execute_recipe_supports_xgboost_autoreg_model(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    result = execute_recipe(
        recipe=_recipe(model_family="xgboost", feature_builder="target_lag_features", benchmark_config={"minimum_train_size": 5}),
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=fixture,
    )
    manifest = json.loads((tmp_path / result.run.artifact_subdir / "manifest.json").read_text())
    assert manifest["model_spec"]["model_family"] == "xgboost"
    assert manifest["prediction_rows"] > 0


def test_execute_recipe_supports_quantile_linear_autoreg_point_median(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    result = execute_recipe(
        recipe=_recipe(model_family="quantile_linear", feature_builder="target_lag_features", forecast_object="point_median", benchmark_config={"minimum_train_size": 5}),
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=fixture,
    )
    manifest = json.loads((tmp_path / result.run.artifact_subdir / "manifest.json").read_text())
    assert manifest["model_spec"]["model_family"] == "quantile_linear"
    assert manifest["prediction_rows"] > 0


def test_execute_recipe_supports_quantile_linear_raw_panel_point_median(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    result = execute_recipe(
        recipe=_recipe(model_family="quantile_linear", feature_builder="raw_feature_panel", forecast_object="point_median", benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5}),
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=fixture,
    )
    manifest = json.loads((tmp_path / result.run.artifact_subdir / "manifest.json").read_text())
    assert manifest["model_spec"]["model_family"] == "quantile_linear"
    assert manifest["prediction_rows"] > 0


def test_execute_recipe_supports_lightgbm_raw_panel_model(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    result = execute_recipe(
        recipe=_recipe(model_family="lightgbm", feature_builder="raw_feature_panel", benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5}),
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=fixture,
    )
    manifest = json.loads((tmp_path / result.run.artifact_subdir / "manifest.json").read_text())
    assert manifest["model_spec"]["model_family"] == "lightgbm"
    assert manifest["prediction_rows"] > 0



def test_execute_recipe_supports_adaptive_lasso_autoreg_model(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    result = execute_recipe(
        recipe=_recipe(model_family="adaptive_lasso", feature_builder="target_lag_features", benchmark_config={"minimum_train_size": 5}),
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=fixture,
    )
    manifest = json.loads((tmp_path / result.run.artifact_subdir / "manifest.json").read_text())
    assert manifest["model_spec"]["model_family"] == "adaptive_lasso"
    assert manifest["prediction_rows"] > 0


def test_execute_recipe_supports_svr_linear_raw_panel_model(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    result = execute_recipe(
        recipe=_recipe(model_family="svr_linear", feature_builder="raw_feature_panel", benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5}),
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=fixture,
    )
    manifest = json.loads((tmp_path / result.run.artifact_subdir / "manifest.json").read_text())
    assert manifest["model_spec"]["model_family"] == "svr_linear"
    assert manifest["prediction_rows"] > 0


def test_execute_recipe_supports_svr_rbf_raw_panel_model(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    result = execute_recipe(
        recipe=_recipe(model_family="svr_rbf", feature_builder="raw_feature_panel", benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5}),
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=fixture,
    )
    manifest = json.loads((tmp_path / result.run.artifact_subdir / "manifest.json").read_text())
    assert manifest["model_spec"]["model_family"] == "svr_rbf"
    assert manifest["prediction_rows"] > 0


def test_execute_recipe_supports_huber_autoreg_model(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    result = execute_recipe(
        recipe=_recipe(model_family="huber", feature_builder="target_lag_features", benchmark_config={"minimum_train_size": 5}),
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=fixture,
    )
    manifest = json.loads((tmp_path / result.run.artifact_subdir / "manifest.json").read_text())
    assert manifest["model_spec"]["model_family"] == "huber"
    assert manifest["prediction_rows"] > 0


def test_execute_recipe_supports_catboost_raw_panel_model(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    result = execute_recipe(
        recipe=_recipe(model_family="catboost", feature_builder="raw_feature_panel", benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5}),
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=fixture,
    )
    manifest = json.loads((tmp_path / result.run.artifact_subdir / "manifest.json").read_text())
    assert manifest["model_spec"]["model_family"] == "catboost"
    assert manifest["prediction_rows"] > 0



def test_execute_recipe_supports_adaptive_lasso_raw_panel_model(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    result = execute_recipe(
        recipe=_recipe(model_family="adaptive_lasso", feature_builder="raw_feature_panel", benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5}),
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=fixture,
    )
    manifest = json.loads((tmp_path / result.run.artifact_subdir / "manifest.json").read_text())
    assert manifest["model_spec"]["model_family"] == "adaptive_lasso"
    assert manifest["prediction_rows"] > 0


def test_execute_recipe_supports_svr_linear_autoreg_model(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    result = execute_recipe(
        recipe=_recipe(model_family="svr_linear", feature_builder="target_lag_features", benchmark_config={"minimum_train_size": 5}),
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=fixture,
    )
    manifest = json.loads((tmp_path / result.run.artifact_subdir / "manifest.json").read_text())
    assert manifest["model_spec"]["model_family"] == "svr_linear"
    assert manifest["prediction_rows"] > 0


def test_execute_recipe_supports_svr_rbf_autoreg_model(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    result = execute_recipe(
        recipe=_recipe(model_family="svr_rbf", feature_builder="target_lag_features", benchmark_config={"minimum_train_size": 5}),
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=fixture,
    )
    manifest = json.loads((tmp_path / result.run.artifact_subdir / "manifest.json").read_text())
    assert manifest["model_spec"]["model_family"] == "svr_rbf"
    assert manifest["prediction_rows"] > 0


def test_execute_recipe_supports_huber_raw_panel_model(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    result = execute_recipe(
        recipe=_recipe(model_family="huber", feature_builder="raw_feature_panel", benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5}),
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=fixture,
    )
    manifest = json.loads((tmp_path / result.run.artifact_subdir / "manifest.json").read_text())
    assert manifest["model_spec"]["model_family"] == "huber"
    assert manifest["prediction_rows"] > 0


def test_execute_recipe_supports_catboost_autoreg_model(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    result = execute_recipe(
        recipe=_recipe(model_family="catboost", feature_builder="target_lag_features", benchmark_config={"minimum_train_size": 5}),
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=fixture,
    )
    manifest = json.loads((tmp_path / result.run.artifact_subdir / "manifest.json").read_text())
    assert manifest["model_spec"]["model_family"] == "catboost"
    assert manifest["prediction_rows"] > 0



def test_execute_recipe_supports_pcr_raw_panel_model(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    recipe = _recipe(model_family="pcr", feature_builder="pca_factor_features", benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5})
    recipe = __import__("dataclasses").replace(
        recipe,
        training_spec={**recipe.training_spec, "fixed_factor_count": 4},
        layer2_representation_spec={
            "feature_blocks": {
                "factor_feature_block": {
                    "value": "pca_static_factors",
                    "factor_count": {
                        "mode": "fixed",
                        "fixed_factor_count": 2,
                        "max_factors": 4,
                        "selection_scope": "train_window",
                    },
                }
            }
        },
    )
    result = execute_recipe(recipe=recipe, preprocess=_preprocess_raw_only(), output_root=tmp_path, local_raw_source=fixture)
    manifest = json.loads((tmp_path / result.run.artifact_subdir / "manifest.json").read_text())
    fit_state = json.loads((tmp_path / result.run.artifact_subdir / "feature_representation_fit_state.json").read_text())
    assert manifest["model_spec"]["model_family"] == "pcr"
    assert fit_state["n_components"] == 2
    assert manifest["prediction_rows"] > 0


def test_execute_recipe_supports_pls_raw_panel_model(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    recipe = _recipe(model_family="pls", feature_builder="pca_factor_features", benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5})
    recipe = __import__("dataclasses").replace(recipe, training_spec={**recipe.training_spec, "fixed_factor_count": 2})
    result = execute_recipe(recipe=recipe, preprocess=_preprocess_raw_only(), output_root=tmp_path, local_raw_source=fixture)
    manifest = json.loads((tmp_path / result.run.artifact_subdir / "manifest.json").read_text())
    assert manifest["model_spec"]["model_family"] == "pls"
    assert manifest["prediction_rows"] > 0


def test_execute_recipe_supports_factor_augmented_linear_raw_panel_model(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    recipe = _recipe(model_family="factor_augmented_linear", feature_builder="factors_plus_target_lags", benchmark_config={"minimum_train_size": 7, "rolling_window_size": 7})
    recipe = __import__("dataclasses").replace(recipe, training_spec={**recipe.training_spec, "fixed_factor_count": 2, "target_lag_count": 2})
    result = execute_recipe(recipe=recipe, preprocess=_preprocess_raw_only(), output_root=tmp_path, local_raw_source=fixture)
    manifest = json.loads((tmp_path / result.run.artifact_subdir / "manifest.json").read_text())
    assert manifest["model_spec"]["model_family"] == "factor_augmented_linear"
    assert manifest["prediction_rows"] > 0



def test_execute_recipe_json_csv_export_writes_sidecar_files(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    recipe = _recipe(benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5})
    result = execute_recipe(
        recipe=recipe,
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=fixture,
        provenance_payload={
            "compiler": {
                "output_spec": {
                    "export_format": "json_csv",
                    "saved_objects": "full_bundle",
                    "provenance_fields": "full",
                    "artifact_granularity": "aggregated",
                },
                "importance_spec": {"importance_method": "none"},
                "stat_test_spec": {"stat_test": "none"},
            }
        },
    )

    run_dir = tmp_path / result.run.artifact_subdir
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["output_spec"]["export_format"] == "json_csv"
    assert manifest["metrics_file"] == "metrics.json"
    assert manifest["metrics_files"]["csv"] == "metrics.csv"
    artifact_manifest = json.loads((run_dir / "artifact_manifest.json").read_text())
    assert manifest["artifact_manifest_file"] == "artifact_manifest.json"
    assert manifest["output_artifact_contract"] == "layer5_output_artifact_manifest_v1"
    assert artifact_manifest["contract_version"] == "layer5_output_artifact_manifest_v1"
    assert any(row["path"] == "metrics.csv" for row in artifact_manifest["artifacts"])
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "comparison_summary.csv").exists()


def test_execute_recipe_predictions_only_saved_objects_writes_minimal_prediction_bundle(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    result = execute_recipe(
        recipe=_recipe(benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5}),
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=fixture,
        provenance_payload={
            "compiler": {
                "output_spec": {
                    "export_format": "json",
                    "saved_objects": "predictions_only",
                    "provenance_fields": "standard",
                    "artifact_granularity": "aggregated",
                },
                "importance_spec": {"importance_method": "none"},
                "stat_test_spec": {"stat_test": "none"},
            }
        },
    )

    run_dir = tmp_path / result.run.artifact_subdir
    manifest = json.loads((run_dir / "manifest.json").read_text())
    artifact_manifest = json.loads((run_dir / "artifact_manifest.json").read_text())
    artifact_paths = {row["path"] for row in artifact_manifest["artifacts"]}

    assert manifest["saved_objects_effective"] == "predictions_only"
    assert manifest["metrics_file"] is None
    assert manifest["evaluation_summary_file"] is None
    assert "predictions.csv" in artifact_paths
    assert "metrics.json" not in artifact_paths
    assert not (run_dir / "metrics.json").exists()
    assert not (run_dir / "evaluation_summary.json").exists()
    assert not (run_dir / "data_preview.csv").exists()
    assert not (run_dir / "tuning_result.json").exists()


def test_execute_recipe_parquet_export_writes_parquet_artifacts(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    recipe = _recipe(benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5})
    result = execute_recipe(
        recipe=recipe,
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=fixture,
        provenance_payload={
            "compiler": {
                "output_spec": {
                    "export_format": "parquet",
                    "saved_objects": "predictions_and_metrics",
                    "provenance_fields": "full",
                    "artifact_granularity": "aggregated",
                },
                "importance_spec": {"importance_method": "none"},
                "stat_test_spec": {"stat_test": "none"},
            }
        },
    )

    run_dir = tmp_path / result.run.artifact_subdir
    manifest = json.loads((run_dir / "manifest.json").read_text())
    artifact_manifest = json.loads((run_dir / "artifact_manifest.json").read_text())
    artifact_paths = {row["path"] for row in artifact_manifest["artifacts"]}
    assert manifest["metrics_file"] == "metrics.parquet"
    assert (run_dir / "metrics.parquet").exists()
    assert (run_dir / "comparison_summary.parquet").exists()
    assert (run_dir / "predictions.parquet").exists()
    assert "metrics.parquet" in artifact_paths
    assert "comparison_summary.parquet" in artifact_paths
    assert "evaluation_summary.json" in artifact_paths
    assert not (run_dir / "data_preview.csv").exists()
    assert not (run_dir / "tuning_result.json").exists()


def test_execute_recipe_rejects_unimplemented_artifact_granularity(tmp_path: Path) -> None:
    with __import__("pytest").raises(ExecutionError, match="artifact_granularity='per_target'"):
        execute_recipe(
            recipe=_recipe(benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5}),
            preprocess=_preprocess_raw_only(),
            output_root=tmp_path,
            local_raw_source=Path("tests/fixtures/fred_md_ar_sample.csv"),
            provenance_payload={
                "compiler": {
                    "output_spec": {
                        "export_format": "json",
                        "saved_objects": "full_bundle",
                        "provenance_fields": "full",
                        "artifact_granularity": "per_target",
                    },
                    "importance_spec": {"importance_method": "none"},
                    "stat_test_spec": {"stat_test": "none"},
                }
            },
        )



def test_execute_recipe_full_provenance_hash_changes_with_recipe_config(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    provenance_payload = {
        "compiler": {
            "output_spec": {
                "export_format": "json",
                "saved_objects": "full_bundle",
                "provenance_fields": "full",
                "artifact_granularity": "aggregated",
            },
            "importance_spec": {"importance_method": "none"},
            "stat_test_spec": {"stat_test": "none"},
        }
    }
    base_recipe = build_recipe_spec(
        recipe_id="hash_probe",
        stage0=_stage0(),
        target="INDPRO",
        horizons=(1, 3),
        raw_dataset="fred_md",
        benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5},
        data_task_spec={"forecast_object": "point_mean"},
        training_spec={},
    )
    alt_recipe = build_recipe_spec(
        recipe_id="hash_probe",
        stage0=_stage0(),
        target="INDPRO",
        horizons=(1, 3),
        raw_dataset="fred_md",
        benchmark_config={"minimum_train_size": 7, "rolling_window_size": 5},
        data_task_spec={"forecast_object": "point_mean"},
        training_spec={},
    )

    base_result = execute_recipe(
        recipe=base_recipe,
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path / "base",
        local_raw_source=fixture,
        provenance_payload=provenance_payload,
    )
    alt_result = execute_recipe(
        recipe=alt_recipe,
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path / "alt",
        local_raw_source=fixture,
        provenance_payload=provenance_payload,
    )

    base_manifest = json.loads((tmp_path / "base" / base_result.run.artifact_subdir / "manifest.json").read_text())
    alt_manifest = json.loads((tmp_path / "alt" / alt_result.run.artifact_subdir / "manifest.json").read_text())

    assert len(base_manifest["git_commit"]) == 40
    assert base_manifest["package_version"] == "0.0.0+local"
    assert base_manifest["config_hash"] != alt_manifest["config_hash"]


def test_execute_recipe_stage6_extended_stat_tests(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    cases = [
        ("dm_hln", "stat_test_dm_hln.json", {"dependence_correction": "nw_hac"}),
        ("dm_modified", "stat_test_dm_modified.json", {"dependence_correction": "nw_hac_auto"}),
        ("mcs", "stat_test_mcs.json", {"dependence_correction": "block_bootstrap"}),
        ("enc_new", "stat_test_enc_new.json", {"dependence_correction": "nw_hac"}),
        ("mse_f", "stat_test_mse_f.json", {}),
        ("mse_t", "stat_test_mse_t.json", {"dependence_correction": "nw_hac"}),
        ("cpa", "stat_test_cpa.json", {"dependence_correction": "nw_hac_auto"}),
        ("rossi", "stat_test_rossi.json", {}),
        ("rolling_dm", "stat_test_rolling_dm.json", {}),
        ("reality_check", "stat_test_reality_check.json", {"dependence_correction": "block_bootstrap"}),
        ("spa", "stat_test_spa.json", {"dependence_correction": "block_bootstrap"}),
        ("full_residual_diagnostics", "stat_test_full_residual_diagnostics.json", {}),
        ("pesaran_timmermann", "stat_test_pesaran_timmermann.json", {}),
        ("binomial_hit", "stat_test_binomial_hit.json", {}),
    ]
    for idx, (stat_name, filename, extra) in enumerate(cases):
        out_root = tmp_path / f"case_{idx}_{stat_name}"
        result = execute_recipe(
            recipe=_recipe(benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5}),
            preprocess=_preprocess_raw_only(),
            output_root=out_root,
            local_raw_source=fixture,
            provenance_payload={"compiler": {"stat_test_spec": {"stat_test": stat_name, **extra}}},
        )
        run_dir = out_root / result.run.artifact_subdir
        manifest = json.loads((run_dir / "manifest.json").read_text())
        payload = json.loads((run_dir / filename).read_text())
        assert manifest["stat_test_file"] == filename
        assert manifest["stat_test_contract"] == "layer6_stat_test_split_v1"
        assert payload["stat_test"] == stat_name


def test_execute_recipe_stage6_stat_test_manifest_preserves_dependence_correction(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    result = execute_recipe(
        recipe=_recipe(benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5}),
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=fixture,
        provenance_payload={"compiler": {"stat_test_spec": {"stat_test": "dm_modified", "dependence_correction": "nw_hac_auto"}}},
    )
    manifest = json.loads((tmp_path / result.run.artifact_subdir / "manifest.json").read_text())
    assert manifest["stat_test_spec"]["dependence_correction"] == "nw_hac_auto"


def test_execute_recipe_stage6_split_stat_test_contract(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    result = execute_recipe(
        recipe=_recipe(benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5}),
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=fixture,
        provenance_payload={
            "compiler": {
                "stat_test_spec": {
                    "equal_predictive": "dm_modified",
                    "dependence_correction": "nw_hac_auto",
                    "overlap_handling": "evaluate_with_hac",
                    "test_scope": "per_target",
                }
            }
        },
    )
    run_dir = tmp_path / result.run.artifact_subdir
    manifest = json.loads((run_dir / "manifest.json").read_text())
    stat_tests = json.loads((run_dir / "stat_tests.json").read_text())
    artifact_manifest = json.loads((run_dir / "artifact_manifest.json").read_text())

    assert manifest["stat_test_contract"] == "layer6_stat_test_split_v1"
    assert manifest["stat_test_spec"]["stat_test"] == "none"
    assert manifest["stat_test_spec"]["equal_predictive"] == "dm_modified"
    assert manifest["stat_test_spec"]["test_scope"] == "per_target"
    assert manifest["stat_test_file"] == "stat_test_dm_modified.json"
    assert stat_tests["equal_predictive"]["stat_test"] == "dm_modified"
    assert any(item["path"] == "stat_tests.json" for item in artifact_manifest["artifacts"])



def test_execute_recipe_stage7_importance_methods(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    cases = [
        ("tree_shap", {"model_family": "random_forest"}, "importance_tree_shap.json"),
        ("kernel_shap", {"model_family": "ridge"}, "importance_kernel_shap.json"),
        ("linear_shap", {"model_family": "ridge"}, "importance_linear_shap.json"),
        ("permutation_importance", {"model_family": "random_forest"}, "importance_permutation_importance.json"),
        ("lime", {"model_family": "ridge"}, "importance_lime.json"),
        ("feature_ablation", {"model_family": "ridge"}, "importance_feature_ablation.json"),
        ("pdp", {"model_family": "ridge"}, "importance_pdp.json"),
        ("ice", {"model_family": "ridge"}, "importance_ice.json"),
        ("ale", {"model_family": "ridge"}, "importance_ale.json"),
        ("grouped_permutation", {"model_family": "random_forest"}, "importance_grouped_permutation.json"),
        ("importance_stability", {"model_family": "random_forest"}, "importance_stability.json"),
    ]
    for idx, (method, overrides, filename) in enumerate(cases):
        out_root = tmp_path / f"importance_{idx}_{method}"
        recipe = _recipe(model_family=overrides.get("model_family", "ridge"), feature_builder=overrides.get("feature_builder", "raw_feature_panel"), benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5})
        result = execute_recipe(
            recipe=recipe,
            preprocess=_preprocess_raw_only(),
            output_root=out_root,
            local_raw_source=fixture,
            provenance_payload={"compiler": {"importance_spec": {"importance_method": method}}},
        )
        run_dir = out_root / result.run.artifact_subdir
        manifest = json.loads((run_dir / "manifest.json").read_text())
        payload = json.loads((run_dir / filename).read_text())
        aggregate = json.loads((run_dir / "importance_artifacts.json").read_text())
        assert manifest["importance_contract"] == "layer7_importance_split_v1"
        assert manifest["importance_file"] == filename
        assert filename in manifest["importance_files"].values()
        assert aggregate["importance_contract"] == "layer7_importance_split_v1"
        assert payload["importance_method"] == method
        assert payload["importance_contract"] == "layer7_importance_split_v1"
        assert payload["feature_runtime_builder"] == "raw_feature_panel"
        assert payload["legacy_feature_builder"] == "raw_feature_panel"
        assert payload["feature_dispatch_source"] == "layer2_feature_blocks"



def test_execute_recipe_parallel_by_oos_date_dispatches_thread_pool(tmp_path: Path) -> None:
    """compute_mode=parallel_by_oos_date spins up a ThreadPoolExecutor for the
    OOS origin loop inside _rows_for_horizon and produces the same prediction
    set as the serial branch."""
    import pandas as pd
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    # baseline (serial) run to capture expected prediction shape
    serial_root = tmp_path / "serial"
    serial_root.mkdir()
    serial_result = execute_recipe(
        recipe=_recipe(framework="expanding", benchmark_config={"minimum_train_size": 5}),
        preprocess=_preprocess_raw_only(),
        output_root=serial_root,
        local_raw_source=fixture,
    )
    serial_preds = pd.read_csv(Path(serial_root) / serial_result.run.artifact_subdir / "predictions.csv")

    # parallel_by_oos_date run — patch ThreadPoolExecutor to count
    from macrocast.execution import build as build_mod
    original = build_mod.ThreadPoolExecutor
    calls: list[int] = []

    class _CountingExecutor(original):
        def __init__(self, *args, max_workers=None, **kwargs):
            calls.append(max_workers or 0)
            super().__init__(*args, max_workers=max_workers, **kwargs)

    parallel_root = tmp_path / "parallel"
    parallel_root.mkdir()
    build_mod.ThreadPoolExecutor = _CountingExecutor
    try:
        parallel_result = execute_recipe(
            recipe=_recipe(framework="expanding", benchmark_config={"minimum_train_size": 5}),
            preprocess=_preprocess_raw_only(),
            output_root=parallel_root,
            local_raw_source=fixture,
            provenance_payload={"compiler": {"compute_mode_spec": {"compute_mode": "parallel_by_oos_date"}}},
        )
    finally:
        build_mod.ThreadPoolExecutor = original

    parallel_preds = pd.read_csv(Path(parallel_root) / parallel_result.run.artifact_subdir / "predictions.csv")
    manifest = json.loads((Path(parallel_root) / parallel_result.run.artifact_subdir / "manifest.json").read_text())

    # At least one ThreadPoolExecutor was built with max_workers capped at 4.
    # Recipe has 2 horizons, each with many OOS origins, so we expect 2 pool constructions
    # (one per horizon iteration; serial horizon loop but parallel origin loop inside each).
    assert calls, "ThreadPoolExecutor was never created for parallel_by_oos_date"
    for n in calls:
        assert 1 <= n <= 4

    assert manifest["compute_mode_spec"]["compute_mode"] == "parallel_by_oos_date"
    # Parallel mode must yield the same prediction count as serial (origin order preserved).
    assert len(parallel_preds) == len(serial_preds)
    assert set(parallel_preds["horizon"].unique()) == set(serial_preds["horizon"].unique())
