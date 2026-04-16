from __future__ import annotations

import json
from pathlib import Path

from macrocast import (
    build_execution_spec,
    build_preprocess_contract,
    build_recipe_spec,
    build_run_spec,
    build_stage0_frame,
    execute_recipe,
)


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
    return build_stage0_frame(
        study_mode="single_path_benchmark_study",
        fixed_design={
            "dataset_adapter": "fred_md",
            "information_set": info_set,
            "sample_split": sample_split,
            "benchmark": benchmark,
            "evaluation_protocol": "point_forecast_core",
            "forecast_task": "single_target_point_forecast",
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
    quantile_level: float | None = None,
):
    data_task_spec = {"forecast_object": forecast_object}
    training_spec = {}
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
        tcode_policy="extra_preprocess_without_tcode",
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


def test_build_execution_spec_with_importance_context() -> None:
    recipe = _recipe()
    run = build_run_spec(recipe)
    preprocess = _preprocess_raw_only()

    execution = build_execution_spec(recipe=recipe, run=run, preprocess=preprocess)

    assert execution.recipe.recipe_id == "fred_md_rolling_ridge_raw_feature_panel"
    assert execution.run.run_id == run.run_id
    assert execution.recipe.stage0.fixed_design.sample_split == "rolling_window_oos"


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
    assert manifest["importance_file"] == "importance_minimal.json"
    assert importance["importance_method"] == "minimal_importance"
    assert importance["model_family"] == "ridge"
    assert len(importance["feature_importance"]) > 0


def test_execute_recipe_writes_minimal_importance_artifact_for_randomforest(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    result = execute_recipe(
        recipe=_recipe(model_family="randomforest", benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5}),
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=fixture,
        provenance_payload={"compiler": {"importance_spec": {"importance_method": "minimal_importance"}}},
    )

    run_dir = tmp_path / result.run.artifact_subdir
    importance = json.loads((run_dir / "importance_minimal.json").read_text())
    assert importance["model_family"] == "randomforest"
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
        feature_builder="autoreg_lagged_target",
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
        feature_builder="autoreg_lagged_target",
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
                    "point_metrics": "RMSE",
                    "relative_metrics": "relative_RMSE",
                    "direction_metrics": "directional_accuracy",
                    "regime_definition": "none",
                    "regime_use": "eval_only",
                    "regime_metrics": "all_main_metrics_by_regime",
                }
            }
        },
    )
    run_dir = tmp_path / result.run.artifact_subdir
    manifest = json.loads((run_dir / "manifest.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    h1 = metrics["metrics_by_horizon"]["h1"]
    assert manifest["evaluation_spec"]["relative_metrics"] == "relative_RMSE"
    assert "relative_rmse" in h1
    assert "relative_mae" in h1
    assert "benchmark_win_rate" in h1
    assert "directional_accuracy" in h1
    assert "sign_accuracy" in h1


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
                    "point_metrics": "MSFE",
                    "relative_metrics": "relative_MSFE",
                    "direction_metrics": "directional_accuracy",
                    "regime_definition": "NBER_recession",
                    "regime_use": "eval_only",
                    "regime_metrics": "all_main_metrics_by_regime",
                }
            }
        },
    )
    run_dir = tmp_path / result.run.artifact_subdir
    manifest = json.loads((run_dir / "manifest.json").read_text())
    regime = json.loads((run_dir / "regime_summary.json").read_text())
    assert manifest["regime_file"] == "regime_summary.json"
    assert regime["regime_definition"] == "NBER_recession"
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
                    "point_metrics": "MSFE",
                    "relative_metrics": "relative_MSFE",
                    "direction_metrics": "directional_accuracy",
                    "regime_definition": "user_defined_regime",
                    "regime_use": "eval_only",
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


def test_execute_recipe_runs_real_time_vintage_slice(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_sample.csv")
    result = execute_recipe(
        recipe=_recipe(
            model_family="ar",
            feature_builder="autoreg_lagged_target",
            framework="expanding",
            benchmark_config={"minimum_train_size": 5},
            info_set="real_time_vintage",
            data_vintage="2020-01",
        ),
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=fixture,
    )

    run_dir = tmp_path / result.run.artifact_subdir
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["raw_dataset"] == "fred_md"
    assert manifest["raw_artifact"].endswith("2020-01.csv")
    assert manifest["route_owner"] == "single_run"


def test_execute_recipe_runs_multi_target_slice(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    stage0 = build_stage0_frame(
        study_mode="single_path_benchmark_study",
        fixed_design={
            "dataset_adapter": "fred_md",
            "information_set": "revised_monthly",
            "sample_split": "expanding_window_oos",
            "benchmark": "zero_change",
            "evaluation_protocol": "point_forecast_core",
            "forecast_task": "multi_target_point_forecast",
        },
        comparison_contract={
            "information_set_policy": "identical",
            "sample_split_policy": "identical",
            "benchmark_policy": "identical",
            "evaluation_policy": "identical",
        },
        varying_design={"model_families": ("ar",), "feature_recipes": ("autoreg_lagged_target",), "horizons": ("h1",)},
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
    stage0 = build_stage0_frame(
        study_mode="single_path_benchmark_study",
        fixed_design={
            "dataset_adapter": "fred_md",
            "information_set": "revised_monthly",
            "sample_split": "expanding_window_oos",
            "benchmark": "zero_change",
            "evaluation_protocol": "point_forecast_core",
            "forecast_task": "single_target_point_forecast",
        },
        comparison_contract={
            "information_set_policy": "identical",
            "sample_split_policy": "identical",
            "benchmark_policy": "identical",
            "evaluation_policy": "identical",
        },
        varying_design={"model_families": ("ar",), "feature_recipes": ("autoreg_lagged_target",), "horizons": ("h1",)},
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
            "study_mode": "single_path_benchmark_study",
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
                "forecast_task": "single_target_point_forecast",
            },
            "varying_design": {
                "model_families": ["ar"],
                "feature_recipes": ["autoreg_lagged_target"],
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
    stage0 = build_stage0_frame(
        study_mode="single_path_benchmark_study",
        fixed_design={
            "dataset_adapter": "fred_md",
            "information_set": "revised_monthly",
            "sample_split": "expanding_window_oos",
            "benchmark": "zero_change",
            "evaluation_protocol": "point_forecast_core",
            "forecast_task": "multi_target_point_forecast",
        },
        comparison_contract={
            "information_set_policy": "identical",
            "sample_split_policy": "identical",
            "benchmark_policy": "identical",
            "evaluation_policy": "identical",
        },
        varying_design={"model_families": ("ar",), "feature_recipes": ("autoreg_lagged_target",), "horizons": ("h1",)},
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
        recipe=_recipe(model_family="ar", feature_builder="autoreg_lagged_target", benchmark_config={"minimum_train_size": 5}),
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


def test_execute_recipe_parallel_by_model_runs_multi_target_slice(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    stage0 = build_stage0_frame(
        study_mode="single_path_benchmark_study",
        fixed_design={
            "dataset_adapter": "fred_md",
            "information_set": "revised_monthly",
            "sample_split": "expanding_window_oos",
            "benchmark": "zero_change",
            "evaluation_protocol": "point_forecast_core",
            "forecast_task": "multi_target_point_forecast",
        },
        comparison_contract={
            "information_set_policy": "identical",
            "sample_split_policy": "identical",
            "benchmark_policy": "identical",
            "evaluation_policy": "identical",
        },
        varying_design={"model_families": ("ar",), "feature_recipes": ("autoreg_lagged_target",), "horizons": ("h1",)},
    )
    recipe = build_recipe_spec(
        recipe_id="fred_md_multi_parallel_model",
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
        provenance_payload={"compiler": {"compute_mode_spec": {"compute_mode": "parallel_by_model"}}},
    )
    run_dir = tmp_path / result.run.artifact_subdir
    manifest = json.loads((run_dir / "manifest.json").read_text())
    predictions = __import__("pandas").read_csv(run_dir / "predictions.csv")
    assert manifest["compute_mode_spec"]["compute_mode"] == "parallel_by_model"
    assert set(predictions["target"].unique()) == {"INDPRO", "RPI"}



def _preprocess_mean_impute_minmax_winsor() -> PreprocessContract:
    return build_preprocess_contract(
        target_transform_policy="raw_level",
        x_transform_policy="raw_level",
        tcode_policy="extra_preprocess_without_tcode",
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
        tcode_policy="extra_preprocess_without_tcode",
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


def _preprocess_lasso_select_contract() -> PreprocessContract:
    return build_preprocess_contract(
        target_transform_policy="raw_level",
        x_transform_policy="raw_level",
        tcode_policy="extra_preprocess_without_tcode",
        target_missing_policy="none",
        x_missing_policy="mean_impute",
        target_outlier_policy="none",
        x_outlier_policy="zscore_clip",
        scaling_policy="standard",
        dimensionality_reduction_policy="none",
        feature_selection_policy="lasso_select",
        preprocess_order="extra_only",
        preprocess_fit_scope="train_only",
        inverse_transform_policy="none",
        evaluation_scale="raw_level",
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
    assert manifest["prediction_rows"] > 0


def test_execute_recipe_supports_lasso_feature_selection_path(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    result = execute_recipe(
        recipe=_recipe(benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5}),
        preprocess=_preprocess_lasso_select_contract(),
        output_root=tmp_path,
        local_raw_source=fixture,
    )
    run_dir = tmp_path / result.run.artifact_subdir
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["preprocess_contract"]["feature_selection_policy"] == "lasso_select"
    assert manifest["prediction_rows"] > 0



def test_execute_recipe_supports_ols_autoreg_model(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    result = execute_recipe(
        recipe=_recipe(model_family="ols", feature_builder="autoreg_lagged_target", benchmark_config={"minimum_train_size": 5}),
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
        recipe=_recipe(model_family="xgboost", feature_builder="autoreg_lagged_target", benchmark_config={"minimum_train_size": 5}),
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
        recipe=_recipe(model_family="quantile_linear", feature_builder="autoreg_lagged_target", forecast_object="point_median", benchmark_config={"minimum_train_size": 5}),
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



def test_execute_recipe_supports_adaptivelasso_autoreg_model(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    result = execute_recipe(
        recipe=_recipe(model_family="adaptivelasso", feature_builder="autoreg_lagged_target", benchmark_config={"minimum_train_size": 5}),
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=fixture,
    )
    manifest = json.loads((tmp_path / result.run.artifact_subdir / "manifest.json").read_text())
    assert manifest["model_spec"]["model_family"] == "adaptivelasso"
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
        recipe=_recipe(model_family="huber", feature_builder="autoreg_lagged_target", benchmark_config={"minimum_train_size": 5}),
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



def test_execute_recipe_supports_adaptivelasso_raw_panel_model(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    result = execute_recipe(
        recipe=_recipe(model_family="adaptivelasso", feature_builder="raw_feature_panel", benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5}),
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=fixture,
    )
    manifest = json.loads((tmp_path / result.run.artifact_subdir / "manifest.json").read_text())
    assert manifest["model_spec"]["model_family"] == "adaptivelasso"
    assert manifest["prediction_rows"] > 0


def test_execute_recipe_supports_svr_linear_autoreg_model(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    result = execute_recipe(
        recipe=_recipe(model_family="svr_linear", feature_builder="autoreg_lagged_target", benchmark_config={"minimum_train_size": 5}),
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
        recipe=_recipe(model_family="svr_rbf", feature_builder="autoreg_lagged_target", benchmark_config={"minimum_train_size": 5}),
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
        recipe=_recipe(model_family="catboost", feature_builder="autoreg_lagged_target", benchmark_config={"minimum_train_size": 5}),
        preprocess=_preprocess_raw_only(),
        output_root=tmp_path,
        local_raw_source=fixture,
    )
    manifest = json.loads((tmp_path / result.run.artifact_subdir / "manifest.json").read_text())
    assert manifest["model_spec"]["model_family"] == "catboost"
    assert manifest["prediction_rows"] > 0



def test_execute_recipe_supports_pcr_raw_panel_model(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    recipe = _recipe(model_family="pcr", feature_builder="factor_pca", benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5})
    recipe = __import__("dataclasses").replace(recipe, training_spec={**recipe.training_spec, "fixed_factor_count": 2})
    result = execute_recipe(recipe=recipe, preprocess=_preprocess_raw_only(), output_root=tmp_path, local_raw_source=fixture)
    manifest = json.loads((tmp_path / result.run.artifact_subdir / "manifest.json").read_text())
    assert manifest["model_spec"]["model_family"] == "pcr"
    assert manifest["prediction_rows"] > 0


def test_execute_recipe_supports_pls_raw_panel_model(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    recipe = _recipe(model_family="pls", feature_builder="factor_pca", benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5})
    recipe = __import__("dataclasses").replace(recipe, training_spec={**recipe.training_spec, "fixed_factor_count": 2})
    result = execute_recipe(recipe=recipe, preprocess=_preprocess_raw_only(), output_root=tmp_path, local_raw_source=fixture)
    manifest = json.loads((tmp_path / result.run.artifact_subdir / "manifest.json").read_text())
    assert manifest["model_spec"]["model_family"] == "pls"
    assert manifest["prediction_rows"] > 0


def test_execute_recipe_supports_factor_augmented_linear_raw_panel_model(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/fred_md_ar_sample.csv")
    recipe = _recipe(model_family="factor_augmented_linear", feature_builder="factors_plus_AR", benchmark_config={"minimum_train_size": 7, "rolling_window_size": 7})
    recipe = __import__("dataclasses").replace(recipe, training_spec={**recipe.training_spec, "fixed_factor_count": 2, "factor_ar_lags": 2})
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
                    "export_format": "json+csv",
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
    assert manifest["output_spec"]["export_format"] == "json+csv"
    assert manifest["metrics_file"] == "metrics.json"
    assert manifest["metrics_files"]["csv"] == "metrics.csv"
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "comparison_summary.csv").exists()


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
    assert manifest["metrics_file"] == "metrics.parquet"
    assert (run_dir / "metrics.parquet").exists()
    assert (run_dir / "comparison_summary.parquet").exists()
    assert (run_dir / "predictions.parquet").exists()



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
        ("diagnostics_full", "stat_test_diagnostics_full.json", {}),
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
