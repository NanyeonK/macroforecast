from __future__ import annotations

import json
from pathlib import Path

from macrocast import (
    CompileValidationError,
    axis_governance_table,
    compile_recipe_dict,
    compile_recipe_yaml,
    get_canonical_layer_order,
    run_compiled_recipe,
)


def test_compile_minimal_importance_recipe_is_executable_for_ridge(tmp_path: Path) -> None:
    recipe = {
        "recipe_id": "importance-ridge-rolling",
        "path": {
            "0_meta": {"fixed_axes": {"study_mode": "single_path_benchmark_study"}},
            "1_data_task": {
                "fixed_axes": {"dataset": "fred_md", "info_set": "revised", "task": "single_target_point_forecast"},
                "leaf_config": {"target": "INDPRO", "horizons": [1, 3]},
            },
            "2_preprocessing": {"fixed_axes": {
                "target_transform_policy": "raw_level", "x_transform_policy": "raw_level", "tcode_policy": "raw_only",
                "target_missing_policy": "none", "x_missing_policy": "none", "target_outlier_policy": "none", "x_outlier_policy": "none",
                "scaling_policy": "none", "dimensionality_reduction_policy": "none", "feature_selection_policy": "none",
                "preprocess_order": "none", "preprocess_fit_scope": "not_applicable", "inverse_transform_policy": "none", "evaluation_scale": "raw_level"
            }},
            "3_training": {"fixed_axes": {
                "framework": "rolling", "benchmark_family": "zero_change", "feature_builder": "raw_feature_panel", "model_family": "ridge"
            }},
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"manifest_mode": "full", "benchmark_config": {"minimum_train_size": 5, "rolling_window_size": 5}}},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "minimal_importance"}},
        },
    }
    compile_result = compile_recipe_dict(recipe)
    assert compile_result.compiled.execution_status == "executable"
    execution = run_compiled_recipe(
        compile_result.compiled,
        output_root=tmp_path,
        local_raw_source=Path("tests/fixtures/fred_md_ar_sample.csv"),
    )
    manifest = json.loads((Path(execution.artifact_dir) / "manifest.json").read_text())
    assert manifest["importance_spec"]["importance_method"] == "minimal_importance"
    assert manifest["importance_file"] == "importance_minimal.json"


def test_compile_minimal_importance_recipe_is_executable_for_randomforest(tmp_path: Path) -> None:
    recipe = {
        "recipe_id": "importance-rf-rolling",
        "path": {
            "0_meta": {"fixed_axes": {"study_mode": "single_path_benchmark_study"}},
            "1_data_task": {
                "fixed_axes": {"dataset": "fred_md", "info_set": "revised", "task": "single_target_point_forecast"},
                "leaf_config": {"target": "INDPRO", "horizons": [1, 3]},
            },
            "2_preprocessing": {"fixed_axes": {
                "target_transform_policy": "raw_level", "x_transform_policy": "raw_level", "tcode_policy": "raw_only",
                "target_missing_policy": "none", "x_missing_policy": "none", "target_outlier_policy": "none", "x_outlier_policy": "none",
                "scaling_policy": "none", "dimensionality_reduction_policy": "none", "feature_selection_policy": "none",
                "preprocess_order": "none", "preprocess_fit_scope": "not_applicable", "inverse_transform_policy": "none", "evaluation_scale": "raw_level"
            }},
            "3_training": {"fixed_axes": {
                "framework": "rolling", "benchmark_family": "zero_change", "feature_builder": "raw_feature_panel", "model_family": "randomforest"
            }},
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"manifest_mode": "full", "benchmark_config": {"minimum_train_size": 5, "rolling_window_size": 5}}},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "minimal_importance"}},
        },
    }
    compile_result = compile_recipe_dict(recipe)
    assert compile_result.compiled.execution_status == "executable"
    execution = run_compiled_recipe(
        compile_result.compiled,
        output_root=tmp_path,
        local_raw_source=Path("tests/fixtures/fred_md_ar_sample.csv"),
    )
    manifest = json.loads((Path(execution.artifact_dir) / "manifest.json").read_text())
    assert manifest["importance_spec"]["importance_method"] == "minimal_importance"
    assert manifest["importance_file"] == "importance_minimal.json"


def test_axis_governance_table_marks_minimal_importance_operational() -> None:
    table = axis_governance_table()
    by_name = {row["axis_name"]: row for row in table}
    assert by_name["importance_method"]["current_status"]["minimal_importance"] == "operational"


def test_compile_dm_recipe_is_executable() -> None:
    compile_result = compile_recipe_yaml("examples/recipes/preprocessing-ablation.yaml")
    assert compile_result.compiled.execution_status == "executable"


def test_compile_recipe_rejects_incompatible_preprocessing_without_silent_fallback() -> None:
    bad_recipe = {
        "recipe_id": "bad-importance-ar",
        "path": {
            "0_meta": {"fixed_axes": {"study_mode": "single_path_benchmark_study"}},
            "1_data_task": {
                "fixed_axes": {"dataset": "fred_md", "info_set": "revised", "task": "single_target_point_forecast"},
                "leaf_config": {"target": "INDPRO", "horizons": [1, 3]},
            },
            "2_preprocessing": {"fixed_axes": {
                "target_transform_policy": "raw_level", "x_transform_policy": "raw_level", "tcode_policy": "raw_only",
                "target_missing_policy": "none", "x_missing_policy": "none", "target_outlier_policy": "none", "x_outlier_policy": "none",
                "scaling_policy": "none", "dimensionality_reduction_policy": "none", "feature_selection_policy": "none",
                "preprocess_order": "none", "preprocess_fit_scope": "not_applicable", "inverse_transform_policy": "none", "evaluation_scale": "raw_level"
            }},
            "3_training": {"fixed_axes": {
                "framework": "rolling", "benchmark_family": "zero_change", "feature_builder": "autoreg_lagged_target", "model_family": "ar"
            }},
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"manifest_mode": "full", "benchmark_config": {"minimum_train_size": 5, "rolling_window_size": 5}}},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "minimal_importance"}},
        },
    }
    compile_result = compile_recipe_dict(bad_recipe)
    assert compile_result.compiled.execution_status == "executable"
    with __import__("pytest").raises(Exception):
        run_compiled_recipe(
            compile_result.compiled,
            output_root=Path("/tmp/macrocast-importance-ar"),
            local_raw_source=Path("tests/fixtures/fred_md_ar_sample.csv"),
        )


def test_canonical_layer_order_is_fixed() -> None:
    assert get_canonical_layer_order() == (
        "0_meta",
        "1_data_task",
        "2_preprocessing",
        "3_training",
        "4_evaluation",
        "5_output_provenance",
        "6_stat_tests",
        "7_importance",
    )


def test_axis_governance_table_marks_cw_operational() -> None:
    table = axis_governance_table()
    by_name = {row["axis_name"]: row for row in table}
    assert by_name["stat_test"]["current_status"]["cw"] == "operational"


def test_compile_cw_recipe_is_executable_and_writes_artifact(tmp_path: Path) -> None:
    recipe = {
        "recipe_id": "cw-ridge-rolling",
        "path": {
            "0_meta": {"fixed_axes": {"study_mode": "single_path_benchmark_study"}},
            "1_data_task": {
                "fixed_axes": {"dataset": "fred_md", "info_set": "revised", "task": "single_target_point_forecast"},
                "leaf_config": {"target": "INDPRO", "horizons": [1, 3]},
            },
            "2_preprocessing": {"fixed_axes": {
                "target_transform_policy": "raw_level", "x_transform_policy": "raw_level", "tcode_policy": "raw_only",
                "target_missing_policy": "none", "x_missing_policy": "none", "target_outlier_policy": "none", "x_outlier_policy": "none",
                "scaling_policy": "none", "dimensionality_reduction_policy": "none", "feature_selection_policy": "none",
                "preprocess_order": "none", "preprocess_fit_scope": "not_applicable", "inverse_transform_policy": "none", "evaluation_scale": "raw_level"
            }},
            "3_training": {"fixed_axes": {
                "framework": "rolling", "benchmark_family": "zero_change", "feature_builder": "raw_feature_panel", "model_family": "ridge"
            }},
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"manifest_mode": "full", "benchmark_config": {"minimum_train_size": 5, "rolling_window_size": 5}}},
            "6_stat_tests": {"fixed_axes": {"stat_test": "cw"}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }
    compile_result = compile_recipe_dict(recipe)
    assert compile_result.compiled.execution_status == "executable"
    execution = run_compiled_recipe(
        compile_result.compiled,
        output_root=tmp_path,
        local_raw_source=Path("tests/fixtures/fred_md_ar_sample.csv"),
    )
    manifest = json.loads((Path(execution.artifact_dir) / "manifest.json").read_text())
    assert manifest["stat_test_spec"]["stat_test"] == "cw"
    assert manifest["stat_test_file"] == "stat_test_cw.json"


def test_axis_governance_table_marks_custom_benchmark_operational() -> None:
    table = axis_governance_table()
    by_name = {row["axis_name"]: row for row in table}
    assert by_name["benchmark_family"]["current_status"]["custom_benchmark"] == "operational"


def test_compile_custom_benchmark_recipe_is_executable(tmp_path: Path) -> None:
    plugin_path = tmp_path / "custom_benchmark_plugin.py"
    plugin_path.write_text(
        "def custom_benchmark(train, horizon, benchmark_config):\n"
        "    return float(train.iloc[-1]) + float(horizon)\n",
        encoding="utf-8",
    )
    recipe = {
        "recipe_id": "custom-benchmark-ridge-expanding",
        "path": {
            "0_meta": {"fixed_axes": {"study_mode": "single_path_benchmark_study"}},
            "1_data_task": {
                "fixed_axes": {"dataset": "fred_md", "info_set": "revised", "task": "single_target_point_forecast"},
                "leaf_config": {"target": "INDPRO", "horizons": [1, 3]},
            },
            "2_preprocessing": {"fixed_axes": {
                "target_transform_policy": "raw_level", "x_transform_policy": "raw_level", "tcode_policy": "raw_only",
                "target_missing_policy": "none", "x_missing_policy": "none", "target_outlier_policy": "none", "x_outlier_policy": "none",
                "scaling_policy": "none", "dimensionality_reduction_policy": "none", "feature_selection_policy": "none",
                "preprocess_order": "none", "preprocess_fit_scope": "not_applicable", "inverse_transform_policy": "none", "evaluation_scale": "raw_level"
            }},
            "3_training": {"fixed_axes": {
                "framework": "expanding", "benchmark_family": "custom_benchmark", "feature_builder": "autoreg_lagged_target", "model_family": "ar"
            }},
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"manifest_mode": "full", "benchmark_config": {"minimum_train_size": 5, "plugin_path": str(plugin_path), "callable_name": "custom_benchmark"}}},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }
    compile_result = compile_recipe_dict(recipe)
    assert compile_result.compiled.execution_status == "executable"
    execution = run_compiled_recipe(
        compile_result.compiled,
        output_root=tmp_path,
        local_raw_source=Path("tests/fixtures/fred_md_ar_sample.csv"),
    )
    manifest = json.loads((Path(execution.artifact_dir) / "manifest.json").read_text())
    assert manifest["benchmark_name"] == "custom_benchmark"
    assert manifest["benchmark_spec"]["plugin_path"] == str(plugin_path)


def test_compile_custom_benchmark_recipe_requires_plugin_contract() -> None:
    recipe = {
        "recipe_id": "custom-benchmark-missing-plugin",
        "path": {
            "0_meta": {"fixed_axes": {"study_mode": "single_path_benchmark_study"}},
            "1_data_task": {
                "fixed_axes": {"dataset": "fred_md", "info_set": "revised", "task": "single_target_point_forecast"},
                "leaf_config": {"target": "INDPRO", "horizons": [1, 3]},
            },
            "2_preprocessing": {"fixed_axes": {
                "target_transform_policy": "raw_level", "x_transform_policy": "raw_level", "tcode_policy": "raw_only",
                "target_missing_policy": "none", "x_missing_policy": "none", "target_outlier_policy": "none", "x_outlier_policy": "none",
                "scaling_policy": "none", "dimensionality_reduction_policy": "none", "feature_selection_policy": "none",
                "preprocess_order": "none", "preprocess_fit_scope": "not_applicable", "inverse_transform_policy": "none", "evaluation_scale": "raw_level"
            }},
            "3_training": {"fixed_axes": {
                "framework": "expanding", "benchmark_family": "custom_benchmark", "feature_builder": "autoreg_lagged_target", "model_family": "ar"
            }},
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"manifest_mode": "full", "benchmark_config": {"minimum_train_size": 5}}},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }
    with __import__("pytest").raises(CompileValidationError):
        compile_recipe_dict(recipe)


def test_compile_recipe_writes_comparison_summary_artifact(tmp_path: Path) -> None:
    recipe = compile_recipe_yaml("examples/recipes/model-benchmark.yaml")
    execution = run_compiled_recipe(
        recipe.compiled,
        output_root=tmp_path,
        local_raw_source=Path("tests/fixtures/fred_md_ar_sample.csv"),
    )
    manifest = json.loads((Path(execution.artifact_dir) / "manifest.json").read_text())
    comparison = json.loads((Path(execution.artifact_dir) / "comparison_summary.json").read_text())
    assert manifest["comparison_file"] == "comparison_summary.json"
    assert comparison["benchmark_name"] == manifest["benchmark_name"]


def test_axis_governance_table_marks_robust_scaling_operational() -> None:
    table = axis_governance_table()
    by_name = {row["axis_name"]: row for row in table}
    assert by_name["scaling_policy"]["current_status"]["robust"] == "operational"


def test_compile_robust_scaling_recipe_is_executable(tmp_path: Path) -> None:
    recipe = {
        "recipe_id": "robust-scaling-ridge-rolling",
        "path": {
            "0_meta": {"fixed_axes": {"study_mode": "single_path_benchmark_study"}},
            "1_data_task": {
                "fixed_axes": {"dataset": "fred_md", "info_set": "revised", "task": "single_target_point_forecast"},
                "leaf_config": {"target": "INDPRO", "horizons": [1, 3]},
            },
            "2_preprocessing": {"fixed_axes": {
                "target_transform_policy": "raw_level", "x_transform_policy": "raw_level", "tcode_policy": "extra_preprocess_without_tcode",
                "target_missing_policy": "none", "x_missing_policy": "em_impute", "target_outlier_policy": "none", "x_outlier_policy": "none",
                "scaling_policy": "robust", "dimensionality_reduction_policy": "none", "feature_selection_policy": "none",
                "preprocess_order": "extra_only", "preprocess_fit_scope": "train_only", "inverse_transform_policy": "none", "evaluation_scale": "raw_level"
            }},
            "3_training": {"fixed_axes": {
                "framework": "rolling", "benchmark_family": "zero_change", "feature_builder": "raw_feature_panel", "model_family": "ridge"
            }},
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"manifest_mode": "full", "benchmark_config": {"minimum_train_size": 5, "rolling_window_size": 5}}},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }
    compile_result = compile_recipe_dict(recipe)
    assert compile_result.compiled.execution_status == "executable"
    execution = run_compiled_recipe(
        compile_result.compiled,
        output_root=tmp_path,
        local_raw_source=Path("tests/fixtures/fred_md_raw_panel_missing.csv"),
    )
    manifest = json.loads((Path(execution.artifact_dir) / "manifest.json").read_text())
    assert manifest["preprocess_contract"]["scaling_policy"] == "robust"


def test_compile_lasso_minimal_importance_recipe_is_executable(tmp_path: Path) -> None:
    recipe = {
        "recipe_id": "importance-lasso-rolling",
        "path": {
            "0_meta": {"fixed_axes": {"study_mode": "single_path_benchmark_study"}},
            "1_data_task": {
                "fixed_axes": {"dataset": "fred_md", "info_set": "revised", "task": "single_target_point_forecast"},
                "leaf_config": {"target": "INDPRO", "horizons": [1, 3]},
            },
            "2_preprocessing": {"fixed_axes": {
                "target_transform_policy": "raw_level", "x_transform_policy": "raw_level", "tcode_policy": "raw_only",
                "target_missing_policy": "none", "x_missing_policy": "none", "target_outlier_policy": "none", "x_outlier_policy": "none",
                "scaling_policy": "none", "dimensionality_reduction_policy": "none", "feature_selection_policy": "none",
                "preprocess_order": "none", "preprocess_fit_scope": "not_applicable", "inverse_transform_policy": "none", "evaluation_scale": "raw_level"
            }},
            "3_training": {"fixed_axes": {
                "framework": "rolling", "benchmark_family": "zero_change", "feature_builder": "raw_feature_panel", "model_family": "lasso"
            }},
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"manifest_mode": "full", "benchmark_config": {"minimum_train_size": 5, "rolling_window_size": 5}}},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "minimal_importance"}},
        },
    }
    compile_result = compile_recipe_dict(recipe)
    assert compile_result.compiled.execution_status == "executable"
    execution = run_compiled_recipe(
        compile_result.compiled,
        output_root=tmp_path,
        local_raw_source=Path("tests/fixtures/fred_md_ar_sample.csv"),
    )
    manifest = json.loads((Path(execution.artifact_dir) / "manifest.json").read_text())
    assert manifest["importance_spec"]["importance_method"] == "minimal_importance"
    assert manifest["importance_file"] == "importance_minimal.json"


def test_compile_real_time_recipe_requires_data_vintage() -> None:
    recipe = {
        "recipe_id": "real-time-missing-vintage",
        "path": {
            "0_meta": {"fixed_axes": {"study_mode": "single_path_benchmark_study"}},
            "1_data_task": {
                "fixed_axes": {"dataset": "fred_md", "info_set": "real_time", "task": "single_target_point_forecast"},
                "leaf_config": {"target": "INDPRO", "horizons": [1, 3]},
            },
            "2_preprocessing": {"fixed_axes": {
                "target_transform_policy": "raw_level", "x_transform_policy": "raw_level", "tcode_policy": "raw_only",
                "target_missing_policy": "none", "x_missing_policy": "none", "target_outlier_policy": "none", "x_outlier_policy": "none",
                "scaling_policy": "none", "dimensionality_reduction_policy": "none", "feature_selection_policy": "none",
                "preprocess_order": "none", "preprocess_fit_scope": "not_applicable", "inverse_transform_policy": "none", "evaluation_scale": "raw_level"
            }},
            "3_training": {"fixed_axes": {
                "framework": "expanding", "benchmark_family": "zero_change", "feature_builder": "autoreg_lagged_target", "model_family": "ar"
            }},
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"manifest_mode": "full", "benchmark_config": {"minimum_train_size": 5}}},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }
    with __import__("pytest").raises(CompileValidationError):
        compile_recipe_dict(recipe)


def test_compile_real_time_recipe_is_executable(tmp_path: Path) -> None:
    recipe = {
        "recipe_id": "real-time-expanding-ar",
        "path": {
            "0_meta": {"fixed_axes": {"study_mode": "single_path_benchmark_study"}},
            "1_data_task": {
                "fixed_axes": {"dataset": "fred_md", "info_set": "real_time", "task": "single_target_point_forecast"},
                "leaf_config": {"target": "INDPRO", "horizons": [1, 3], "data_vintage": "2020-01"},
            },
            "2_preprocessing": {"fixed_axes": {
                "target_transform_policy": "raw_level", "x_transform_policy": "raw_level", "tcode_policy": "raw_only",
                "target_missing_policy": "none", "x_missing_policy": "none", "target_outlier_policy": "none", "x_outlier_policy": "none",
                "scaling_policy": "none", "dimensionality_reduction_policy": "none", "feature_selection_policy": "none",
                "preprocess_order": "none", "preprocess_fit_scope": "not_applicable", "inverse_transform_policy": "none", "evaluation_scale": "raw_level"
            }},
            "3_training": {"fixed_axes": {
                "framework": "expanding", "benchmark_family": "zero_change", "feature_builder": "autoreg_lagged_target", "model_family": "ar"
            }},
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"manifest_mode": "full", "benchmark_config": {"minimum_train_size": 5}}},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }
    compile_result = compile_recipe_dict(recipe)
    assert compile_result.compiled.execution_status == "executable"
    execution = run_compiled_recipe(
        compile_result.compiled,
        output_root=tmp_path,
        local_raw_source=Path("tests/fixtures/fred_md_sample.csv"),
    )
    manifest = json.loads((Path(execution.artifact_dir) / "manifest.json").read_text())
    assert manifest["raw_artifact"].endswith("2020-01.csv")


def test_compile_wrapper_bundle_requires_wrapper_metadata() -> None:
    recipe = {
        "recipe_id": "wrapper-missing-metadata",
        "path": {
            "0_meta": {"fixed_axes": {"study_mode": "orchestrated_bundle_study"}},
            "1_data_task": {
                "fixed_axes": {"dataset": "fred_md", "info_set": "revised", "task": "single_target_point_forecast"},
                "leaf_config": {"target": "INDPRO", "horizons": [1, 3]},
            },
            "2_preprocessing": {"fixed_axes": {
                "target_transform_policy": "raw_level", "x_transform_policy": "raw_level", "tcode_policy": "raw_only",
                "target_missing_policy": "none", "x_missing_policy": "none", "target_outlier_policy": "none", "x_outlier_policy": "none",
                "scaling_policy": "none", "dimensionality_reduction_policy": "none", "feature_selection_policy": "none",
                "preprocess_order": "none", "preprocess_fit_scope": "not_applicable", "inverse_transform_policy": "none", "evaluation_scale": "raw_level"
            }},
            "3_training": {"fixed_axes": {
                "framework": "expanding", "benchmark_family": "zero_change", "feature_builder": "autoreg_lagged_target", "model_family": "ar"
            }},
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"manifest_mode": "full", "benchmark_config": {"minimum_train_size": 5}}},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }
    with __import__("pytest").raises(CompileValidationError):
        compile_recipe_dict(recipe)


def test_compile_wrapper_bundle_emits_handoff_contract() -> None:
    recipe = {
        "recipe_id": "wrapper-benchmark-suite",
        "path": {
            "0_meta": {"fixed_axes": {"study_mode": "orchestrated_bundle_study"}, "leaf_config": {"wrapper_family": "benchmark_suite", "bundle_label": "fred-md-baselines"}},
            "1_data_task": {
                "fixed_axes": {"dataset": "fred_md", "info_set": "revised", "task": "single_target_point_forecast"},
                "leaf_config": {"target": "INDPRO", "horizons": [1, 3]},
            },
            "2_preprocessing": {"fixed_axes": {
                "target_transform_policy": "raw_level", "x_transform_policy": "raw_level", "tcode_policy": "raw_only",
                "target_missing_policy": "none", "x_missing_policy": "none", "target_outlier_policy": "none", "x_outlier_policy": "none",
                "scaling_policy": "none", "dimensionality_reduction_policy": "none", "feature_selection_policy": "none",
                "preprocess_order": "none", "preprocess_fit_scope": "not_applicable", "inverse_transform_policy": "none", "evaluation_scale": "raw_level"
            }},
            "3_training": {"fixed_axes": {
                "framework": "expanding", "benchmark_family": "zero_change", "feature_builder": "autoreg_lagged_target", "model_family": "ar"
            }},
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"manifest_mode": "full", "benchmark_config": {"minimum_train_size": 5}}},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }
    compile_result = compile_recipe_dict(recipe)
    assert compile_result.compiled.execution_status == "representable_but_not_executable"
    handoff = compile_result.manifest["wrapper_handoff"]
    tree_context = compile_result.manifest["tree_context"]
    assert handoff["wrapper_family"] == "benchmark_suite"
    assert handoff["bundle_label"] == "fred-md-baselines"
    assert handoff["route_owner"] == "wrapper"
    assert handoff["execution_posture"] == "wrapper_bundle_plan"
    assert tree_context["route_owner"] == "wrapper"
    assert tree_context["fixed_axes"]["study_mode"] == "orchestrated_bundle_study"
    assert tree_context["leaf_config"]["bundle_label"] == "fred-md-baselines"


def test_compile_multi_target_recipe_requires_targets() -> None:
    recipe = {
        "recipe_id": "multi-target-missing-targets",
        "path": {
            "0_meta": {"fixed_axes": {"study_mode": "single_path_benchmark_study"}},
            "1_data_task": {
                "fixed_axes": {"dataset": "fred_md", "info_set": "revised", "task": "multi_target_point_forecast"},
                "leaf_config": {"target": "INDPRO", "horizons": [1, 3]},
            },
            "2_preprocessing": {"fixed_axes": {
                "target_transform_policy": "raw_level", "x_transform_policy": "raw_level", "tcode_policy": "raw_only",
                "target_missing_policy": "none", "x_missing_policy": "none", "target_outlier_policy": "none", "x_outlier_policy": "none",
                "scaling_policy": "none", "dimensionality_reduction_policy": "none", "feature_selection_policy": "none",
                "preprocess_order": "none", "preprocess_fit_scope": "not_applicable", "inverse_transform_policy": "none", "evaluation_scale": "raw_level"
            }},
            "3_training": {"fixed_axes": {
                "framework": "expanding", "benchmark_family": "zero_change", "feature_builder": "autoreg_lagged_target", "model_family": "ar"
            }},
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"manifest_mode": "full", "benchmark_config": {"minimum_train_size": 5}}},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }
    with __import__("pytest").raises(CompileValidationError):
        compile_recipe_dict(recipe)


def test_compile_multi_target_recipe_is_executable(tmp_path: Path) -> None:
    recipe = {
        "recipe_id": "multi-target-expanding-ar",
        "path": {
            "0_meta": {"fixed_axes": {"study_mode": "single_path_benchmark_study"}},
            "1_data_task": {
                "fixed_axes": {"dataset": "fred_md", "info_set": "revised", "task": "multi_target_point_forecast"},
                "leaf_config": {"targets": ["INDPRO", "RPI"], "horizons": [1, 3]},
            },
            "2_preprocessing": {"fixed_axes": {
                "target_transform_policy": "raw_level", "x_transform_policy": "raw_level", "tcode_policy": "raw_only",
                "target_missing_policy": "none", "x_missing_policy": "none", "target_outlier_policy": "none", "x_outlier_policy": "none",
                "scaling_policy": "none", "dimensionality_reduction_policy": "none", "feature_selection_policy": "none",
                "preprocess_order": "none", "preprocess_fit_scope": "not_applicable", "inverse_transform_policy": "none", "evaluation_scale": "raw_level"
            }},
            "3_training": {"fixed_axes": {
                "framework": "expanding", "benchmark_family": "zero_change", "feature_builder": "autoreg_lagged_target", "model_family": "ar"
            }},
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"manifest_mode": "full", "benchmark_config": {"minimum_train_size": 5}}},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }
    compile_result = compile_recipe_dict(recipe)
    assert compile_result.compiled.execution_status == "executable"
    execution = run_compiled_recipe(
        compile_result.compiled,
        output_root=tmp_path,
        local_raw_source=Path("tests/fixtures/fred_md_ar_sample.csv"),
    )
    manifest = json.loads((Path(execution.artifact_dir) / "manifest.json").read_text())
    assert manifest["targets"] == ["INDPRO", "RPI"]
    assert manifest["tree_context"]["fixed_design"]["forecast_task"] == "multi_target_point_forecast"
    assert manifest["tree_context"]["leaf_config"]["targets"] == ["INDPRO", "RPI"]

def test_compile_tree_context_groups_fixed_and_sweep_axes() -> None:
    recipe = {
        "recipe_id": "tree-context-controlled-variation",
        "path": {
            "0_meta": {"fixed_axes": {"study_mode": "controlled_variation_study"}},
            "1_data_task": {
                "fixed_axes": {"dataset": "fred_md", "info_set": "revised", "task": "single_target_point_forecast"},
                "leaf_config": {"target": "INDPRO", "horizons": [1, 3]},
            },
            "2_preprocessing": {"fixed_axes": {
                "target_transform_policy": "raw_level", "x_transform_policy": "raw_level", "tcode_policy": "raw_only",
                "target_missing_policy": "none", "x_missing_policy": "none", "target_outlier_policy": "none", "x_outlier_policy": "none",
                "scaling_policy": "none", "dimensionality_reduction_policy": "none", "feature_selection_policy": "none",
                "preprocess_order": "none", "preprocess_fit_scope": "not_applicable", "inverse_transform_policy": "none", "evaluation_scale": "raw_level"
            }},
            "3_training": {
                "fixed_axes": {"framework": "expanding", "benchmark_family": "zero_change", "feature_builder": "autoreg_lagged_target"},
                "sweep_axes": {"model_family": ["ar", "ridge"]},
            },
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"manifest_mode": "full", "benchmark_config": {"minimum_train_size": 5}}},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }
    compile_result = compile_recipe_dict(recipe)
    tree_context = compile_result.manifest["tree_context"]
    assert compile_result.compiled.execution_status == "representable_but_not_executable"
    assert tree_context["study_mode"] == "controlled_variation_study"
    assert tree_context["route_owner"] == "single_run"
    assert tree_context["fixed_design"]["dataset_adapter"] == "fred_md"
    assert tree_context["varying_design"]["model_families"] == ["ar", "ridge"]
    assert tree_context["fixed_axes"]["framework"] == "expanding"
    assert tree_context["sweep_axes"]["model_family"] == ["ar", "ridge"]
    assert tree_context["leaf_config"]["horizons"] == [1, 3]




def test_axis_governance_table_includes_experiment_unit_axis() -> None:
    table = axis_governance_table()
    by_name = {row["axis_name"]: row for row in table}
    assert by_name["experiment_unit"]["current_status"]["single_target_model_grid"] == "operational"
    assert by_name["experiment_unit"]["current_status"]["benchmark_suite"] == "planned"


def test_compile_recipe_preserves_explicit_experiment_unit() -> None:
    recipe = {
        "recipe_id": "experiment-unit-single-model",
        "path": {
            "0_meta": {"fixed_axes": {
                "study_mode": "single_path_benchmark_study",
                "experiment_unit": "single_target_single_model",
            }},
            "1_data_task": {
                "fixed_axes": {"dataset": "fred_md", "info_set": "revised", "task": "single_target_point_forecast"},
                "leaf_config": {"target": "INDPRO", "horizons": [1, 3]},
            },
            "2_preprocessing": {"fixed_axes": {
                "target_transform_policy": "raw_level", "x_transform_policy": "raw_level", "tcode_policy": "raw_only",
                "target_missing_policy": "none", "x_missing_policy": "none", "target_outlier_policy": "none", "x_outlier_policy": "none",
                "scaling_policy": "none", "dimensionality_reduction_policy": "none", "feature_selection_policy": "none",
                "preprocess_order": "none", "preprocess_fit_scope": "not_applicable", "inverse_transform_policy": "none", "evaluation_scale": "raw_level"
            }},
            "3_training": {"fixed_axes": {
                "framework": "expanding", "benchmark_family": "zero_change", "feature_builder": "autoreg_lagged_target", "model_family": "ar"
            }},
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"manifest_mode": "full", "benchmark_config": {"minimum_train_size": 5}}},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }
    compile_result = compile_recipe_dict(recipe)
    assert compile_result.manifest["tree_context"]["experiment_unit"] == "single_target_single_model"
    assert compile_result.compiled.stage0.experiment_unit == "single_target_single_model"



def test_axis_governance_table_includes_axis_type() -> None:
    table = axis_governance_table()
    by_name = {row["axis_name"]: row for row in table}
    assert by_name["axis_type"]["current_status"]["fixed"] == "operational"
    assert by_name["axis_type"]["current_status"]["nested_sweep"] == "planned"


def test_compile_warns_when_fixed_policy_axis_is_placed_in_sweep_axes() -> None:
    recipe = {
        "recipe_id": "fixed-policy-swept-axis",
        "path": {
            "0_meta": {"fixed_axes": {"study_mode": "single_path_benchmark_study"}},
            "1_data_task": {
                "fixed_axes": {"dataset": "fred_md", "info_set": "revised", "task": "single_target_point_forecast"},
                "leaf_config": {"target": "INDPRO", "horizons": [1, 3]},
            },
            "2_preprocessing": {"fixed_axes": {
                "target_transform_policy": "raw_level", "x_transform_policy": "raw_level", "tcode_policy": "raw_only",
                "target_missing_policy": "none", "x_missing_policy": "none", "target_outlier_policy": "none", "x_outlier_policy": "none",
                "scaling_policy": "none", "dimensionality_reduction_policy": "none", "feature_selection_policy": "none",
                "preprocess_order": "none", "preprocess_fit_scope": "not_applicable", "inverse_transform_policy": "none", "evaluation_scale": "raw_level"
            }},
            "3_training": {
                "fixed_axes": {
                    "framework": "expanding", "benchmark_family": "zero_change", "model_family": "ar"
                },
                "sweep_axes": {
                    "feature_builder": ["autoreg_lagged_target", "raw_feature_panel"]
                },
            },
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"manifest_mode": "full", "benchmark_config": {"minimum_train_size": 5}}},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }
    compile_result = compile_recipe_dict(recipe)
    assert any("fixed-policy axis 'feature_builder' placed in sweep_axes" in warning for warning in compile_result.manifest["warnings"])
    assert compile_result.compiled.execution_status == "representable_but_not_executable"



def test_axis_governance_table_includes_registry_type() -> None:
    table = axis_governance_table()
    by_name = {row["axis_name"]: row for row in table}
    assert by_name["registry_type"]["current_status"]["enum_registry"] == "operational"
    assert by_name["registry_type"]["current_status"]["custom_plugin"] == "planned"



def test_axis_governance_table_includes_reproducibility_mode() -> None:
    table = axis_governance_table()
    by_name = {row["axis_name"]: row for row in table}
    assert by_name["reproducibility_mode"]["current_status"]["seeded_reproducible"] == "operational"
    assert by_name["reproducibility_mode"]["current_status"]["strict_reproducible"] == "planned"


def test_compile_seeded_reproducible_requires_random_seed() -> None:
    recipe = {
        "recipe_id": "seed-required",
        "path": {
            "0_meta": {"fixed_axes": {"study_mode": "single_path_benchmark_study", "reproducibility_mode": "seeded_reproducible"}},
            "1_data_task": {
                "fixed_axes": {"dataset": "fred_md", "info_set": "revised", "task": "single_target_point_forecast"},
                "leaf_config": {"target": "INDPRO", "horizons": [1, 3]},
            },
            "2_preprocessing": {"fixed_axes": {
                "target_transform_policy": "raw_level", "x_transform_policy": "raw_level", "tcode_policy": "raw_only",
                "target_missing_policy": "none", "x_missing_policy": "none", "target_outlier_policy": "none", "x_outlier_policy": "none",
                "scaling_policy": "none", "dimensionality_reduction_policy": "none", "feature_selection_policy": "none",
                "preprocess_order": "none", "preprocess_fit_scope": "not_applicable", "inverse_transform_policy": "none", "evaluation_scale": "raw_level"
            }},
            "3_training": {"fixed_axes": {
                "framework": "expanding", "benchmark_family": "zero_change", "feature_builder": "autoreg_lagged_target", "model_family": "ar"
            }},
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"manifest_mode": "full", "benchmark_config": {"minimum_train_size": 5}}},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }
    with __import__("pytest").raises(CompileValidationError):
        compile_recipe_dict(recipe)


def test_compile_reproducibility_spec_preserved_in_manifest() -> None:
    recipe = {
        "recipe_id": "seeded-provenance",
        "path": {
            "0_meta": {"fixed_axes": {"study_mode": "single_path_benchmark_study", "reproducibility_mode": "seeded_reproducible"}},
            "1_data_task": {
                "fixed_axes": {"dataset": "fred_md", "info_set": "revised", "task": "single_target_point_forecast"},
                "leaf_config": {"target": "INDPRO", "horizons": [1, 3], "random_seed": 42},
            },
            "2_preprocessing": {"fixed_axes": {
                "target_transform_policy": "raw_level", "x_transform_policy": "raw_level", "tcode_policy": "raw_only",
                "target_missing_policy": "none", "x_missing_policy": "none", "target_outlier_policy": "none", "x_outlier_policy": "none",
                "scaling_policy": "none", "dimensionality_reduction_policy": "none", "feature_selection_policy": "none",
                "preprocess_order": "none", "preprocess_fit_scope": "not_applicable", "inverse_transform_policy": "none", "evaluation_scale": "raw_level"
            }},
            "3_training": {"fixed_axes": {
                "framework": "expanding", "benchmark_family": "zero_change", "feature_builder": "autoreg_lagged_target", "model_family": "ar"
            }},
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"manifest_mode": "full", "benchmark_config": {"minimum_train_size": 5}}},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }
    compile_result = compile_recipe_dict(recipe)
    assert compile_result.manifest["reproducibility_spec"]["reproducibility_mode"] == "seeded_reproducible"
    assert compile_result.manifest["reproducibility_spec"]["random_seed"] == 42
    assert compile_result.manifest["tree_context"]["reproducibility_mode"] == "seeded_reproducible"



def test_axis_governance_table_includes_failure_policy() -> None:
    table = axis_governance_table()
    by_name = {row["axis_name"]: row for row in table}
    assert by_name["failure_policy"]["current_status"]["fail_fast"] == "operational"
    assert by_name["failure_policy"]["current_status"]["skip_failed_model"] == "operational"


def test_compile_failure_policy_spec_preserved_in_manifest() -> None:
    recipe = {
        "recipe_id": "fail-fast-provenance",
        "path": {
            "0_meta": {"fixed_axes": {"study_mode": "single_path_benchmark_study", "failure_policy": "fail_fast"}},
            "1_data_task": {
                "fixed_axes": {"dataset": "fred_md", "info_set": "revised", "task": "single_target_point_forecast"},
                "leaf_config": {"target": "INDPRO", "horizons": [1, 3]},
            },
            "2_preprocessing": {"fixed_axes": {
                "target_transform_policy": "raw_level", "x_transform_policy": "raw_level", "tcode_policy": "raw_only",
                "target_missing_policy": "none", "x_missing_policy": "none", "target_outlier_policy": "none", "x_outlier_policy": "none",
                "scaling_policy": "none", "dimensionality_reduction_policy": "none", "feature_selection_policy": "none",
                "preprocess_order": "none", "preprocess_fit_scope": "not_applicable", "inverse_transform_policy": "none", "evaluation_scale": "raw_level"
            }},
            "3_training": {"fixed_axes": {
                "framework": "expanding", "benchmark_family": "zero_change", "feature_builder": "autoreg_lagged_target", "model_family": "ar"
            }},
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"manifest_mode": "full", "benchmark_config": {"minimum_train_size": 5}}},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }
    compile_result = compile_recipe_dict(recipe)
    assert compile_result.manifest["failure_policy_spec"]["failure_policy"] == "fail_fast"
    assert compile_result.manifest["tree_context"]["failure_policy"] == "fail_fast"


def test_compile_warn_only_is_representable_not_executable() -> None:
    recipe = {
        "recipe_id": "warn-only-provenance",
        "path": {
            "0_meta": {"fixed_axes": {"study_mode": "single_path_benchmark_study", "failure_policy": "warn_only"}},
            "1_data_task": {
                "fixed_axes": {"dataset": "fred_md", "info_set": "revised", "task": "single_target_point_forecast"},
                "leaf_config": {"target": "INDPRO", "horizons": [1, 3]},
            },
            "2_preprocessing": {"fixed_axes": {
                "target_transform_policy": "raw_level", "x_transform_policy": "raw_level", "tcode_policy": "raw_only",
                "target_missing_policy": "none", "x_missing_policy": "none", "target_outlier_policy": "none", "x_outlier_policy": "none",
                "scaling_policy": "none", "dimensionality_reduction_policy": "none", "feature_selection_policy": "none",
                "preprocess_order": "none", "preprocess_fit_scope": "not_applicable", "inverse_transform_policy": "none", "evaluation_scale": "raw_level"
            }},
            "3_training": {"fixed_axes": {
                "framework": "expanding", "benchmark_family": "zero_change", "feature_builder": "autoreg_lagged_target", "model_family": "ar"
            }},
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"manifest_mode": "full", "benchmark_config": {"minimum_train_size": 5}}},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }
    compile_result = compile_recipe_dict(recipe)
    assert compile_result.compiled.execution_status == "representable_but_not_executable"
    assert any("status=registry_only" in warning or "representable but not executable" in warning for warning in compile_result.manifest["warnings"])



def test_axis_governance_table_marks_skip_failed_model_operational() -> None:
    table = axis_governance_table()
    by_name = {row["axis_name"]: row for row in table}
    assert by_name["failure_policy"]["current_status"]["skip_failed_model"] == "operational"
    assert by_name["failure_policy"]["current_status"]["save_partial_results"] == "operational"


def test_compile_skip_failed_model_recipe_is_executable() -> None:
    recipe = {
        "recipe_id": "skip-failed-model-executable",
        "path": {
            "0_meta": {"fixed_axes": {"study_mode": "single_path_benchmark_study", "failure_policy": "skip_failed_model"}},
            "1_data_task": {
                "fixed_axes": {"dataset": "fred_md", "info_set": "revised", "task": "single_target_point_forecast"},
                "leaf_config": {"target": "INDPRO", "horizons": [1, 3]},
            },
            "2_preprocessing": {"fixed_axes": {
                "target_transform_policy": "raw_level", "x_transform_policy": "raw_level", "tcode_policy": "raw_only",
                "target_missing_policy": "none", "x_missing_policy": "none", "target_outlier_policy": "none", "x_outlier_policy": "none",
                "scaling_policy": "none", "dimensionality_reduction_policy": "none", "feature_selection_policy": "none",
                "preprocess_order": "none", "preprocess_fit_scope": "not_applicable", "inverse_transform_policy": "none", "evaluation_scale": "raw_level"
            }},
            "3_training": {"fixed_axes": {
                "framework": "expanding", "benchmark_family": "zero_change", "feature_builder": "autoreg_lagged_target", "model_family": "ar"
            }},
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"manifest_mode": "full", "benchmark_config": {"minimum_train_size": 5}}},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }
    compile_result = compile_recipe_dict(recipe)
    assert compile_result.compiled.execution_status == "executable"
    assert compile_result.manifest["failure_policy_spec"]["failure_policy"] == "skip_failed_model"
