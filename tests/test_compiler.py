from __future__ import annotations

import json
from pathlib import Path

import pytest

from macrocast import (
    CompileValidationError,
    axis_governance_table,
    clear_custom_extensions,
    compile_recipe_dict,
    compile_recipe_yaml,
    custom_feature_block,
    get_canonical_layer_order,
    run_compiled_recipe,
)


def _layer2_level_block_recipe(
    *,
    feature_builder: str = "raw_feature_panel",
    model_family: str = "ridge",
    level_feature_block: str = "target_level_addback",
    contemporaneous_x_rule: str | None = None,
    selected_level_addback_columns: list[str] | None = None,
    level_growth_pair_columns: list[str] | None = None,
) -> dict:
    data_axes = {
        "dataset": "fred_md",
        "information_set_type": "revised",
        "target_structure": "single_target_point_forecast",
    }
    if contemporaneous_x_rule is not None:
        data_axes["contemporaneous_x_rule"] = contemporaneous_x_rule
    leaf_config = {
        "target": "INDPRO",
        "horizons": [1],
    }
    if selected_level_addback_columns is not None:
        leaf_config["selected_level_addback_columns"] = selected_level_addback_columns
    if level_growth_pair_columns is not None:
        leaf_config["level_growth_pair_columns"] = level_growth_pair_columns
    return {
        "recipe_id": f"l2-level-block-{feature_builder}",
        "path": {
            "0_meta": {"fixed_axes": {"research_design": "single_path_benchmark"}},
            "1_data_task": {
                "fixed_axes": data_axes,
                "leaf_config": leaf_config,
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
                    "level_feature_block": level_feature_block,
                }
            },
            "3_training": {
                "fixed_axes": {
                    "framework": "expanding",
                    "benchmark_family": "zero_change",
                    "feature_builder": feature_builder,
                    "model_family": model_family,
                }
            },
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"benchmark_config": {"minimum_train_size": 5}}},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }


def _layer2_temporal_block_recipe(
    *,
    feature_builder: str = "raw_feature_panel",
    model_family: str = "ridge",
    temporal_feature_block: str = "moving_average_features",
    rotation_feature_block: str | None = None,
    x_lag_feature_block: str | None = None,
    marx_max_lag: int | None = None,
) -> dict:
    preprocessing_axes = {
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
        "temporal_feature_block": temporal_feature_block,
    }
    if rotation_feature_block is not None:
        preprocessing_axes["rotation_feature_block"] = rotation_feature_block
    if x_lag_feature_block is not None:
        preprocessing_axes["tcode_policy"] = "extra_preprocess_without_tcode"
        preprocessing_axes["preprocess_order"] = "extra_only"
        preprocessing_axes["preprocess_fit_scope"] = "train_only"
        preprocessing_axes["x_lag_feature_block"] = x_lag_feature_block
    leaf_config = {
        "target": "INDPRO",
        "horizons": [1],
    }
    if marx_max_lag is not None:
        leaf_config["marx_max_lag"] = marx_max_lag
    return {
        "recipe_id": f"l2-temporal-block-{feature_builder}",
        "path": {
            "0_meta": {"fixed_axes": {"research_design": "single_path_benchmark"}},
            "1_data_task": {
                "fixed_axes": {
                    "dataset": "fred_md",
                    "information_set_type": "revised",
                    "target_structure": "single_target_point_forecast",
                },
                "leaf_config": leaf_config,
            },
            "2_preprocessing": {"fixed_axes": preprocessing_axes},
            "3_training": {
                "fixed_axes": {
                    "framework": "expanding",
                    "benchmark_family": "zero_change",
                    "feature_builder": feature_builder,
                    "model_family": model_family,
                }
            },
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"benchmark_config": {"minimum_train_size": 5}}},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }


def test_compile_minimal_importance_recipe_is_executable_for_ridge(tmp_path: Path) -> None:
    recipe = {
        "recipe_id": "importance-ridge-rolling",
        "path": {
            "0_meta": {"fixed_axes": {"research_design": "single_path_benchmark"}},
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
            "0_meta": {"fixed_axes": {"research_design": "single_path_benchmark"}},
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
            "0_meta": {"fixed_axes": {"research_design": "single_path_benchmark"}},
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
            "0_meta": {"fixed_axes": {"research_design": "single_path_benchmark"}},
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
            "0_meta": {"fixed_axes": {"research_design": "single_path_benchmark"}},
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
            "0_meta": {"fixed_axes": {"research_design": "single_path_benchmark"}},
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
            "0_meta": {"fixed_axes": {"research_design": "single_path_benchmark"}},
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


def test_compile_dataset_tcode_then_train_only_extra_is_executable(tmp_path: Path) -> None:
    recipe = {
        "recipe_id": "dataset-tcode-then-extra-ridge",
        "path": {
            "0_meta": {"fixed_axes": {"research_design": "single_path_benchmark"}},
            "1_data_task": {
                "fixed_axes": {
                    "dataset": "fred_md",
                    "information_set_type": "revised",
                    "target_structure": "single_target_point_forecast",
                    "official_transform_policy": "dataset_tcode",
                    "official_transform_scope": "apply_tcode_to_both",
                },
                "leaf_config": {"target": "INDPRO", "horizons": [1, 3]},
            },
            "2_preprocessing": {"fixed_axes": {
                "target_missing_policy": "none", "x_missing_policy": "mean_impute", "target_outlier_policy": "none", "x_outlier_policy": "winsorize",
                "scaling_policy": "standard", "dimensionality_reduction_policy": "none", "feature_selection_policy": "none",
                "preprocess_fit_scope": "train_only", "inverse_transform_policy": "none", "evaluation_scale": "raw_level",
                "additional_preprocessing": "none", "x_lag_creation": "no_x_lags", "feature_grouping": "none",
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
    assert compile_result.manifest["data_task_spec"]["official_transform_policy"] == "dataset_tcode"
    contract = compile_result.manifest["preprocess_contract"]
    assert contract["tcode_policy"] == "tcode_then_extra_preprocess"
    assert contract["preprocess_order"] == "tcode_then_extra"
    assert contract["representation_policy"] == "tcode_only"

    execution = run_compiled_recipe(
        compile_result.compiled,
        output_root=tmp_path,
        local_raw_source=Path("tests/fixtures/fred_md_raw_panel_missing.csv"),
    )
    manifest = json.loads((Path(execution.artifact_dir) / "manifest.json").read_text())
    assert manifest["preprocess_contract"]["x_missing_policy"] == "mean_impute"
    assert manifest["preprocess_contract"]["scaling_policy"] == "standard"


def test_compile_lasso_minimal_importance_recipe_is_executable(tmp_path: Path) -> None:
    recipe = {
        "recipe_id": "importance-lasso-rolling",
        "path": {
            "0_meta": {"fixed_axes": {"research_design": "single_path_benchmark"}},
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




def test_compile_wrapper_bundle_requires_wrapper_metadata() -> None:
    recipe = {
        "recipe_id": "wrapper-missing-metadata",
        "path": {
            "0_meta": {"fixed_axes": {"research_design": "orchestrated_bundle"}},
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
            "0_meta": {"fixed_axes": {"research_design": "orchestrated_bundle"}, "leaf_config": {"wrapper_family": "benchmark_suite", "bundle_label": "fred-md-baselines"}},
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
    assert compile_result.compiled.execution_status == "not_supported"
    handoff = compile_result.manifest["wrapper_handoff"]
    tree_context = compile_result.manifest["tree_context"]
    assert handoff["wrapper_family"] == "benchmark_suite"
    assert handoff["bundle_label"] == "fred-md-baselines"
    assert handoff["route_owner"] == "wrapper"
    assert handoff["execution_posture"] == "wrapper_bundle_plan"
    assert tree_context["route_owner"] == "wrapper"
    assert tree_context["fixed_axes"]["research_design"] == "orchestrated_bundle"
    assert tree_context["leaf_config"]["bundle_label"] == "fred-md-baselines"


def test_compile_multi_target_recipe_requires_targets() -> None:
    recipe = {
        "recipe_id": "multi-target-missing-targets",
        "path": {
            "0_meta": {"fixed_axes": {"research_design": "single_path_benchmark"}},
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
            "0_meta": {"fixed_axes": {"research_design": "single_path_benchmark"}},
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
            "0_meta": {"fixed_axes": {"research_design": "controlled_variation"}},
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
    assert compile_result.compiled.execution_status == "ready_for_sweep_runner"
    assert tree_context["research_design"] == "controlled_variation"
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
    assert by_name["experiment_unit"]["current_status"]["benchmark_suite"] == "registry_only"


def test_compile_recipe_preserves_explicit_experiment_unit() -> None:
    recipe = {
        "recipe_id": "experiment-unit-single-model",
        "path": {
            "0_meta": {"fixed_axes": {
                "research_design": "single_path_benchmark",
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
    assert by_name["axis_type"]["current_status"]["nested_sweep"] == "operational"


def test_compile_warns_when_fixed_policy_axis_is_placed_in_sweep_axes() -> None:
    recipe = {
        "recipe_id": "fixed-policy-swept-axis",
        "path": {
            "0_meta": {"fixed_axes": {"research_design": "single_path_benchmark"}},
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
    assert compile_result.compiled.execution_status == "ready_for_sweep_runner"



def test_axis_governance_table_includes_reproducibility_mode() -> None:
    table = axis_governance_table()
    by_name = {row["axis_name"]: row for row in table}
    assert by_name["reproducibility_mode"]["current_status"]["seeded_reproducible"] == "operational"
    assert by_name["reproducibility_mode"]["current_status"]["strict_reproducible"] == "operational"


def test_compile_seeded_reproducible_requires_random_seed() -> None:
    recipe = {
        "recipe_id": "seed-required",
        "path": {
            "0_meta": {"fixed_axes": {"research_design": "single_path_benchmark", "reproducibility_mode": "seeded_reproducible"}},
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
            "0_meta": {"fixed_axes": {"research_design": "single_path_benchmark", "reproducibility_mode": "seeded_reproducible"}},
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
            "0_meta": {"fixed_axes": {"research_design": "single_path_benchmark", "failure_policy": "fail_fast"}},
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


def test_compile_warn_only_is_now_executable() -> None:
    """warn_only flipped to operational in the 0.4 cleanup — compile should pass as executable."""
    recipe = {
        "recipe_id": "warn-only-provenance",
        "path": {
            "0_meta": {"fixed_axes": {"research_design": "single_path_benchmark", "failure_policy": "warn_only"}},
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
    assert compile_result.manifest["warnings"] == []



def test_axis_governance_table_marks_skip_failed_model_operational() -> None:
    table = axis_governance_table()
    by_name = {row["axis_name"]: row for row in table}
    assert by_name["failure_policy"]["current_status"]["skip_failed_model"] == "operational"
    assert by_name["failure_policy"]["current_status"]["save_partial_results"] == "operational"


def test_compile_skip_failed_model_recipe_is_executable() -> None:
    recipe = {
        "recipe_id": "skip-failed-model-executable",
        "path": {
            "0_meta": {"fixed_axes": {"research_design": "single_path_benchmark", "failure_policy": "skip_failed_model"}},
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



def test_axis_governance_table_includes_compute_mode() -> None:
    table = axis_governance_table()
    by_name = {row["axis_name"]: row for row in table}
    assert by_name["compute_mode"]["current_status"]["serial"] == "operational"
    assert by_name["compute_mode"]["current_status"]["parallel_by_model"] == "operational"


def test_compile_compute_mode_spec_defaults_to_serial() -> None:
    recipe = {
        "recipe_id": "default-serial-compute-mode",
        "path": {
            "0_meta": {"fixed_axes": {"research_design": "single_path_benchmark"}},
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
    assert compile_result.manifest["compute_mode_spec"]["compute_mode"] == "serial"
    assert compile_result.manifest["tree_context"]["compute_mode"] == "serial"


def test_compile_parallel_by_model_is_executable() -> None:
    recipe = {
        "recipe_id": "parallel-by-model-provenance",
        "path": {
            "0_meta": {"fixed_axes": {"research_design": "single_path_benchmark", "compute_mode": "parallel_by_model"}},
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
    assert compile_result.manifest["compute_mode_spec"]["compute_mode"] == "parallel_by_model"
    assert compile_result.manifest["warnings"] == []



def test_axis_governance_table_marks_parallel_compute_modes_operational() -> None:
    table = axis_governance_table()
    by_name = {row["axis_name"]: row for row in table}
    assert by_name["compute_mode"]["current_status"]["parallel_by_model"] == "operational"
    assert by_name["compute_mode"]["current_status"]["parallel_by_horizon"] == "operational"


def test_compile_parallel_by_model_recipe_is_executable() -> None:
    recipe = {
        "recipe_id": "parallel-by-model-executable",
        "path": {
            "0_meta": {"fixed_axes": {"research_design": "single_path_benchmark", "compute_mode": "parallel_by_model"}},
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
    assert compile_result.manifest["compute_mode_spec"]["compute_mode"] == "parallel_by_model"



def test_compile_recipe_accepts_legacy_info_set_alias() -> None:
    recipe = {
        "recipe_id": "legacy-info-set-alias",
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
                "preprocess_order": "none", "preprocess_fit_scope": "not_applicable", "inverse_transform_policy": "none", "evaluation_scale": "raw_level"
            }},
            "3_training": {"fixed_axes": {"framework": "expanding", "benchmark_family": "historical_mean", "feature_builder": "autoreg_lagged_target", "model_family": "ar"}},
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"manifest_mode": "full", "benchmark_config": {"minimum_train_size": 5}}},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }
    compile_result = compile_recipe_dict(recipe)
    assert compile_result.compiled.execution_status == "executable"
    assert compile_result.manifest["data_task_spec"]["information_set_type"] == "revised"



def test_compile_recipe_rejects_conflicting_predictor_family_and_feature_builder() -> None:
    recipe = {
        "recipe_id": "bad-predictor-family",
        "path": {
            "0_meta": {"fixed_axes": {"research_design": "single_path_benchmark"}},
            "1_data_task": {
                "fixed_axes": {
                    "dataset": "fred_md",
                    "information_set_type": "revised",
                    "task": "single_target_point_forecast",
                    "predictor_family": "target_lags_only",
                },
                "leaf_config": {"target": "INDPRO", "horizons": [1]},
            },
            "2_preprocessing": {"fixed_axes": {
                "target_transform_policy": "raw_level", "x_transform_policy": "raw_level", "tcode_policy": "raw_only",
                "target_missing_policy": "none", "x_missing_policy": "none", "target_outlier_policy": "none", "x_outlier_policy": "none",
                "scaling_policy": "none", "dimensionality_reduction_policy": "none", "feature_selection_policy": "none",
                "preprocess_order": "none", "preprocess_fit_scope": "not_applicable", "inverse_transform_policy": "none", "evaluation_scale": "raw_level"
            }},
            "3_training": {"fixed_axes": {"framework": "rolling", "benchmark_family": "zero_change", "feature_builder": "raw_feature_panel", "model_family": "ridge"}},
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"manifest_mode": "full", "benchmark_config": {"minimum_train_size": 5, "rolling_window_size": 5}}},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }
    compile_result = compile_recipe_dict(recipe)
    assert compile_result.compiled.execution_status == "blocked_by_incompatibility"
    assert any("predictor_family='target_lags_only'" in reason for reason in compile_result.compiled.blocked_reasons)


def test_compiled_manifest_records_stage1_data_task_defaults() -> None:
    compile_result = compile_recipe_yaml("examples/recipes/model-benchmark.yaml")
    spec = compile_result.manifest["data_task_spec"]
    assert spec["source_adapter"] == "fred_md"
    assert "dataset_source" not in spec
    assert spec["target_structure"] == "single_target_point_forecast"
    assert "task" not in spec
    assert spec["information_set_type"] == "revised"
    assert spec["forecast_type"] == "iterated"  # dynamic default for autoreg_lagged_target



def test_compile_recipe_records_stage2_preprocess_governance_defaults() -> None:
    compile_result = compile_recipe_yaml("examples/recipes/model-benchmark.yaml")
    contract = compile_result.manifest["preprocess_contract"]
    assert contract["representation_policy"] == "raw_only"
    assert contract["tcode_application_scope"] == "apply_tcode_to_none"


def test_compile_recipe_accepts_stage2_preprocess_axes() -> None:
    recipe = {
        "recipe_id": "stage2-preprocess-governance",
        "path": {
            "0_meta": {"fixed_axes": {"research_design": "single_path_benchmark"}},
            "1_data_task": {
                "fixed_axes": {"dataset": "fred_md", "information_set_type": "revised", "task": "single_target_point_forecast"},
                "leaf_config": {"target": "INDPRO", "horizons": [1, 3]},
            },
            "2_preprocessing": {"fixed_axes": {
                "target_transform_policy": "raw_level", "x_transform_policy": "raw_level", "tcode_policy": "extra_preprocess_without_tcode",
                "target_missing_policy": "none", "x_missing_policy": "mean_impute", "target_outlier_policy": "none", "x_outlier_policy": "winsorize",
                "scaling_policy": "minmax", "dimensionality_reduction_policy": "none", "feature_selection_policy": "none",
                "preprocess_order": "extra_only", "preprocess_fit_scope": "train_only", "inverse_transform_policy": "none", "evaluation_scale": "raw_level",
                "representation_policy": "raw_only", "tcode_application_scope": "apply_tcode_to_none",
                "target_transform": "level", "target_normalization": "none", "target_domain": "unconstrained", "scaling_scope": "columnwise",
                "additional_preprocessing": "none", "x_lag_creation": "no_x_lags", "feature_grouping": "none",
            }},
            "3_training": {"fixed_axes": {"framework": "rolling", "benchmark_family": "zero_change", "feature_builder": "raw_feature_panel", "model_family": "ridge"}},
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"manifest_mode": "full", "benchmark_config": {"minimum_train_size": 5, "rolling_window_size": 5}}},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }
    compile_result = compile_recipe_dict(recipe)
    assert compile_result.compiled.execution_status == "executable"
    assert compile_result.manifest["preprocess_contract"]["x_missing_policy"] == "mean_impute"
    assert compile_result.manifest["preprocess_contract"]["representation_policy"] == "raw_only"



def test_compiled_manifest_records_stage3_training_defaults() -> None:
    compile_result = compile_recipe_yaml("examples/recipes/model-benchmark.yaml")
    spec = compile_result.manifest["training_spec"]
    assert spec["outer_window"] == "expanding"
    assert spec["refit_policy"] == "refit_every_step"
    assert spec["horizon_modelization"] == "separate_model_per_h"


def test_compiled_manifest_records_layer2_representation_provenance() -> None:
    compile_result = compile_recipe_yaml("examples/recipes/model-benchmark.yaml")
    spec = compile_result.manifest["layer2_representation_spec"]
    assert spec["schema_version"] == "layer2_representation_v1"
    assert spec["runtime_effect"] == "provenance_plus_runtime_block_dispatch"
    assert spec["compatibility_source"]["source_kind"] == "legacy_bridge"
    assert spec["compatibility_source"]["legacy_manifest_alias"] == "source_bridge"
    assert spec["compatibility_source"]["feature_builder"] == "autoreg_lagged_target"
    assert spec["source_bridge"]["feature_builder"] == "autoreg_lagged_target"
    assert spec["source_bridge"]["data_richness_mode"] == "target_lags_only"
    assert spec["source_bridge"]["target_lag_selection"] == "ic_select"
    assert spec["source_bridge"]["legacy_y_lag_count"] == "IC_select"
    assert spec["target_lag_config"]["selection"] == "ic_select"
    assert spec["target_lag_config"]["selection_source_axis"] == "y_lag_count"
    assert spec["target_representation"]["horizon_target_construction"] == "future_target_level_t_plus_h"
    assert spec["feature_blocks"]["feature_block_set"]["value"] == "target_lags_only"
    assert spec["feature_blocks"]["target_lag_block"]["value"] == "ic_selected_target_lags"
    assert spec["feature_blocks"]["x_lag_feature_block"]["value"] == "none"
    assert (
        "Feature-block specs drive executor-family dispatch, fixed target-lag matrix composition, "
        "fixed X-lag matrix composition, PCA static-factor matrix composition, "
        "and fixed target-lag concatenation with raw-panel/factor-panel direct Z."
    ) == spec["compatibility_notes"][0]
    assert compile_result.compiled.recipe_spec.layer2_representation_spec == spec


def test_layer2_path_average_protocol_records_layer3_gate() -> None:
    recipe = {
        "recipe_id": "l2-path-average-protocol",
        "path": {
            "0_meta": {"fixed_axes": {"research_design": "single_path_benchmark"}},
            "1_data_task": {
                "fixed_axes": {
                    "dataset": "fred_md",
                    "information_set_type": "revised",
                    "target_structure": "single_target_point_forecast",
                    "horizon_target_construction": "path_average_log_growth_1_to_h",
                },
                "leaf_config": {"target": "INDPRO", "horizons": [1, 3]},
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
                    "framework": "expanding",
                    "benchmark_family": "zero_change",
                    "feature_builder": "autoreg_lagged_target",
                    "model_family": "ar",
                }
            },
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"benchmark_config": {"minimum_train_size": 5}}},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }
    result = compile_recipe_dict(recipe)
    assert result.compiled.execution_status == "not_supported"
    assert any("Layer 3 multi-step fit/aggregation" in warning for warning in result.compiled.warnings)
    target_repr = result.manifest["layer2_representation_spec"]["target_representation"]
    assert target_repr["horizon_target_construction"] == "path_average_log_growth_1_to_h"
    assert target_repr["target_construction_scale"] == "path_average_log_growth"
    protocol = target_repr["path_average_protocol"]
    assert protocol["runtime_effect"] == "protocol_only"
    assert protocol["formula_owner"] == "2_preprocessing"
    assert protocol["execution_owner"] == "3_training"
    assert "1" in protocol["protocols_by_horizon"]
    assert protocol["protocols_by_horizon"]["3"]["step_count"] == 3
    assert len(protocol["protocols_by_horizon"]["3"]["stepwise_target_specs"]) == 3


def test_layer2_representation_provenance_maps_feature_builder_bridge_values() -> None:
    def _recipe(feature_builder: str, model_family: str, **axes: str) -> dict:
        preprocessing_axes = {
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
        training_axes = {
            "framework": "expanding",
            "benchmark_family": "zero_change",
            "feature_builder": feature_builder,
            "model_family": model_family,
        }
        for key, value in axes.items():
            if key in preprocessing_axes:
                preprocessing_axes[key] = value
            else:
                training_axes[key] = value
        return {
            "recipe_id": f"l2-provenance-{feature_builder}",
            "path": {
                "0_meta": {"fixed_axes": {"research_design": "single_path_benchmark"}},
                "1_data_task": {
                    "fixed_axes": {
                        "dataset": "fred_md",
                        "information_set_type": "revised",
                        "target_structure": "single_target_point_forecast",
                    },
                    "leaf_config": {"target": "INDPRO", "horizons": [1]},
                },
                "2_preprocessing": {"fixed_axes": preprocessing_axes},
                "3_training": {"fixed_axes": training_axes},
                "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
                "5_output_provenance": {"leaf_config": {"benchmark_config": {"minimum_train_size": 5}}},
                "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
                "7_importance": {"fixed_axes": {"importance_method": "none"}},
            },
        }

    cases = [
        ("autoreg_lagged_target", "ar", {}, "target_lags_only", "ic_selected_target_lags", "none"),
        ("raw_feature_panel", "ridge", {}, "high_dimensional_x", "none", "none"),
        ("raw_X_only", "ridge", {"data_richness_mode": "selected_sparse_X"}, "selected_sparse_x", "none", "none"),
        ("factor_pca", "pcr", {}, "factor_blocks_only", "none", "pca_static_factors"),
        ("factors_plus_AR", "factor_augmented_linear", {}, "factors_plus_target_lags", "fixed_target_lags", "pca_static_factors"),
    ]
    for feature_builder, model_family, axes, block_set, target_lag_block, factor_block in cases:
        result = compile_recipe_dict(_recipe(feature_builder, model_family, **axes))
        spec = result.manifest["layer2_representation_spec"]
        assert spec["compatibility_source"]["source_kind"] == "legacy_bridge"
        assert spec["compatibility_source"]["legacy_manifest_alias"] == "source_bridge"
        assert spec["compatibility_source"]["feature_builder"] == feature_builder
        assert spec["source_bridge"]["feature_builder"] == feature_builder
        assert spec["feature_blocks"]["feature_block_set"]["value"] == block_set
        assert spec["feature_blocks"]["target_lag_block"]["value"] == target_lag_block
        assert spec["feature_blocks"]["factor_feature_block"]["value"] == factor_block


def test_layer2_target_lag_selection_axis_records_target_language_provenance() -> None:
    recipe = {
        "recipe_id": "l2-target-lag-selection",
        "path": {
            "0_meta": {"fixed_axes": {"research_design": "single_path_benchmark"}},
            "1_data_task": {
                "fixed_axes": {
                    "dataset": "fred_md",
                    "information_set_type": "revised",
                    "target_structure": "single_target_point_forecast",
                },
                "leaf_config": {
                    "target": "INDPRO",
                    "horizons": [1],
                    "training_config": {"target_lag_count": 2},
                },
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
                    "target_lag_selection": "fixed",
                }
            },
            "3_training": {
                "fixed_axes": {
                    "framework": "expanding",
                    "benchmark_family": "zero_change",
                    "feature_builder": "factors_plus_AR",
                    "model_family": "factor_augmented_linear",
                }
            },
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"benchmark_config": {"minimum_train_size": 5}}},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }
    result = compile_recipe_dict(recipe)
    spec = result.manifest["layer2_representation_spec"]
    assert "target_lag_selection" not in result.manifest["training_spec"]
    assert "target_lag_count" not in result.manifest["training_spec"]
    assert result.manifest["training_spec"]["factor_ar_lags"] == 2
    assert spec["target_lag_config"] == {
        "selection": "fixed",
        "selection_source_axis": "target_lag_selection",
        "selection_source_value": "fixed",
        "count": 2,
        "count_source": "target_lag_count",
    }
    assert spec["feature_blocks"]["target_lag_block"]["value"] == "fixed_target_lags"


def test_layer2_explicit_target_lag_block_lowers_to_ar_bridge() -> None:
    recipe = {
        "recipe_id": "l2-explicit-target-lag-block",
        "path": {
            "0_meta": {"fixed_axes": {"research_design": "single_path_benchmark"}},
            "1_data_task": {
                "fixed_axes": {
                    "dataset": "fred_md",
                    "information_set_type": "revised",
                    "target_structure": "single_target_point_forecast",
                },
                "leaf_config": {
                    "target": "INDPRO",
                    "horizons": [1],
                    "training_config": {"target_lag_count": 2},
                },
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
                    "target_lag_block": "fixed_target_lags",
                }
            },
            "3_training": {
                "fixed_axes": {
                    "framework": "expanding",
                    "benchmark_family": "zero_change",
                    "feature_builder": "autoreg_lagged_target",
                    "model_family": "ar",
                }
            },
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"benchmark_config": {"minimum_train_size": 5}}},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }
    result = compile_recipe_dict(recipe)
    assert result.compiled.execution_status == "executable"
    assert result.manifest["preprocess_contract"]["x_lag_creation"] == "no_x_lags"
    assert "target_lag_selection" not in result.manifest["training_spec"]
    assert "target_lag_count" not in result.manifest["training_spec"]
    assert result.manifest["layer2_representation_spec"]["target_lag_config"]["selection"] == "fixed"
    assert result.manifest["layer2_representation_spec"]["target_lag_config"]["count"] == 2
    assert result.manifest["benchmark_spec"]["max_ar_lag"] == 2
    blocks = result.manifest["layer2_representation_spec"]["feature_blocks"]
    assert blocks["target_lag_block"]["source_axis"] == "target_lag_block"
    assert blocks["target_lag_block"]["feature_names"] == ["target_lag_1", "target_lag_2"]
    assert blocks["target_lag_block"]["runtime_block"] == {
        "matrix_composition": "fixed_target_lags",
        "lag_count": 2,
    }
    assert blocks["target_lag_block"]["alignment"]["lookahead"] == "forbidden"


def test_layer2_explicit_x_lag_block_lowers_to_raw_panel_bridge() -> None:
    recipe = {
        "recipe_id": "l2-explicit-x-lag-block",
        "path": {
            "0_meta": {"fixed_axes": {"research_design": "single_path_benchmark"}},
            "1_data_task": {
                "fixed_axes": {
                    "dataset": "fred_md",
                    "information_set_type": "revised",
                    "target_structure": "single_target_point_forecast",
                },
                "leaf_config": {
                    "target": "INDPRO",
                    "horizons": [1],
                },
            },
            "2_preprocessing": {
                "fixed_axes": {
                    "target_transform_policy": "raw_level",
                    "x_transform_policy": "raw_level",
                    "tcode_policy": "extra_preprocess_without_tcode",
                    "target_missing_policy": "none",
                    "x_missing_policy": "none",
                    "target_outlier_policy": "none",
                    "x_outlier_policy": "none",
                    "scaling_policy": "none",
                    "dimensionality_reduction_policy": "none",
                    "feature_selection_policy": "none",
                    "preprocess_order": "extra_only",
                    "preprocess_fit_scope": "train_only",
                    "inverse_transform_policy": "none",
                    "evaluation_scale": "raw_level",
                    "x_lag_feature_block": "fixed_x_lags",
                }
            },
            "3_training": {
                "fixed_axes": {
                    "framework": "expanding",
                    "benchmark_family": "zero_change",
                    "feature_builder": "raw_feature_panel",
                    "model_family": "ridge",
                }
            },
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"benchmark_config": {"minimum_train_size": 5}}},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }
    result = compile_recipe_dict(recipe)
    assert result.compiled.execution_status == "executable"
    assert result.manifest["preprocess_contract"]["x_lag_creation"] == "fixed_x_lags"
    blocks = result.manifest["layer2_representation_spec"]["feature_blocks"]
    assert blocks["x_lag_feature_block"]["source_axis"] == "x_lag_feature_block"
    assert blocks["x_lag_feature_block"]["runtime_bridge"] == {"x_lag_creation": "fixed_x_lags"}
    assert blocks["x_lag_feature_block"]["alignment"]["lookahead"] == "forbidden"


def test_layer2_explicit_level_block_lowers_to_raw_panel_bridge() -> None:
    result = compile_recipe_dict(_layer2_level_block_recipe())
    assert result.compiled.execution_status == "executable"
    blocks = result.manifest["layer2_representation_spec"]["feature_blocks"]
    block = blocks["level_feature_block"]
    assert block["value"] == "target_level_addback"
    assert block["source_axis"] == "level_feature_block"
    assert block["feature_names"] == ["target_level_origin"]
    assert block["runtime_feature_name"] == "__target_level_origin"
    assert block["runtime_bridge"] == {"raw_panel_level_addback": "target_level_addback"}
    assert block["alignment"] == {
        "train_row_t_uses": "target_t",
        "prediction_origin_uses": "target_origin",
        "lookahead": "forbidden",
    }


def test_layer2_explicit_x_level_block_lowers_to_raw_panel_bridge() -> None:
    result = compile_recipe_dict(_layer2_level_block_recipe(level_feature_block="x_level_addback"))
    assert result.compiled.execution_status == "executable"
    blocks = result.manifest["layer2_representation_spec"]["feature_blocks"]
    block = blocks["level_feature_block"]
    assert block["value"] == "x_level_addback"
    assert block["source_axis"] == "level_feature_block"
    assert block["feature_name_pattern"] == "{predictor}_level"
    assert block["runtime_feature_name_pattern"] == "{predictor}__level"
    assert block["runtime_bridge"] == {"raw_panel_level_addback": "x_level_addback"}
    assert block["alignment"] == {
        "train_row_t_uses": "H_{t}",
        "prediction_origin_uses": "H_{origin}",
        "lookahead": "forbidden",
    }


def test_layer2_explicit_selected_level_block_lowers_to_raw_panel_bridge() -> None:
    result = compile_recipe_dict(
        _layer2_level_block_recipe(
            level_feature_block="selected_level_addbacks",
            selected_level_addback_columns=["RPI", "UNRATE"],
        )
    )
    assert result.compiled.execution_status == "executable"
    blocks = result.manifest["layer2_representation_spec"]["feature_blocks"]
    block = blocks["level_feature_block"]
    assert block["value"] == "selected_level_addbacks"
    assert block["selected_columns"] == ["RPI", "UNRATE"]
    assert block["feature_names"] == ["RPI_level", "UNRATE_level"]
    assert block["runtime_feature_names"] == ["RPI__level", "UNRATE__level"]
    assert block["runtime_bridge"] == {"raw_panel_level_addback": "selected_level_addbacks"}
    assert block["alignment"]["lookahead"] == "forbidden"


def test_layer2_explicit_selected_level_block_requires_columns() -> None:
    recipe = _layer2_level_block_recipe(level_feature_block="selected_level_addbacks")
    with pytest.raises(CompileValidationError, match="selected_level_addback_columns"):
        compile_recipe_dict(recipe)


def test_layer2_explicit_level_growth_pairs_lower_to_raw_panel_bridge() -> None:
    result = compile_recipe_dict(
        _layer2_level_block_recipe(
            level_feature_block="level_growth_pairs",
            level_growth_pair_columns=["RPI", "UNRATE"],
        )
    )
    assert result.compiled.execution_status == "executable"
    blocks = result.manifest["layer2_representation_spec"]["feature_blocks"]
    block = blocks["level_feature_block"]
    assert block["value"] == "level_growth_pairs"
    assert block["pair_columns"] == ["RPI", "UNRATE"]
    assert block["transformed_feature_names"] == ["RPI", "UNRATE"]
    assert block["level_feature_names"] == ["RPI_level", "UNRATE_level"]
    assert block["runtime_level_feature_names"] == ["RPI__level", "UNRATE__level"]
    assert block["runtime_bridge"] == {"raw_panel_level_addback": "level_growth_pairs"}
    assert block["alignment"]["lookahead"] == "forbidden"


def test_layer2_explicit_level_growth_pairs_require_columns() -> None:
    recipe = _layer2_level_block_recipe(level_feature_block="level_growth_pairs")
    with pytest.raises(CompileValidationError, match="level_growth_pair_columns"):
        compile_recipe_dict(recipe)


def test_layer2_explicit_level_block_requires_raw_panel_bridge() -> None:
    result = compile_recipe_dict(
        _layer2_level_block_recipe(feature_builder="autoreg_lagged_target", model_family="ar")
    )
    assert result.compiled.execution_status == "blocked_by_incompatibility"
    assert any("raw_feature_panel is not compatible with model_family='ar'" in reason for reason in result.compiled.blocked_reasons)


def test_layer2_explicit_level_block_rejects_contemporaneous_oracle_alignment() -> None:
    result = compile_recipe_dict(
        _layer2_level_block_recipe(contemporaneous_x_rule="allow_contemporaneous")
    )
    assert result.compiled.execution_status == "not_supported"
    assert any("requires contemporaneous_x_rule='forbid_contemporaneous'" in warning for warning in result.compiled.warnings)


def test_layer2_explicit_temporal_block_lowers_to_raw_panel_bridge() -> None:
    result = compile_recipe_dict(_layer2_temporal_block_recipe())
    assert result.compiled.execution_status == "executable"
    blocks = result.manifest["layer2_representation_spec"]["feature_blocks"]
    block = blocks["temporal_feature_block"]
    assert block["value"] == "moving_average_features"
    assert block["source_axis"] == "temporal_feature_block"
    assert block["window"] == 3
    assert block["feature_name_pattern"] == "{predictor}_ma3"
    assert block["runtime_feature_name_pattern"] == "{predictor}__ma3"
    assert block["runtime_bridge"] == {"raw_panel_temporal_features": "moving_average_features"}
    assert block["alignment"] == {
        "train_row_t_uses": "X_{t}, X_{t-1}, X_{t-2}",
        "prediction_origin_uses": "X_{origin}, X_{origin-1}, X_{origin-2}",
        "lookahead": "forbidden",
    }


def test_layer2_explicit_volatility_block_lowers_to_raw_panel_bridge() -> None:
    result = compile_recipe_dict(
        _layer2_temporal_block_recipe(temporal_feature_block="volatility_features")
    )
    assert result.compiled.execution_status == "executable"
    blocks = result.manifest["layer2_representation_spec"]["feature_blocks"]
    block = blocks["temporal_feature_block"]
    assert block["value"] == "volatility_features"
    assert block["source_axis"] == "temporal_feature_block"
    assert block["window"] == 3
    assert block["feature_name_pattern"] == "{predictor}_vol3"
    assert block["runtime_feature_name_pattern"] == "{predictor}__vol3"
    assert block["runtime_bridge"] == {"raw_panel_temporal_features": "volatility_features"}
    assert block["alignment"]["lookahead"] == "forbidden"


def test_layer2_explicit_rolling_moments_block_lowers_to_raw_panel_bridge() -> None:
    result = compile_recipe_dict(
        _layer2_temporal_block_recipe(temporal_feature_block="rolling_moments")
    )
    assert result.compiled.execution_status == "executable"
    blocks = result.manifest["layer2_representation_spec"]["feature_blocks"]
    block = blocks["temporal_feature_block"]
    assert block["value"] == "rolling_moments"
    assert block["source_axis"] == "temporal_feature_block"
    assert block["window"] == 3
    assert block["moments"] == ["mean", "variance"]
    assert block["feature_name_patterns"] == ["{predictor}_mean3", "{predictor}_var3"]
    assert block["runtime_feature_name_patterns"] == ["{predictor}__mean3", "{predictor}__var3"]
    assert block["runtime_bridge"] == {"raw_panel_temporal_features": "rolling_moments"}
    assert block["alignment"]["lookahead"] == "forbidden"


def test_layer2_explicit_local_temporal_factor_block_lowers_to_raw_panel_bridge() -> None:
    result = compile_recipe_dict(
        _layer2_temporal_block_recipe(temporal_feature_block="local_temporal_factors")
    )
    assert result.compiled.execution_status == "executable"
    blocks = result.manifest["layer2_representation_spec"]["feature_blocks"]
    block = blocks["temporal_feature_block"]
    assert block["value"] == "local_temporal_factors"
    assert block["source_axis"] == "temporal_feature_block"
    assert block["window"] == 3
    assert block["factor_construction"] == "deterministic cross-sectional summaries with trailing time smoothing"
    assert block["feature_names"] == ["local_temporal_factor_mean3", "local_temporal_factor_dispersion3"]
    assert block["runtime_feature_names"] == ["__local_temporal_factor_mean3", "__local_temporal_factor_dispersion3"]
    assert block["runtime_bridge"] == {"raw_panel_temporal_features": "local_temporal_factors"}
    assert block["alignment"]["lookahead"] == "forbidden"


def test_layer2_explicit_rotation_none_records_no_rotation_block() -> None:
    result = compile_recipe_dict(
        _layer2_temporal_block_recipe(
            temporal_feature_block="none",
            rotation_feature_block="none",
        )
    )
    assert result.compiled.execution_status == "executable"
    blocks = result.manifest["layer2_representation_spec"]["feature_blocks"]
    assert blocks["rotation_feature_block"] == {
        "value": "none",
        "source_axis": "rotation_feature_block",
        "source_value": "none",
    }


def test_layer2_explicit_moving_average_rotation_lowers_to_raw_panel_bridge() -> None:
    result = compile_recipe_dict(
        _layer2_temporal_block_recipe(
            temporal_feature_block="none",
            rotation_feature_block="moving_average_rotation",
        )
    )
    assert result.compiled.execution_status == "executable"
    blocks = result.manifest["layer2_representation_spec"]["feature_blocks"]
    block = blocks["rotation_feature_block"]
    assert block["value"] == "moving_average_rotation"
    assert block["source_axis"] == "rotation_feature_block"
    assert block["windows"] == [3, 6]
    assert block["feature_name_patterns"] == ["{predictor}_rotma3", "{predictor}_rotma6"]
    assert block["runtime_feature_name_patterns"] == ["{predictor}__rotma3", "{predictor}__rotma6"]
    assert block["runtime_bridge"] == {"raw_panel_rotation_features": "moving_average_rotation"}
    assert block["alignment"]["lookahead"] == "forbidden"
    assert "MARX uses lag-polynomial basis replacement" in block["scope_note"]


@pytest.mark.parametrize(
    ("rotation_block", "required_contract", "scope_phrase"),
    [
        (
            "maf_rotation",
            "factor_rotation_block_composer",
            "not a raw-X moving-average append",
        ),
        (
            "custom_rotation",
            "custom_feature_block_callable_v1",
            "custom_preprocessor hook is not enough",
        ),
    ],
)
def test_layer2_advanced_rotation_blocks_record_registry_only_boundary(
    rotation_block: str,
    required_contract: str,
    scope_phrase: str,
) -> None:
    result = compile_recipe_dict(
        _layer2_temporal_block_recipe(
            temporal_feature_block="none",
            rotation_feature_block=rotation_block,
        )
    )
    assert result.compiled.execution_status == "not_supported"
    assert any(
        f"axis rotation_feature_block value {rotation_block} is not supported" in warning
        and "status=registry_only" in warning
        for warning in result.compiled.warnings
    )
    block = result.manifest["layer2_representation_spec"]["feature_blocks"]["rotation_feature_block"]
    assert block["value"] == rotation_block
    assert block["source_axis"] == "rotation_feature_block"
    assert block["source_value"] == rotation_block
    assert block["runtime_status"] == "registry_only"
    assert block["required_runtime_contract"] == required_contract
    assert scope_phrase in block["scope_note"]
    assert "runtime_bridge" not in block
    if rotation_block == "custom_rotation":
        assert block["callable_contract"]["schema_version"] == "custom_feature_block_callable_v1"
        assert block["callable_contract"]["block_kind"] == "rotation"


def test_layer2_explicit_marx_rotation_requires_lag_order() -> None:
    with pytest.raises(CompileValidationError, match="marx_max_lag"):
        compile_recipe_dict(
            _layer2_temporal_block_recipe(
                temporal_feature_block="none",
                rotation_feature_block="marx_rotation",
            )
        )


def test_layer2_explicit_marx_rotation_lowers_to_raw_panel_bridge() -> None:
    result = compile_recipe_dict(
        _layer2_temporal_block_recipe(
            temporal_feature_block="none",
            rotation_feature_block="marx_rotation",
            marx_max_lag=3,
        )
    )
    assert result.compiled.execution_status == "executable"
    assert result.manifest["data_task_spec"]["marx_max_lag"] == 3
    block = result.manifest["layer2_representation_spec"]["feature_blocks"]["rotation_feature_block"]
    assert block["value"] == "marx_rotation"
    assert block["source_axis"] == "rotation_feature_block"
    assert block["runtime_status"] == "operational"
    assert block["max_lag"] == 3
    assert block["rotation_orders"] == [1, 2, 3]
    assert block["feature_name_pattern"] == "{predictor}_marx_ma_lag1_to_lag{p}"
    assert block["runtime_feature_name_pattern"] == "{predictor}__marx_ma_lag1_to_lag{p}"
    assert block["runtime_bridge"] == {"raw_panel_rotation_features": "marx_rotation"}
    assert block["basis_policy"] == "replace_lag_polynomial_basis"
    assert block["initial_lag_fill_policy"] == "zero_fill_before_start"
    assert block["composition_modes"]["operational"] == [
        "replace_lag_polynomial_basis",
        "marx_append_to_x",
        "marx_then_factor",
        "marx_with_external_x_lag_append",
        "marx_with_temporal_append",
    ]
    assert block["composition_modes"]["gated"] == ["factor_then_marx"]
    assert block["alignment"]["lookahead"] == "forbidden"
    assert "replaces the X lag-polynomial basis" in block["scope_note"]
    composer = block["composer_contract"]
    assert composer["schema_version"] == "lag_polynomial_rotation_contract_v1"
    assert composer["runtime_status"] == "operational"
    assert composer["runtime_builder"] == "build_marx_rotation_frame"
    assert composer["max_lag"] == 3
    assert composer["rotation_orders"] == [1, 2, 3]
    assert composer["source_feature_name_pattern"] == "{predictor}_lag_{k}"
    assert composer["rotated_feature_name_pattern"] == "{predictor}_marx_ma_lag1_to_lag{p}"
    assert composer["basis_policy"] == "replace_lag_polynomial_basis"


def test_layer2_explicit_marx_rotation_runs_raw_panel_bridge(tmp_path: Path) -> None:
    result = compile_recipe_dict(
        _layer2_temporal_block_recipe(
            temporal_feature_block="none",
            rotation_feature_block="marx_rotation",
            marx_max_lag=3,
        )
    )

    execution = run_compiled_recipe(
        result.compiled,
        output_root=tmp_path,
        local_raw_source=Path("tests/fixtures/fred_md_ar_sample.csv"),
    )

    manifest = json.loads((Path(execution.artifact_dir) / "manifest.json").read_text())
    block = manifest["layer2_representation_spec"]["feature_blocks"]["rotation_feature_block"]
    assert block["value"] == "marx_rotation"
    assert block["max_lag"] == 3
    assert block["runtime_bridge"] == {"raw_panel_rotation_features": "marx_rotation"}


def test_layer2_explicit_marx_rotation_supports_static_factor_composition(tmp_path: Path) -> None:
    recipe = _layer2_temporal_block_recipe(
        temporal_feature_block="none",
        rotation_feature_block="marx_rotation",
        marx_max_lag=3,
    )
    recipe["path"]["2_preprocessing"]["fixed_axes"].update(
        {
            "factor_feature_block": "pca_static_factors",
            "dimensionality_reduction_policy": "pca",
            "tcode_policy": "extra_preprocess_without_tcode",
            "preprocess_order": "extra_only",
            "preprocess_fit_scope": "train_only",
            "scaling_policy": "standard",
        }
    )

    result = compile_recipe_dict(recipe)

    assert result.compiled.execution_status == "executable"
    factor_block = result.manifest["layer2_representation_spec"]["feature_blocks"]["factor_feature_block"]
    rotation_interaction = factor_block["rotation_interaction"]
    assert rotation_interaction["rotation_feature_block"] == "marx_rotation"
    assert rotation_interaction["supported_semantics"] == ["marx_then_factor"]
    assert rotation_interaction["active_semantic"] == "marx_then_factor"
    assert "factor_then_marx" in rotation_interaction["composition_modes"]["gated"]

    execution = run_compiled_recipe(
        result.compiled,
        output_root=tmp_path,
        local_raw_source=Path("tests/fixtures/fred_md_ar_sample.csv"),
    )
    manifest = json.loads((Path(execution.artifact_dir) / "manifest.json").read_text())
    fit_state = json.loads((Path(execution.artifact_dir) / "feature_representation_fit_state.json").read_text())

    assert manifest["layer2_representation_spec"]["feature_blocks"]["rotation_feature_block"]["value"] == "marx_rotation"
    assert fit_state["block"] == "pca_static_factors"
    assert all("__marx_ma_lag1_to_lag" in name for name in fit_state["source_feature_names"])


def test_layer2_pca_factor_lags_run_as_factor_block(tmp_path: Path) -> None:
    recipe = _layer2_temporal_block_recipe(
        temporal_feature_block="none",
        rotation_feature_block="none",
    )
    recipe["path"]["2_preprocessing"]["fixed_axes"].update(
        {
            "factor_feature_block": "pca_factor_lags",
            "tcode_policy": "extra_preprocess_without_tcode",
            "preprocess_order": "extra_only",
            "preprocess_fit_scope": "train_only",
            "scaling_policy": "standard",
        }
    )

    result = compile_recipe_dict(recipe)

    assert result.compiled.execution_status == "executable"
    execution = run_compiled_recipe(
        result.compiled,
        output_root=tmp_path,
        local_raw_source=Path("tests/fixtures/fred_md_ar_sample.csv"),
    )
    fit_state = json.loads((Path(execution.artifact_dir) / "feature_representation_fit_state.json").read_text())
    assert fit_state["block"] == "pca_factor_lags"
    assert fit_state["factor_lag_count"] >= 1
    assert any(name.startswith("factor_1_lag_") for name in fit_state["factor_lag_feature_names"])


def test_layer2_supervised_factors_run_as_factor_block(tmp_path: Path) -> None:
    recipe = _layer2_temporal_block_recipe(
        temporal_feature_block="none",
        rotation_feature_block="none",
    )
    recipe["path"]["2_preprocessing"]["fixed_axes"].update(
        {
            "factor_feature_block": "supervised_factors",
            "tcode_policy": "extra_preprocess_without_tcode",
            "preprocess_order": "extra_only",
            "preprocess_fit_scope": "train_only",
            "scaling_policy": "standard",
        }
    )

    result = compile_recipe_dict(recipe)

    assert result.compiled.execution_status == "executable"
    execution = run_compiled_recipe(
        result.compiled,
        output_root=tmp_path,
        local_raw_source=Path("tests/fixtures/fred_md_ar_sample.csv"),
    )
    fit_state = json.loads((Path(execution.artifact_dir) / "feature_representation_fit_state.json").read_text())
    assert fit_state["block"] == "supervised_factors"
    assert fit_state["supervision_target"] == "train_window_y"


def test_layer2_explicit_marx_rotation_rejects_x_lag_composition() -> None:
    result = compile_recipe_dict(
        _layer2_temporal_block_recipe(
            temporal_feature_block="none",
            rotation_feature_block="marx_rotation",
            x_lag_feature_block="fixed_x_lags",
            marx_max_lag=3,
        )
    )
    assert result.compiled.execution_status == "not_supported"
    assert any("cannot yet be combined with x_lag_feature_block" in warning for warning in result.compiled.warnings)


def test_layer2_explicit_marx_rotation_rejects_temporal_composition() -> None:
    result = compile_recipe_dict(
        _layer2_temporal_block_recipe(
            rotation_feature_block="marx_rotation",
            marx_max_lag=3,
        )
    )
    assert result.compiled.execution_status == "not_supported"
    assert any("cannot yet be combined with temporal_feature_block" in warning for warning in result.compiled.warnings)


def test_layer2_marx_rotation_can_append_with_temporal_block(tmp_path: Path) -> None:
    recipe = _layer2_temporal_block_recipe(
        temporal_feature_block="moving_average_features",
        rotation_feature_block="marx_rotation",
        marx_max_lag=2,
    )
    recipe["path"]["2_preprocessing"]["fixed_axes"]["feature_block_combination"] = "append_to_base_x"

    result = compile_recipe_dict(recipe)

    assert result.compiled.execution_status == "executable"
    combination = result.manifest["layer2_representation_spec"]["feature_blocks"]["feature_block_combination"]
    assert combination["value"] == "append_to_base_x"
    execution = run_compiled_recipe(
        result.compiled,
        output_root=tmp_path,
        local_raw_source=Path("tests/fixtures/fred_md_ar_sample.csv"),
    )
    manifest = json.loads((Path(execution.artifact_dir) / "manifest.json").read_text())
    assert manifest["prediction_rows"] > 0


def test_layer2_marx_rotation_can_append_with_fixed_x_lags(tmp_path: Path) -> None:
    recipe = _layer2_temporal_block_recipe(
        temporal_feature_block="none",
        rotation_feature_block="marx_rotation",
        x_lag_feature_block="fixed_x_lags",
        marx_max_lag=2,
    )
    recipe["path"]["2_preprocessing"]["fixed_axes"]["feature_block_combination"] = "append_to_base_x"

    result = compile_recipe_dict(recipe)

    assert result.compiled.execution_status == "executable"
    execution = run_compiled_recipe(
        result.compiled,
        output_root=tmp_path,
        local_raw_source=Path("tests/fixtures/fred_md_ar_sample.csv"),
    )
    manifest = json.loads((Path(execution.artifact_dir) / "manifest.json").read_text())
    assert manifest["prediction_rows"] > 0


def test_layer2_append_to_target_lags_combination_is_executable(tmp_path: Path) -> None:
    recipe = _layer2_temporal_block_recipe(
        temporal_feature_block="none",
        rotation_feature_block="none",
    )
    recipe["path"]["1_data_task"]["leaf_config"]["training_config"] = {"target_lag_count": 2}
    recipe["path"]["2_preprocessing"]["fixed_axes"].update(
        {
            "factor_feature_block": "pca_static_factors",
            "target_lag_block": "fixed_target_lags",
            "feature_block_combination": "append_to_target_lags",
            "tcode_policy": "extra_preprocess_without_tcode",
            "preprocess_order": "extra_only",
            "preprocess_fit_scope": "train_only",
            "scaling_policy": "standard",
            "dimensionality_reduction_policy": "pca",
        }
    )

    result = compile_recipe_dict(recipe)

    assert result.compiled.execution_status == "executable"
    combination = result.manifest["layer2_representation_spec"]["feature_blocks"]["feature_block_combination"]
    assert combination["value"] == "append_to_target_lags"
    assert combination["runtime_status"] == "operational"
    execution = run_compiled_recipe(
        result.compiled,
        output_root=tmp_path,
        local_raw_source=Path("tests/fixtures/fred_md_ar_sample.csv"),
    )
    manifest = json.loads((Path(execution.artifact_dir) / "manifest.json").read_text())
    assert manifest["prediction_rows"] > 0


def test_layer2_custom_temporal_block_records_callable_contract() -> None:
    result = compile_recipe_dict(
        _layer2_temporal_block_recipe(
            temporal_feature_block="custom_temporal_features",
            rotation_feature_block="none",
        )
    )
    assert result.compiled.execution_status == "not_supported"
    block = result.manifest["layer2_representation_spec"]["feature_blocks"]["temporal_feature_block"]
    assert block["value"] == "custom_temporal_features"
    assert block["runtime_status"] == "registry_only"
    assert block["required_runtime_contract"] == "custom_feature_block_callable_v1"
    assert block["callable_contract"]["block_kind"] == "temporal"
    assert "custom_preprocessor is a broader matrix hook" in block["scope_note"]


def test_layer2_registered_custom_temporal_block_is_executable() -> None:
    clear_custom_extensions()

    @custom_feature_block("temporal_spread", block_kind="temporal")
    def _temporal_spread(context):
        raise AssertionError("compile should not execute custom feature blocks")

    recipe = _layer2_temporal_block_recipe(
        temporal_feature_block="custom_temporal_features",
        rotation_feature_block="none",
    )
    recipe["path"]["1_data_task"]["leaf_config"]["custom_temporal_feature_block"] = "temporal_spread"

    result = compile_recipe_dict(recipe)

    assert result.compiled.execution_status == "executable"
    block = result.manifest["layer2_representation_spec"]["feature_blocks"]["temporal_feature_block"]
    assert block["runtime_status"] == "operational"
    assert block["custom_feature_block"] == "temporal_spread"
    assert block["runtime_bridge"] == {"custom_feature_block": "temporal_spread", "block_kind": "temporal"}
    clear_custom_extensions()


def test_layer2_explicit_rotation_block_requires_raw_panel_bridge() -> None:
    result = compile_recipe_dict(
        _layer2_temporal_block_recipe(
            feature_builder="autoreg_lagged_target",
            model_family="ar",
            temporal_feature_block="none",
            rotation_feature_block="moving_average_rotation",
        )
    )
    assert result.compiled.execution_status == "blocked_by_incompatibility"
    assert any("raw_feature_panel is not compatible with model_family='ar'" in reason for reason in result.compiled.blocked_reasons)


def test_layer2_explicit_moving_average_rotation_allows_temporal_composition() -> None:
    result = compile_recipe_dict(
        _layer2_temporal_block_recipe(rotation_feature_block="moving_average_rotation")
    )
    assert result.compiled.execution_status == "executable"
    blocks = result.manifest["layer2_representation_spec"]["feature_blocks"]
    assert blocks["temporal_feature_block"]["value"] == "moving_average_features"
    assert blocks["rotation_feature_block"]["value"] == "moving_average_rotation"
    assert blocks["temporal_feature_block"]["runtime_bridge"] == {
        "raw_panel_temporal_features": "moving_average_features"
    }
    assert blocks["rotation_feature_block"]["runtime_bridge"] == {
        "raw_panel_rotation_features": "moving_average_rotation"
    }


def test_layer2_explicit_temporal_block_requires_raw_panel_bridge() -> None:
    result = compile_recipe_dict(
        _layer2_temporal_block_recipe(feature_builder="autoreg_lagged_target", model_family="ar")
    )
    assert result.compiled.execution_status == "blocked_by_incompatibility"
    assert any("raw_feature_panel is not compatible with model_family='ar'" in reason for reason in result.compiled.blocked_reasons)


def test_layer2_explicit_temporal_block_allows_fixed_x_lag_composition() -> None:
    result = compile_recipe_dict(
        _layer2_temporal_block_recipe(x_lag_feature_block="fixed_x_lags")
    )
    assert result.compiled.execution_status == "executable"
    blocks = result.manifest["layer2_representation_spec"]["feature_blocks"]
    assert blocks["x_lag_feature_block"]["value"] == "fixed_x_lags"
    assert blocks["x_lag_feature_block"]["runtime_bridge"] == {"x_lag_creation": "fixed_x_lags"}
    assert blocks["temporal_feature_block"]["value"] == "moving_average_features"
    assert blocks["temporal_feature_block"]["runtime_bridge"] == {
        "raw_panel_temporal_features": "moving_average_features"
    }


def test_layer2_explicit_target_and_x_lag_blocks_execute_with_raw_panel_composer(tmp_path) -> None:
    recipe = {
        "recipe_id": "l2-explicit-lag-block-composition",
        "path": {
            "0_meta": {"fixed_axes": {"research_design": "single_path_benchmark"}},
            "1_data_task": {
                "fixed_axes": {
                    "dataset": "fred_md",
                    "information_set_type": "revised",
                    "target_structure": "single_target_point_forecast",
                },
                "leaf_config": {
                    "target": "INDPRO",
                    "horizons": [1],
                    "training_config": {"target_lag_count": 2},
                },
            },
            "2_preprocessing": {
                "fixed_axes": {
                    "target_transform_policy": "raw_level",
                    "x_transform_policy": "raw_level",
                    "tcode_policy": "extra_preprocess_without_tcode",
                    "target_missing_policy": "none",
                    "x_missing_policy": "none",
                    "target_outlier_policy": "none",
                    "x_outlier_policy": "none",
                    "scaling_policy": "none",
                    "dimensionality_reduction_policy": "none",
                    "feature_selection_policy": "none",
                    "preprocess_order": "extra_only",
                    "preprocess_fit_scope": "train_only",
                    "inverse_transform_policy": "none",
                    "evaluation_scale": "raw_level",
                    "target_lag_block": "fixed_target_lags",
                    "x_lag_feature_block": "fixed_x_lags",
                }
            },
            "3_training": {
                "fixed_axes": {
                    "framework": "expanding",
                    "benchmark_family": "zero_change",
                    "feature_builder": "raw_feature_panel",
                    "model_family": "ridge",
                }
            },
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"benchmark_config": {"minimum_train_size": 5}}},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }
    result = compile_recipe_dict(recipe)
    assert result.compiled.execution_status == "executable"
    blocks = result.manifest["layer2_representation_spec"]["feature_blocks"]
    assert blocks["target_lag_block"]["value"] == "fixed_target_lags"
    assert blocks["x_lag_feature_block"]["value"] == "fixed_x_lags"
    assert blocks["target_lag_block"]["alignment"]["train_row_t_uses"] == "target_{origin_t-k+1}"
    execution = run_compiled_recipe(
        result.compiled,
        output_root=tmp_path,
        local_raw_source="tests/fixtures/fred_md_ar_sample.csv",
    )
    assert (Path(execution.artifact_dir) / "predictions.csv").exists()


def test_layer2_explicit_target_lag_and_static_factor_blocks_execute(tmp_path) -> None:
    recipe = {
        "recipe_id": "l2-target-lag-factor-composition",
        "path": {
            "0_meta": {"fixed_axes": {"research_design": "single_path_benchmark"}},
            "1_data_task": {
                "fixed_axes": {
                    "dataset": "fred_md",
                    "information_set_type": "revised",
                    "target_structure": "single_target_point_forecast",
                },
                "leaf_config": {
                    "target": "INDPRO",
                    "horizons": [1],
                    "training_config": {"target_lag_count": 2},
                },
            },
            "2_preprocessing": {
                "fixed_axes": {
                    "target_transform_policy": "raw_level",
                    "x_transform_policy": "raw_level",
                    "tcode_policy": "extra_preprocess_without_tcode",
                    "target_missing_policy": "none",
                    "x_missing_policy": "em_impute",
                    "target_outlier_policy": "none",
                    "x_outlier_policy": "none",
                    "scaling_policy": "standard",
                    "dimensionality_reduction_policy": "none",
                    "feature_selection_policy": "none",
                    "preprocess_order": "extra_only",
                    "preprocess_fit_scope": "train_only",
                    "inverse_transform_policy": "none",
                    "evaluation_scale": "raw_level",
                    "target_lag_block": "fixed_target_lags",
                    "factor_feature_block": "pca_static_factors",
                }
            },
            "3_training": {
                "fixed_axes": {
                    "framework": "expanding",
                    "benchmark_family": "zero_change",
                    "feature_builder": "raw_feature_panel",
                    "model_family": "ridge",
                }
            },
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"benchmark_config": {"minimum_train_size": 5}}},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }
    result = compile_recipe_dict(recipe)
    assert result.compiled.execution_status == "executable"
    blocks = result.manifest["layer2_representation_spec"]["feature_blocks"]
    assert blocks["target_lag_block"]["value"] == "fixed_target_lags"
    assert blocks["factor_feature_block"]["value"] == "pca_static_factors"
    execution = run_compiled_recipe(
        result.compiled,
        output_root=tmp_path,
        local_raw_source="tests/fixtures/fred_md_ar_sample.csv",
    )
    assert (Path(execution.artifact_dir) / "tuning_result.json").exists()


def test_layer2_explicit_x_lag_block_rejects_conflicting_legacy_bridge() -> None:
    recipe = {
        "recipe_id": "l2-x-lag-conflict",
        "path": {
            "0_meta": {"fixed_axes": {"research_design": "single_path_benchmark"}},
            "1_data_task": {
                "fixed_axes": {
                    "dataset": "fred_md",
                    "information_set_type": "revised",
                    "target_structure": "single_target_point_forecast",
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
                    "x_lag_feature_block": "fixed_x_lags",
                    "x_lag_creation": "no_x_lags",
                }
            },
            "3_training": {
                "fixed_axes": {
                    "framework": "expanding",
                    "benchmark_family": "zero_change",
                    "feature_builder": "raw_feature_panel",
                    "model_family": "ridge",
                }
            },
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"benchmark_config": {"minimum_train_size": 5}}},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }
    with pytest.raises(CompileValidationError, match="x_lag_feature_block conflicts"):
        compile_recipe_dict(recipe)


def test_layer2_explicit_target_lag_block_rejects_conflicting_selection() -> None:
    recipe = {
        "recipe_id": "l2-target-lag-conflict",
        "path": {
            "0_meta": {"fixed_axes": {"research_design": "single_path_benchmark"}},
            "1_data_task": {
                "fixed_axes": {
                    "dataset": "fred_md",
                    "information_set_type": "revised",
                    "target_structure": "single_target_point_forecast",
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
                    "target_lag_block": "fixed_target_lags",
                    "target_lag_selection": "ic_select",
                }
            },
            "3_training": {
                "fixed_axes": {
                    "framework": "expanding",
                    "benchmark_family": "zero_change",
                    "feature_builder": "autoreg_lagged_target",
                    "model_family": "ar",
                }
            },
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"benchmark_config": {"minimum_train_size": 5}}},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }
    with pytest.raises(CompileValidationError, match="target_lag_block conflicts"):
        compile_recipe_dict(recipe)


def test_layer2_explicit_factor_block_lowers_to_dimred_bridge() -> None:
    recipe = {
        "recipe_id": "l2-explicit-factor-block",
        "path": {
            "0_meta": {"fixed_axes": {"research_design": "single_path_benchmark"}},
            "1_data_task": {
                "fixed_axes": {
                    "dataset": "fred_md",
                    "information_set_type": "revised",
                    "target_structure": "single_target_point_forecast",
                },
                "leaf_config": {
                    "target": "INDPRO",
                    "horizons": [1],
                    "training_config": {"fixed_factor_count": 2, "max_factors": 4},
                },
            },
            "2_preprocessing": {
                "fixed_axes": {
                    "target_transform_policy": "raw_level",
                    "x_transform_policy": "raw_level",
                    "tcode_policy": "extra_preprocess_without_tcode",
                    "target_missing_policy": "none",
                    "x_missing_policy": "mean_impute",
                    "target_outlier_policy": "none",
                    "x_outlier_policy": "none",
                    "scaling_policy": "standard",
                    "dimensionality_reduction_policy": "pca",
                    "feature_selection_policy": "none",
                    "preprocess_order": "extra_only",
                    "preprocess_fit_scope": "train_only",
                    "inverse_transform_policy": "none",
                    "evaluation_scale": "raw_level",
                    "factor_feature_block": "pca_static_factors",
                }
            },
            "3_training": {
                "fixed_axes": {
                    "framework": "expanding",
                    "benchmark_family": "zero_change",
                    "feature_builder": "raw_feature_panel",
                    "model_family": "ridge",
                    "factor_count": "fixed",
                }
            },
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"benchmark_config": {"minimum_train_size": 5}}},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }
    result = compile_recipe_dict(recipe)
    assert result.compiled.execution_status == "executable"
    assert "factor_count" not in result.manifest["training_spec"]
    assert "fixed_factor_count" not in result.manifest["training_spec"]
    assert "max_factors" not in result.manifest["training_spec"]
    assert result.manifest["preprocess_contract"]["dimensionality_reduction_policy"] == "pca"
    block = result.manifest["layer2_representation_spec"]["feature_blocks"]["factor_feature_block"]
    assert block["value"] == "pca_static_factors"
    assert block["source_axis"] == "factor_feature_block"
    assert block["runtime_bridge"] == {"dimensionality_reduction_policy": "pca"}
    assert block["runtime_block"] == {
        "matrix_composition": "pca_static_factors",
        "default_dimensionality_reduction_policy": "pca",
    }
    assert block["factor_count"] == {
        "mode": "fixed",
        "fixed_factor_count": 2,
        "max_factors": 4,
        "selection_scope": "train_window",
    }
    assert block["feature_names"] == ["factor_1", "factor_2"]
    assert block["loadings_artifact"] == "feature_representation_fit_state.json"
    assert block["alignment"]["lookahead"] == "forbidden"


def test_layer2_factor_block_lowers_without_dimred_bridge() -> None:
    recipe = {
        "recipe_id": "l2-factor-block-missing-bridge",
        "path": {
            "0_meta": {"fixed_axes": {"research_design": "single_path_benchmark"}},
            "1_data_task": {
                "fixed_axes": {
                    "dataset": "fred_md",
                    "information_set_type": "revised",
                    "target_structure": "single_target_point_forecast",
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
                    "factor_feature_block": "pca_static_factors",
                }
            },
            "3_training": {
                "fixed_axes": {
                    "framework": "expanding",
                    "benchmark_family": "zero_change",
                    "feature_builder": "raw_feature_panel",
                    "model_family": "ridge",
                }
            },
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"benchmark_config": {"minimum_train_size": 5}}},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }
    result = compile_recipe_dict(recipe)
    assert result.compiled.execution_status == "executable"
    assert not any("factor_feature_block='pca_static_factors' requires" in warning for warning in result.compiled.warnings)
    block = result.manifest["layer2_representation_spec"]["feature_blocks"]["factor_feature_block"]
    assert block["value"] == "pca_static_factors"
    assert block["runtime_bridge"] == {}
    assert block["runtime_block"] == {
        "matrix_composition": "pca_static_factors",
        "default_dimensionality_reduction_policy": "pca",
    }


def test_layer2_factor_block_accepts_select_before_factor_mix() -> None:
    recipe = {
        "recipe_id": "l2-factor-block-selection-mix",
        "path": {
            "0_meta": {"fixed_axes": {"research_design": "single_path_benchmark"}},
            "1_data_task": {
                "fixed_axes": {
                    "dataset": "fred_md",
                    "information_set_type": "revised",
                    "target_structure": "single_target_point_forecast",
                },
                "leaf_config": {"target": "INDPRO", "horizons": [1]},
            },
            "2_preprocessing": {
                "fixed_axes": {
                    "target_transform_policy": "raw_level",
                    "x_transform_policy": "raw_level",
                    "tcode_policy": "extra_preprocess_without_tcode",
                    "target_missing_policy": "none",
                    "x_missing_policy": "mean_impute",
                    "target_outlier_policy": "none",
                    "x_outlier_policy": "none",
                    "scaling_policy": "standard",
                    "dimensionality_reduction_policy": "pca",
                    "feature_selection_policy": "lasso_select",
                    "preprocess_order": "extra_only",
                    "preprocess_fit_scope": "train_only",
                    "inverse_transform_policy": "none",
                    "evaluation_scale": "raw_level",
                    "factor_feature_block": "pca_static_factors",
                }
            },
            "3_training": {
                "fixed_axes": {
                    "framework": "expanding",
                    "benchmark_family": "zero_change",
                    "feature_builder": "raw_feature_panel",
                    "model_family": "ridge",
                }
            },
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"benchmark_config": {"minimum_train_size": 5}}},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }
    result = compile_recipe_dict(recipe)
    assert result.compiled.execution_status == "executable"
    block = result.manifest["layer2_representation_spec"]["feature_blocks"]["factor_feature_block"]
    interaction = block["feature_selection_interaction"]
    assert interaction["feature_selection_policy"] == "lasso_select"
    assert interaction["active_semantic"] == "select_before_factor"
    assert interaction["supported_semantics"] == ["select_before_factor", "select_after_factor"]


def test_layer2_factor_block_accepts_select_after_factor_mix() -> None:
    recipe = {
        "recipe_id": "l2-factor-block-selection-mix-after-factor",
        "path": {
            "0_meta": {"fixed_axes": {"research_design": "single_path_benchmark"}},
            "1_data_task": {
                "fixed_axes": {
                    "dataset": "fred_md",
                    "information_set_type": "revised",
                    "target_structure": "single_target_point_forecast",
                },
                "leaf_config": {"target": "INDPRO", "horizons": [1]},
            },
            "2_preprocessing": {
                "fixed_axes": {
                    "target_transform_policy": "raw_level",
                    "x_transform_policy": "raw_level",
                    "tcode_policy": "extra_preprocess_without_tcode",
                    "target_missing_policy": "none",
                    "x_missing_policy": "mean_impute",
                    "target_outlier_policy": "none",
                    "x_outlier_policy": "none",
                    "scaling_policy": "standard",
                    "dimensionality_reduction_policy": "pca",
                    "feature_selection_policy": "lasso_select",
                    "feature_selection_semantics": "select_after_factor",
                    "preprocess_order": "extra_only",
                    "preprocess_fit_scope": "train_only",
                    "inverse_transform_policy": "none",
                    "evaluation_scale": "raw_level",
                    "factor_feature_block": "pca_static_factors",
                }
            },
            "3_training": {
                "fixed_axes": {
                    "framework": "expanding",
                    "benchmark_family": "zero_change",
                    "feature_builder": "raw_feature_panel",
                    "model_family": "ridge",
                }
            },
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"benchmark_config": {"minimum_train_size": 5}}},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }
    result = compile_recipe_dict(recipe)
    assert result.compiled.execution_status == "executable"
    block = result.manifest["layer2_representation_spec"]["feature_blocks"]["factor_feature_block"]
    interaction = block["feature_selection_interaction"]
    assert interaction["feature_selection_policy"] == "lasso_select"
    assert interaction["active_semantic"] == "select_after_factor"
    assert interaction["supported_semantics"] == ["select_before_factor", "select_after_factor"]
    assert interaction["gated_semantics"] == ["select_after_custom_blocks"]


@pytest.mark.parametrize("factor_block", ["pca_factor_lags", "supervised_factors"])
def test_layer2_non_pca_factor_blocks_accept_selection_semantics(factor_block: str) -> None:
    recipe = _layer2_temporal_block_recipe(
        temporal_feature_block="none",
        rotation_feature_block="none",
    )
    recipe["path"]["2_preprocessing"]["fixed_axes"].update(
        {
            "factor_feature_block": factor_block,
            "feature_selection_policy": "lasso_select",
            "feature_selection_semantics": "select_after_factor",
            "tcode_policy": "extra_preprocess_without_tcode",
            "preprocess_order": "extra_only",
            "preprocess_fit_scope": "train_only",
            "scaling_policy": "standard",
        }
    )

    result = compile_recipe_dict(recipe)

    assert result.compiled.execution_status == "executable"
    block = result.manifest["layer2_representation_spec"]["feature_blocks"]["factor_feature_block"]
    interaction = block["feature_selection_interaction"]
    assert interaction["feature_selection_policy"] == "lasso_select"
    assert interaction["active_semantic"] == "select_after_factor"
    assert interaction["gated_semantics"] == ["select_after_custom_blocks"]


def test_layer2_target_representation_records_scale_contract() -> None:
    recipe = _layer2_temporal_block_recipe(
        temporal_feature_block="none",
        rotation_feature_block="none",
    )
    recipe["path"]["2_preprocessing"]["fixed_axes"].update(
        {
            "tcode_policy": "extra_preprocess_without_tcode",
            "preprocess_order": "extra_only",
            "preprocess_fit_scope": "train_only",
            "x_missing_policy": "mean_impute",
            "scaling_policy": "standard",
            "target_transform": "log",
            "target_normalization": "zscore_train_only",
            "inverse_transform_policy": "target_only",
            "evaluation_scale": "both",
        }
    )

    result = compile_recipe_dict(recipe)

    assert result.compiled.execution_status == "executable"
    target_rep = result.manifest["layer2_representation_spec"]["target_representation"]
    scale = target_rep["target_scale_contract"]
    assert target_rep["target_transform"] == "log"
    assert target_rep["target_normalization"] == "zscore_train_only"
    assert scale["schema_version"] == "target_scale_contract_v1"
    assert scale["runtime_status"] == "operational"
    assert scale["model_target_scale"] == "transformed_target_scale"
    assert scale["forecast_scale"] == "original_target_scale"
    assert scale["blockers"] == []


def test_layer2_custom_factor_block_records_callable_contract() -> None:
    recipe = _layer2_temporal_block_recipe(
        temporal_feature_block="none",
        rotation_feature_block="none",
    )
    recipe["path"]["2_preprocessing"]["fixed_axes"]["factor_feature_block"] = "custom_factors"

    result = compile_recipe_dict(recipe)

    assert result.compiled.execution_status == "not_supported"
    block = result.manifest["layer2_representation_spec"]["feature_blocks"]["factor_feature_block"]
    assert block["value"] == "custom_factors"
    assert block["runtime_status"] == "registry_only"
    assert block["required_runtime_contract"] == "custom_feature_block_callable_v1"
    assert block["callable_contract"]["block_kind"] == "factor"


def test_compile_recipe_accepts_stage3_training_axes() -> None:
    recipe = {
        "recipe_id": "stage3-training-axes",
        "path": {
            "0_meta": {"fixed_axes": {"research_design": "single_path_benchmark"}},
            "1_data_task": {
                "fixed_axes": {"dataset": "fred_md", "information_set_type": "revised", "task": "single_target_point_forecast"},
                "leaf_config": {"target": "INDPRO", "horizons": [1, 3]},
            },
            "2_preprocessing": {"fixed_axes": {
                "target_transform_policy": "raw_level", "x_transform_policy": "raw_level", "tcode_policy": "raw_only",
                "target_missing_policy": "none", "x_missing_policy": "none", "target_outlier_policy": "none", "x_outlier_policy": "none",
                "scaling_policy": "none", "dimensionality_reduction_policy": "none", "feature_selection_policy": "none",
                "preprocess_order": "none", "preprocess_fit_scope": "not_applicable", "inverse_transform_policy": "none", "evaluation_scale": "raw_level"
            }},
            "3_training": {"fixed_axes": {
                "framework": "expanding", "outer_window": "expanding", "refit_policy": "refit_every_step", "benchmark_family": "historical_mean",
                "feature_builder": "autoreg_lagged_target", "model_family": "ols", "search_algorithm": "grid_search", "seed_policy": "fixed_seed"
            }},
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"manifest_mode": "full", "benchmark_config": {"minimum_train_size": 5}}},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }
    compile_result = compile_recipe_dict(recipe)
    assert compile_result.compiled.execution_status == "executable"
    assert compile_result.manifest["training_spec"]["search_algorithm"] == "grid_search"



def test_compiled_manifest_records_training_config_passthrough_defaults() -> None:
    compile_result = compile_recipe_yaml("examples/recipes/model-benchmark.yaml")
    spec = compile_result.manifest["training_spec"]
    assert spec["validation_ratio"] == 0.2
    assert spec["max_trials"] == 6
    assert "factor_count" not in spec
    assert "fixed_factor_count" not in spec
    assert "max_factors" not in spec


def test_axis_governance_table_marks_quantile_linear_and_point_median_operational() -> None:
    table = axis_governance_table()
    by_name = {row["axis_name"]: row for row in table}
    assert by_name["model_family"]["current_status"]["quantile_linear"] == "operational"
    assert by_name["forecast_object"]["current_status"]["point_median"] == "operational"


def test_compile_quantile_linear_point_median_recipe_is_executable(tmp_path: Path) -> None:
    recipe = {
        "recipe_id": "quantile-linear-median-rolling",
        "path": {
            "0_meta": {"fixed_axes": {"research_design": "single_path_benchmark"}},
            "1_data_task": {
                "fixed_axes": {"dataset": "fred_md", "info_set": "revised", "task": "single_target_point_forecast", "forecast_object": "point_median"},
                "leaf_config": {"target": "INDPRO", "horizons": [1, 3]},
            },
            "2_preprocessing": {"fixed_axes": {
                "target_transform_policy": "raw_level", "x_transform_policy": "raw_level", "tcode_policy": "raw_only",
                "target_missing_policy": "none", "x_missing_policy": "none", "target_outlier_policy": "none", "x_outlier_policy": "none",
                "scaling_policy": "none", "dimensionality_reduction_policy": "none", "feature_selection_policy": "none",
                "preprocess_order": "none", "preprocess_fit_scope": "not_applicable", "inverse_transform_policy": "none", "evaluation_scale": "raw_level"
            }},
            "3_training": {"fixed_axes": {
                "framework": "rolling", "benchmark_family": "zero_change", "feature_builder": "autoreg_lagged_target", "model_family": "quantile_linear"
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
        local_raw_source=Path("tests/fixtures/fred_md_ar_sample.csv"),
    )
    manifest = json.loads((Path(execution.artifact_dir) / "manifest.json").read_text())
    assert manifest["model_spec"]["model_family"] == "quantile_linear"


def test_axis_governance_table_marks_stage4_eval_axes_operational() -> None:
    table = axis_governance_table()
    by_name = {row["axis_name"]: row for row in table}
    assert by_name["primary_metric"]["current_status"]["rmse"] == "operational"
    assert by_name["primary_metric"]["current_status"]["mae"] == "operational"
    assert by_name["primary_metric"]["current_status"]["mape"] == "operational"
    assert by_name["relative_metrics"]["current_status"]["relative_RMSE"] == "operational"
    assert by_name["direction_metrics"]["current_status"]["directional_accuracy"] == "operational"
    assert by_name["regime_definition"]["current_status"]["NBER_recession"] == "operational"
    assert by_name["regime_definition"]["current_status"]["user_defined_regime"] == "operational"


def test_compiled_manifest_records_stage4_evaluation_defaults() -> None:
    compile_result = compile_recipe_yaml("examples/recipes/model-benchmark.yaml")
    spec = compile_result.manifest["evaluation_spec"]
    assert spec["primary_metric"] == "msfe"
    assert spec["relative_metrics"] == "relative_MSFE"
    assert spec["direction_metrics"] == "directional_accuracy"
    assert spec["regime_definition"] == "none"


def test_compile_primary_metric_rmse_recipe_is_executable() -> None:
    recipe = {
        "recipe_id": "stage4-primary-metric-rmse",
        "path": {
            "0_meta": {"fixed_axes": {"research_design": "single_path_benchmark"}},
            "1_data_task": {
                "fixed_axes": {"dataset": "fred_md", "information_set_type": "revised", "task": "single_target_point_forecast"},
                "leaf_config": {"target": "INDPRO", "horizons": [1, 3]},
            },
            "2_preprocessing": {"fixed_axes": {
                "target_transform_policy": "raw_level", "x_transform_policy": "raw_level", "tcode_policy": "raw_only",
                "target_missing_policy": "none", "x_missing_policy": "none", "target_outlier_policy": "none", "x_outlier_policy": "none",
                "scaling_policy": "none", "dimensionality_reduction_policy": "none", "feature_selection_policy": "none",
                "preprocess_order": "none", "preprocess_fit_scope": "not_applicable", "inverse_transform_policy": "none", "evaluation_scale": "raw_level"
            }},
            "3_training": {"fixed_axes": {"framework": "rolling", "benchmark_family": "zero_change", "feature_builder": "raw_feature_panel", "model_family": "ridge"}},
            "4_evaluation": {"fixed_axes": {"primary_metric": "rmse"}},
            "5_output_provenance": {"leaf_config": {"manifest_mode": "full", "benchmark_config": {"minimum_train_size": 5, "rolling_window_size": 5}}},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }
    compile_result = compile_recipe_dict(recipe)
    assert compile_result.compiled.execution_status == "executable"
    assert compile_result.manifest["evaluation_spec"]["primary_metric"] == "rmse"



def test_compile_manifest_includes_output_spec_defaults() -> None:
    compile_result = compile_recipe_yaml("examples/recipes/model-benchmark.yaml")
    output_spec = compile_result.manifest["output_spec"]
    assert output_spec == {
        "export_format": "json",
        "saved_objects": "full_bundle",
        "provenance_fields": "full",
        "artifact_granularity": "aggregated",
    }



def test_compile_extended_stage6_stat_test_manifest() -> None:
    compile_result = compile_recipe_yaml("examples/recipes/model-benchmark.yaml")
    stat_test_spec = compile_result.manifest["stat_test_spec"]
    assert stat_test_spec["stat_test"] == "none"
    assert stat_test_spec["dependence_correction"] == "none"



def test_compile_stage7_importance_defaults() -> None:
    compile_result = compile_recipe_yaml("examples/recipes/model-benchmark.yaml")
    assert compile_result.manifest["importance_spec"]["importance_method"] == "none"


def test_multi_target_derives_shared_design_experiment_unit() -> None:
    """Multi-target recipes with no explicit experiment_unit must auto-derive
    multi_target_shared_design (operational, single_run route). Prior to
    2026-04-18 this returned the registry_only multi_target_separate_runs,
    producing an executable status but a latent label mismatch."""
    recipe = {
        "recipe_id": "multi-target-shared-derive",
        "path": {
            "0_meta": {"fixed_axes": {"research_design": "single_path_benchmark"}},
            "1_data_task": {
                "fixed_axes": {
                    "dataset": "fred_md",
                    "info_set": "revised",
                    "task": "multi_target_point_forecast",
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
                "feature_builder": "autoreg_lagged_target", "model_family": "ar",
            }},
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"manifest_mode": "full", "benchmark_config": {"minimum_train_size": 5, "rolling_window_size": 5}}},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }
    compile_result = compile_recipe_dict(recipe)
    assert compile_result.compiled.execution_status == "executable"
    assert compile_result.compiled.stage0.experiment_unit == "multi_target_shared_design"


def test_wizard_options_filter_registry_only_entries() -> None:
    from macrocast.registry.stage0.experiment_unit import experiment_unit_options_for_wizard
    # Multi-target options should include both operational units (shared_design,
    # separate_runs) after the PR #27 cleanup. multi_output_joint_model was
    # dropped from the registry entirely in the same PR; filter would exclude
    # any registry_only future additions.
    options = experiment_unit_options_for_wizard(
        research_design="single_path_benchmark",
        task="multi_target_point_forecast",
    )
    assert set(options) == {"multi_target_shared_design", "multi_target_separate_runs"}


def test_wizard_options_single_target_returns_operational_only() -> None:
    from macrocast.registry.stage0.experiment_unit import experiment_unit_options_for_wizard
    options = experiment_unit_options_for_wizard(
        research_design="single_path_benchmark",
        task="single_target_point_forecast",
    )
    # unsupported full-sweep wrapper routes are not offered by the wizard.
    assert set(options) == {
        "single_target_single_model",
        "single_target_model_grid",
    }


# --- 0.6 research_design compile status per-value ---


def _study_mode_recipe_base() -> dict:
    return {
        "recipe_id": "study-mode-compile-status",
        "path": {
            "0_meta": {"fixed_axes": {"research_design": "PLACEHOLDER"}},
            "1_data_task": {
                "fixed_axes": {"dataset": "fred_md", "info_set": "revised", "task": "single_target_point_forecast"},
                "leaf_config": {"target": "INDPRO", "horizons": [1]},
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
                "feature_builder": "autoreg_lagged_target", "model_family": "ar",
            }},
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"manifest_mode": "full", "benchmark_config": {"minimum_train_size": 5, "rolling_window_size": 5}}},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }


def test_study_mode_single_path_benchmark_compiles_executable() -> None:
    recipe = _study_mode_recipe_base()
    recipe["path"]["0_meta"]["fixed_axes"]["research_design"] = "single_path_benchmark"
    r = compile_recipe_dict(recipe)
    assert r.compiled.execution_status == "executable"


def test_study_mode_controlled_variation_compiles_executable() -> None:
    recipe = _study_mode_recipe_base()
    recipe["path"]["0_meta"]["fixed_axes"]["research_design"] = "controlled_variation"
    r = compile_recipe_dict(recipe)
    assert r.compiled.execution_status == "executable"


def test_study_mode_replication_override_requires_replication_runner() -> None:
    """replication_override is a replication handoff, not a direct recipe run."""
    recipe = _study_mode_recipe_base()
    recipe["path"]["0_meta"]["fixed_axes"]["research_design"] = "replication_override"
    r = compile_recipe_dict(recipe)
    assert r.compiled.execution_status == "ready_for_replication_runner"
    assert r.manifest["tree_context"]["route_owner"] == "replication"
    assert r.manifest["tree_context"]["route_contract"] == "replication_handoff"


def test_study_mode_orchestrated_bundle_without_runner_contract_is_not_supported() -> None:
    """orchestrated_bundle stays in the grammar, but unsupported wrapper
    families must not be reported as runnable."""
    recipe = _study_mode_recipe_base()
    recipe["path"]["0_meta"]["fixed_axes"]["research_design"] = "orchestrated_bundle"
    recipe["path"]["5_output_provenance"]["leaf_config"]["wrapper_family"] = "benchmark_suite"
    recipe["path"]["5_output_provenance"]["leaf_config"]["bundle_label"] = "bundle-test"
    r = compile_recipe_dict(recipe)
    assert r.compiled.execution_status == "not_supported"
    assert any(
        "benchmark_suite" in w and "no executable wrapper runner contract" in w
        for w in r.manifest.get("warnings", [])
    )
