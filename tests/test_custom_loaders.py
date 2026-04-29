"""Smoke tests for custom_csv / custom_parquet loaders."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from macrocast import load_custom_csv, load_custom_parquet
from macrocast.compiler.errors import CompileValidationError
from macrocast.raw.errors import RawParseError


def _write_sample_csv(path: Path) -> None:
    df = pd.DataFrame(
        {"INDPRO": [101.0, 102.0, 103.5, 104.2], "RPI": [200.1, 200.5, 201.0, 201.8]},
        index=pd.date_range("2020-01-01", periods=4, freq="MS"),
    )
    df.index.name = "date"
    df.to_csv(path)


def _write_sample_parquet(path: Path) -> None:
    df = pd.DataFrame(
        {"INDPRO": [101.0, 102.0, 103.5, 104.2], "RPI": [200.1, 200.5, 201.0, 201.8]},
        index=pd.date_range("2020-01-01", periods=4, freq="MS"),
    )
    df.index.name = "date"
    df.to_parquet(path)


def test_load_custom_csv_basic(tmp_path: Path) -> None:
    csv_path = tmp_path / "custom.csv"
    _write_sample_csv(csv_path)
    result = load_custom_csv(csv_path, dataset="fred_md")
    assert result.dataset_metadata.source_family == "custom-csv"
    assert result.dataset_metadata.dataset == "fred_md"
    assert result.dataset_metadata.frequency == "monthly"
    assert list(result.data.columns) == ["INDPRO", "RPI"]
    assert len(result.data) == 4


def test_load_custom_parquet_basic(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    pq_path = tmp_path / "custom.parquet"
    _write_sample_parquet(pq_path)
    result = load_custom_parquet(pq_path, dataset="fred_md")
    assert result.dataset_metadata.source_family == "custom-parquet"
    assert result.dataset_metadata.dataset == "fred_md"
    assert list(result.data.columns) == ["INDPRO", "RPI"]


def test_load_custom_csv_rejects_unsupported_dataset(tmp_path: Path) -> None:
    csv_path = tmp_path / "custom.csv"
    _write_sample_csv(csv_path)
    with pytest.raises(RawParseError, match="not a supported schema"):
        load_custom_csv(csv_path, dataset="not_a_real_schema")


def test_load_custom_csv_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(RawParseError, match="does not exist"):
        load_custom_csv(tmp_path / "missing.csv", dataset="fred_md")


def test_custom_csv_via_execute_recipe_requires_custom_source_path() -> None:
    """Compiler-level validation: custom CSV without
    leaf_config.custom_source_path raises CompileValidationError."""
    from macrocast.compiler.build import compile_recipe_dict
    from macrocast.compiler.errors import CompileValidationError

    recipe = {
        "recipe_id": "custom-csv-missing-path",
        "path": {
            "0_meta": {"fixed_axes": {"study_scope": "one_target_one_method"}},
            "1_data_task": {
                "fixed_axes": {
                    "dataset": "fred_md",
                    "custom_source_policy": "custom_panel_only",
                    "information_set_type": "final_revised_data",
                    "target_structure": "single_target",
                },
                "leaf_config": {
                    "target": "INDPRO",
                    "horizons": [1],
                },  # no custom_source_path!
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
                "framework": "expanding", "benchmark_family": "zero_change",
                "feature_builder": "target_lag_features", "model_family": "ar",
            }},
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"manifest_mode": "full", "benchmark_config": {"minimum_train_size": 5}}},
            "6_stat_tests": {"fixed_axes": {}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }
    with pytest.raises(CompileValidationError, match="custom_source_path"):
        compile_recipe_dict(recipe)


def test_custom_csv_source_infers_custom_source_schema(tmp_path: Path) -> None:
    from macrocast.compiler.build import compile_recipe_dict

    csv_path = tmp_path / "custom.csv"
    _write_sample_csv(csv_path)
    recipe = {
        "recipe_id": "custom-csv-missing-schema",
        "path": {
            "0_meta": {"fixed_axes": {"study_scope": "one_target_one_method"}},
            "1_data_task": {
                "fixed_axes": {
                    "dataset": "fred_md",
                    "custom_source_policy": "custom_panel_only",
                    "information_set_type": "final_revised_data",
                    "target_structure": "single_target",
                },
                "leaf_config": {
                    "target": "INDPRO",
                    "horizons": [1],
                    "custom_source_path": str(csv_path),
                },
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
                "framework": "expanding", "benchmark_family": "zero_change",
                "feature_builder": "target_lag_features", "model_family": "ar",
            }},
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"manifest_mode": "full", "benchmark_config": {"minimum_train_size": 5}}},
            "6_stat_tests": {"fixed_axes": {}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }
    result = compile_recipe_dict(recipe)
    spec = result.compiled.recipe_spec.data_task_spec
    assert spec["custom_source_format"] == "csv"
    assert spec["custom_source_schema"] == "fred_md"


def test_custom_csv_source_compiles_to_schema(tmp_path: Path) -> None:
    from macrocast.compiler.build import compile_recipe_dict

    csv_path = tmp_path / "custom.csv"
    _write_sample_csv(csv_path)
    recipe = {
        "recipe_id": "custom-csv-dataset",
        "path": {
            "0_meta": {"fixed_axes": {"study_scope": "one_target_one_method"}},
            "1_data_task": {
                "fixed_axes": {
                    "dataset": "fred_md",
                    "custom_source_policy": "custom_panel_only",
                    "frequency": "monthly",
                    "information_set_type": "final_revised_data",
                    "target_structure": "single_target",
                },
                "leaf_config": {
                    "target": "INDPRO",
                    "horizons": [1],
                    "custom_source_path": str(csv_path),
                },
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
                "framework": "expanding", "benchmark_family": "zero_change",
                "feature_builder": "target_lag_features", "model_family": "ar",
            }},
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"manifest_mode": "full", "benchmark_config": {"minimum_train_size": 5}}},
            "6_stat_tests": {"fixed_axes": {}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }

    result = compile_recipe_dict(recipe)
    recipe_spec = result.compiled.recipe_spec
    assert recipe_spec.raw_dataset == "fred_md"
    assert recipe_spec.data_task_spec["dataset"] == "fred_md"
    assert recipe_spec.data_task_spec["dataset_schema"] == "fred_md"
    assert recipe_spec.data_task_spec["custom_source_policy"] == "custom_panel_only"
    assert recipe_spec.data_task_spec["custom_source_format"] == "csv"
    assert "source_adapter" not in recipe_spec.data_task_spec


def test_custom_source_replace_runtime_loads_custom_csv(tmp_path: Path) -> None:
    from macrocast.compiler.build import compile_recipe_dict
    from macrocast.execution.build import _load_raw_for_recipe

    csv_path = tmp_path / "custom.csv"
    _write_sample_csv(csv_path)
    recipe = {
        "recipe_id": "custom-replace-runtime",
        "path": {
            "0_meta": {"fixed_axes": {"study_scope": "one_target_one_method"}},
            "1_data_task": {
                "fixed_axes": {
                    "dataset": "fred_md",
                    "custom_source_policy": "custom_panel_only",
                    "frequency": "monthly",
                    "information_set_type": "final_revised_data",
                    "target_structure": "single_target",
                },
                "leaf_config": {
                    "target": "INDPRO",
                    "horizons": [1],
                    "custom_source_path": str(csv_path),
                },
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
                "framework": "expanding", "benchmark_family": "zero_change",
                "feature_builder": "target_lag_features", "model_family": "ar",
            }},
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"manifest_mode": "full", "benchmark_config": {"minimum_train_size": 5}}},
            "6_stat_tests": {"fixed_axes": {}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }

    compiled = compile_recipe_dict(recipe)
    raw = _load_raw_for_recipe(compiled.compiled.recipe_spec, None, tmp_path / "cache")
    assert raw.dataset_metadata.source_family == "custom-csv"
    assert raw.dataset_metadata.dataset == "fred_md"
    assert list(raw.data.columns) == ["INDPRO", "RPI"]


def test_default_recipe_accepts_custom_csv_replacement(tmp_path: Path) -> None:
    from macrocast.defaults import build_default_recipe_dict

    csv_path = tmp_path / "custom.csv"
    _write_sample_csv(csv_path)

    recipe = build_default_recipe_dict(
        dataset="fred_md",
        target="INDPRO",
        start="2020-01",
        end="2020-04",
        horizons=[1],
        custom_source_policy="custom_panel_only",
        custom_source_path=str(csv_path),
    )

    data_task = recipe["path"]["1_data_task"]
    assert data_task["fixed_axes"]["dataset"] == "fred_md"
    assert data_task["fixed_axes"]["custom_source_policy"] == "custom_panel_only"
    assert "custom_source_format" not in data_task["fixed_axes"]
    assert "custom_source_schema" not in data_task["fixed_axes"]
    assert data_task["fixed_axes"]["frequency"] == "monthly"
    assert data_task["leaf_config"]["custom_source_path"] == str(csv_path)


def test_custom_source_append_compiles_to_data_task_spec(tmp_path: Path) -> None:
    from macrocast.compiler.build import compile_recipe_dict

    csv_path = tmp_path / "custom.csv"
    _write_sample_csv(csv_path)
    recipe = {
        "recipe_id": "custom-append",
        "path": {
            "0_meta": {"fixed_axes": {"study_scope": "one_target_one_method"}},
            "1_data_task": {
                "fixed_axes": {
                    "dataset": "fred_md",
                    "custom_source_policy": "official_plus_custom",
                    "frequency": "monthly",
                    "information_set_type": "final_revised_data",
                    "target_structure": "single_target",
                },
                "leaf_config": {
                    "target": "INDPRO",
                    "horizons": [1],
                    "custom_source_path": str(csv_path),
                },
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
                "framework": "expanding", "benchmark_family": "zero_change",
                "feature_builder": "target_lag_features", "model_family": "ar",
            }},
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"manifest_mode": "full", "benchmark_config": {"minimum_train_size": 5}}},
            "6_stat_tests": {"fixed_axes": {}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }

    result = compile_recipe_dict(recipe)
    spec = result.compiled.recipe_spec.data_task_spec
    assert spec["dataset"] == "fred_md"
    assert spec["dataset_schema"] == "fred_md"
    assert spec["custom_source_policy"] == "official_plus_custom"
    assert spec["custom_source_format"] == "csv"
    assert spec["custom_source_schema"] == "fred_md"


def test_dataset_source_alias_is_rejected(tmp_path: Path) -> None:
    from macrocast.compiler.build import compile_recipe_dict

    csv_path = tmp_path / "custom.csv"
    _write_sample_csv(csv_path)
    recipe = {
        "recipe_id": "legacy-dataset-source",
        "path": {
            "0_meta": {"fixed_axes": {"study_scope": "one_target_one_method"}},
            "1_data_task": {
                "fixed_axes": {
                    "dataset": "fred_md",
                    "dataset_source": "custom_csv",
                    "information_set_type": "final_revised_data",
                    "target_structure": "single_target",
                },
                "leaf_config": {
                    "target": "INDPRO",
                    "horizons": [1],
                    "custom_source_path": str(csv_path),
                },
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
                "framework": "expanding", "benchmark_family": "zero_change",
                "feature_builder": "target_lag_features", "model_family": "ar",
            }},
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"manifest_mode": "full", "benchmark_config": {"minimum_train_size": 5}}},
            "6_stat_tests": {"fixed_axes": {}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }
    with pytest.raises(CompileValidationError, match="unknown registry axis 'dataset_source'"):
        compile_recipe_dict(recipe)


def test_source_adapter_axis_is_rejected() -> None:
    from macrocast.compiler.build import compile_recipe_dict
    from macrocast.compiler.errors import CompileValidationError

    recipe = {
        "recipe_id": "removed-source-adapter",
        "path": {
            "0_meta": {"fixed_axes": {"study_scope": "one_target_one_method"}},
            "1_data_task": {
                "fixed_axes": {
                    "dataset": "fred_md",
                    "source_adapter": "custom_csv",
                    "information_set_type": "final_revised_data",
                    "target_structure": "single_target",
                },
                "leaf_config": {
                    "target": "INDPRO",
                    "horizons": [1],
                    "custom_source_path": "/tmp/unused.csv",
                },
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
                "framework": "expanding", "benchmark_family": "zero_change",
                "feature_builder": "target_lag_features", "model_family": "ar",
            }},
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"manifest_mode": "full", "benchmark_config": {"minimum_train_size": 5}}},
            "6_stat_tests": {"fixed_axes": {}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }
    with pytest.raises(CompileValidationError, match="unknown registry axis 'source_adapter'"):
        compile_recipe_dict(recipe)
