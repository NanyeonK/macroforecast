"""Smoke tests for custom_csv / custom_parquet loaders."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from macrocast import load_custom_csv, load_custom_parquet
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


def test_custom_csv_via_execute_recipe_requires_custom_data_path() -> None:
    """Compiler-level validation: source_adapter=custom_csv without
    leaf_config.custom_data_path raises CompileValidationError."""
    from macrocast.compiler.build import compile_recipe_dict
    from macrocast.compiler.errors import CompileValidationError

    recipe = {
        "recipe_id": "custom-csv-missing-path",
        "path": {
            "0_meta": {"fixed_axes": {"research_design": "single_path_benchmark"}},
            "1_data_task": {
                "fixed_axes": {
                    "dataset": "fred_md",
                    "source_adapter": "custom_csv",
                    "info_set": "revised",
                    "task": "single_target_point_forecast",
                },
                "leaf_config": {"target": "INDPRO", "horizons": [1]},  # no custom_data_path!
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
                "feature_builder": "autoreg_lagged_target", "model_family": "ar",
            }},
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"manifest_mode": "full", "benchmark_config": {"minimum_train_size": 5}}},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }
    with pytest.raises(CompileValidationError, match="custom_data_path"):
        compile_recipe_dict(recipe)


def test_legacy_dataset_source_alias_compiles_to_source_adapter(tmp_path: Path) -> None:
    """Old recipes may still say dataset_source during the compatibility window."""
    from macrocast.compiler.build import compile_recipe_dict

    csv_path = tmp_path / "custom.csv"
    _write_sample_csv(csv_path)
    recipe = {
        "recipe_id": "legacy-dataset-source",
        "path": {
            "0_meta": {"fixed_axes": {"research_design": "single_path_benchmark"}},
            "1_data_task": {
                "fixed_axes": {
                    "dataset": "fred_md",
                    "dataset_source": "custom_csv",
                    "info_set": "revised",
                    "task": "single_target_point_forecast",
                },
                "leaf_config": {
                    "target": "INDPRO",
                    "horizons": [1],
                    "custom_data_path": str(csv_path),
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
                "feature_builder": "autoreg_lagged_target", "model_family": "ar",
            }},
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"manifest_mode": "full", "benchmark_config": {"minimum_train_size": 5}}},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }
    compile_result = compile_recipe_dict(recipe)
    spec = compile_result.manifest["data_task_spec"]
    assert spec["source_adapter"] == "custom_csv"
    assert "dataset_source" not in spec


def test_dataset_source_alias_conflicts_with_source_adapter() -> None:
    from macrocast.compiler.build import compile_recipe_dict
    from macrocast.compiler.errors import CompileValidationError

    recipe = {
        "recipe_id": "conflicting-source-adapter",
        "path": {
            "0_meta": {"fixed_axes": {"research_design": "single_path_benchmark"}},
            "1_data_task": {
                "fixed_axes": {
                    "dataset": "fred_md",
                    "source_adapter": "custom_csv",
                    "dataset_source": "custom_parquet",
                    "info_set": "revised",
                    "task": "single_target_point_forecast",
                },
                "leaf_config": {
                    "target": "INDPRO",
                    "horizons": [1],
                    "custom_data_path": "/tmp/unused.csv",
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
                "feature_builder": "autoreg_lagged_target", "model_family": "ar",
            }},
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"manifest_mode": "full", "benchmark_config": {"minimum_train_size": 5}}},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }
    with pytest.raises(CompileValidationError, match="canonical/legacy aliases"):
        compile_recipe_dict(recipe)
