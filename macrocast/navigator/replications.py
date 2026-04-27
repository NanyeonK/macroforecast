from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

REPLICATION_LIBRARY_VERSION = "replication_library_v1"


@dataclass(frozen=True)
class ReplicationEntry:
    id: str
    paper_name: str
    short_description: str
    exact_tree_path: tuple[str, ...]
    recipe_yaml: str
    command: str
    notebook_snippet: str
    expected_outputs: tuple[str, ...]
    deviations_from_original_paper: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["recipe"] = yaml.safe_load(self.recipe_yaml)
        return payload


_GOULET_COULOMBE_2021_RECIPE: dict[str, Any] = {
    "recipe_id": "goulet-coulombe-2021-fred-md-ridge",
    "path": {
        "0_meta": {
            "fixed_axes": {
                "research_design": "single_forecast_run",
                "failure_policy": "fail_fast",
                "compute_mode": "serial",
                "reproducibility_mode": "seeded_reproducible",
            },
            "leaf_config": {"random_seed": 42},
        },
        "1_data_task": {
            "fixed_axes": {
                "dataset": "fred_md",
                "source_adapter": "fred_md",
                "frequency": "monthly",
                "information_set_type": "final_revised_data",
                "target_structure": "single_target",
                "variable_universe": "all_variables",
                "missing_availability": "zero_fill_leading_predictor_gaps",
                "release_lag_rule": "ignore_release_lag",
                "contemporaneous_x_rule": "forbid_same_period_predictors",
                "official_transform_policy": "apply_official_tcode",
                "official_transform_scope": "target_and_predictors",
                "raw_missing_policy": "zero_fill_leading_predictor_missing_before_tcode",
                "raw_outlier_policy": "preserve_raw_outliers",
            },
            "leaf_config": {"target": "INDPRO", "horizons": [1, 3, 6, 12]},
        },
        "2_preprocessing": {
            "fixed_axes": {
                "horizon_target_construction": "future_target_level_t_plus_h",
                "target_transform_policy": "official_tcode_transformed",
                "x_transform_policy": "official_tcode_transformed",
                "tcode_policy": "official_tcode_then_extra_preprocess",
                "target_missing_policy": "none",
                "x_missing_policy": "none",
                "target_outlier_policy": "none",
                "x_outlier_policy": "none",
                "scaling_policy": "standard",
                "dimensionality_reduction_policy": "none",
                "feature_selection_policy": "none",
                "preprocess_order": "official_tcode_then_extra",
                "preprocess_fit_scope": "train_only",
                "inverse_transform_policy": "target_only",
                "evaluation_scale": "raw_level",
                "target_lag_block": "none",
                "x_lag_feature_block": "none",
                "factor_feature_block": "none",
                "level_feature_block": "none",
                "temporal_feature_block": "none",
                "rotation_feature_block": "none",
                "feature_block_combination": "replace_with_blocks",
            }
        },
        "3_training": {
            "fixed_axes": {
                "framework": "expanding",
                "benchmark_family": "ar_bic",
                "feature_builder": "raw_feature_panel",
                "model_family": "ridge",
                "forecast_type": "direct",
                "forecast_object": "point_mean",
            }
        },
        "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
        "5_output_provenance": {
            "fixed_axes": {
                "export_format": "json",
                "saved_objects": "full_bundle",
                "provenance_fields": "full",
                "artifact_granularity": "aggregated",
            },
            "leaf_config": {"manifest_mode": "full", "benchmark_config": {"minimum_train_size": 60}},
        },
        "6_stat_tests": {"fixed_axes": {"equal_predictive": "dm"}},
        "7_importance": {"fixed_axes": {"importance_method": "none"}},
    },
}

_SYNTHETIC_REPLICATION_RECIPE: dict[str, Any] = {
    "recipe_id": "synthetic-replication-roundtrip-navigator",
    "path": {
        "0_meta": {
            "fixed_axes": {
                "research_design": "single_forecast_run",
                "experiment_unit": "single_target_single_generator",
                "failure_policy": "fail_fast",
                "compute_mode": "serial",
                "reproducibility_mode": "seeded_reproducible",
            },
            "leaf_config": {"random_seed": 42},
        },
        "1_data_task": {
            "fixed_axes": {
                "dataset": "fred_md",
                "source_adapter": "fred_md",
                "information_set_type": "final_revised_data",
                "target_structure": "single_target",
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
                "framework": "expanding",
                "benchmark_family": "zero_change",
                "feature_builder": "autoreg_lagged_target",
                "model_family": "ar",
                "forecast_type": "iterated",
                "forecast_object": "point_mean",
            }
        },
        "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
        "5_output_provenance": {"leaf_config": {"manifest_mode": "full", "benchmark_config": {"minimum_train_size": 5}}},
        "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
        "7_importance": {"fixed_axes": {"importance_method": "none"}},
    },
}

_FRED_SD_MIDASR_ALMONP_RECIPE: dict[str, Any] = {
    "recipe_id": "fred-sd-midasr-almonp-direct",
    "path": {
        "0_meta": {
            "fixed_axes": {
                "research_design": "single_forecast_run",
                "experiment_unit": "single_target_single_generator",
                "failure_policy": "fail_fast",
                "compute_mode": "serial",
                "reproducibility_mode": "seeded_reproducible",
            },
            "leaf_config": {"random_seed": 42},
        },
        "1_data_task": {
            "fixed_axes": {
                "dataset": "fred_sd",
                "source_adapter": "fred_sd",
                "frequency": "monthly",
                "information_set_type": "final_revised_data",
                "target_structure": "single_target",
                "fred_sd_frequency_policy": "report_only",
                "official_transform_policy": "keep_official_raw_scale",
                "raw_missing_policy": "preserve_raw_missing",
                "raw_outlier_policy": "preserve_raw_outliers",
            },
            "leaf_config": {"target": "UR_CA", "horizons": [1]},
        },
        "2_preprocessing": {
            "fixed_axes": {
                "fred_sd_mixed_frequency_representation": "mixed_frequency_model_adapter",
                "horizon_target_construction": "future_target_level_t_plus_h",
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
                "feature_block_set": "transformed_x",
            }
        },
        "3_training": {
            "fixed_axes": {
                "framework": "expanding",
                "benchmark_family": "zero_change",
                "feature_builder": "raw_feature_panel",
                "model_family": "midasr",
                "midasr_weight_family": "almonp",
                "forecast_type": "direct",
                "forecast_object": "point_mean",
            },
            "leaf_config": {
                "training_config": {
                    "midas_max_lag": 3,
                    "midasr_almonp_degree": 2,
                    "midasr_max_terms": 12,
                    "midasr_max_nfev": 500,
                }
            },
        },
        "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
        "5_output_provenance": {
            "leaf_config": {"manifest_mode": "full", "benchmark_config": {"minimum_train_size": 5}}
        },
        "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
        "7_importance": {"fixed_axes": {"importance_method": "none"}},
    },
}


def _dump_recipe(recipe: dict[str, Any]) -> str:
    return yaml.safe_dump(recipe, sort_keys=False)


_ENTRIES: dict[str, ReplicationEntry] = {
    "goulet-coulombe-2021-fred-md-ridge": ReplicationEntry(
        id="goulet-coulombe-2021-fred-md-ridge",
        paper_name="Goulet Coulombe et al. (2021), International Journal of Forecasting",
        short_description=(
            "FRED-MD style macro forecasting path with official transformations, "
            "raw-panel ridge generator, AR-BIC benchmark, and MSFE evaluation."
        ),
        exact_tree_path=(
            "0_meta.research_design=single_forecast_run",
            "1_data_task.dataset=fred_md",
            "1_data_task.official_transform_policy=apply_official_tcode",
            "2_preprocessing.tcode_policy=official_tcode_then_extra_preprocess",
            "2_preprocessing.scaling_policy=standard",
            "3_training.feature_builder=raw_feature_panel",
            "3_training.model_family=ridge",
            "3_training.benchmark_family=ar_bic",
            "4_evaluation.primary_metric=msfe",
            "6_stat_tests.equal_predictive=dm",
        ),
        recipe_yaml=_dump_recipe(_GOULET_COULOMBE_2021_RECIPE),
        command=(
            "macrocast-navigate run examples/recipes/replications/"
            "goulet-coulombe-2021-fred-md-ridge.yaml --output-root results/gc2021"
        ),
        notebook_snippet=(
            "from macrocast.navigator import get_replication_entry, replication_recipe_yaml\n"
            "from macrocast import compile_recipe_dict, run_compiled_recipe\n"
            "import yaml\n"
            "recipe = yaml.safe_load(replication_recipe_yaml('goulet-coulombe-2021-fred-md-ridge'))\n"
            "compiled = compile_recipe_dict(recipe)\n"
            "result = run_compiled_recipe(compiled.compiled, output_root='results/gc2021')\n"
        ),
        expected_outputs=(
            "manifest.json",
            "predictions.csv",
            "metrics.json",
            "comparison_summary.json",
            "stat_tests.json",
            "stat_test_dm.json",
        ),
        deviations_from_original_paper=(
            "This entry is a package-native runnable path, not a byte-identical replication package.",
            "Vintage availability is revised/pseudo-OOS unless a real-time source adapter is supplied.",
            "The model grid is represented as individual navigator recipes or a sweep, not as the full paper table bundle.",
        ),
    ),
    "synthetic-replication-roundtrip": ReplicationEntry(
        id="synthetic-replication-roundtrip",
        paper_name="Synthetic replication round-trip",
        short_description="Small fixture-safe replication route used to verify recipe lowering and artifact contracts.",
        exact_tree_path=(
            "0_meta.research_design=single_forecast_run",
            "0_meta.experiment_unit=single_target_single_generator",
            "1_data_task.dataset=fred_md",
            "2_preprocessing.tcode_policy=raw_only",
            "3_training.feature_builder=autoreg_lagged_target",
            "3_training.model_family=ar",
        ),
        recipe_yaml=_dump_recipe(_SYNTHETIC_REPLICATION_RECIPE),
        command=(
            "macrocast-navigate run examples/recipes/replications/"
            "synthetic-replication-roundtrip.yaml --local-raw-source tests/fixtures/fred_md_ar_sample.csv"
        ),
        notebook_snippet=(
            "from macrocast.navigator import replication_recipe_yaml\n"
            "import yaml\n"
            "recipe = yaml.safe_load(replication_recipe_yaml('synthetic-replication-roundtrip'))\n"
        ),
        expected_outputs=("manifest.json", "predictions.csv", "metrics.json", "comparison_summary.json"),
        deviations_from_original_paper=("Synthetic fixture route; no external paper claims.",),
    ),
    "fred-sd-midasr-almonp-direct": ReplicationEntry(
        id="fred-sd-midasr-almonp-direct",
        paper_name="FRED-SD mixed-frequency MIDAS runtime route",
        short_description=(
            "Fixture-safe FRED-SD direct forecast using Layer 2 native-frequency adapter payloads "
            "and the built-in Layer 3 midasr/almonp restricted MIDAS executor."
        ),
        exact_tree_path=(
            "1_data_task.dataset=fred_sd",
            "1_data_task.frequency=monthly",
            "2_preprocessing.fred_sd_mixed_frequency_representation=mixed_frequency_model_adapter",
            "3_training.feature_builder=raw_feature_panel",
            "3_training.model_family=midasr",
            "3_training.midasr_weight_family=almonp",
            "3_training.forecast_type=direct",
        ),
        recipe_yaml=_dump_recipe(_FRED_SD_MIDASR_ALMONP_RECIPE),
        command=(
            "macrocast-navigate run examples/recipes/replications/"
            "fred-sd-midasr-almonp-direct.yaml --local-raw-source tests/fixtures/fred_sd_sample.csv "
            "--output-root results/fred_sd_midasr_almonp"
        ),
        notebook_snippet=(
            "from macrocast.navigator import replication_recipe_yaml\n"
            "from macrocast import compile_recipe_dict, run_compiled_recipe\n"
            "import yaml\n"
            "recipe = yaml.safe_load(replication_recipe_yaml('fred-sd-midasr-almonp-direct'))\n"
            "compiled = compile_recipe_dict(recipe)\n"
            "result = run_compiled_recipe(\n"
            "    compiled.compiled,\n"
            "    output_root='results/fred_sd_midasr_almonp',\n"
            "    local_raw_source='tests/fixtures/fred_sd_sample.csv',\n"
            ")\n"
        ),
        expected_outputs=(
            "manifest.json",
            "predictions.csv",
            "metrics.json",
            "fred_sd_series_metadata.json",
            "fred_sd_frequency_report.json",
            "fred_sd_mixed_frequency_representation.json",
            "fred_sd_mixed_frequency_model_adapter.json",
        ),
        deviations_from_original_paper=(
            "Runtime fixture route; no paper-identical claims.",
            "Uses current package FRED-SD fixture unless an official vintage is supplied.",
            "Demonstrates the almonp MIDAS weight-family branch, not a full MIDAS grid.",
        ),
    ),
}


def list_replication_entries() -> list[dict[str, Any]]:
    return [entry.to_dict() for entry in _ENTRIES.values()]


def get_replication_entry(replication_id: str) -> dict[str, Any]:
    try:
        return _ENTRIES[replication_id].to_dict()
    except KeyError as exc:
        raise KeyError(f"unknown replication id {replication_id!r}; available: {sorted(_ENTRIES)}") from exc


def replication_recipe_yaml(replication_id: str) -> str:
    try:
        return _ENTRIES[replication_id].recipe_yaml
    except KeyError as exc:
        raise KeyError(f"unknown replication id {replication_id!r}; available: {sorted(_ENTRIES)}") from exc


def write_replication_recipe(replication_id: str, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(replication_recipe_yaml(replication_id), encoding="utf-8")
    return output
