from __future__ import annotations

from macrocast.compiler import compile_recipe_dict
from macrocast.defaults import build_default_recipe_dict
from macrocast.registry.naming import (
    AXIS_NAME_ALIASES,
    AXIS_VALUE_ALIASES,
    canonical_axis_name,
    canonical_axis_value,
    canonicalize_recipe_path,
)


def test_naming_alias_maps_are_closed() -> None:
    assert AXIS_NAME_ALIASES == {}
    assert AXIS_VALUE_ALIASES == {}


def test_canonical_axis_helpers_are_identity_only() -> None:
    assert canonical_axis_name("information_set_type") == "information_set_type"
    assert canonical_axis_name("info_set") == "info_set"
    assert canonical_axis_value("relative_metrics", "relative_msfe") == "relative_msfe"
    assert canonical_axis_value("relative_metrics", "relative_MSFE") == "relative_MSFE"


def test_canonicalize_recipe_path_preserves_canonical_ids() -> None:
    recipe = build_default_recipe_dict(
        dataset="fred_md",
        target="INDPRO",
        start="1960-01-01",
        end="1970-01-01",
    )
    recipe["path"]["4_evaluation"]["fixed_axes"]["relative_metrics"] = "relative_msfe"
    recipe["path"]["4_evaluation"]["sweep_axes"] = {"agg_time": ["none", "full_out_of_sample_average"]}

    canonical = canonicalize_recipe_path(recipe)

    assert canonical["path"]["4_evaluation"]["fixed_axes"]["relative_metrics"] == "relative_msfe"
    assert canonical["path"]["4_evaluation"]["sweep_axes"]["agg_time"] == [
        "none",
        "full_out_of_sample_average",
    ]


def test_canonical_recipe_compiles_to_canonical_ids() -> None:
    recipe = build_default_recipe_dict(
        dataset="fred_md",
        target="INDPRO",
        start="1960-01-01",
        end="1970-01-01",
    )
    recipe["path"]["0_meta"]["fixed_axes"]["study_scope"] = "one_target_one_method"
    recipe["path"]["1_data_task"]["fixed_axes"]["information_set_type"] = "pseudo_oos_on_revised_data"
    recipe["path"]["1_data_task"]["fixed_axes"]["target_structure"] = "single_target"
    recipe["path"]["1_data_task"]["fixed_axes"]["official_transform_scope"] = "target_and_predictors"
    recipe["path"]["2_preprocessing"]["fixed_axes"]["feature_block_set"] = "transformed_predictor_lags"
    recipe["path"]["2_preprocessing"]["fixed_axes"]["x_lag_feature_block"] = "fixed_predictor_lags"
    recipe["path"]["2_preprocessing"]["fixed_axes"]["feature_block_combination"] = "append_to_base_predictors"
    recipe["path"]["2_preprocessing"]["fixed_axes"]["preprocess_fit_scope"] = "train_only"
    recipe["path"]["3_training"]["fixed_axes"]["feature_builder"] = "factors_plus_target_lags"
    recipe["path"]["3_training"]["fixed_axes"]["model_family"] = "random_forest"
    recipe["path"]["3_training"]["fixed_axes"]["benchmark_family"] = "autoregressive_bic"
    recipe["path"]["4_evaluation"]["fixed_axes"]["point_metrics"] = "msfe"
    recipe["path"]["4_evaluation"]["fixed_axes"]["relative_metrics"] = "relative_msfe"
    recipe["path"]["4_evaluation"]["fixed_axes"]["agg_time"] = "full_out_of_sample_average"
    recipe["path"]["4_evaluation"]["fixed_axes"]["regime_definition"] = "nber_recession"
    recipe["path"]["4_evaluation"]["fixed_axes"]["regime_use"] = "evaluation_only"
    recipe["path"]["5_output_provenance"].setdefault("fixed_axes", {})["export_format"] = "json_csv"
    recipe["path"]["6_stat_tests"]["fixed_axes"]["density_interval"] = "pit_uniformity"
    recipe["path"]["6_stat_tests"]["fixed_axes"]["residual_diagnostics"] = "full_residual_diagnostics"

    result = compile_recipe_dict(recipe)

    assert result.compiled.stage0.study_scope == "one_target_one_method"
    assert result.manifest["data_task_spec"]["information_set_type"] == "pseudo_oos_on_revised_data"
    assert result.manifest["data_task_spec"]["target_structure"] == "single_target"
    assert result.manifest["model_spec"]["model_family"] == "random_forest"
    assert result.manifest["benchmark_spec"]["benchmark_family"] == "autoregressive_bic"
    assert result.manifest["evaluation_spec"]["relative_metrics"] == "relative_msfe"
    assert result.manifest["output_spec"]["export_format"] == "json_csv"
    assert result.manifest["stat_test_spec"]["density_interval"] == "pit_uniformity"
    assert result.manifest["stat_test_spec"]["residual_diagnostics"] == "full_residual_diagnostics"
