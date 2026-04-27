from __future__ import annotations

from macrocast.compiler import compile_recipe_dict
from macrocast.defaults import build_default_recipe_dict
from macrocast.registry.naming import (
    canonical_axis_value,
    canonicalize_recipe_path,
    rename_ledger,
)


def test_stage0_legacy_value_aliases_canonicalize() -> None:
    assert canonical_axis_value("research_design", "single_path_benchmark") == "single_forecast_run"
    assert canonical_axis_value("research_design", "orchestrated_bundle") == "study_bundle"
    assert canonical_axis_value("research_design", "replication_override") == "replication_recipe"
    assert (
        canonical_axis_value("experiment_unit", "single_target_single_model")
        == "single_target_single_generator"
    )
    assert (
        canonical_axis_value("experiment_unit", "single_target_model_grid")
        == "single_target_generator_grid"
    )


def test_stage1_legacy_value_aliases_canonicalize() -> None:
    assert canonical_axis_value("information_set_type", "revised") == "final_revised_data"
    assert (
        canonical_axis_value("information_set_type", "pseudo_oos_revised")
        == "pseudo_oos_on_revised_data"
    )
    assert canonical_axis_value("target_structure", "single_target_point_forecast") == "single_target"
    assert canonical_axis_value("target_structure", "multi_target_point_forecast") == "multi_target"
    assert canonical_axis_value("official_transform_policy", "dataset_tcode") == "apply_official_tcode"
    assert (
        canonical_axis_value("official_transform_policy", "raw_official_frame")
        == "keep_official_raw_scale"
    )
    assert canonical_axis_value("official_transform_scope", "apply_tcode_to_X") == "predictors_only"
    assert (
        canonical_axis_value("contemporaneous_x_rule", "forbid_contemporaneous")
        == "forbid_same_period_predictors"
    )
    assert canonical_axis_value("variable_universe", "handpicked_set") == "explicit_variable_list"
    assert (
        canonical_axis_value("missing_availability", "zero_fill_before_start")
        == "zero_fill_leading_predictor_gaps"
    )
    assert canonical_axis_value("raw_missing_policy", "x_impute_raw") == "impute_raw_predictors"
    assert (
        canonical_axis_value("raw_outlier_policy", "raw_outlier_to_missing")
        == "set_raw_outliers_to_missing"
    )


def test_shared_tcode_and_predictor_legacy_aliases_canonicalize() -> None:
    assert (
        canonical_axis_value("target_transform_policy", "tcode_transformed")
        == "official_tcode_transformed"
    )
    assert (
        canonical_axis_value("x_transform_policy", "dataset_tcode_transformed")
        == "official_tcode_transformed"
    )
    assert (
        canonical_axis_value("x_transform_policy", "apply_official_tcode_transformed")
        == "official_tcode_transformed"
    )
    assert canonical_axis_value("tcode_policy", "tcode_only") == "official_tcode_only"
    assert (
        canonical_axis_value("tcode_policy", "tcode_then_extra_preprocess")
        == "official_tcode_then_extra_preprocess"
    )
    assert (
        canonical_axis_value("tcode_policy", "extra_preprocess_without_tcode")
        == "extra_preprocess_only"
    )
    assert (
        canonical_axis_value("tcode_policy", "extra_then_tcode")
        == "extra_preprocess_then_official_tcode"
    )
    assert canonical_axis_value("tcode_application_scope", "apply_tcode_to_X") == "predictors_only"
    assert canonical_axis_value("preprocess_order", "tcode_only") == "official_tcode_only"
    assert (
        canonical_axis_value("preprocess_order", "tcode_then_extra")
        == "official_tcode_then_extra"
    )
    assert canonical_axis_value("predictor_family", "handpicked_set") == "explicit_variable_list"


def test_canonicalize_recipe_path_rewrites_legacy_stage0_values() -> None:
    recipe = build_default_recipe_dict(dataset="fred_md", target="INDPRO", start="1960-01-01", end="1970-01-01")
    recipe["path"]["0_meta"]["fixed_axes"]["research_design"] = "single_path_benchmark"
    recipe["path"]["0_meta"]["fixed_axes"]["experiment_unit"] = "single_target_model_grid"

    canonical = canonicalize_recipe_path(recipe)

    assert canonical["path"]["0_meta"]["fixed_axes"]["research_design"] == "single_forecast_run"
    assert canonical["path"]["0_meta"]["fixed_axes"]["experiment_unit"] == "single_target_generator_grid"


def test_legacy_stage0_recipe_compiles_to_canonical_ids() -> None:
    recipe = build_default_recipe_dict(dataset="fred_md", target="INDPRO", start="1960-01-01", end="1970-01-01")
    recipe["path"]["0_meta"]["fixed_axes"]["research_design"] = "single_path_benchmark"
    recipe["path"]["0_meta"]["fixed_axes"]["experiment_unit"] = "single_target_single_model"

    result = compile_recipe_dict(recipe)

    assert result.compiled.stage0.research_design == "single_forecast_run"
    assert result.compiled.stage0.experiment_unit == "single_target_single_generator"
    assert result.manifest["tree_context"]["fixed_axes"]["research_design"] == "single_forecast_run"
    assert result.manifest["tree_context"]["fixed_axes"]["experiment_unit"] == "single_target_single_generator"


def test_legacy_stage1_recipe_compiles_to_canonical_ids() -> None:
    recipe = build_default_recipe_dict(dataset="fred_md", target="INDPRO", start="1960-01-01", end="1970-01-01")
    axes = recipe["path"]["1_data_task"]["fixed_axes"]
    axes["information_set_type"] = "pseudo_oos_revised"
    axes["target_structure"] = "single_target_point_forecast"
    axes["official_transform_policy"] = "dataset_tcode"
    axes["official_transform_scope"] = "apply_tcode_to_X"
    axes["contemporaneous_x_rule"] = "forbid_contemporaneous"
    axes["variable_universe"] = "handpicked_set"
    axes["missing_availability"] = "zero_fill_before_start"
    axes["raw_missing_policy"] = "x_impute_raw"
    axes["raw_outlier_policy"] = "raw_outlier_to_missing"
    recipe["path"]["1_data_task"]["leaf_config"]["raw_x_imputation"] = "ffill"
    recipe["path"]["1_data_task"]["leaf_config"]["variable_universe_columns"] = ["RPI", "UNRATE"]

    result = compile_recipe_dict(recipe)
    spec = result.manifest["data_task_spec"]

    assert spec["information_set_type"] == "pseudo_oos_on_revised_data"
    assert spec["target_structure"] == "single_target"
    assert spec["official_transform_policy"] == "apply_official_tcode"
    assert spec["official_transform_scope"] == "predictors_only"
    assert (
        result.manifest["layer2_representation_spec"]["input_panel"]["contemporaneous_x_rule"]
        == "forbid_same_period_predictors"
    )
    assert spec["variable_universe_columns"] == ["RPI", "UNRATE"]
    assert spec["missing_availability"] == "zero_fill_leading_predictor_gaps"
    assert spec["raw_missing_policy"] == "impute_raw_predictors"
    assert spec["raw_outlier_policy"] == "set_raw_outliers_to_missing"


def test_rename_ledger_lists_stage0_aliases() -> None:
    aliases = {
        (item["axis"], item["legacy_id"]): item["canonical_id"]
        for item in rename_ledger()["axis_value_aliases"]
    }

    assert aliases[("research_design", "single_path_benchmark")] == "single_forecast_run"
    assert aliases[("experiment_unit", "single_target_model_grid")] == "single_target_generator_grid"


def test_rename_ledger_lists_stage1_aliases() -> None:
    aliases = {
        (item["axis"], item["legacy_id"]): item["canonical_id"]
        for item in rename_ledger()["axis_value_aliases"]
    }

    assert aliases[("information_set_type", "revised")] == "final_revised_data"
    assert aliases[("target_structure", "single_target_point_forecast")] == "single_target"
    assert aliases[("official_transform_scope", "apply_tcode_to_X")] == "predictors_only"
    assert aliases[("variable_universe", "handpicked_set")] == "explicit_variable_list"
