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


def test_layer2_feature_composer_legacy_aliases_canonicalize() -> None:
    assert canonical_axis_value("feature_builder", "autoreg_lagged_target") == "target_lag_features"
    assert canonical_axis_value("feature_builder", "factors_plus_AR") == "factors_plus_target_lags"
    assert canonical_axis_value("feature_builder", "raw_X_only") == "raw_predictors_only"
    assert canonical_axis_value("feature_builder", "factor_pca") == "pca_factor_features"
    assert (
        canonical_axis_value("data_richness_mode", "full_high_dimensional_X")
        == "high_dimensional_predictors"
    )
    assert canonical_axis_value("data_richness_mode", "selected_sparse_X") == "selected_sparse_predictors"
    assert (
        canonical_axis_value("feature_block_set", "legacy_feature_builder_bridge")
        == "feature_builder_compatibility_bridge"
    )
    assert canonical_axis_value("feature_block_set", "transformed_x") == "transformed_predictors"
    assert canonical_axis_value("feature_block_set", "transformed_x_lags") == "transformed_predictor_lags"
    assert canonical_axis_value("feature_block_set", "custom_blocks") == "custom_feature_blocks"
    assert canonical_axis_value("x_lag_creation", "no_x_lags") == "no_predictor_lags"
    assert canonical_axis_value("x_lag_creation", "fixed_x_lags") == "fixed_predictor_lags"
    assert canonical_axis_value("x_lag_feature_block", "fixed_x_lags") == "fixed_predictor_lags"
    assert (
        canonical_axis_value("x_lag_feature_block", "cv_selected_x_lags")
        == "cv_selected_predictor_lags"
    )
    assert (
        canonical_axis_value("feature_block_combination", "append_to_base_x")
        == "append_to_base_predictors"
    )
    assert canonical_axis_value("feature_selection_policy", "lasso_select") == "lasso_selection"
    assert (
        canonical_axis_value("feature_selection_semantics", "select_after_custom_blocks")
        == "select_after_custom_feature_blocks"
    )


def test_layer3_training_legacy_aliases_canonicalize() -> None:
    assert canonical_axis_value("model_family", "bayesianridge") == "bayesian_ridge"
    assert canonical_axis_value("model_family", "adaptivelasso") == "adaptive_lasso"
    assert canonical_axis_value("model_family", "randomforest") == "random_forest"
    assert canonical_axis_value("model_family", "extratrees") == "extra_trees"
    assert canonical_axis_value("model_family", "gbm") == "gradient_boosting"
    assert canonical_axis_value("benchmark_family", "ar_bic") == "autoregressive_bic"
    assert canonical_axis_value("benchmark_family", "ar_fixed_p") == "autoregressive_fixed_lag"
    assert canonical_axis_value("benchmark_family", "ardi") == "autoregressive_diffusion_index"
    assert canonical_axis_value("benchmark_family", "factor_model") == "factor_model_benchmark"
    assert canonical_axis_value("benchmark_family", "multi_benchmark_suite") == "benchmark_suite"


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


def test_legacy_layer2_feature_composer_recipe_compiles_to_canonical_ids() -> None:
    recipe = build_default_recipe_dict(dataset="fred_md", target="INDPRO", start="1960-01-01", end="1970-01-01")
    axes = recipe["path"]["2_preprocessing"]["fixed_axes"]
    axes["data_richness_mode"] = "full_high_dimensional_X"
    axes["feature_block_set"] = "transformed_x_lags"
    axes["x_lag_creation"] = "fixed_x_lags"
    axes["x_lag_feature_block"] = "fixed_x_lags"
    axes["feature_block_combination"] = "append_to_base_x"
    axes["preprocess_fit_scope"] = "train_only"
    recipe["path"]["3_training"]["fixed_axes"]["feature_builder"] = "factors_plus_AR"
    recipe["path"]["3_training"]["fixed_axes"]["model_family"] = "factor_augmented_linear"
    recipe["path"]["3_training"]["fixed_axes"]["forecast_type"] = "direct"

    result = compile_recipe_dict(recipe)
    layer2 = result.manifest["layer2_representation_spec"]

    assert result.manifest["model_spec"]["feature_builder"] == "factors_plus_target_lags"
    assert layer2["source_bridge"]["feature_builder"] == "factors_plus_target_lags"
    assert layer2["source_bridge"]["data_richness_mode"] == "high_dimensional_predictors"
    assert layer2["feature_blocks"]["feature_block_set"]["value"] == "transformed_predictor_lags"
    assert layer2["feature_blocks"]["x_lag_feature_block"]["value"] == "fixed_predictor_lags"
    assert layer2["feature_blocks"]["feature_block_combination"]["value"] == "append_to_base_predictors"


def test_legacy_layer3_training_recipe_compiles_to_canonical_ids() -> None:
    recipe = build_default_recipe_dict(dataset="fred_md", target="INDPRO", start="1960-01-01", end="1970-01-01")
    axes = recipe["path"]["3_training"]["fixed_axes"]
    axes["benchmark_family"] = "ar_bic"
    axes["model_family"] = "randomforest"

    result = compile_recipe_dict(recipe)

    assert result.manifest["model_spec"]["model_family"] == "random_forest"
    assert result.manifest["benchmark_spec"]["benchmark_family"] == "autoregressive_bic"


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


def test_rename_ledger_lists_layer2_feature_composer_aliases() -> None:
    aliases = {
        (item["axis"], item["legacy_id"]): item["canonical_id"]
        for item in rename_ledger()["axis_value_aliases"]
    }

    assert aliases[("feature_builder", "autoreg_lagged_target")] == "target_lag_features"
    assert aliases[("feature_builder", "factors_plus_AR")] == "factors_plus_target_lags"
    assert aliases[("feature_block_set", "transformed_x_lags")] == "transformed_predictor_lags"
    assert aliases[("x_lag_creation", "fixed_x_lags")] == "fixed_predictor_lags"
    assert aliases[("feature_block_combination", "append_to_base_x")] == "append_to_base_predictors"
    assert aliases[("feature_selection_semantics", "select_after_custom_blocks")] == "select_after_custom_feature_blocks"


def test_rename_ledger_lists_layer3_training_aliases() -> None:
    aliases = {
        (item["axis"], item["legacy_id"]): item["canonical_id"]
        for item in rename_ledger()["axis_value_aliases"]
    }

    assert aliases[("model_family", "randomforest")] == "random_forest"
    assert aliases[("model_family", "extratrees")] == "extra_trees"
    assert aliases[("model_family", "gbm")] == "gradient_boosting"
    assert aliases[("benchmark_family", "ar_bic")] == "autoregressive_bic"
    assert aliases[("benchmark_family", "ar_fixed_p")] == "autoregressive_fixed_lag"
    assert aliases[("benchmark_family", "ardi")] == "autoregressive_diffusion_index"
