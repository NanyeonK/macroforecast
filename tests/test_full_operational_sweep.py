"""Systematic verification that every 'operational' registry value actually executes end-to-end."""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from macrocast.compiler import compile_recipe_dict, run_compiled_recipe


def _base_recipe(overrides_3=None, overrides_1=None, overrides_4=None, overrides_6=None, overrides_7=None):
    r = {
        "recipe_id": "sweep_test",
        "path": {
            "0_meta": {"fixed_axes": {"research_design": "single_forecast_run", "experiment_unit": "single_target_generator_grid"}},
            "1_data_task": {
                "fixed_axes": {"dataset": "fred_md", "information_set_type": "final_revised_data", "task": "single_target", "benchmark_family": "autoregressive_bic", "evaluation_scale": "raw_level"},
                "leaf_config": {"target": "INDPRO", "horizons": [1]},
            },
            "2_preprocessing": {"fixed_axes": {
                "target_transform_policy": "raw_level", "x_transform_policy": "raw_level",
                "tcode_policy": "raw_only", "target_missing_policy": "none",
                "x_missing_policy": "none", "target_outlier_policy": "none",
                "x_outlier_policy": "none", "scaling_policy": "none",
                "dimensionality_reduction_policy": "none", "feature_selection_policy": "none",
                "preprocess_order": "none", "preprocess_fit_scope": "not_applicable",
                "inverse_transform_policy": "none", "representation_policy": "raw_only",
                "tcode_application_scope": "none",
                "target_transform": "level", "target_normalization": "none",
                "target_domain": "unconstrained", "scaling_scope": "columnwise",
                "additional_preprocessing": "none", "x_lag_creation": "no_predictor_lags",
                "feature_grouping": "none",
            }},
            "3_training": {
                "fixed_axes": {"framework": "expanding", "feature_builder": "target_lag_features"},
                "sweep_axes": {"model_family": ["ar"]},
            },
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }
    if overrides_3:
        r["path"]["3_training"]["fixed_axes"].update(overrides_3.get("fixed_axes", {}))
        if "sweep_axes" in overrides_3:
            r["path"]["3_training"]["sweep_axes"] = overrides_3["sweep_axes"]
        if "leaf_config" in overrides_3:
            r["path"]["3_training"].setdefault("leaf_config", {}).update(overrides_3["leaf_config"])
    if overrides_1:
        r["path"]["1_data_task"]["fixed_axes"].update(overrides_1.get("fixed_axes", {}))
        if "leaf_config" in overrides_1:
            r["path"]["1_data_task"]["leaf_config"].update(overrides_1["leaf_config"])
    if overrides_4:
        r["path"]["4_evaluation"]["fixed_axes"].update(overrides_4)
    if overrides_6:
        r["path"]["6_stat_tests"]["fixed_axes"].update(overrides_6)
    if overrides_7:
        r["path"]["7_importance"]["fixed_axes"].update(overrides_7)
    return r


def _compile_and_run(recipe, tmp_path):
    result = compile_recipe_dict(recipe)
    c = result.compiled
    assert c.execution_status == "executable", f"not executable: {c.execution_status}, warnings={list(c.warnings)}, blocked={list(c.blocked_reasons)}"
    ex = run_compiled_recipe(c, output_root=tmp_path, local_raw_source="tests/fixtures/fred_md_ar_sample.csv")
    manifest = json.loads((Path(ex.artifact_dir) / "manifest.json").read_text())
    return manifest, ex.artifact_dir


# ============================================================
# Model family sweep (autoreg path)
# ============================================================
AUTOREG_MODELS = ["ar", "ols", "ridge", "lasso", "elasticnet", "bayesian_ridge", "huber",
                  "adaptive_lasso", "svr_linear", "svr_rbf", "random_forest", "extra_trees",
                  "gradient_boosting", "xgboost", "lightgbm", "catboost", "mlp",
                  "componentwise_boosting", "boosting_ridge", "boosting_lasso", "quantile_linear"]

@pytest.mark.parametrize("model", AUTOREG_MODELS)
def test_model_autoreg_executes(model, tmp_path):
    overrides_1 = {}
    if model == "quantile_linear":
        overrides_1 = {"fixed_axes": {"forecast_object": "point_median"}}
    recipe = _base_recipe(
        overrides_3={"sweep_axes": {"model_family": [model]}},
        overrides_1=overrides_1,
    )
    manifest, art_dir = _compile_and_run(recipe, tmp_path)
    assert (Path(art_dir) / "predictions.csv").exists()
    assert (Path(art_dir) / "metrics.json").exists()
    assert (Path(art_dir) / "tuning_result.json").exists()


# ============================================================
# Model family sweep (raw_panel path)
# ============================================================
RAW_PANEL_MODELS = ["ols", "ridge", "lasso", "elasticnet", "bayesian_ridge", "huber",
                    "adaptive_lasso", "svr_linear", "svr_rbf", "random_forest", "extra_trees",
                    "gradient_boosting", "xgboost", "lightgbm", "catboost", "mlp",
                    "componentwise_boosting", "boosting_ridge", "boosting_lasso"]

@pytest.mark.parametrize("model", RAW_PANEL_MODELS)
def test_model_raw_panel_executes(model, tmp_path):
    recipe = _base_recipe(overrides_3={
        "fixed_axes": {"feature_builder": "raw_feature_panel"},
        "sweep_axes": {"model_family": [model]},
    })
    # Need preprocessing for raw panel
    recipe["path"]["2_preprocessing"]["fixed_axes"].update({
        "tcode_policy": "extra_preprocess_only",
        "x_missing_policy": "em_impute",
        "scaling_policy": "standard",
        "preprocess_order": "extra_only",
        "preprocess_fit_scope": "train_only",
    })
    manifest, art_dir = _compile_and_run(recipe, tmp_path)
    assert (Path(art_dir) / "predictions.csv").exists()


# ============================================================
# Factor model path (factors_plus_target_lags / pca_factor_features)
# ============================================================
FACTOR_MODELS = ["pcr", "pls", "factor_augmented_linear"]

@pytest.mark.parametrize("model", FACTOR_MODELS)
def test_model_factor_executes(model, tmp_path):
    recipe = _base_recipe(overrides_3={
        "fixed_axes": {"feature_builder": "factors_plus_target_lags"},
        "sweep_axes": {"model_family": [model]},
    })
    recipe["path"]["2_preprocessing"]["fixed_axes"].update({
        "tcode_policy": "extra_preprocess_only",
        "x_missing_policy": "em_impute",
        "scaling_policy": "standard",
        "preprocess_order": "extra_only",
        "preprocess_fit_scope": "train_only",
    })
    manifest, art_dir = _compile_and_run(recipe, tmp_path)
    assert (Path(art_dir) / "predictions.csv").exists()


# ============================================================
# Statistical tests sweep
# ============================================================
STAT_TESTS = ["dm", "dm_hln", "dm_modified", "cw", "mcs", "enc_new", "mse_f", "mse_t",
              "cpa", "rossi", "rolling_dm", "reality_check", "spa",
              "mincer_zarnowitz", "ljung_box", "arch_lm", "bias_test",
              "pesaran_timmermann", "binomial_hit", "full_residual_diagnostics"]

@pytest.mark.parametrize("test_name", STAT_TESTS)
def test_stat_test_executes(test_name, tmp_path):
    recipe = _base_recipe(
        overrides_3={"sweep_axes": {"model_family": ["ridge"]}},
        overrides_6={"stat_test": test_name},
    )
    manifest, art_dir = _compile_and_run(recipe, tmp_path)
    expected_file = f"stat_test_{test_name}.json"
    assert (Path(art_dir) / expected_file).exists(), f"missing {expected_file}"
    stat_data = json.loads((Path(art_dir) / expected_file).read_text())
    assert isinstance(stat_data, (dict, list))


# ============================================================
# Dependence corrections
# ============================================================
DEP_CORRECTIONS = ["none", "nw_hac", "nw_hac_auto", "block_bootstrap"]

@pytest.mark.parametrize("dep", DEP_CORRECTIONS)
def test_dependence_correction_executes(dep, tmp_path):
    recipe = _base_recipe(
        overrides_3={"sweep_axes": {"model_family": ["ridge"]}},
        overrides_6={"stat_test": "dm", "dependence_correction": dep},
    )
    manifest, art_dir = _compile_and_run(recipe, tmp_path)
    assert (Path(art_dir) / "stat_test_dm.json").exists()


# ============================================================
# Importance methods sweep
# ============================================================
IMPORTANCE_MODEL_PAIRS = [
    ("tree_shap", "random_forest"),
    ("tree_shap", "xgboost"),
    ("tree_shap", "lightgbm"),
    ("kernel_shap", "ridge"),
    ("linear_shap", "ridge"),
    ("linear_shap", "lasso"),
    ("permutation_importance", "ridge"),
    ("permutation_importance", "xgboost"),
    ("lime", "ridge"),
    ("feature_ablation", "random_forest"),
    ("pdp", "ridge"),
    ("ice", "ridge"),
    ("ale", "ridge"),
    ("grouped_permutation", "ridge"),
    ("importance_stability", "ridge"),
    ("minimal_importance", "ridge"),
    ("minimal_importance", "random_forest"),
]

@pytest.mark.parametrize("method,model", IMPORTANCE_MODEL_PAIRS)
def test_importance_method_executes(method, model, tmp_path):
    recipe = _base_recipe(
        overrides_3={
            "fixed_axes": {"feature_builder": "raw_feature_panel"},
            "sweep_axes": {"model_family": [model]},
        },
        overrides_7={"importance_method": method},
    )
    recipe["path"]["2_preprocessing"]["fixed_axes"].update({
        "tcode_policy": "extra_preprocess_only",
        "x_missing_policy": "em_impute",
        "scaling_policy": "standard",
        "preprocess_order": "extra_only",
        "preprocess_fit_scope": "train_only",
    })
    manifest, art_dir = _compile_and_run(recipe, tmp_path)
    _fn_map = {"minimal_importance": "importance_minimal.json", "importance_stability": "importance_stability.json"}
    expected_file = _fn_map.get(method, f"importance_{method}.json")
    assert (Path(art_dir) / expected_file).exists(), f"missing {expected_file}"


# ============================================================
# Export formats
# ============================================================
EXPORT_FORMATS = ["json", "csv", "parquet", "json_csv", "all"]

@pytest.mark.parametrize("fmt", EXPORT_FORMATS)
def test_export_format_executes(fmt, tmp_path):
    recipe = _base_recipe(overrides_3={"sweep_axes": {"model_family": ["ridge"]}})
    recipe["path"]["5_output_provenance"] = {"fixed_axes": {"export_format": fmt, "saved_objects": "full_bundle", "provenance_fields": "standard", "artifact_granularity": "aggregated"}}
    manifest, art_dir = _compile_and_run(recipe, tmp_path)
    if "parquet" in fmt or fmt == "all":
        assert (Path(art_dir) / "predictions.parquet").exists()
    if "csv" in fmt or fmt == "all":
        assert (Path(art_dir) / "predictions.csv").exists()


# ============================================================
# Primary metrics
# ============================================================
PRIMARY_METRICS = ["msfe", "relative_msfe", "oos_r2", "csfe", "rmse", "mae", "mape"]

@pytest.mark.parametrize("metric", PRIMARY_METRICS)
def test_primary_metric_executes(metric, tmp_path):
    recipe = _base_recipe(
        overrides_3={"sweep_axes": {"model_family": ["ridge"]}},
        overrides_4={"primary_metric": metric},
    )
    manifest, art_dir = _compile_and_run(recipe, tmp_path)
    assert (Path(art_dir) / "metrics.json").exists()
