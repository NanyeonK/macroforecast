from __future__ import annotations

import pandas as pd
import pytest

from macrocast.core import execute_minimal_forecast
from macrocast.core.types import L6TestsArtifact, L7ImportanceArtifact, L8ArtifactsArtifact


MINIMAL_RECIPE = """
1_data:
  fixed_axes:
    custom_source_policy: custom_panel_only
    frequency: monthly
    horizon_set: custom_list
  leaf_config:
    target: y
    target_horizons: [1]
    custom_panel_inline:
      date: [2020-01-01, 2020-02-01, 2020-03-01, 2020-04-01, 2020-05-01, 2020-06-01]
      y: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
      x1: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
      x2: [2.0, 1.0, 2.0, 1.0, 2.0, 1.0]
2_preprocessing:
  fixed_axes:
    transform_policy: no_transform
    outlier_policy: none
    imputation_policy: none_propagate
    frame_edge_policy: keep_unbalanced
3_feature_engineering:
  nodes:
    - {id: src_X, type: source, selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: predictors}}}
    - {id: src_y, type: source, selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: target}}}
    - {id: lag_x, type: step, op: lag, params: {n_lag: 1}, inputs: [src_X]}
    - {id: y_h, type: step, op: target_construction, params: {mode: point_forecast, method: direct, horizon: 1}, inputs: [src_y]}
  sinks:
    l3_features_v1: {X_final: lag_x, y_final: y_h}
    l3_metadata_v1: auto
4_forecasting_model:
  nodes:
    - {id: src_X, type: source, selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: X_final}}}
    - {id: src_y, type: source, selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: y_final}}}
    - id: fit_ridge
      type: step
      op: fit_model
      params: {family: ridge, alpha: 1.0, min_train_size: 2, forecast_strategy: direct, training_start_rule: expanding, refit_policy: every_origin, search_algorithm: none}
      inputs: [src_X, src_y]
    - {id: predict_ridge, type: step, op: predict, inputs: [fit_ridge, src_X]}
  sinks:
    l4_forecasts_v1: predict_ridge
    l4_model_artifacts_v1: fit_ridge
    l4_training_metadata_v1: auto
5_evaluation:
  fixed_axes:
    primary_metric: mse
    point_metrics: [mse, rmse, mae]
"""


def test_execute_minimal_forecast_materializes_l3_l4_l5():
    result = execute_minimal_forecast(MINIMAL_RECIPE)

    l3_features = result.sink("l3_features_v1")
    l4_forecasts = result.sink("l4_forecasts_v1")
    l4_models = result.sink("l4_model_artifacts_v1")
    l4_training = result.sink("l4_training_metadata_v1")
    l5_eval = result.sink("l5_evaluation_v1")

    assert l3_features.X_final.shape == (4, 2)
    assert l3_features.X_final.column_names == ("x1_lag1", "x2_lag1")
    assert l3_features.sample_index[0] == pd.Timestamp("2020-02-01")
    assert l4_forecasts.model_ids == ("fit_ridge",)
    assert len(l4_forecasts.forecasts) == 2
    assert tuple(l4_forecasts.sample_index) == (pd.Timestamp("2020-04-01"), pd.Timestamp("2020-05-01"))
    assert l4_models.artifacts["fit_ridge"].family == "ridge"
    assert l4_models.artifacts["fit_ridge"].framework == "sklearn"
    assert l4_models.artifacts["fit_ridge"].fit_metadata["runtime"] == "expanding_direct"
    assert l4_training.forecast_origins == (pd.Timestamp("2020-04-01"), pd.Timestamp("2020-05-01"))
    assert l4_training.training_window_per_origin[("fit_ridge", pd.Timestamp("2020-04-01"))] == (
        pd.Timestamp("2020-02-01"),
        pd.Timestamp("2020-03-01"),
    )
    assert l5_eval.metrics_table[["model_id", "target", "horizon"]].to_dict("records") == [
        {"model_id": "fit_ridge", "target": "y", "horizon": 1}
    ]
    assert set(["mse", "rmse", "mae"]) <= set(l5_eval.metrics_table.columns)
    assert l5_eval.ranking_table.iloc[0]["rank_value"] == 1


def test_execute_minimal_forecast_computes_l5_relative_metrics_against_benchmark():
    yaml_text = MINIMAL_RECIPE.replace(
        """    - id: fit_ridge
      type: step
      op: fit_model
      params: {family: ridge, alpha: 1.0, min_train_size: 2, forecast_strategy: direct, training_start_rule: expanding, refit_policy: every_origin, search_algorithm: none}
      inputs: [src_X, src_y]""",
        """    - id: fit_ridge
      type: step
      op: fit_model
      params: {family: ridge, alpha: 1.0, min_train_size: 2, forecast_strategy: direct, training_start_rule: expanding, refit_policy: every_origin, search_algorithm: none}
      inputs: [src_X, src_y]
    - id: fit_ols
      type: step
      op: fit_model
      is_benchmark: true
      params: {family: ols, min_train_size: 2, forecast_strategy: direct, training_start_rule: expanding, refit_policy: every_origin, search_algorithm: none}
      inputs: [src_X, src_y]""",
    ).replace("l4_model_artifacts_v1: fit_ridge", "l4_model_artifacts_v1: [fit_ridge, fit_ols]").replace(
        "primary_metric: mse",
        "primary_metric: relative_mse\n    relative_metrics: [relative_mse, r2_oos, relative_mae, mse_reduction]",
    )

    result = execute_minimal_forecast(yaml_text)
    l4_forecasts = result.sink("l4_forecasts_v1")
    l4_models = result.sink("l4_model_artifacts_v1")
    l5_eval = result.sink("l5_evaluation_v1")

    assert set(l4_forecasts.model_ids) == {"fit_ridge", "fit_ols"}
    assert l4_models.is_benchmark == {"fit_ridge": False, "fit_ols": True}
    assert {"relative_mse", "r2_oos", "relative_mae", "mse_reduction"} <= set(l5_eval.metrics_table.columns)
    benchmark_row = l5_eval.metrics_table.loc[l5_eval.metrics_table["model_id"] == "fit_ols"].iloc[0]
    assert benchmark_row["relative_mse"] == pytest.approx(1.0)
    assert benchmark_row["r2_oos"] == pytest.approx(0.0)
    assert l5_eval.ranking_table.iloc[0]["rank_method"] == "by_primary_metric"


def test_execute_minimal_forecast_materializes_disabled_consumption_artifacts():
    yaml_text = (
        MINIMAL_RECIPE
        + """
1_5_data_summary:
  enabled: true
2_5_pre_post_preprocessing:
  enabled: true
3_5_feature_diagnostics:
  enabled: true
4_5_generator_diagnostics:
  enabled: true
6_statistical_tests: {}
7_interpretation:
  enabled: false
8_output:
  fixed_axes: {}
"""
    )

    result = execute_minimal_forecast(yaml_text)

    assert isinstance(result.sink("l6_tests_v1"), L6TestsArtifact)
    assert result.sink("l6_tests_v1").l6_axis_resolved["enabled"] is False
    assert isinstance(result.sink("l7_importance_v1"), L7ImportanceArtifact)
    assert isinstance(result.sink("l8_artifacts_v1"), L8ArtifactsArtifact)
    assert result.sink("l1_5_diagnostic_v1").enabled is True
    assert result.sink("l2_5_diagnostic_v1").enabled is True
    assert result.sink("l3_5_diagnostic_v1").enabled is True
    assert result.sink("l4_5_diagnostic_v1").enabled is True
    assert result.sink("l8_artifacts_v1").output_directory.as_posix().startswith("macrocast_output/default_recipe/")
    assert "l1_5_diagnostic_v1" in result.sink("l8_artifacts_v1").upstream_hashes
    assert "l2_5_diagnostic_v1" in result.sink("l8_artifacts_v1").upstream_hashes
    assert "l3_5_diagnostic_v1" in result.sink("l8_artifacts_v1").upstream_hashes
    assert "l4_5_diagnostic_v1" in result.sink("l8_artifacts_v1").upstream_hashes
    assert "l6_tests_v1" in result.sink("l8_artifacts_v1").upstream_hashes
    assert result.resolved_axes["l8"]["export_format"] == "json_csv"


def test_execute_minimal_forecast_writes_l8_runtime_exports(tmp_path):
    output_dir = tmp_path / "macrocast_export"
    yaml_text = (
        MINIMAL_RECIPE
        + f"""
6_statistical_tests: {{}}
8_output:
  fixed_axes:
    saved_objects: [forecasts, metrics, ranking, tests]
  leaf_config:
    output_directory: {output_dir.as_posix()}
"""
    )

    result = execute_minimal_forecast(yaml_text)
    l8_artifacts = result.sink("l8_artifacts_v1")

    assert l8_artifacts.artifact_count >= 5
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "recipe.json").exists()
    assert (output_dir / "summary" / "metrics_all_cells.csv").exists()
    assert (output_dir / "summary" / "ranking.csv").exists()
    assert (output_dir / "cell_001" / "forecasts.csv").exists()
    assert (output_dir / "tests_summary.json").exists()


def test_execute_minimal_forecast_materializes_l3_5_diagnostic():
    yaml_text = (
        MINIMAL_RECIPE
        + """
3_5_feature_diagnostics:
  enabled: true
"""
    )

    result = execute_minimal_forecast(yaml_text)
    diagnostic = result.sink("l3_5_diagnostic_v1")

    assert diagnostic.enabled is True
    assert diagnostic.layer_hooked == "l1+l2+l3"
    assert diagnostic.metadata["comparison"]["feature_shape"] == (4, 2)
    assert diagnostic.metadata["feature_summary"]["n_features"] == 2
    assert diagnostic.metadata["lag_block"]["active"] is True
    assert diagnostic.metadata["lag_block"]["lag_feature_count"] == 2
    assert "feature_correlation" in diagnostic.metadata


def test_execute_minimal_forecast_materializes_l4_5_diagnostic():
    yaml_text = (
        MINIMAL_RECIPE
        + """
4_5_generator_diagnostics:
  enabled: true
"""
    )

    result = execute_minimal_forecast(yaml_text)
    diagnostic = result.sink("l4_5_diagnostic_v1")

    assert diagnostic.enabled is True
    assert diagnostic.layer_hooked == "l4"
    assert diagnostic.metadata["forecast_summary"]["n_forecasts"] == 2
    assert diagnostic.metadata["model_summary"]["fit_ridge"]["family"] == "ridge"
    assert diagnostic.metadata["training_summary"]["refit_origin_count"] == {"fit_ridge": 2}
    assert diagnostic.metadata["fit_summary"]["fit_ridge|y|h1"]["n"] == 2


def test_execute_minimal_forecast_materializes_enabled_l6_runtime():
    yaml_text = (
        MINIMAL_RECIPE.replace(
            """    - id: fit_ridge
      type: step
      op: fit_model
      params: {family: ridge, alpha: 1.0, min_train_size: 2, forecast_strategy: direct, training_start_rule: expanding, refit_policy: every_origin, search_algorithm: none}
      inputs: [src_X, src_y]""",
            """    - id: fit_ridge
      type: step
      op: fit_model
      params: {family: ridge, alpha: 1.0, min_train_size: 2, forecast_strategy: direct, training_start_rule: expanding, refit_policy: every_origin, search_algorithm: none}
      inputs: [src_X, src_y]
    - id: fit_ols
      type: step
      op: fit_model
      is_benchmark: true
      params: {family: ols, min_train_size: 2, forecast_strategy: direct, training_start_rule: expanding, refit_policy: every_origin, search_algorithm: none}
      inputs: [src_X, src_y]""",
        ).replace("l4_model_artifacts_v1: fit_ridge", "l4_model_artifacts_v1: [fit_ridge, fit_ols]")
        + """
6_statistical_tests:
  enabled: true
  sub_layers:
    L6_A_equal_predictive:
      enabled: true
    L6_D_multiple_model:
      enabled: true
    L6_F_direction:
      enabled: true
    L6_G_residual:
      enabled: true
"""
    )

    result = execute_minimal_forecast(yaml_text)
    l6_tests = result.sink("l6_tests_v1")

    assert isinstance(l6_tests, L6TestsArtifact)
    assert l6_tests.test_metadata["runtime"] == "core_l6_minimal"
    assert l6_tests.equal_predictive_results
    assert "mcs_inclusion" in l6_tests.multiple_model_results
    assert l6_tests.direction_results
    assert l6_tests.residual_results


def test_execute_minimal_forecast_materializes_enabled_l7_importance():
    yaml_text = (
        MINIMAL_RECIPE
        + """
7_interpretation:
  enabled: true
  nodes:
    - id: src_model
      type: source
      selector: {layer_ref: l4, sink_name: l4_model_artifacts_v1, subset: {model_id: fit_ridge}}
    - id: src_X
      type: source
      selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: X_final}}
    - id: linear_imp
      type: step
      op: model_native_linear_coef
      params: {model_family: ridge}
      inputs: [src_model, src_X]
  sinks:
    l7_importance_v1:
      global: linear_imp
"""
    )

    result = execute_minimal_forecast(yaml_text)
    importance = result.sink("l7_importance_v1")

    assert importance.computation_metadata["runtime"] == "core_l7_minimal"
    assert len(importance.global_importance) == 1
    table = next(iter(importance.global_importance.values()))
    assert set(table["feature"]) == {"x1_lag1", "x2_lag1"}
    assert "importance" in table.columns


def test_execute_minimal_forecast_rejects_unknown_family():
    yaml_text = MINIMAL_RECIPE.replace("family: ridge", "family: nonexistent_family_xyz")

    # The L4 schema rejects via the params_schema options check first
    # ("unknown model family 'X'") and the validator hard rule second
    # ("model family is future or unknown ..."). Either is fine; both mean
    # the recipe was rejected before ever reaching ``_build_l4_model``.
    with pytest.raises(ValueError, match=r"(unknown model family|model family is future)"):
        execute_minimal_forecast(yaml_text)


def test_execute_minimal_forecast_rejects_demoted_future_family():
    # PR-B of the v0.1 honesty pass: macroeconomic_random_forest /
    # bvar_minnesota / etc. are now ``future`` -- the validator must
    # hard-reject before the runtime tries to build the estimator.
    yaml_text = MINIMAL_RECIPE.replace("family: ridge", "family: macroeconomic_random_forest")
    with pytest.raises(ValueError, match=r"future or unknown"):
        execute_minimal_forecast(yaml_text)


@pytest.mark.parametrize(
    "family",
    [
        "ols",
        "lasso",
        "elastic_net",
        "ar_p",
        "decision_tree",
        "random_forest",
        "extra_trees",
        "knn",
        "svr_linear",
        "gradient_boosting",
        "huber",
    ],
)
def test_execute_minimal_forecast_supports_operational_families(family):
    yaml_text = MINIMAL_RECIPE.replace("family: ridge", f"family: {family}")

    result = execute_minimal_forecast(yaml_text)
    l4_models = result.sink("l4_model_artifacts_v1")
    l5_eval = result.sink("l5_evaluation_v1")

    assert l4_models.artifacts["fit_ridge"].family == family
    assert l4_models.artifacts["fit_ridge"].framework == "sklearn"
    assert l5_eval.metrics_table.iloc[0]["model_id"] == "fit_ridge"


def test_execute_minimal_forecast_rejects_exhaustive_min_train_size():
    yaml_text = MINIMAL_RECIPE.replace("min_train_size: 2", "min_train_size: 4")

    with pytest.raises(ValueError, match="min_train_size < aligned observation count"):
        execute_minimal_forecast(yaml_text)


def test_execute_minimal_forecast_runs_l3_concat_and_scale_dag():
    yaml_text = MINIMAL_RECIPE.replace(
        "    - {id: lag_x, type: step, op: lag, params: {n_lag: 1}, inputs: [src_X]}",
        """    - {id: lag_1, type: step, op: lag, params: {n_lag: 1}, inputs: [src_X]}
    - {id: lag_2, type: step, op: lag, params: {n_lag: 2}, inputs: [src_X]}
    - {id: joined_x, type: combine, op: concat, inputs: [lag_1, lag_2]}
    - {id: lag_x, type: step, op: scale, params: {method: zscore}, inputs: [joined_x]}""",
    )
    result = execute_minimal_forecast(yaml_text)
    l3_features = result.sink("l3_features_v1")

    assert l3_features.X_final.shape == (3, 6)
    assert "x1_lag2" in l3_features.X_final.column_names
    assert result.sink("l5_evaluation_v1").metrics_table.iloc[0]["model_id"] == "fit_ridge"


def test_execute_minimal_forecast_runs_l3_transform_before_lag():
    yaml_text = MINIMAL_RECIPE.replace(
        "    - {id: lag_x, type: step, op: lag, params: {n_lag: 1}, inputs: [src_X]}",
        """    - {id: logged_x, type: step, op: log, inputs: [src_X]}
    - {id: diffed_x, type: step, op: diff, params: {n_diff: 1}, inputs: [logged_x]}
    - {id: lag_x, type: step, op: lag, params: {n_lag: 1, include_contemporaneous: true}, inputs: [diffed_x]}""",
    )
    result = execute_minimal_forecast(yaml_text)
    l3_features = result.sink("l3_features_v1")

    assert l3_features.X_final.shape[1] == 4
    assert "x1_lag0" in l3_features.X_final.column_names


def test_execute_minimal_forecast_runs_l3_temporal_feature_ops():
    yaml_text = MINIMAL_RECIPE.replace(
        "    - {id: lag_x, type: step, op: lag, params: {n_lag: 1}, inputs: [src_X]}",
        """    - {id: season_x, type: step, op: seasonal_lag, params: {seasonal_period: 2, n_seasonal_lags: 1}, inputs: [src_X]}
    - {id: ma_x, type: step, op: ma_window, params: {window: 2}, inputs: [src_X]}
    - {id: dummy_x, type: step, op: season_dummy, inputs: [src_X]}
    - {id: lag_x, type: combine, op: concat, inputs: [season_x, ma_x, dummy_x]}""",
    )
    result = execute_minimal_forecast(yaml_text)
    l3_features = result.sink("l3_features_v1")

    assert "x1_s2_lag1" in l3_features.X_final.column_names
    assert "month_1" in l3_features.X_final.column_names
    assert l3_features.X_final.shape[0] == 3


def test_execute_minimal_forecast_runs_l3_polynomial_and_interactions():
    yaml_text = MINIMAL_RECIPE.replace(
        "    - {id: lag_x, type: step, op: lag, params: {n_lag: 1}, inputs: [src_X]}",
        """    - {id: poly_x, type: step, op: polynomial_expansion, params: {degree: 2}, inputs: [src_X]}
    - {id: inter_x, type: step, op: interaction, inputs: [src_X]}
    - {id: lag_x, type: combine, op: concat, inputs: [poly_x, inter_x]}""",
    )
    result = execute_minimal_forecast(yaml_text)
    l3_features = result.sink("l3_features_v1")

    assert "x1_pow2" in l3_features.X_final.column_names
    assert "x1_x_x2" in l3_features.X_final.column_names
    assert l3_features.X_final.shape[1] == 5


def test_execute_minimal_forecast_runs_l3_cumsum_and_ma_increasing_order():
    yaml_text = MINIMAL_RECIPE.replace(
        "    - {id: lag_x, type: step, op: lag, params: {n_lag: 1}, inputs: [src_X]}",
        """    - {id: cumulative_x, type: step, op: cumsum, inputs: [src_X]}
    - {id: lag_x, type: step, op: ma_increasing_order, params: {max_order: 3}, inputs: [cumulative_x]}""",
    )
    result = execute_minimal_forecast(yaml_text)
    l3_features = result.sink("l3_features_v1")

    assert "x1_ma2" in l3_features.X_final.column_names
    assert "x2_ma3" in l3_features.X_final.column_names
    assert l3_features.X_final.shape == (3, 4)
