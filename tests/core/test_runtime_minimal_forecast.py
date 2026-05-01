from __future__ import annotations

import pandas as pd
import pytest

from macrocast.core import execute_minimal_forecast


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


def test_execute_minimal_forecast_rejects_non_ridge_family():
    yaml_text = MINIMAL_RECIPE.replace("family: ridge", "family: xgboost")

    with pytest.raises(NotImplementedError, match="supports family=ridge only"):
        execute_minimal_forecast(yaml_text)


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
