# Your First Study: Ridge With Diagnostics, Tests, Importance, And Export

This walkthrough builds a complete core layer-contract study. It uses the current supported runtime path: custom panel data, L3 lag features, L4 linear sklearn forecasting, L5 point metrics, optional diagnostic layers, lightweight L6 tests, L7 linear importance, and L8 file export.

For the exact support boundary, see [Runtime Support Matrix](runtime_support.md).

## Study Design

Fixed design:

- Data: custom monthly panel
- Target: `y`
- Horizon: 1 month ahead
- Features: one lag of all predictors
- Model: expanding-window ridge regression
- Evaluation: MSE/RMSE/MAE
- Output: forecasts, metrics, ranking, tests, importance, diagnostics

## Recipe

```yaml
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

1_5_data_summary:
  enabled: true
2_5_pre_post_preprocessing:
  enabled: true
3_5_feature_diagnostics:
  enabled: true
4_5_generator_diagnostics:
  enabled: true

6_statistical_tests:
  enabled: true
  sub_layers:
    L6_F_direction:
      enabled: true
    L6_G_residual:
      enabled: true

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

8_output:
  fixed_axes:
    saved_objects: [forecasts, metrics, ranking, tests, importance, diagnostics_all]
  leaf_config:
    output_directory: ./macrocast_output/first_study/
```

## Execute

```python
from macroforecast.core import execute_minimal_forecast

result = execute_minimal_forecast(open("my_study.yaml").read())
```

## Inspect Results

```python
print(result.sink("l5_evaluation_v1").metrics_table)
print(result.sink("l5_evaluation_v1").ranking_table)
print(result.sink("l6_tests_v1").direction_results)
print(result.sink("l7_importance_v1").global_importance)
print(result.sink("l8_artifacts_v1").exported_files)
```

## What You Learned

- L1-L4 construct forecasts from a fixed information set.
- L5 evaluates forecast accuracy.
- L1.5-L4.5 provide non-blocking diagnostic artifacts.
- L6 adds lightweight inference artifacts when enabled.
- L7 adds importance artifacts when enabled.
- L8 writes a directory that can be inspected without rerunning the study.

## Next Steps

- [Understanding Output](understanding_output.md) — every current core runtime artifact
- [Runtime Support Matrix](runtime_support.md) — what is runtime-supported today
