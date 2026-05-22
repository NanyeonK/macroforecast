# Recipe API quickstart

Run a core layer-contract forecast in under 5 minutes.

This quickstart uses `mf.run(...)`, the v0.9.0 public entry point. It executes the core L1-L8 runtime: custom panels, deterministic L3 features, linear sklearn L4 models, point metrics, lightweight L6/L7 artifacts, diagnostics, and L8 file export.

For the exact support boundary, read [Runtime Support Matrix](runtime_support.md).

## Run A Minimal Core Recipe

```python
import macroforecast as mf

recipe = """
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

8_output:
  fixed_axes:
    saved_objects: [forecasts, metrics, ranking]
  leaf_config:
    output_directory: ./macroforecast_output/quickstart/
"""

import macroforecast as mf
result = mf.run(recipe)
print(result.sink("l5_evaluation_v1").metrics_table)
print(result.sink("l8_artifacts_v1").output_directory)
```

`mf.run` accepts both YAML string and path.

## What Gets Materialized

The result object is an in-memory sink map:

```python
forecasts = result.sink("l4_forecasts_v1")
metrics = result.sink("l5_evaluation_v1").metrics_table
ranking = result.sink("l5_evaluation_v1").ranking_table
output = result.sink("l8_artifacts_v1")
```

With `8_output` enabled, the runtime writes:

```text
macroforecast_output/quickstart/
  manifest.json
  recipe.json
  summary/
    metrics_all_cells.csv
    ranking.csv
  cell_001/
    forecasts.csv
```

## Enabling Diagnostics, Tests, And Importance

Add these blocks when needed:

```yaml
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
```

## Next Steps

- [Runtime Support Matrix](runtime_support.md) -- current support boundary
- [Your First Study](first_study.md) -- extend the quickstart to diagnostics, L6, L7, and L8 export
- [Understanding Output](understanding_output.md) -- output directory and artifact guide

---

**Note on FRED data and real-time vintages**: macroforecast v0.9.x uses final-revised FRED data
(current vintage) when `custom_source_policy: official_only` is set. Real-time vintage tracking
(ALFRED) is planned for v1.x; `vintage_policy: real_time_alfred` raises `NotImplementedError`
in v0.9.x. For details and workarounds, see
[Your First Study — Real-Time Data Caveat](first_study.md#real-time-data-caveat).

---

**Note on output directory naming**: The runtime default is `./macrocast_output/<recipe_id>/<timestamp>/` (defined in `macroforecast.core.types` and `macroforecast.core.layers.l8`). To override, set `output_directory` in the `8_output.leaf_config` block, as the examples above do (`./macroforecast_output/quickstart/`). The historical `macrocast_output/` default is preserved for backward compatibility with v0.1-era manifests.
