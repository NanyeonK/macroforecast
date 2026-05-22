# A complete benchmarking study

In this tutorial you will extend the AR(2) baseline from {doc}`01_first_forecast` into
a complete benchmarking study: multiple model families, a parameter sweep, the
Diebold-Mariano significance test, and an importance figure. By the end you will have
a recipe that is ready for a paper.

---

## Adding a second model family

Start with the same 20-row synthetic panel from tutorial 1. Extend the recipe by
adding a ridge regression node alongside the AR(2) node and marking AR(2) as the
benchmark via `is_benchmark: true`.

The `is_benchmark: true` flag tells L6 tests which model to use as the baseline
when computing the Diebold-Mariano statistic. All other models are compared against it.

The full `4_forecasting_model` block with both nodes:

```yaml
4_forecasting_model:
  nodes:
    - id: src_X
      type: source
      selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: X_final}}
    - id: src_y
      type: source
      selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: y_final}}
    - id: fit_ar
      type: step
      op: fit_model
      params: {family: ar_p, n_lag: 2, forecast_strategy: direct,
               training_start_rule: expanding, refit_policy: every_origin,
               search_algorithm: none, min_train_size: 6}
      is_benchmark: true
      inputs: [src_y]
    - id: predict_ar
      type: step
      op: predict
      inputs: [fit_ar]
    - id: fit_ridge
      type: step
      op: fit_model
      params: {family: ridge, alpha: 1.0, forecast_strategy: direct,
               training_start_rule: expanding, refit_policy: every_origin,
               search_algorithm: none, min_train_size: 6}
      inputs: [src_X, src_y]
    - id: predict_ridge
      type: step
      op: predict
      inputs: [fit_ridge, src_X]
  sinks:
    l4_forecasts_v1: predict_ridge
    l4_model_artifacts_v1: [fit_ar, fit_ridge]
    l4_training_metadata_v1: auto
```

Notice that `fit_ar` takes only `[src_y]` as input — AR(p) models use only the
target history, not the predictor matrix. Ridge uses both `[src_X, src_y]`.

---

## Sweeping over lag orders

To compare AR models with different lag orders, use a sweep marker on the `n_lag`
parameter. The runtime runs each lag order as an independent cell.

```python
import macroforecast as mf

recipe = """
0_meta:
  fixed_axes:
    failure_policy: fail_fast
    reproducibility_mode: seeded_reproducible
  leaf_config:
    random_seed: 0

1_data:
  fixed_axes:
    custom_source_policy: custom_panel_only
    frequency: monthly
    horizon_set: custom_list
  leaf_config:
    target: gdp_growth
    target_horizons: [1]
    custom_panel_inline:
      date:
        [2015-01-01, 2015-02-01, 2015-03-01, 2015-04-01, 2015-05-01,
         2015-06-01, 2015-07-01, 2015-08-01, 2015-09-01, 2015-10-01,
         2015-11-01, 2015-12-01, 2016-01-01, 2016-02-01, 2016-03-01,
         2016-04-01, 2016-05-01, 2016-06-01, 2016-07-01, 2016-08-01]
      gdp_growth:
        [0.3, 0.5, 0.4, 0.6, 0.5, 0.7, 0.6, 0.8, 0.7, 0.9,
         0.8, 1.0, 0.9, 1.1, 1.0, 1.2, 1.1, 1.3, 1.2, 1.4]
      ip_index:
        [100.0, 100.5, 101.0, 101.5, 102.0, 102.5, 103.0, 103.5, 104.0, 104.5,
         105.0, 105.5, 106.0, 106.5, 107.0, 107.5, 108.0, 108.5, 109.0, 109.5]

2_preprocessing:
  fixed_axes:
    transform_policy: no_transform
    outlier_policy: none
    imputation_policy: none_propagate
    frame_edge_policy: keep_unbalanced

3_feature_engineering:
  nodes:
    - id: src_X
      type: source
      selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: predictors}}
    - id: src_y
      type: source
      selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: target}}
    - id: lag_x
      type: step
      op: lag
      params: {n_lag: 1}
      inputs: [src_X]
    - id: y_h
      type: step
      op: target_construction
      params: {mode: point_forecast, method: direct, horizon: 1}
      inputs: [src_y]
  sinks:
    l3_features_v1: {X_final: lag_x, y_final: y_h}
    l3_metadata_v1: auto

4_forecasting_model:
  nodes:
    - id: src_X
      type: source
      selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: X_final}}
    - id: src_y
      type: source
      selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: y_final}}
    - id: fit_ar
      type: step
      op: fit_model
      params:
        family: ar_p
        n_lag: {sweep: [1, 2, 3]}
        forecast_strategy: direct
        training_start_rule: expanding
        refit_policy: every_origin
        search_algorithm: none
        min_train_size: 6
      inputs: [src_y]
    - id: predict_ar
      type: step
      op: predict
      inputs: [fit_ar]
  sinks:
    l4_forecasts_v1: predict_ar
    l4_model_artifacts_v1: fit_ar
    l4_training_metadata_v1: auto

5_evaluation:
  fixed_axes:
    primary_metric: mse
    point_metrics: [mse, rmse, mae]
"""

result = mf.run(recipe)
print(f"Number of cells: {len(result.cells)}")
for cell in result.cells:
    metrics = cell.runtime_result.artifacts["l5_evaluation_v1"].metrics_table
    print(cell.cell_id, "->", round(metrics["mse"].values[0], 6))
```

Each cell ran independently with its own seed. The sweep expands combinatorially
when you have multiple sweep axes.

---

## The Diebold-Mariano test

Add a `6_statistical_tests` block to run the DM test. The test asks: does ridge
significantly outperform AR(2)?

```python
recipe_with_dm = """
0_meta:
  fixed_axes:
    failure_policy: fail_fast
    reproducibility_mode: seeded_reproducible
  leaf_config:
    random_seed: 0

1_data:
  fixed_axes:
    custom_source_policy: custom_panel_only
    frequency: monthly
    horizon_set: custom_list
  leaf_config:
    target: gdp_growth
    target_horizons: [1]
    custom_panel_inline:
      date:
        [2015-01-01, 2015-02-01, 2015-03-01, 2015-04-01, 2015-05-01,
         2015-06-01, 2015-07-01, 2015-08-01, 2015-09-01, 2015-10-01,
         2015-11-01, 2015-12-01, 2016-01-01, 2016-02-01, 2016-03-01,
         2016-04-01, 2016-05-01, 2016-06-01, 2016-07-01, 2016-08-01]
      gdp_growth:
        [0.3, 0.5, 0.4, 0.6, 0.5, 0.7, 0.6, 0.8, 0.7, 0.9,
         0.8, 1.0, 0.9, 1.1, 1.0, 1.2, 1.1, 1.3, 1.2, 1.4]
      ip_index:
        [100.0, 100.5, 101.0, 101.5, 102.0, 102.5, 103.0, 103.5, 104.0, 104.5,
         105.0, 105.5, 106.0, 106.5, 107.0, 107.5, 108.0, 108.5, 109.0, 109.5]

2_preprocessing:
  fixed_axes:
    transform_policy: no_transform
    outlier_policy: none
    imputation_policy: none_propagate
    frame_edge_policy: keep_unbalanced

3_feature_engineering:
  nodes:
    - id: src_X
      type: source
      selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: predictors}}
    - id: src_y
      type: source
      selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: target}}
    - id: lag_x
      type: step
      op: lag
      params: {n_lag: 1}
      inputs: [src_X]
    - id: y_h
      type: step
      op: target_construction
      params: {mode: point_forecast, method: direct, horizon: 1}
      inputs: [src_y]
  sinks:
    l3_features_v1: {X_final: lag_x, y_final: y_h}
    l3_metadata_v1: auto

4_forecasting_model:
  nodes:
    - id: src_X
      type: source
      selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: X_final}}
    - id: src_y
      type: source
      selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: y_final}}
    - id: fit_ar
      type: step
      op: fit_model
      params: {family: ar_p, n_lag: 2, forecast_strategy: direct,
               training_start_rule: expanding, refit_policy: every_origin,
               search_algorithm: none, min_train_size: 6}
      is_benchmark: true
      inputs: [src_y]
    - id: predict_ar
      type: step
      op: predict
      inputs: [fit_ar]
    - id: fit_ridge
      type: step
      op: fit_model
      params: {family: ridge, alpha: 1.0, forecast_strategy: direct,
               training_start_rule: expanding, refit_policy: every_origin,
               search_algorithm: none, min_train_size: 6}
      inputs: [src_X, src_y]
    - id: predict_ridge
      type: step
      op: predict
      inputs: [fit_ridge, src_X]
  sinks:
    l4_forecasts_v1: predict_ridge
    l4_model_artifacts_v1: [fit_ar, fit_ridge]
    l4_training_metadata_v1: auto

5_evaluation:
  fixed_axes:
    primary_metric: mse
    point_metrics: [mse, rmse, mae]

6_statistical_tests:
  enabled: true
  sub_layers:
    L6_A_equal_predictive:
      enabled: true
      fixed_axes:
        equal_predictive_test: dm_diebold_mariano
        loss_function: squared
        model_pair_strategy: vs_benchmark_only
        hln_correction: true
"""

result_dm = mf.run(recipe_with_dm)
cell = result_dm.cells[0]
arts = cell.runtime_result.artifacts

tests = arts["l6_tests_v1"]
print(tests.equal_predictive_results)
```

The HLN correction accounts for the fact that walk-forward errors at adjacent origins
are correlated. Always use `hln_correction: true` for multi-step horizons.

```{note}
The DM test requires enough out-of-sample origins to be meaningful. With a 20-row
panel and `min_train_size: 6`, you get about 12 origins. Interpret p-values
cautiously on such short samples.
```

---

## An importance figure

Add a `7_interpretation` block to extract ridge coefficients as an importance measure.

```python
recipe_with_l7 = """
0_meta:
  fixed_axes:
    failure_policy: fail_fast
    reproducibility_mode: seeded_reproducible
  leaf_config:
    random_seed: 0

1_data:
  fixed_axes:
    custom_source_policy: custom_panel_only
    frequency: monthly
    horizon_set: custom_list
  leaf_config:
    target: gdp_growth
    target_horizons: [1]
    custom_panel_inline:
      date:
        [2015-01-01, 2015-02-01, 2015-03-01, 2015-04-01, 2015-05-01,
         2015-06-01, 2015-07-01, 2015-08-01, 2015-09-01, 2015-10-01,
         2015-11-01, 2015-12-01, 2016-01-01, 2016-02-01, 2016-03-01,
         2016-04-01, 2016-05-01, 2016-06-01, 2016-07-01, 2016-08-01]
      gdp_growth:
        [0.3, 0.5, 0.4, 0.6, 0.5, 0.7, 0.6, 0.8, 0.7, 0.9,
         0.8, 1.0, 0.9, 1.1, 1.0, 1.2, 1.1, 1.3, 1.2, 1.4]
      ip_index:
        [100.0, 100.5, 101.0, 101.5, 102.0, 102.5, 103.0, 103.5, 104.0, 104.5,
         105.0, 105.5, 106.0, 106.5, 107.0, 107.5, 108.0, 108.5, 109.0, 109.5]

2_preprocessing:
  fixed_axes:
    transform_policy: no_transform
    outlier_policy: none
    imputation_policy: none_propagate
    frame_edge_policy: keep_unbalanced

3_feature_engineering:
  nodes:
    - id: src_X
      type: source
      selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: predictors}}
    - id: src_y
      type: source
      selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: target}}
    - id: lag_x
      type: step
      op: lag
      params: {n_lag: 1}
      inputs: [src_X]
    - id: y_h
      type: step
      op: target_construction
      params: {mode: point_forecast, method: direct, horizon: 1}
      inputs: [src_y]
  sinks:
    l3_features_v1: {X_final: lag_x, y_final: y_h}
    l3_metadata_v1: auto

4_forecasting_model:
  nodes:
    - id: src_X
      type: source
      selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: X_final}}
    - id: src_y
      type: source
      selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: y_final}}
    - id: fit_ar
      type: step
      op: fit_model
      params: {family: ar_p, n_lag: 2, forecast_strategy: direct,
               training_start_rule: expanding, refit_policy: every_origin,
               search_algorithm: none, min_train_size: 6}
      is_benchmark: true
      inputs: [src_y]
    - id: predict_ar
      type: step
      op: predict
      inputs: [fit_ar]
    - id: fit_ridge
      type: step
      op: fit_model
      params: {family: ridge, alpha: 1.0, forecast_strategy: direct,
               training_start_rule: expanding, refit_policy: every_origin,
               search_algorithm: none, min_train_size: 6}
      inputs: [src_X, src_y]
    - id: predict_ridge
      type: step
      op: predict
      inputs: [fit_ridge, src_X]
  sinks:
    l4_forecasts_v1: predict_ridge
    l4_model_artifacts_v1: [fit_ar, fit_ridge]
    l4_training_metadata_v1: auto

5_evaluation:
  fixed_axes:
    primary_metric: mse
    point_metrics: [mse, rmse, mae]

6_statistical_tests:
  enabled: true
  sub_layers:
    L6_A_equal_predictive:
      enabled: true
      fixed_axes:
        equal_predictive_test: dm_diebold_mariano
        loss_function: squared
        model_pair_strategy: vs_benchmark_only
        hln_correction: true

7_interpretation:
  enabled: true
  nodes:
    - id: src_model
      type: source
      selector:
        layer_ref: l4
        sink_name: l4_model_artifacts_v1
        subset: {model_id: fit_ridge}
    - id: src_X
      type: source
      selector:
        layer_ref: l3
        sink_name: l3_features_v1
        subset: {component: X_final}
    - id: coef_importance
      type: step
      op: model_native_linear_coef
      params: {model_family: ridge}
      inputs: [src_model, src_X]
  sinks:
    l7_importance_v1:
      global: coef_importance
"""

result_l7 = mf.run(recipe_with_l7, output_directory="./tutorial_output/full_study/")
cell = result_l7.cells[0]
arts = cell.runtime_result.artifacts

importance = arts["l7_importance_v1"]
print(importance.global_importance)
```

The global importance table shows the average absolute coefficient across forecast
origins. The L7 runtime writes a bar chart to the output directory automatically.

---

## The complete study — reading the output directory

After the run above, the output directory contains:

```text
tutorial_output/full_study/
  manifest.json           <- full provenance + recipe + hash fingerprints
  recipe.json             <- human-readable recipe snapshot
  cell_001/
    forecasts.csv         <- per-origin forecasts
    raw_panel.csv
    clean_panel.csv
    feature_metadata.json
  summary/
    metrics_all_cells.csv <- metrics table for all cells
    ranking.csv
```

---

## What to do next

- Continue to {doc}`03_custom_model` to register your own model callable and plug
  it in as a third contender in this study.
- See {doc}`../how_to/sweep_over_models` for a focused recipe on multi-model sweeps.
- See {doc}`../how_to/tune_hyperparameters` for BIC/AIC/grid-search-based tuning.
