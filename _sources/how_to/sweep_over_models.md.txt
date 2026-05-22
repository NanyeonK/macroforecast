# How to sweep over models

Run a sweep over multiple model families and collect per-cell metrics.

---

## Sweep over families with the `{sweep}` marker

Place a `{sweep: [...]}` marker on the `family` parameter to run each family as
an independent cell. Each cell gets its own seed and produces its own evaluation
artifact.

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
    target: y
    target_horizons: [1]
    custom_panel_inline:
      date:
        [2018-01-01, 2018-02-01, 2018-03-01, 2018-04-01, 2018-05-01,
         2018-06-01, 2018-07-01, 2018-08-01, 2018-09-01, 2018-10-01,
         2018-11-01, 2018-12-01, 2019-01-01, 2019-02-01, 2019-03-01,
         2019-04-01, 2019-05-01, 2019-06-01, 2019-07-01, 2019-08-01,
         2019-09-01, 2019-10-01, 2019-11-01, 2019-12-01]
      y:
        [1.0, 1.2, 1.4, 1.1, 1.3, 1.5, 1.7, 1.6, 1.8, 2.0,
         1.9, 2.1, 2.3, 2.2, 2.4, 2.6, 2.5, 2.7, 2.9, 3.0,
         2.8, 3.1, 3.3, 3.2]
      x1:
        [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
         5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0,
         10.5, 11.0, 11.5, 12.0]

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
    - id: fit_model
      type: step
      op: fit_model
      params:
        family: {sweep: [ar_p, ridge, random_forest]}
        n_lag: 2
        alpha: 1.0
        n_estimators: 50
        forecast_strategy: direct
        training_start_rule: expanding
        refit_policy: every_origin
        search_algorithm: none
        min_train_size: 10
      inputs: [src_y]
    - id: predict_model
      type: step
      op: predict
      inputs: [fit_model]
  sinks:
    l4_forecasts_v1: predict_model
    l4_model_artifacts_v1: fit_model
    l4_training_metadata_v1: auto

5_evaluation:
  fixed_axes:
    primary_metric: mse
    point_metrics: [mse, rmse, mae]
"""

result = mf.run(recipe)
for cell in result.cells:
    m = cell.runtime_result.artifacts["l5_evaluation_v1"].metrics_table
    print(cell.cell_id, m[["model_id", "mse"]].to_string(index=False))
```

When sweeping across families that have different input requirements, provide the
union of inputs and let each family select what it needs. AR(p) uses only `src_y`;
ridge and random_forest also use `src_X`. By passing only `[src_y]` here, all three
families use the same target-only input path.

---

## Sweeping over hyperparameter values

Sweep `alpha` across values for ridge regression:

```yaml
    - id: fit_ridge
      type: step
      op: fit_model
      params:
        family: ridge
        alpha: {sweep: [0.01, 0.1, 1.0, 10.0]}
        forecast_strategy: direct
        training_start_rule: expanding
        refit_policy: every_origin
        search_algorithm: none
        min_train_size: 10
      inputs: [src_X, src_y]
```

Combining a family sweep with a parameter sweep creates a combinatorial grid. With
3 families and 4 alpha values, you get 12 cells.

---

## Reading sweep results into a summary table

```python
import pandas as pd

rows = []
for cell in result.cells:
    m = cell.runtime_result.artifacts["l5_evaluation_v1"].metrics_table
    for _, row in m.iterrows():
        rows.append({"cell": cell.cell_id, "model": row["model_id"],
                     "mse": row["mse"], "rmse": row["rmse"]})

summary = pd.DataFrame(rows).sort_values("mse")
print(summary.to_string(index=False))
```

---

See {doc}`../tutorial/02_full_study` for a full narrative context.
