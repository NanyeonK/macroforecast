# Bring your own model

In this tutorial you will register a custom model callable, add it as a contender
in the benchmarking study from {doc}`02_full_study`, and compare its performance
against the AR(2) baseline.

---

## The custom model contract

Before writing any code, understand the contract macroforecast expects. Your
function must accept exactly four arguments:

```python
def my_model(
    X_train,    # pd.DataFrame: n_train rows x n_features cols (training predictors)
    y_train,    # pd.Series: n_train training target values
    X_test,     # pd.DataFrame: exactly 1 row (one forecast origin)
    context,    # dict: runtime metadata (model_name, target, horizon, feature_names, ...)
) -> float:
    ...         # return a single float — the one-step forecast
```

Four rules:

1. Fit only on the data supplied in `X_train` and `y_train`. The runtime has
   already applied the expanding-window split for you.
2. Return a scalar (Python `float`) or a one-element sequence. The runtime coerces
   both via `float(value)`.
3. Never read future rows, the global panel, or any state outside the function.
4. If your estimator fails on a degenerate window, raise `ValueError` — the
   `failure_policy` in L0 decides what to do.

---

## A naive baseline

Start with the simplest possible custom model — a last-value naive baseline — to
see the registration and recipe mechanics in one go.

```python
import macroforecast as mf

@mf.register_model("naive_last_value")
def naive_last_value(X_train, y_train, X_test, context):
    """Return the last observed target value as the forecast."""
    return float(y_train.iloc[-1])

# Confirm registration
print(mf.list_custom_models())   # ('naive_last_value',)
```

The `@mf.register_model` decorator registers the function under the name
`'naive_last_value'` in the current Python process. You reference this name in
your recipe with `family: naive_last_value`.

```{warning}
The registry lives in the Python process. If you call `mf.run()` in a different
script without importing this registration code first, the recipe will fail with
`ValueError: unknown model family 'naive_last_value'`. Always import your
registration module before calling `mf.run()`.
```

---

## Add the custom model to the recipe

The simplest way to compare multiple model families is to sweep over them using
the `{sweep: [...]}` marker on the `family` parameter. Each value becomes an
independent cell.

```python
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
    - id: fit_model
      type: step
      op: fit_model
      params:
        family: {sweep: [ar_p, naive_last_value]}
        n_lag: 2                    # used by ar_p; ignored by naive_last_value
        forecast_strategy: direct
        training_start_rule: expanding
        refit_policy: every_origin
        search_algorithm: none
        min_train_size: 6
      inputs: [src_y]              # both models only need the target series
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
```

---

## Run and compare

Register the model, then run:

```python
@mf.register_model("naive_last_value")
def naive_last_value(X_train, y_train, X_test, context):
    return float(y_train.iloc[-1])

result = mf.run(recipe)
print(f"Cells: {len(result.cells)}")   # 2 cells (one per sweep value)

for cell in result.cells:
    metrics = cell.runtime_result.artifacts["l5_evaluation_v1"].metrics_table
    print(cell.cell_id, "MSE:", round(metrics["mse"].values[0], 6))
```

The naive last-value baseline typically has higher MSE than AR(2) on a trending
series. The point is to see the custom model integrated into the same evaluation
pipeline.

Inspect the model artifact to confirm `framework = "custom"`:

```python
cell_naive = result.cells[-1]   # last cell: naive_last_value
art = cell_naive.runtime_result.artifacts["l4_model_artifacts_v1"]
fitted = next(iter(art.artifacts.values()))
print("framework:", fitted.framework)   # "custom"
print("family:", fitted.family)         # "naive_last_value"
```

---

## A more realistic custom model

The naive baseline is easy to verify. Here is a slightly more realistic example:
a custom exponential-smoothing forecast that uses only the target series.

```python
import numpy as np

@mf.register_model("exp_smooth_alpha04")
def exp_smooth(X_train, y_train, X_test, context):
    """Simple exponential smoothing with fixed alpha=0.4."""
    alpha = 0.4
    y = y_train.values
    smoothed = y[0]
    for v in y[1:]:
        smoothed = alpha * v + (1 - alpha) * smoothed
    return float(smoothed)
```

This function fits entirely from `y_train.values`. It ignores `X_train` and
`context`. For a production custom model, you might use `context["feature_names"]`
to select predictors by name, or `context["horizon"]` to adjust the forecast
horizon.

Swap it into the sweep: `family: {sweep: [ar_p, exp_smooth_alpha04]}` and re-run.

---

## Debugging tips

If your custom model silently returns `NaN`, check three things:

1. **Return value shape.** The runtime calls `float(return_value)`. If
   `return_value` is a NumPy array with more than one element, `float()` raises.
   Return `float(arr[0])` explicitly.
2. **Registration order.** Import your registration module before `mf.run()`. Use
   `mf.list_custom_models()` to confirm the name is registered.
3. **Training window.** If `y_train` is shorter than your model requires, raise
   `ValueError` rather than returning a degenerate value. The runtime's
   `failure_policy` will handle it.

Clean up the registry between sessions:

```python
mf.clear_custom_models()
print(mf.list_custom_models())   # ()
```

---

## What to do next

- See {doc}`../how_to/add_custom_model` for the terse task-recipe version of this
  tutorial.
- See {doc}`../how_to/use_custom_hooks` for all five extension points: custom feature
  blocks, combiners, preprocessors, target transformers, and models.
- See {doc}`02_full_study` to revisit the full benchmarking study setup.
