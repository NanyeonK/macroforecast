# Your first forecast

By the end of this tutorial you will have run your first AR(p) forecast, inspected
the output manifest, and confirmed the bit-exact replicate guarantee. It takes about
five minutes.

---

## Before you start

Verify that macroforecast is installed:

```python
import macroforecast as mf
print(mf.__version__)   # 0.9.2b1 or later
```

If the import fails, see {doc}`00_install`.

---

## Your first recipe

A recipe is a YAML document that describes your full forecasting study. You pass it
to `mf.run()` and get back a result object containing all your forecasts, metrics,
and artifacts.

The recipe below uses a 20-row synthetic panel with a linear trend so the output
is visually interpretable. Every layer from L0 (study setup) through L8 (output)
is represented in full.

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
        n_lag: 2
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
```

Notice that `3_feature_engineering` takes a one-period lag of the predictors. This
ensures the model never sees future data — a hard leakage rule enforced by the runtime.

---

## Run it and inspect the results

```python
result = mf.run(recipe, output_directory="./tutorial_output/first_forecast/")
```

The call returns a `ManifestExecutionResult` object. Access metrics and artifacts
through `result.cells[0].runtime_result.artifacts`:

```python
# The artifacts dict maps sink names to artifact objects
cell = result.cells[0]
arts = cell.runtime_result.artifacts

# Metrics table
metrics = arts["l5_evaluation_v1"].metrics_table
print(metrics[["model_id", "mse", "rmse", "mae"]])
```

You will see one row for the AR(2) model with columns `model_id`, `mse`, `rmse`,
and `mae`. Because this panel has a clear linear trend and the lag features capture
it, the MSE is very small.

You can also use the aggregate `result.metrics` property, which adds a `cell_id`
column and concatenates all cells:

```python
print(result.metrics[["cell_id", "model_id", "mse"]])
```

Inspect individual forecasts:

```python
forecasts_artifact = arts["l4_forecasts_v1"]
# The forecasts dict maps (model_id, target, horizon, origin) -> float
for key, val in list(forecasts_artifact.forecasts.items())[:3]:
    print(key, "->", round(val, 4))
```

Each key encodes the model ID, target series name, forecast horizon, and the origin
date — the last date the model saw before making its prediction.

---

## The manifest and bit-exact replication

The most important file in the output directory is `manifest.json`. It is the
recipe's receipt — a machine-readable record of exactly what was run, which data
was used, and a hash fingerprint of every artifact.

```python
import pathlib

manifest_path = pathlib.Path("./tutorial_output/first_forecast/manifest.json")
replication = mf.replicate(str(manifest_path))
print("Recipe match:", replication.recipe_match)
print("Artifacts match:", replication.sink_hashes_match)
```

If both print `True`, you have confirmed that re-running the recipe produces
bit-identical artifacts. Share `manifest.json` with a colleague and they can run
the same check.

```{note}
The replicate guarantee requires the same macroforecast version and the same
dependency lockfile. Use `pip freeze > requirements.txt` to record your
environment at run time.
```

---

## What to do next

- Continue to {doc}`02_full_study` to extend this recipe with multiple models, a
  model sweep, the DM test, and importance figures.
- See {doc}`../how_to/add_custom_dataset` to use your own panel data instead of
  the inline fixture.
- See {doc}`../how_to/replicate_a_study` for the replication workflow in production.
