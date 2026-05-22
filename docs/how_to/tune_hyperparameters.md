# How to tune hyperparameters

Use grid search, random search, or information criteria to tune model parameters
in L4.

---

## Grid search over alpha

Set `search_algorithm: grid` and provide `param_grid` to sweep regularization
strengths within a single L4 cell. The runtime re-selects the best alpha at each
forecast origin using the training window only — strictly in-sample tuning, no
future data leaks.

```yaml
4_forecasting_model:
  nodes:
    - id: src_X
      type: source
      selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: X_final}}
    - id: src_y
      type: source
      selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: y_final}}
    - id: fit_ridge
      type: step
      op: fit_model
      params:
        family: ridge
        forecast_strategy: direct
        training_start_rule: expanding
        refit_policy: every_origin
        search_algorithm: grid
        param_grid:
          alpha: [0.01, 0.1, 1.0, 10.0, 100.0]
        cv_folds: 3
        scoring: neg_mean_squared_error
        min_train_size: 10
      inputs: [src_X, src_y]
    - id: predict_ridge
      type: step
      op: predict
      inputs: [fit_ridge, src_X]
  sinks:
    l4_forecasts_v1: predict_ridge
    l4_model_artifacts_v1: fit_ridge
    l4_training_metadata_v1: auto
```

Read the selected alpha from the tuning log:

```python
cell = result.cells[0]
tuning_log = cell.runtime_result.artifacts["l4_training_metadata_v1"].tuning_log
for origin, info in list(tuning_log.items())[:3]:
    print(origin, info.get("best_params"))
```

---

## Random search over a continuous range

For large grids, use `search_algorithm: random` to sample a subset of parameter
combinations:

```yaml
        search_algorithm: random
        param_distributions:
          alpha: {distribution: log_uniform, low: 0.001, high: 1000.0}
        n_iter: 20
```

This requires `pip install macroforecast[tuning]` for optuna-backed distributions.
It falls back to scipy log-uniform when optuna is not installed.

---

## Information criteria for AR(p) lag selection

For AR(p) lag selection, use `search_algorithm: bic` or `aic`. The runtime
fits AR models with lags from 1 to `max_lag` and picks the order that minimizes
the criterion at each origin:

```yaml
    - id: fit_ar
      type: step
      op: fit_model
      params:
        family: ar_p
        forecast_strategy: direct
        training_start_rule: expanding
        refit_policy: every_origin
        search_algorithm: bic
        max_lag: 6
        min_train_size: 10
      inputs: [src_y]
```

---

See {doc}`../tutorial/02_full_study` for a full recipe context.
