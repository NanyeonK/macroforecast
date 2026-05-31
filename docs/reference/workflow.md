# Workflow Contract

[Back to reference](index.md)

`macroforecast` keeps statistical functions callable and puts workflow
composition in the forecasting runner.

## Ownership

| Module | Owns | Does not own |
| --- | --- | --- |
| `macroforecast.data` | Canonical pandas panels and metadata. | Model-ready matrices. |
| `macroforecast.preprocessing` | Callable data-cleaning transforms and reusable preprocessing specs. | Expanding/rolling schedule decisions. |
| `macroforecast.feature_engineering` | Callable feature/target transforms and reusable feature specs. | Forecast-origin scheduling. |
| `macroforecast.window` | Estimation mode, validation/test time frames, retrain/retune cadence, and reusable stage policies. | Low-level transformation formulas. |
| `macroforecast.models` | Model callables and model-owned hyperparameter spaces. | Forecast loops or forecast combination. |
| `macroforecast.selection` | Parameter search over supplied data and validation windows. | Global run orchestration. |
| `macroforecast.forecasting` | Runner-level composition and forecast combination. | Low-level transformation formulas. |
| `macroforecast.metrics` | Forecast scoring, forecast ranking, and metric resolution. | Data splits, model fitting, or statistical tests. |
| `macroforecast.tests` | Forecast-comparison statistical tests and residual diagnostics. | Forecast scoring or model fitting. |
| `macroforecast.evaluation` | Namespace wrapper for `metrics` and `tests`. | Callable metric/test functions. |
| `macroforecast.interpretation` | Post-fit importance and effect summaries. | Model fitting, feature construction, or forecast testing. |
| `macroforecast.output` | Writing artifacts and schema-aware provenance manifests. | Deciding model/window/evaluation design. |

## Runner Loop

The runner is the only module that combines stages:

```python
for origin in window.iter_origins(panel.index):
    preprocessing_fit_panel = rows_allowed_by(preprocessing_policy, origin)
    fitted_preprocessing = preprocess_spec.fit(
        preprocessing_fit_panel,
        policy=preprocessing_policy.scope,
    )
    processed = fitted_preprocessing.transform(rows_needed_by_runner, ...)

    feature_fit_panel = rows_allowed_by(feature_policy, origin)
    builder = feature_spec.fit(feature_fit_panel)
    train_features = builder.transform(processed, index=train_dates)
    test_features = builder.transform(processed, index=test_dates)

    selection_features = rows_allowed_by(selection_policy, origin)
    selected_params = selection.select_params(model, selection_features, ...)
    fit = model(train_features.X, train_features.y, **selected_params)
    forecast = fit.predict(test_features.X)
```

This keeps expanding, rolling, fixed-sample, retrain cadence, and retune cadence
in `macroforecast.window` and `macroforecast.forecasting`, not inside individual
preprocessing or feature functions. Full-sample preprocessing remains available
through `reprocess(...)`; origin-local preprocessing uses `preprocess_spec(...)`
inside the runner.

## Stage Policies

`window.stage_policy(...)` is the shared timing grammar for runner stages.

```python
mf.window.stage_policy("origin_available")
mf.window.stage_policy("fit_window")
mf.window.stage_policy(
    "fixed_reference",
    reference_start="2000-01-01",
    reference_end="2019-12-31",
    update="never",
)
```

The same policy object can be supplied as `preprocessing_policy`,
`feature_policy`, or `selection_policy` in `forecasting.run(...)`. This makes
full-sample, expanding, rolling, fixed-reference, and scheduled-refit designs
expressible without putting time logic inside low-level transformation
functions.

## One-Shot Convenience

The direct one-shot helpers remain useful:

```python
processed = mf.preprocessing.reprocess(panel)
features = mf.feature_engineering.build_features(processed, target="INDPRO", horizon=1)
fit = mf.models.ridge(features)
```

They are convenience calls. For strict out-of-sample work, use
`preprocess_spec(...)`, `feature_spec(...)`, and `forecasting.run(...)` so fitted
transforms are learned inside each training window.

## Fit Policies

Public time policy should come from `window`, not from feature functions. Older
one-shot feature helpers may still expose narrow convenience arguments, but new
runner-compatible code should use `feature_spec(...)` and let the runner decide
which rows belong to each fit.
