# Custom Interpretation And Analysis

[Back to custom extensions](index.md)

Use these hooks after models, features, or forecasts already exist. They do
not refit models, rebuild features, or change predictions.

## custom_interpretation

```python
mf.interpretation.custom_interpretation(
    model,
    X,
    func,
    *,
    y=None,
    name=None,
    **params,
) -> pandas.DataFrame
```

### Callable Signature

```python
func(model, X, *, y=None, metadata=None, **params)
```

Accepted return types are `DataFrame`, `Series`, mapping, or a sequence that
can be converted to a `DataFrame`. The output table receives
`attrs["macroforecast_metadata_schema"]["kind"] == "custom_interpretation"`.

## custom_feature_diagnostic

```python
mf.feature_analysis.custom_feature_diagnostic(
    features,
    func,
    *,
    name=None,
    **params,
) -> pandas.DataFrame
```

### Callable Signature

```python
func(X, *, feature_metadata=None, metadata=None, **params)
```

Use this for feature-matrix checks: missingness by block, custom stability
statistics, custom factor summaries, or project-local data-quality flags.

## custom_forecast_diagnostic

```python
mf.forecast_analysis.custom_forecast_diagnostic(
    forecasts,
    func,
    *,
    name=None,
    **params,
) -> pandas.DataFrame
```

### Callable Signature

```python
func(forecasts, *, metadata=None, **params)
```

Use this for forecast-output checks: horizon bias, model-level summary tables,
origin-level errors, custom stability summaries, or project-local reporting
tables.

## Example

```python
feature_diag = mf.feature_analysis.custom_feature_diagnostic(
    feature_set,
    lambda X, **_: {"n_features": X.shape[1], "missing_cells": int(X.isna().sum().sum())},
    name="shape_check",
)

forecast_diag = mf.forecast_analysis.custom_forecast_diagnostic(
    result,
    lambda forecasts, **_: forecasts.groupby("model", as_index=False)["prediction"].mean(),
    name="mean_prediction_by_model",
)
```

## Output Flow

```python
mf.output.write_artifacts(
    {
        "custom_feature_diagnostic": feature_diag,
        "custom_forecast_diagnostic": forecast_diag,
    },
    "results/custom_diagnostics",
)
```
