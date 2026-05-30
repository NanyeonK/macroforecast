# Evaluation

[Back to reference](index.md)

`macroforecast.evaluation` contains scoring functions and forecast-result
evaluation helpers. Selection uses the point metrics. Forecasting output can
be evaluated directly with `evaluate_forecasts()`.

## Metrics

All metrics accept `y_true` and `y_pred`, align them as pandas Series, drop
missing paired observations, and return a float.

### mse

```python
macroforecast.evaluation.mse(y_true, y_pred)
```

Output: mean squared error.

### rmse

```python
macroforecast.evaluation.rmse(y_true, y_pred)
```

Output: root mean squared error.

### mae

```python
macroforecast.evaluation.mae(y_true, y_pred)
```

Output: mean absolute error.

### pinball_loss

```python
macroforecast.evaluation.pinball_loss(y_true, y_quantile, *, quantile)
```

Output: mean quantile pinball loss. `quantile` must be strictly between 0 and
1.

### gaussian_nll

```python
macroforecast.evaluation.gaussian_nll(y_true, y_pred, variance)
```

Output: Gaussian negative log likelihood using the supplied point forecast and
predictive variance.

### coverage_rate

```python
macroforecast.evaluation.coverage_rate(y_true, lower, upper)
```

Output: share of observations between lower and upper forecasts.

### interval_width

```python
macroforecast.evaluation.interval_width(lower, upper)
```

Output: mean interval width.

## Forecast Table Evaluation

### evaluate_forecasts

```python
macroforecast.evaluation.evaluate_forecasts(
    forecasts,
    *,
    by=("model", "horizon"),
    metrics=("mse", "rmse", "mae"),
    actual="actual",
    prediction="prediction",
    variance_prediction="variance_prediction",
    quantile_predictions="quantile_predictions",
)
```

Input: a `ForecastResult` or a forecast table with at least `actual` and
`prediction`. If present, `variance_prediction` is scored with
`gaussian_nll`. If present, `quantile_predictions` is expected to contain
per-row dictionaries keyed by quantile level and is scored with pinball loss.
Lower/upper quantile pairs such as `0.1` and `0.9` also produce coverage and
interval width.

Output: one DataFrame row per `by` group.

Example:

```python
result = mf.forecasting.run(panel, "ridge", target="y")
scores = mf.evaluation.evaluate_forecasts(result)
scores2 = result.evaluate()
```

### get_metric

```python
macroforecast.evaluation.get_metric(metric)
```

Input:

| Argument | Type | Meaning |
| --- | --- | --- |
| `metric` | str or callable | `"mse"`, `"rmse"`, `"mae"`, legacy validation aliases, or custom callable. |

Output: callable metric.

Supported names:

| Name | Function |
| --- | --- |
| `mse`, `validation_mse` | `mse()` |
| `rmse`, `validation_rmse` | `rmse()` |
| `mae`, `validation_mae` | `mae()` |

Example:

```python
score = mf.evaluation.rmse(y_true, y_pred)

result = mf.selection.select_params(
    "ridge",
    X,
    y,
    search=mf.selection.grid({"alpha": [0.1, 1.0]}),
    window=mf.window.last_block(validation_size=24),
    metric=mf.evaluation.rmse,
)
```
