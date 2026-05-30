# Evaluation

[Back to reference](index.md)

`macroforecast.evaluation` contains scoring functions. Selection uses these
metrics, and later forecast-evaluation code will build on the same namespace.

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
