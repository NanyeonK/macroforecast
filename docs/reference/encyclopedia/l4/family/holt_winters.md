# `holt_winters` -- Holt-Winters additive / multiplicative seasonal exponential smoothing.

[Back to `family` axis](../axes/family.md) | [Back to L4](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `family`, sub-layer `L4_A_model_selection`, layer `l4`.
> Standalone callable: `mf.functions.holt_winters_fit`.

## Function signature

```python
mf.functions.holt_winters_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> HoltWintersFitResult
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `X` | `np.ndarray | pd.DataFrame` | — | — | Feature matrix. Shape (n_samples, n_features). Accepts numpy arrays or DataFrames. |
| `y` | `np.ndarray | pd.Series` | — | — | Target vector. Shape (n_samples,). Accepts numpy arrays or Series. |

## Returns

`HoltWintersFitResult` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `.seasonal` | `str` | Seasonal component type (add or mul). |
| `.seasonal_periods` | `int` | Number of periods per season. |
| `.n_obs` | `int` | Number of observations. |
| `.predict(X)` | `np.ndarray` | Forecast, len(X) steps ahead. |
| `.summary()` | `str` | Table: seasonal type, periods, observation count. |

## Behavior

Wraps ``statsmodels.tsa.holtwinters.ExponentialSmoothing``. Fits level / trend / seasonal smoothing parameters by MLE (``optimized=True``). Supports additive and multiplicative trend and seasonal components plus an optional damped trend (Hyndman et al. 2008 §3).

**Defaults**: ``seasonal = 'add'``, ``seasonal_periods = 12``, ``trend = 'add'``, ``damped_trend = False``. Auto-disables seasonal fitting when ``len(y) < 2 · seasonal_periods``.

**When to use**

Seasonal univariate baselines; M-competition style benchmarking; standard reference forecast for monthly / quarterly macro series.

**When NOT to use**

Without a clear seasonal pattern (use ``ets`` AAN instead); covariate-driven forecasting.

## In recipe context

Set ``params.family = "holt_winters"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L4 recipe fragment
params:
  family: holt_winters
```

## References

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Holt (2004 / orig. 1957) 'Forecasting seasonals and trends by exponentially weighted moving averages', International Journal of Forecasting 20(1): 5-10.
* Winters (1960) 'Forecasting Sales by Exponentially Weighted Moving Averages', Management Science 6(3): 324-342.
* Hyndman & Athanasopoulos (2018) 'Forecasting: Principles and Practice', 2nd ed., OTexts §7.

## Related ops

See also: `ets`, `theta_method`, `ar_p` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
