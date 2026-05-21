# `theil_u2` -- Theil's U2 inequality coefficient -- ratio of forecast MSE to no-change MSE.

[Back to `point_metrics` axis](../axes/point_metrics.md) | [Back to L5](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `point_metrics`, sub-layer `L5_A_metric_specification`, layer `l5`.
> Standalone callable: `mf.functions.theil_u2`.

## Function signature

```python
mf.functions.theil_u2(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_prev: np.ndarray | pd.Series,
) -> float
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `y_true` | `np.ndarray | pd.Series` | — | — | Actual values at time t. 1-D array of length N. |
| `y_pred` | `np.ndarray | pd.Series` | — | — | Forecast values at time t. Same length as y_true. |
| `y_prev` | `np.ndarray | pd.Series` | — | — | Actual values at time t-1 (random-walk baseline). Same length as y_true. Pass np.nan for missing rows. |

## Returns

`float` — scalar result.

## Behavior

Point-forecast metric ``theil_u2``. ``U₂ = √(Σ (ŷ_t - y_t)² / Σ (y_{t-1} - y_t)²)``. ``U₂ < 1`` means the forecast beats the random-walk benchmark. Standard sanity-check ratio in macro forecasting -- if ``U₂ ≥ 1`` the model is no better than 'tomorrow looks like today'.

**When to use**

Sanity-checking against the random-walk benchmark; macro-forecasting tradition.

**When NOT to use**

When a custom benchmark (not random walk) is preferred -- use ``relative_mse`` instead.

## In recipe context

Set ``params.point_metrics = "theil_u2"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L5 recipe fragment
params:
  point_metrics: theil_u2
```

## References

* macroforecast design Part 3, L5: 'evaluation = (metric × benchmark × aggregation × decomposition × ranking).'
* Theil (1966) 'Applied Economic Forecasting', North-Holland (Chapter 2: Inequality coefficients).

## Related ops

See also: `mse`, `rmse`, `mae`, `medae`, `mape`, `theil_u1` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
