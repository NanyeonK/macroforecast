# `mape` -- Mean absolute percentage error -- ``(100/N) Σ |y_t - ŷ_t| / |y_t|``.

[Back to `point_metrics` axis](../axes/point_metrics.md) | [Back to L5](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `point_metrics`, sub-layer `L5_A_metric_specification`, layer `l5`.
> Standalone callable: `mf.functions.mape`.

## Function signature

```python
mf.functions.mape(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    *,
    eps: float = 1e-10,
) -> float
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `y_true` | `np.ndarray | pd.Series` | — | — | Actual (realised) values. 1-D float array of length N. |
| `y_pred` | `np.ndarray | pd.Series` | — | — | Forecast values. Must be the same length as y_true. |
| `eps` | `float` | `1e-10` | — | Small positive value added to |y_true| to avoid division by zero when targets are near zero. |

## Returns

`float` — scalar result.

## Behavior

Point-forecast metric ``mape``. Scale-free percentage version of MAE. Allows comparing forecasts for targets on different scales (US GDP vs Korean GDP). Pathological when targets can be zero or near-zero -- the metric blows up. Hyndman & Koehler (2006) recommend MASE / sMAPE in those cases.

**When to use**

Cross-target / cross-country comparisons; reporting forecast accuracy in percentage terms.

**When NOT to use**

Targets that can be near zero (rates, growth rates) -- division by tiny ``|y_t|`` makes the metric explode.

## In recipe context

Set ``params.point_metrics = "mape"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L5 recipe fragment
params:
  point_metrics: mape
```

## References

* macroforecast design Part 3, L5: 'evaluation = (metric × benchmark × aggregation × decomposition × ranking).'
* Diebold (2017) 'Forecasting in Economics, Business, Finance and Beyond', University of Pennsylvania (free online). <https://www.sas.upenn.edu/~fdiebold/Textbooks.html>
* Hyndman & Koehler (2006) 'Another look at measures of forecast accuracy', International Journal of Forecasting 22(4): 679-688. (doi:10.1016/j.ijforecast.2006.03.001)

## Related ops

See also: `mse`, `rmse`, `mae`, `medae`, `theil_u1`, `theil_u2` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
