# `medae` -- Median absolute error -- ``median |y_t - ŷ_t|``.

[Back to `point_metrics` axis](../axes/point_metrics.md) | [Back to L5](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `point_metrics`, sub-layer `L5_A_metric_specification`, layer `l5`.
> Standalone callable: `mf.functions.medae`.

## Function signature

```python
mf.functions.medae(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> float
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `y_true` | `np.ndarray | pd.Series` | — | — | Actual (realised) values. 1-D float array of length N. |
| `y_pred` | `np.ndarray | pd.Series` | — | — | Forecast values. Must be the same length as y_true. |

## Returns

`float` — scalar result.

## Behavior

Point-forecast metric ``medae``. Maximally robust point-forecast metric: substitution by median completely insulates the score from a constant-share of extreme residuals. Common in robust-statistics papers; rarer in mainstream forecasting.

**When to use**

Pathologically heavy-tailed errors (financial crises, regime shifts).

**When NOT to use**

Standard reporting -- mean-based metrics are the convention.

## In recipe context

Set ``params.point_metrics = "medae"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L5 recipe fragment
params:
  point_metrics: medae
```

## References

* macroforecast design Part 3, L5: 'evaluation = (metric × benchmark × aggregation × decomposition × ranking).'
* Diebold (2017) 'Forecasting in Economics, Business, Finance and Beyond', University of Pennsylvania (free online). <https://www.sas.upenn.edu/~fdiebold/Textbooks.html>

## Related ops

See also: `mse`, `rmse`, `mae`, `mape`, `theil_u1`, `theil_u2` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
