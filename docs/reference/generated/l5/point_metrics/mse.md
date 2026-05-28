# `mse` -- Mean squared error -- ``(1/N) Σ (y_t - ŷ_t)²``.

[Back to `point_metrics` axis](../axes/point_metrics.md) | [Back to L5](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `point_metrics`, sub-layer `L5_A_metric_specification`, layer `l5`.
> Standalone callable: `mf.functions.mse`.

## Function signature

```python
mf.functions.mse(
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

Point-forecast metric ``mse``. The classical quadratic-loss metric. Optimal under Gaussian-residual / squared-loss decision theory; the L4 fit objective for OLS / ridge / elastic net is its in-sample version. MSE penalises large residuals super-linearly, so a single outlier in the OOS sample can dominate the score.

**When to use**

Default for Gaussian-residual problems; horse-race ranking under squared-loss decision rules.

**When NOT to use**

Heavy-tailed forecast errors -- a single outlier dominates the score; consider MAE or MedAE instead.

## In recipe context

Set ``params.point_metrics = "mse"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L5 recipe fragment
params:
  point_metrics: mse
```

## References

* macroforecast design Part 3, L5: 'evaluation = (metric × benchmark × aggregation × decomposition × ranking).'
* Diebold (2017) 'Forecasting in Economics, Business, Finance and Beyond', University of Pennsylvania (free online). <https://www.sas.upenn.edu/~fdiebold/Textbooks.html>

## Related ops

See also: `rmse`, `mae`, `medae`, `mape`, `theil_u1`, `theil_u2` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
