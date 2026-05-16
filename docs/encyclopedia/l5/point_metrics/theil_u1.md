# `theil_u1` -- Theil's U1 inequality coefficient -- bounded in ``[0, 1]``.

[Back to `point_metrics` axis](../axes/point_metrics.md) | [Back to L5](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `point_metrics`, sub-layer `L5_A_metric_specification`, layer `l5`.
> Standalone callable: `mf.functions.theil_u1`.

## Function signature

```python
mf.functions.theil_u1(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> float
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `y_true` | `np.ndarray | pd.Series` | — | — | Actual (realised) values. 1-D array of length N. |
| `y_pred` | `np.ndarray | pd.Series` | — | — | Forecast values. Must be the same length as y_true. |

## Returns

`float` — scalar result.

## Behavior

Point-forecast metric ``theil_u1``. ``U₁ = √MSE / (√(1/N Σ y²) + √(1/N Σ ŷ²))``. Bounded between 0 (perfect forecast) and 1 (worst possible). Theil's original 1966 metric; less commonly used today than U2 because the denominator's interpretation is less intuitive.

**When to use**

Long-run macro forecasting tradition; comparability with Theil-1966-era papers.

**When NOT to use**

Modern reporting -- U2 is more interpretable as a ratio against the no-change benchmark.

## In recipe context

Set ``params.point_metrics = "theil_u1"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L5 recipe fragment
params:
  point_metrics: theil_u1
```

## References

* macroforecast design Part 3, L5: 'evaluation = (metric × benchmark × aggregation × decomposition × ranking).'
* Theil (1966) 'Applied Economic Forecasting', North-Holland (Chapter 2: Inequality coefficients).

## Related ops

See also: `mse`, `rmse`, `mae`, `medae`, `mape`, `theil_u2` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
