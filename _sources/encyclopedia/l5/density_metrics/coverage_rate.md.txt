# `coverage_rate` -- Empirical coverage rate -- share of OOS observations falling within the nominal-α interval.

[Back to `density_metrics` axis](../axes/density_metrics.md) | [Back to L5](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `density_metrics`, sub-layer `L5_A_metric_specification`, layer `l5`.
> Standalone callable: `mf.functions.coverage_rate`.

## Function signature

```python
mf.functions.coverage_rate(
    y_true: np.ndarray | pd.Series,
    y_lower: np.ndarray | pd.Series,
    y_upper: np.ndarray | pd.Series,
) -> float
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `y_true` | `np.ndarray | pd.Series` | — | — | Actual (realised) values. 1-D float array of length N. |
| `y_lower` | `np.ndarray | pd.Series` | — | — | Lower bound of the prediction interval. Same length as y_true. |
| `y_upper` | `np.ndarray | pd.Series` | — | — | Upper bound of the prediction interval. Same length as y_true. |

## Returns

`float` — scalar result.

## Behavior

Density-forecast metric ``coverage_rate``. Should equal α (1 - α miscoverage) if the model is well-calibrated. Deviations indicate miscalibration: low coverage = intervals too narrow; high coverage = intervals too wide. Pair with ``interval_score`` to capture both calibration and sharpness.

**When to use**

Interval-calibration audits; reporting alongside interval_score.

## In recipe context

Set ``params.density_metrics = "coverage_rate"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L5 recipe fragment
params:
  density_metrics: coverage_rate
```

## References

* macroforecast design Part 3, L5: 'evaluation = (metric × benchmark × aggregation × decomposition × ranking).'
* Gneiting & Raftery (2007) 'Strictly Proper Scoring Rules, Prediction, and Estimation', JASA 102(477): 359-378. (doi:10.1198/016214506000001437)
* Gneiting & Katzfuss (2014) 'Probabilistic Forecasting', Annual Review of Statistics and Its Application 1: 125-151.

## Related ops

See also: `log_score`, `crps`, `interval_score` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
