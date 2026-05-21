# `interval_score` -- Winkler (1972) interval score -- jointly penalises miscoverage + interval width.

[Back to `density_metrics` axis](../axes/density_metrics.md) | [Back to L5](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `density_metrics`, sub-layer `L5_A_metric_specification`, layer `l5`.
> Standalone callable: `mf.functions.interval_score`.

## Function signature

```python
mf.functions.interval_score(
    y_true: np.ndarray | pd.Series,
    y_lower: np.ndarray | pd.Series,
    y_upper: np.ndarray | pd.Series,
    *,
    alpha: float = 0.05,
) -> float
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `y_true` | `np.ndarray | pd.Series` | — | — | Actual (realised) values. 1-D float array of length N. |
| `y_lower` | `np.ndarray | pd.Series` | — | — | Lower bound of the prediction interval. Same length as y_true. |
| `y_upper` | `np.ndarray | pd.Series` | — | — | Upper bound of the prediction interval. Same length as y_true. |
| `alpha` | `float` | `0.05` | — | Miscoverage level (1 - confidence level). E.g. alpha=0.05 for 95% prediction intervals. |

## Returns

`float` — scalar result.

## Behavior

Density-forecast metric ``interval_score``. For a nominal-α interval ``[L, U]``: ``IS_α = (U - L) + (2/α)(L - y) 1{y < L} + (2/α)(y - U) 1{y > U}``. Lower = better. Strictly-proper for the α-level prediction interval; the natural metric when L4 emits ``forecast_object = interval``.

**When to use**

Prediction-interval evaluation; balancing tightness against coverage.

## In recipe context

Set ``params.density_metrics = "interval_score"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L5 recipe fragment
params:
  density_metrics: interval_score
```

## References

* macroforecast design Part 3, L5: 'evaluation = (metric × benchmark × aggregation × decomposition × ranking).'
* Gneiting & Raftery (2007) 'Strictly Proper Scoring Rules, Prediction, and Estimation', JASA 102(477): 359-378. (doi:10.1198/016214506000001437)
* Gneiting & Katzfuss (2014) 'Probabilistic Forecasting', Annual Review of Statistics and Its Application 1: 125-151.
* Winkler (1972) 'A Decision-Theoretic Approach to Interval Estimation', JASA 67(337): 187-191.

## Related ops

See also: `log_score`, `crps`, `coverage_rate` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
