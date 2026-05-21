# `success_ratio` -- Hit-rate of correct directional forecasts -- ``(1/N) Σ 1{sign(ŷ_t) = sign(y_t)}``.

[Back to `direction_metrics` axis](../axes/direction_metrics.md) | [Back to L5](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `direction_metrics`, sub-layer `L5_A_metric_specification`, layer `l5`.
> Standalone callable: `mf.functions.success_ratio`.

## Function signature

```python
mf.functions.success_ratio(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_prev: np.ndarray | pd.Series,
) -> float
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `y_true` | `np.ndarray | pd.Series` | — | — | Actual (realised) values. 1-D float array of length N. |
| `y_pred` | `np.ndarray | pd.Series` | — | — | Forecast values. Same length as y_true. |
| `y_prev` | `np.ndarray | pd.Series` | — | — | Lagged actual values (y at t-1). Same length as y_true. Pass np.nan for rows where the previous value is unavailable. |

## Returns

`float` — scalar result.

## Behavior

Directional-accuracy metric ``success_ratio``. Naive directional accuracy, bounded in ``[0, 1]``. Does not adjust for the unconditional direction frequency, so a constant 'always positive' forecast can score 0.7 on a growth target. For statistical significance, pair with ``pesaran_timmermann_metric`` and the L6.F PT test.

**When to use**

Quick directional-accuracy reporting; reporting the raw hit-rate alongside the PT statistic.

**When NOT to use**

Standalone significance testing -- needs PT correction for unconditional direction frequency.

## In recipe context

Set ``params.direction_metrics = "success_ratio"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L5 recipe fragment
params:
  direction_metrics: success_ratio
```

## References

* macroforecast design Part 3, L5: 'evaluation = (metric × benchmark × aggregation × decomposition × ranking).'
* Pesaran & Timmermann (1992) 'A simple nonparametric test of predictive performance', JBES 10(4): 461-465.

## Related ops

See also: `pesaran_timmermann_metric` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
