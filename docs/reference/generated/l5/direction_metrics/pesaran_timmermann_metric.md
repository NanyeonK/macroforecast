# `pesaran_timmermann_metric` -- Pesaran-Timmermann (1992) directional-accuracy statistic.

[Back to `direction_metrics` axis](../axes/direction_metrics.md) | [Back to L5](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `direction_metrics`, sub-layer `L5_A_metric_specification`, layer `l5`.
> Standalone callable: `mf.functions.pesaran_timmermann_metric`.

## Function signature

```python
mf.functions.pesaran_timmermann_metric(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    *,
    threshold: float = 0.0,
) -> float
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `y_true` | `np.ndarray | pd.Series` | — | — | Actual (realised) values. 1-D float array of length N. |
| `y_pred` | `np.ndarray | pd.Series` | — | — | Forecast values. Same length as y_true. |
| `threshold` | `float` | `0.0` | — | Threshold for computing directional binary series. A forecast above threshold = directional 'up'. |

## Returns

`float` — scalar result.

## Behavior

Directional-accuracy metric ``pesaran_timmermann_metric``. Adjusts the success ratio for the joint probability of agreement under independence (so a constant-sign forecast no longer scores high). Asymptotically standard normal under the null of no directional skill; the L6.F test computes the corresponding p-value.

**When to use**

Formal directional-accuracy reporting (paired with the L6 PT test).

## In recipe context

Set ``params.direction_metrics = "pesaran_timmermann_metric"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L5 recipe fragment
params:
  direction_metrics: pesaran_timmermann_metric
```

## References

* macroforecast design Part 3, L5: 'evaluation = (metric × benchmark × aggregation × decomposition × ranking).'
* Pesaran & Timmermann (1992) 'A simple nonparametric test of predictive performance', JBES 10(4): 461-465.

## Related ops

See also: `success_ratio` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
