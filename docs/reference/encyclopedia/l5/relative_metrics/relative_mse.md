# `relative_mse` -- Forecast MSE divided by the L4 ``is_benchmark`` model's MSE.

[Back to `relative_metrics` axis](../axes/relative_metrics.md) | [Back to L5](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `relative_metrics`, sub-layer `L5_A_metric_specification`, layer `l5`.
> Standalone callable: `mf.functions.relative_mse`.

## Function signature

```python
mf.functions.relative_mse(
    y_true: np.ndarray | pd.Series,
    y_model: np.ndarray | pd.Series,
    y_benchmark: np.ndarray | pd.Series,
) -> float
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `y_true` | `np.ndarray | pd.Series` | — | — | Actual (realised) values. 1-D float array of length N. |
| `y_model` | `np.ndarray | pd.Series` | — | — | Candidate model forecast values. Same length as y_true. |
| `y_benchmark` | `np.ndarray | pd.Series` | — | — | Benchmark model forecast values. Same length as y_true. |

## Returns

`float` — scalar result.

## Behavior

Relative-loss metric ``relative_mse``. ``MSE_model / MSE_benchmark``. The standard horse-race ratio. Below 1 means the candidate beats the benchmark; the L5.E ranking tables surface this column by default. Requires exactly one L4 model with ``is_benchmark = true`` (validator hard-rejects 0 or > 1 benchmarks).

**When to use**

Default reporting metric in horse-race tables; comparing candidate models against a fixed benchmark.

## In recipe context

Set ``params.relative_metrics = "relative_mse"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L5 recipe fragment
params:
  relative_metrics: relative_mse
```

## References

* macroforecast design Part 3, L5: 'evaluation = (metric × benchmark × aggregation × decomposition × ranking).'
* Diebold (2017) 'Forecasting in Economics, Business, Finance and Beyond', University of Pennsylvania (free online). <https://www.sas.upenn.edu/~fdiebold/Textbooks.html>

## Related ops

See also: `relative_mae`, `mse_reduction`, `r2_oos` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
