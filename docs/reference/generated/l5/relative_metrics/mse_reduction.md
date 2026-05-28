# `mse_reduction` -- ``MSE_benchmark - MSE_model`` (absolute MSE reduction) -- positive means the candidate beats the benchmark.

[Back to `relative_metrics` axis](../axes/relative_metrics.md) | [Back to L5](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `relative_metrics`, sub-layer `L5_A_metric_specification`, layer `l5`.
> Standalone callable: `mf.functions.mse_reduction`.

## Function signature

```python
mf.functions.mse_reduction(
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

Relative-loss metric ``mse_reduction``. Absolute difference: ``MSE_benchmark - MSE_model``. A positive value means the model produces lower MSE than the benchmark. Common in macro-forecasting papers (e.g. Stock-Watson 2002 reports MSE reduction in %). Note: some documentation describes this as ``1 - relative_mse`` (ratio form); the computation uses the absolute difference, matching the recipe-path runtime.

**When to use**

Default reporting in horse-race tables when 'positive = better' is preferred.

## In recipe context

Set ``params.relative_metrics = "mse_reduction"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L5 recipe fragment
params:
  relative_metrics: mse_reduction
```

## References

* macroforecast design Part 3, L5: 'evaluation = (metric × benchmark × aggregation × decomposition × ranking).'
* Campbell & Thompson (2008) 'Predicting Excess Stock Returns Out of Sample: Can Anything Beat the Historical Average?', Review of Financial Studies 21(4): 1509-1531. (doi:10.1093/rfs/hhm055)

## Related ops

See also: `relative_mse`, `relative_mae`, `r2_oos` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
