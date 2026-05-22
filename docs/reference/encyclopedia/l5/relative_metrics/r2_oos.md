# `r2_oos` -- Out-of-sample R² (Campbell-Thompson 2008) -- ``1 - SSE_model / SSE_benchmark``.

[Back to `relative_metrics` axis](../axes/relative_metrics.md) | [Back to L5](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `relative_metrics`, sub-layer `L5_A_metric_specification`, layer `l5`.
> Standalone callable: `mf.functions.r2_oos`.

## Function signature

```python
mf.functions.r2_oos(
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

Relative-loss metric ``r2_oos``. Standard return-predictability metric in finance (and increasingly in macro). Identical formula to ``mse_reduction`` when the benchmark is the historical mean. Campbell & Thompson (2008) popularised the metric for the empirical-asset-pricing literature.

**When to use**

Macro / financial forecasting tradition; literature-compatibility with CT-2008-era papers.

## In recipe context

Set ``params.relative_metrics = "r2_oos"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L5 recipe fragment
params:
  relative_metrics: r2_oos
```

## References

* macroforecast design Part 3, L5: 'evaluation = (metric × benchmark × aggregation × decomposition × ranking).'
* Campbell & Thompson (2008) 'Predicting Excess Stock Returns Out of Sample: Can Anything Beat the Historical Average?', Review of Financial Studies 21(4): 1509-1531. (doi:10.1093/rfs/hhm055)

## Related ops

See also: `relative_mse`, `relative_mae`, `mse_reduction` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
