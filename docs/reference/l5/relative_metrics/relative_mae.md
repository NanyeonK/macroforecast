# `relative_mae` -- Forecast MAE divided by the L4 ``is_benchmark`` model's MAE.

[Back to `relative_metrics` axis](../axes/relative_metrics.md) | [Back to L5](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `relative_metrics`, sub-layer `L5_A_metric_specification`, layer `l5`.
> Standalone callable: `mf.functions.relative_mae`.

## Function signature

```python
mf.functions.relative_mae(
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

Relative-loss metric ``relative_mae``. L1-loss analogue of ``relative_mse``. Below 1 means the candidate beats the benchmark on absolute-loss criterion. Robust to heavy-tailed forecast errors.

**When to use**

Heavy-tailed targets where MSE is too sensitive to outliers.

## In recipe context

Set ``params.relative_metrics = "relative_mae"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L5 recipe fragment
params:
  relative_metrics: relative_mae
```

## References

* macroforecast design Part 3, L5: 'evaluation = (metric × benchmark × aggregation × decomposition × ranking).'
* Diebold (2017) 'Forecasting in Economics, Business, Finance and Beyond', University of Pennsylvania (free online). <https://www.sas.upenn.edu/~fdiebold/Textbooks.html>

## Related ops

See also: `relative_mse`, `mse_reduction`, `r2_oos` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
