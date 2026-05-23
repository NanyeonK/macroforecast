# `scale` -- Standardise to zero mean and unit variance.

[Back to `op` axis](../axes/op.md) | [Back to L3](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L3_A_step_op`, layer `l3`.
> Standalone callable: `mf.functions.scale_transform`.

## Function signature

```python
mf.functions.scale_transform(
    panel: pd.DataFrame,
    method: str enum {"zscore", "standard", "standardize", "robust", "minmax"},
) -> pd.DataFrame
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `panel` | `pd.DataFrame` | — | — | Input panel. Each column is a variable; rows are time periods. Series is promoted to a single-column DataFrame internally. |
| `method` | `str enum {"zscore", "standard", "standardize", "robust", "minmax"}` | `'zscore'` | — | Scaling method. "zscore"/"standard"/"standardize" for zero-mean/unit-std; "robust" for median/IQR; "minmax" for [0, 1] range. |

## Returns

`pd.DataFrame` — scalar result.

## Behavior

Computes ``(y - μ) / σ`` over the temporal_rule window (``expanding_window_per_origin`` by default to avoid look-ahead). Required pre-step for distance-based learners (kNN, SVM, NN); ridge/lasso also benefit when columns are on different scales.

**When to use**

Pre-conditioning for distance-based or regularised learners; mandatory for SVM/NN.

**When NOT to use**

Tree-based models (RF/XGBoost/LightGBM) -- scale-invariant by construction.

## In recipe context

Set ``params.op = "scale"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L3 recipe fragment
params:
  op: scale
```

## References

* macroforecast design Part 2, L3: 'feature engineering is a pipeline of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'

## Related ops

See also: `pca`, `kernel_features` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
