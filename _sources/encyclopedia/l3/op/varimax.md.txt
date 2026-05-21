# `varimax` -- Varimax-rotated factors (orthogonal rotation for interpretability).

[Back to `op` axis](../axes/op.md) | [Back to L3](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L3_A_step_op`, layer `l3`.
> Standalone callable: `mf.functions.varimax_transform`.

## Function signature

```python
mf.functions.varimax_transform(
    panel: pd.DataFrame,
) -> pd.DataFrame
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `panel` | `pd.DataFrame` | — | — | Input panel. Each column is a variable; rows are time periods. Series is promoted to a single-column DataFrame internally. |

## Returns

`pd.DataFrame` — scalar result.

## Behavior

Applies a varimax rotation to PCA loadings, maximising the variance of squared loadings within each factor. Produces factors that load heavily on a small subset of original predictors -- useful for naming / labelling factors.

**When to use**

Factor analysis where downstream interpretation requires distinct, well-named factors.

## In recipe context

Set ``params.op = "varimax"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L3 recipe fragment
params:
  op: varimax
```

## References

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'

## Related ops

See also: `pca`, `sparse_pca` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
