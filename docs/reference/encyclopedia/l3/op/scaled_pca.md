# `scaled_pca` -- Scaled / weighted PCA (target-aware factor extraction).

[Back to `op` axis](../axes/op.md) | [Back to L3](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L3_A_step_op`, layer `l3`.
> Standalone callable: `mf.functions.scaled_pca_transform`.

## Function signature

```python
mf.functions.scaled_pca_transform(
    panel: pd.DataFrame,
    target: pd.Series,
    n_components: int,
) -> pd.DataFrame
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `panel` | `pd.DataFrame` | — | — | Input panel. Each column is a variable; rows are time periods. Series is promoted to a single-column DataFrame internally. |
| `target` | `pd.Series` | — | — | Supervisory signal aligned to the panel index. Must share at least one index value with panel; raises ValueError if the intersection is empty. |
| `n_components` | `int` | `3` | >= 1 | Number of principal components to extract. Clamped internally to min(T_clean, K) - 1. |

## Returns

`pd.DataFrame` — scalar result.

## Behavior

Weights each column by its predictive correlation with the target before performing PCA. Implements the Huang-Jiang-Tu-Zhou (2022) scaled PCA for forecasting macro variables.

Reduces to plain PCA when all weights are equal.

**When to use**

When standard PCA's leading factor is dominated by predictively-irrelevant variance.

## In recipe context

Set ``params.op = "scaled_pca"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L3 recipe fragment
params:
  op: scaled_pca
```

## References

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'
* Huang, Jiang, Tu & Zhou (2022) 'Scaled PCA: A New Approach to Dimension Reduction', Management Science 68(3): 1678-1695.

## Related ops

See also: `pca`, `partial_least_squares` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
