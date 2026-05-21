# `random_projection` -- Johnson-Lindenstrauss random Gaussian projection.

[Back to `op` axis](../axes/op.md) | [Back to L3](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L3_A_step_op`, layer `l3`.
> Standalone callable: `mf.functions.random_projection_transform`.

## Function signature

```python
mf.functions.random_projection_transform(
    panel: pd.DataFrame,
    n_components: int,
) -> pd.DataFrame
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `panel` | `pd.DataFrame` | — | — | Input panel. Each column is a variable; rows are time periods. Series is promoted to a single-column DataFrame internally. |
| `n_components` | `int` | `8` | >= 1 | Number of random projection output dimensions. Clamped internally to min(n_components, K). |

## Returns

`pd.DataFrame` — scalar result.

## Behavior

Reduces dimensionality by multiplying with a random Gaussian matrix scaled to (approximately) preserve pairwise distances. Cheap baseline for dimensionality reduction; sklearn's ``GaussianRandomProjection``.

**When to use**

Sweep baselines / sanity checks against PCA's structured reduction.

## In recipe context

Set ``params.op = "random_projection"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L3 recipe fragment
params:
  op: random_projection
```

## References

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'

## Related ops

See also: `pca`, `kernel_features` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
