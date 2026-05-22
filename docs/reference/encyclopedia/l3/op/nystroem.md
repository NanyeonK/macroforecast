# `nystroem` -- Nyström kernel approximation -- subset-based feature map.

[Back to `op` axis](../axes/op.md) | [Back to L3](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L3_A_step_op`, layer `l3`.
> Standalone callable: `mf.functions.nystroem_transform`.

## Function signature

```python
mf.functions.nystroem_transform(
    panel: pd.DataFrame,
    n_components: int,
) -> pd.DataFrame
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `panel` | `pd.DataFrame` | — | — | Input panel. Each column is a variable; rows are time periods. Series is promoted to a single-column DataFrame internally. |
| `n_components` | `int` | `32` | >= 1 | Number of landmark points for Nystroem approximation. Clamped internally to min(n_components, T_clean). |

## Returns

`pd.DataFrame` — scalar result.

## Behavior

sklearn ``Nystroem`` constructs a low-rank approximation of an arbitrary kernel matrix using a random subsample of training points. More accurate than Random Fourier features for non-RBF kernels but with a larger memory footprint.

**When to use**

Non-RBF kernel-augmented linear models (poly / sigmoid).

## In recipe context

Set ``params.op = "nystroem"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L3 recipe fragment
params:
  op: nystroem
```

## References

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'

## Related ops

See also: `kernel_features`, `kernel` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
