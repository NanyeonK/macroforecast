# `nystroem_features` -- Alias for ``nystroem`` -- explicit feature-stage name.

[Back to `op` axis](../axes/op.md) | [Back to L3](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L3_A_step_op`, layer `l3`.
> Standalone callable: `mf.functions.nystroem_transform`.

## Function signature

```python
mf.functions.nystroem_transform(
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

Identical to ``nystroem``; preferred when a multi-stage pipeline names its kernel approximation explicitly in the lineage graph.

**When to use**

Multi-stage pipelines that separate kernel approximation from downstream linear fits.

## In recipe context

Set ``params.op = "nystroem_features"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L3 recipe fragment
params:
  op: nystroem_features
```

## References

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'

## Related ops

See also: `nystroem` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
