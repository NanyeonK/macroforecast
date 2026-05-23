# `polynomial_expansion` -- Alias for ``polynomial`` -- explicit expansion node in cascade pipelines.

[Back to `op` axis](../axes/op.md) | [Back to L3](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L3_A_step_op`, layer `l3`.
> Standalone callable: `mf.functions.polynomial_expansion_transform`.

## Function signature

```python
mf.functions.polynomial_expansion_transform(
    panel: pd.DataFrame,
    degree: int,
) -> pd.DataFrame
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `panel` | `pd.DataFrame` | — | — | Input panel. Each column is a variable; rows are time periods. Series is promoted to a single-column DataFrame internally. |
| `degree` | `int` | `2` | >= 1 | Maximum polynomial degree. Degree 1 returns the panel unchanged; degree 2 appends _pow2 columns; etc. |

## Returns

`pd.DataFrame` — scalar result.

## Behavior

Identical to ``polynomial`` but with a name that reads more clearly as a stage in a multi-step expansion pipeline.

**When to use**

Pipelines that explicitly stage `expand → reduce` sequences.

## In recipe context

Set ``params.op = "polynomial_expansion"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L3 recipe fragment
params:
  op: polynomial_expansion
```

## References

* macroforecast design Part 2, L3: 'feature engineering is a pipeline of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'

## Related ops

See also: `polynomial` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
