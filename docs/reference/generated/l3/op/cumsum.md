# `cumsum` -- Cumulative sum of a series.

[Back to `op` axis](../axes/op.md) | [Back to L3](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L3_A_step_op`, layer `l3`.
> Standalone callable: `mf.functions.cumsum_transform`.

## Function signature

```python
mf.functions.cumsum_transform(
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

Running total ``Σ_{s ≤ t} y_s``. Inverts ``diff`` (modulo an initial constant). Used to recover level forecasts from differenced predictions or to build cumulative-shock features.

**When to use**

Building cumulative-impact features; recovering levels from differenced forecasts.

## In recipe context

Set ``params.op = "cumsum"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L3 recipe fragment
params:
  op: cumsum
```

## References

* macroforecast design Part 2, L3: 'feature engineering is a pipeline of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'

## Related ops

See also: `diff`, `level` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
