# `holiday` -- Holiday / event dummy variables.

[Back to `op` axis](../axes/op.md) | [Back to L3](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L3_A_step_op`, layer `l3`.
> Standalone callable: `mf.functions.holiday_transform`.

## Function signature

```python
mf.functions.holiday_transform(
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

0/1 indicators for calendar holidays (US federal by default; ``params.country`` selects locale via the ``holidays`` package). For business / financial macro series.

**When to use**

Daily / weekly business-cycle series where holidays create discrete level shifts.

**When NOT to use**

Pure macro series at monthly+ frequency where holidays are absorbed by ``season_dummy``.

## In recipe context

Set ``params.op = "holiday"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L3 recipe fragment
params:
  op: holiday
```

## References

* macroforecast design Part 2, L3: 'feature engineering is a pipeline of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'

## Related ops

See also: `season_dummy` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
