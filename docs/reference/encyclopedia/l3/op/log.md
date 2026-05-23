# `log` -- Natural logarithm: ``ln(y)``.

[Back to `op` axis](../axes/op.md) | [Back to L3](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L3_A_step_op`, layer `l3`.
> Standalone callable: `mf.functions.log_transform`.

## Function signature

```python
mf.functions.log_transform(
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

Element-wise natural log. Strictly positive series only; raises if any input is non-positive. Often paired with ``diff`` to produce log-changes (which are approximately equal to percentage changes for small movements).

**When to use**

Strictly-positive macro series (price levels, employment counts, GDP) before differencing.

**When NOT to use**

Series that can be negative or zero (rates, growth rates, balances).

## In recipe context

Set ``params.op = "log"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L3 recipe fragment
params:
  op: log
```

## References

* macroforecast design Part 2, L3: 'feature engineering is a pipeline of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'

## Related ops

See also: `log_diff`, `level`, `pct_change` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
