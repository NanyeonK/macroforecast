# `pct_change` -- Period-over-period percentage change: ``(y_t / y_{t-1}) - 1``.

[Back to `op` axis](../axes/op.md) | [Back to L3](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L3_A_step_op`, layer `l3`.
> Standalone callable: `mf.functions.pct_change_transform`.

## Function signature

```python
mf.functions.pct_change_transform(
    panel: pd.DataFrame,
    periods: int,
) -> pd.DataFrame
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `panel` | `pd.DataFrame` | — | — | Input panel. Each column is a variable; rows are time periods. Series is promoted to a single-column DataFrame internally. |
| `periods` | `int` | `1` | >= 1 | Number of lag periods for the percentage change. |

## Returns

`pd.DataFrame` — scalar result.

## Behavior

Strict simple growth rate; not equivalent to log_diff for large movements. Returns NaN where the previous observation is zero or NaN.

**When to use**

When a literal percentage interpretation is required (returns, inflation rates).

**When NOT to use**

Trend-following analysis where log_diff's symmetry is preferable.

## In recipe context

Set ``params.op = "pct_change"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L3 recipe fragment
params:
  op: pct_change
```

## References

* macroforecast design Part 2, L3: 'feature engineering is a pipeline of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'

## Related ops

See also: `log_diff`, `diff` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
