# `diff` -- First difference: ``y_t - y_{t-1}``.

[Back to `op` axis](../axes/op.md) | [Back to L3](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L3_A_step_op`, layer `l3`.
> Standalone callable: `mf.functions.diff_transform`.

## Function signature

```python
mf.functions.diff_transform(
    panel: pd.DataFrame,
    periods: int,
) -> pd.DataFrame
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `panel` | `pd.DataFrame` | — | — | Input panel. Each column is a variable; rows are time periods. Series is promoted to a single-column DataFrame internally. |
| `periods` | `int` | `1` | >= 1 | Number of lag periods to difference. |

## Returns

`pd.DataFrame` — scalar result.

## Behavior

Computes the simple first difference on the input column. The first observation becomes NaN. Combine with ``lag`` to recover level features when the L2 layer already differenced the panel.

**When to use**

I(1) variables that need a stationary representation in addition to the L2-applied tcode.

**When NOT to use**

When the panel is already differenced by L2.B (avoids double-differencing).

## In recipe context

Set ``params.op = "diff"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L3 recipe fragment
params:
  op: diff
```

## References

* macroforecast design Part 2, L3: 'feature engineering is a pipeline of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'

## Related ops

See also: `level`, `log_diff`, `pct_change` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
