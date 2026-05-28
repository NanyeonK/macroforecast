# `ma_window` -- Trailing moving average over a fixed window.

[Back to `op` axis](../axes/op.md) | [Back to L3](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L3_A_step_op`, layer `l3`.
> Standalone callable: `mf.functions.ma_window_transform`.

## Function signature

```python
mf.functions.ma_window_transform(
    panel: pd.DataFrame,
    window: int,
) -> pd.DataFrame
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `panel` | `pd.DataFrame` | — | — | Input panel. Each column is a variable; rows are time periods. Series is promoted to a single-column DataFrame internally. |
| `window` | `int` | `3` | >= 1 | Rolling window size in periods. First window-1 rows are NaN. |

## Returns

`pd.DataFrame` — scalar result.

## Behavior

Computes ``mean(y_{t-w+1..t})`` for a user-specified window ``params.window``. ``temporal_rule`` controls expanding vs rolling vs block-wise refit semantics. The first ``w-1`` rows are NaN.

**When to use**

Smoothing noisy series; building short / medium / long-term momentum features.

## In recipe context

Set ``params.op = "ma_window"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L3 recipe fragment
params:
  op: ma_window
```

## References

* macroforecast design Part 2, L3: 'feature engineering is a pipeline of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'

## Related ops

See also: `ma_increasing_order`, `diff`, `scale` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
