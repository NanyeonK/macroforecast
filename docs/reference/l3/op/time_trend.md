# `time_trend` -- Deterministic linear time trend (``t = 1, 2, ...``).

[Back to `op` axis](../axes/op.md) | [Back to L3](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L3_A_step_op`, layer `l3`.
> Standalone callable: `mf.functions.time_trend_transform`.

## Function signature

```python
mf.functions.time_trend_transform(
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

Adds a column ``time_trend`` to the panel; with ``params.degree > 1`` appends polynomial trends. Deterministic complement to stochastic detrending (HP / Hamilton).

**When to use**

Trend-stationary linear models where a deterministic trend is part of the DGP.

**When NOT to use**

Series with structural breaks -- use ``regime_indicator`` or stochastic detrending instead.

## In recipe context

Set ``params.op = "time_trend"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L3 recipe fragment
params:
  op: time_trend
```

## References

* macroforecast design Part 2, L3: 'feature engineering is a pipeline of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'

## Related ops

See also: `hp_filter`, `hamilton_filter` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
