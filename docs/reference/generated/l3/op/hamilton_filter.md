# `hamilton_filter` -- Hamilton (2018) regression-based detrend (HP-filter alternative).

[Back to `op` axis](../axes/op.md) | [Back to L3](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L3_A_step_op`, layer `l3`.
> Standalone callable: `mf.functions.hamilton_filter_transform`.

## Function signature

```python
mf.functions.hamilton_filter_transform(
    panel: pd.DataFrame,
    h: int,
    p: int,
) -> pd.DataFrame
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `panel` | `pd.DataFrame` | — | — | Input panel. Each column is a variable; rows are time periods. Series is promoted to a single-column DataFrame internally. |
| `h` | `int` | `8` | >= 1 | Forecast horizon (periods ahead). Hamilton (2018) recommends h=8 for quarterly, h=24 for monthly. |
| `p` | `int` | `4` | >= 1 | Number of lags used in the regression. |

## Returns

`pd.DataFrame` — scalar result.

## Behavior

Regression-based two-sided alternative to the HP filter advocated by Hamilton (2018) for its better real-time properties. Default lookback h = 8 (quarterly) / 24 (monthly). Uses statsmodels ``hamilton_filter``.

**When to use**

Real-time / one-sided detrending where HP's two-sided smoothing is inappropriate.

## In recipe context

Set ``params.op = "hamilton_filter"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L3 recipe fragment
params:
  op: hamilton_filter
```

## References

* macroforecast design Part 2, L3: 'feature engineering is a pipeline of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'
* Hamilton (2018) 'Why You Should Never Use the Hodrick-Prescott Filter', RES 100(5): 831-843.

## Related ops

See also: `hp_filter` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
