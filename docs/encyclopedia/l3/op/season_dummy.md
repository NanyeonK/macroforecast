# `season_dummy` -- Calendar dummy variables (month-of-year, quarter-of-year).

[Back to `op` axis](../axes/op.md) | [Back to L3](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L3_A_step_op`, layer `l3`.
> Standalone callable: `mf.functions.season_dummy_transform`.

## Function signature

```python
mf.functions.season_dummy_transform(
    panel: pd.DataFrame,
    season: str enum {"quarter", "month"},
) -> pd.DataFrame
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `panel` | `pd.DataFrame` | — | — | Input panel. Each column is a variable; rows are time periods. Series is promoted to a single-column DataFrame internally. |
| `season` | `str enum {"quarter", "month"}` | `'"quarter"'` | — | Seasonal granularity hint. Accepted values: "quarter" and "month". Currently validated but has no effect on output (deprecated -- kept for API compatibility). Non-DatetimeIndex inputs produce season_* columns; DatetimeIndex inputs produce month_* columns. |

## Returns

`pd.DataFrame` — scalar result.

## Behavior

Generates ``params.n - 1`` 0/1 indicators for the calendar period (drops one to avoid multicollinearity with intercept). Standard frequentist seasonality control.

**When to use**

Capturing calendar seasonality in linear models when a smooth Fourier basis would over-shrink discrete jumps.

## In recipe context

Set ``params.op = "season_dummy"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L3 recipe fragment
params:
  op: season_dummy
```

## References

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'

## Related ops

See also: `fourier`, `seasonal_lag` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
