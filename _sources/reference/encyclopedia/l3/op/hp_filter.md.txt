# `hp_filter` -- Hodrick-Prescott filter -- trend / cycle decomposition.

[Back to `op` axis](../axes/op.md) | [Back to L3](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L3_A_step_op`, layer `l3`.
> Standalone callable: `mf.functions.hp_filter_transform`.

## Function signature

```python
mf.functions.hp_filter_transform(
    panel: pd.DataFrame,
    lambda_: float,
) -> pd.DataFrame
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `panel` | `pd.DataFrame` | — | — | Input panel. Each column is a variable; rows are time periods. Series is promoted to a single-column DataFrame internally. |
| `lambda_` | `float` | `1600` | > 0 | HP smoothing parameter. Convention: 1600 for quarterly, 129600 for monthly (Ravn-Uhlig 2002). |

## Returns

`pd.DataFrame` — scalar result.

## Behavior

statsmodels ``hpfilter`` with smoothing parameter ``params.lamb`` (1600 for quarterly, 129600 for monthly per Ravn-Uhlig 2002). Returns the cyclical component by default; the trend can also be returned via ``params.return = 'trend'``.

**When to use**

Extracting business-cycle gaps from trending series.

**When NOT to use**

Real-time / one-sided forecasting -- HP introduces look-ahead bias unless restricted to ``expanding_window_per_origin``.

## In recipe context

Set ``params.op = "hp_filter"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L3 recipe fragment
params:
  op: hp_filter
```

## References

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'
* Hodrick & Prescott (1997) 'Postwar U.S. Business Cycles: An Empirical Investigation', JMCB 29(1): 1-16.

## Related ops

See also: `hamilton_filter`, `diff` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
