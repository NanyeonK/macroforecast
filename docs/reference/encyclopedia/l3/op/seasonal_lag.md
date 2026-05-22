# `seasonal_lag` -- Lag at a seasonal period (e.g. y_{t-12} for monthly data).

[Back to `op` axis](../axes/op.md) | [Back to L3](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L3_A_step_op`, layer `l3`.
> Standalone callable: `mf.functions.seasonal_lag_matrix`.

## Function signature

```python
mf.functions.seasonal_lag_matrix(
    panel: pd.DataFrame,
    seasonal_period: int,
    n_seasonal_lags: int,
) -> pd.DataFrame
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `panel` | `pd.DataFrame` | — | — | Input panel. Each column is a variable; rows are time periods. Series is promoted to a single-column DataFrame internally. |
| `seasonal_period` | `int` | `12` | >= 2 | Seasonal cycle length (12 for monthly, 4 for quarterly). |
| `n_seasonal_lags` | `int` | `1` | >= 1 | Number of seasonal lags. Shifts by seasonal_period * i. |

## Returns

`pd.DataFrame` — scalar result.

## Behavior

Standard ``lag`` op restricted to the seasonal index (``params.lag = 12`` for monthly, ``4`` for quarterly). Useful for year-over-year features and seasonal AR terms.

**When to use**

Capturing year-over-year persistence; seasonal AR baselines.

**When NOT to use**

When season_dummy or X-13 deseasonalisation is preferred over lag-based seasonality.

## In recipe context

Set ``params.op = "seasonal_lag"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L3 recipe fragment
params:
  op: seasonal_lag
```

## References

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'

## Related ops

See also: `season_dummy`, `ma_window` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
