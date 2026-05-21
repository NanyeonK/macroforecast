# `log_diff` -- Log first difference: ``ln(y_t) - ln(y_{t-1})``.

[Back to `op` axis](../axes/op.md) | [Back to L3](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L3_A_step_op`, layer `l3`.
> Standalone callable: `mf.functions.log_diff_transform`.

## Function signature

```python
mf.functions.log_diff_transform(
    panel: pd.DataFrame,
    periods: int,
) -> pd.DataFrame
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `panel` | `pd.DataFrame` | — | — | Input panel. Each column is a variable; rows are time periods. Series is promoted to a single-column DataFrame internally. |
| `periods` | `int` | `1` | >= 1 | Number of lag periods to difference after taking logs. |

## Returns

`pd.DataFrame` — scalar result.

## Behavior

Composite of ``log`` then ``diff``. The standard FRED-MD transformation code 5/6 representation; produces a stationary approximation of the percentage change and is symmetric in expansions vs contractions.

**When to use**

Strictly-positive trending series (real GDP, employment, prices); FRED-MD tcode 5/6 default.

**When NOT to use**

Series that take non-positive values.

## In recipe context

Set ``params.op = "log_diff"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L3 recipe fragment
params:
  op: log_diff
```

## References

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'
* McCracken & Ng (2016) 'FRED-MD: A Monthly Database for Macroeconomic Research', JBES 34(4): 574-589. <https://doi.org/10.1080/07350015.2015.1086655>

## Related ops

See also: `log`, `diff`, `pct_change` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
