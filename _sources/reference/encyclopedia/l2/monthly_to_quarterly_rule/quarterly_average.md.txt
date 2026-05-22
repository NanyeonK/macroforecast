# `quarterly_average` -- Aggregate to quarterly via mean of the three monthly observations.

[Back to `monthly_to_quarterly_rule` axis](../axes/monthly_to_quarterly_rule.md) | [Back to L2](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `monthly_to_quarterly_rule`, sub-layer `l2_a`, layer `l2`.
> Standalone callable: `mf.functions.freq_align_monthly_to_quarterly_clean`.

## Function signature

```python
mf.functions.freq_align_monthly_to_quarterly_clean(
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

Standard NIPA aggregation for stocks / averages.

Configures the ``monthly_to_quarterly_rule`` axis on ``l2_a`` (layer ``l2``); the ``quarterly_average`` value is materialised in the recipe's ``fixed_axes`` block under that sub-layer.

**When to use**

Default. Stock variables (interest rates, prices, employment levels).

## In recipe context

Set ``params.monthly_to_quarterly_rule = "quarterly_average"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L2 recipe fragment
params:
  monthly_to_quarterly_rule: quarterly_average
```

## References

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'

## Related ops

See also: `quarterly_endpoint`, `quarterly_sum` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
