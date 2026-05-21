# `step_backward` -- Step-function: each month inherits the most-recent published quarterly value.

[Back to `quarterly_to_monthly_rule` axis](../axes/quarterly_to_monthly_rule.md) | [Back to L2](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `quarterly_to_monthly_rule`, sub-layer `l2_a`, layer `l2`.
> Standalone callable: `mf.functions.freq_align_quarterly_to_monthly_clean`.

## Function signature

```python
mf.functions.freq_align_quarterly_to_monthly_clean(
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

When a quarterly series needs to align with a monthly target, macroforecast holds the quarterly observation constant for all three months of the quarter (with a 1-quarter publication lag where appropriate). Conservative: no smoothing, no extrapolation.

**When to use**

Default for FRED-SD mixed-frequency studies.

## In recipe context

Set ``params.quarterly_to_monthly_rule = "step_backward"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L2 recipe fragment
params:
  quarterly_to_monthly_rule: step_backward
```

## References

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'

## Related ops

See also: `step_forward`, `linear_interpolation`, `chow_lin` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
