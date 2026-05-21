# `step_backward` -- Align quarterly series to a monthly grid using a chosen interpolation rule.

[Back to `quarterly_to_monthly_rule` axis](../axes/quarterly_to_monthly_rule.md) | [Back to L2](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `quarterly_to_monthly_rule`, sub-layer `l2_a`, layer `l2`.
> Standalone callable: `mf.functions.freq_align_quarterly_to_monthly_clean`.

## Function signature

```python
mf.functions.freq_align_quarterly_to_monthly_clean(
    panel: pd.DataFrame,
    quarterly_columns: list[str],
    rule: str = 'step_backward',
) -> pd.DataFrame
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `panel` | `pd.DataFrame` | — | — | — |
| `quarterly_columns` | `list[str]` | — | — | — |
| `rule` | `str` | `'step_backward'` | — | — |

## Returns

`pd.DataFrame` — scalar result.

## Behavior

(See standalone callable docstring.)

## In recipe context

Set ``params.quarterly_to_monthly_rule = "step_backward"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L2 recipe fragment
params:
  quarterly_to_monthly_rule: step_backward
```

## References

* macroforecast design, L2: see design docs for step_backward.

## Related ops

See also: `chow_lin_disaggregation` (on the same axis).

_Last reviewed 2026-05-22 by macroforecast author._
