# `quarterly_average` -- Aggregate monthly columns to quarterly frequency.

[Back to `monthly_to_quarterly_rule` axis](../axes/monthly_to_quarterly_rule.md) | [Back to L2](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `monthly_to_quarterly_rule`, sub-layer `l2_a`, layer `l2`.
> Standalone callable: `mf.functions.freq_align_monthly_to_quarterly_clean`.

## Function signature

```python
mf.functions.freq_align_monthly_to_quarterly_clean(
    panel: pd.DataFrame,
    monthly_columns: list[str],
    rule: str = 'quarterly_average',
) -> pd.DataFrame
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `panel` | `pd.DataFrame` | — | — | — |
| `monthly_columns` | `list[str]` | — | — | — |
| `rule` | `str` | `'quarterly_average'` | — | — |

## Returns

`pd.DataFrame` — scalar result.

## Behavior

(See standalone callable docstring.)

## In recipe context

Set ``params.monthly_to_quarterly_rule = "quarterly_average"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L2 recipe fragment
params:
  monthly_to_quarterly_rule: quarterly_average
```

## References

* macroforecast design, L2: see design docs for quarterly_average.

## Related ops

See the layer index for related ops.

_Last reviewed 2026-05-22 by macroforecast author._
