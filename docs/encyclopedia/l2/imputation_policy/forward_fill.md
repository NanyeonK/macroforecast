# `forward_fill` -- Carry the last observed value forward.

[Back to `imputation_policy` axis](../axes/imputation_policy.md) | [Back to L2](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `imputation_policy`, sub-layer `l2_d`, layer `l2`.
> Standalone callable: `mf.functions.forward_fill_clean`.

## Function signature

```python
mf.functions.forward_fill_clean(
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

Standard pandas ffill. Appropriate for series where the most recent observation is the best forecast of the missing value.

**When to use**

Slowly-moving series (interest rates, ratios); release-lag handling.

## In recipe context

Set ``params.imputation_policy = "forward_fill"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L2 recipe fragment
params:
  imputation_policy: forward_fill
```

## References

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'

## Related ops

See also: `linear_interpolation`, `em_factor` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
