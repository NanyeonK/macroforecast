# `linear_interpolation` -- Linear interpolation between adjacent observations.

[Back to `imputation_policy` axis](../axes/imputation_policy.md) | [Back to L2](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `imputation_policy`, sub-layer `l2_d`, layer `l2`.
> Standalone callable: `mf.functions.linear_interpolate_clean`.

## Function signature

```python
mf.functions.linear_interpolate_clean(
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

Smooths over isolated missing observations. Not appropriate for leading / trailing missings.

**When to use**

Interior missing observations in well-behaved series.

## In recipe context

Set ``params.imputation_policy = "linear_interpolation"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L2 recipe fragment
params:
  imputation_policy: linear_interpolation
```

## References

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'
* Chow & Lin (1971) 'Best Linear Unbiased Interpolation, Distribution, and Extrapolation of Time Series by Related Series', RES 53(4).

## Related ops

See also: `forward_fill`, `em_factor` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
