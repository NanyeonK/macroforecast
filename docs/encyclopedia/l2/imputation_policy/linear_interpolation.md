# `linear_interpolation` -- Impute missing cells by linear interpolation between adjacent observations.

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
| `panel` | `pd.DataFrame` | — | — | — |

## Returns

`pd.DataFrame` — scalar result.

## Behavior

(See standalone callable docstring.)

## In recipe context

Set ``params.imputation_policy = "linear_interpolation"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L2 recipe fragment
params:
  imputation_policy: linear_interpolation
```

## References

* macroforecast design, L2: see design docs for linear_interpolation.

## Related ops

See also: `em_factor`, `em_multivariate`, `forward_fill`, `mean` (on the same axis).

_Last reviewed 2026-05-22 by macroforecast author._
