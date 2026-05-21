# `mean` -- Replace missing cells with the per-column full-sample mean.

[Back to `imputation_policy` axis](../axes/imputation_policy.md) | [Back to L2](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `imputation_policy`, sub-layer `l2_d`, layer `l2`.
> Standalone callable: `mf.functions.mean_impute_clean`.

## Function signature

```python
mf.functions.mean_impute_clean(
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

Set ``params.imputation_policy = "mean"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L2 recipe fragment
params:
  imputation_policy: mean
```

## References

* macroforecast design, L2: see design docs for mean.

## Related ops

See also: `em_factor`, `em_multivariate`, `forward_fill`, `linear_interpolation` (on the same axis).

_Last reviewed 2026-05-22 by macroforecast author._
