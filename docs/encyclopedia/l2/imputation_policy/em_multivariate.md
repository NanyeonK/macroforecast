# `em_multivariate` -- Impute missing values using the PCA-EM algorithm (uncapped rank).

[Back to `imputation_policy` axis](../axes/imputation_policy.md) | [Back to L2](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `imputation_policy`, sub-layer `l2_d`, layer `l2`.
> Standalone callable: `mf.functions.em_multivariate_impute_clean`.

## Function signature

```python
mf.functions.em_multivariate_impute_clean(
    panel: pd.DataFrame,
    max_iter: int = 20,
    tol: float = 0.0001,
) -> pd.DataFrame
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `panel` | `pd.DataFrame` | — | — | — |
| `max_iter` | `int` | `20` | — | — |
| `tol` | `float` | `0.0001` | — | — |

## Returns

`pd.DataFrame` — scalar result.

## Behavior

(See standalone callable docstring.)

## In recipe context

Set ``params.imputation_policy = "em_multivariate"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L2 recipe fragment
params:
  imputation_policy: em_multivariate
```

## References

* macroforecast design, L2: see design docs for em_multivariate.

## Related ops

See also: `em_factor`, `forward_fill`, `linear_interpolation`, `mean` (on the same axis).

_Last reviewed 2026-05-22 by macroforecast author._
