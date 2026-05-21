# `em_multivariate` -- Multivariate-Gaussian EM imputation.

[Back to `imputation_policy` axis](../axes/imputation_policy.md) | [Back to L2](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `imputation_policy`, sub-layer `l2_d`, layer `l2`.
> Standalone callable: `mf.functions.em_multivariate_impute_clean`.

## Function signature

```python
mf.functions.em_multivariate_impute_clean(
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

Models the full panel as multivariate Gaussian and imputes missing cells via Schur-complement conditioning. More flexible than ``em_factor`` (no rank cap) but more expensive on large panels (O(p²) per iteration).

**When to use**

Smaller panels (≤ 50 series) where the full covariance is tractable.

## In recipe context

Set ``params.imputation_policy = "em_multivariate"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L2 recipe fragment
params:
  imputation_policy: em_multivariate
```

## References

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'

## Related ops

See also: `em_factor`, `mean` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
