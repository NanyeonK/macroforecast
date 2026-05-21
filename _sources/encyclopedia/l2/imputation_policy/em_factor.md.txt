# `em_factor` -- EM-factor imputation (McCracken-Ng default).

[Back to `imputation_policy` axis](../axes/imputation_policy.md) | [Back to L2](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `imputation_policy`, sub-layer `l2_d`, layer `l2`.
> Standalone callable: `mf.functions.em_factor_impute_clean`.

## Function signature

```python
mf.functions.em_factor_impute_clean(
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

Iterative EM algorithm: alternates between (1) fitting a factor model to the currently-imputed panel and (2) imputing missing cells from the factor model's prediction. Converges to a low-rank fill consistent with the cross-series factor structure.

Used per-origin under ``imputation_temporal_rule = expanding_window_per_origin`` so the imputation respects the walk-forward information set.

**When to use**

Default for FRED-MD/QD high-dimensional panels.

## In recipe context

Set ``params.imputation_policy = "em_factor"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L2 recipe fragment
params:
  imputation_policy: em_factor
```

## References

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'
* Stock & Watson (2002) 'Macroeconomic Forecasting Using Diffusion Indexes', JBES 20(2).

## Related ops

See also: `em_multivariate`, `mean`, `forward_fill`, `linear_interpolation` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
