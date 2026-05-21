# `imputation_policy`

[Back to L2](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``imputation_policy`` on sub-layer ``l2_d`` (layer ``l2``).

## Sub-layer

**l2_d**

## Axis metadata

- Default: `'em_factor'`
- Sweepable: True
- Status: operational

## Operational status summary

- Operational: 6 option(s)
- Future: 0 option(s)

## Options

### `em_factor`  --  operational

EM-factor imputation (McCracken-Ng default).

See [em_factor function page](../imputation_policy/em_factor.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.em_factor_impute_clean``.

### `em_multivariate`  --  operational

Multivariate-Gaussian EM imputation.

See [em_multivariate function page](../imputation_policy/em_multivariate.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.em_multivariate_impute_clean``.

### `mean`  --  operational

Replace missing cells with the per-series rolling mean.

See [mean function page](../imputation_policy/mean.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.mean_impute_clean``.

### `forward_fill`  --  operational

Carry the last observed value forward.

See [forward_fill function page](../imputation_policy/forward_fill.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.forward_fill_clean``.

### `linear_interpolation`  --  operational

Linear interpolation between adjacent observations.

See [linear_interpolation function page](../imputation_policy/linear_interpolation.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.linear_interpolate_clean``.

### `none_propagate`  --  operational

Pass NaN through; downstream layers handle it.

Useful when the recipe wants L3 / L4 to see the missing pattern (e.g., for missingness-as-feature studies).

**When to use**

Studies that treat missingness as informative; or panels with no missings.

**References**

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'

**Related options**: [`em_factor`](#em-factor), [`mean`](#mean), [`forward_fill`](#forward-fill)

_Last reviewed 2026-05-04 by macroforecast author._
