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

Iterative EM algorithm: alternates between (1) fitting a factor model to the currently-imputed panel and (2) imputing missing cells from the factor model's prediction. Converges to a low-rank fill consistent with the cross-series factor structure.

Used per-origin under ``imputation_temporal_rule = expanding_window_per_origin`` so the imputation respects the walk-forward information set.

**When to use**

Default for FRED-MD/QD high-dimensional panels.

**References**

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'
* Stock & Watson (2002) 'Macroeconomic Forecasting Using Diffusion Indexes', JBES 20(2).

**Related options**: [`em_multivariate`](#em-multivariate), [`mean`](#mean), [`forward_fill`](#forward-fill), [`linear_interpolation`](#linear-interpolation)

_Last reviewed 2026-05-04 by macroforecast author._

### `em_multivariate`  --  operational

Multivariate-Gaussian EM imputation.

Models the full panel as multivariate Gaussian and imputes missing cells via Schur-complement conditioning. More flexible than ``em_factor`` (no rank cap) but more expensive on large panels (O(p²) per iteration).

**When to use**

Smaller panels (≤ 50 series) where the full covariance is tractable.

**References**

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'

**Related options**: [`em_factor`](#em-factor), [`mean`](#mean)

_Last reviewed 2026-05-04 by macroforecast author._

### `mean`  --  operational

Replace missing cells with the per-series rolling mean.

Simple, fast, deterministic. No iteration. Useful when the missing pattern is sparse.

**When to use**

Sparse missingness; quick smoke tests.

**References**

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'

**Related options**: [`em_factor`](#em-factor), [`forward_fill`](#forward-fill)

_Last reviewed 2026-05-04 by macroforecast author._

### `forward_fill`  --  operational

Carry the last observed value forward.

Standard pandas ffill. Appropriate for series where the most recent observation is the best forecast of the missing value.

**When to use**

Slowly-moving series (interest rates, ratios); release-lag handling.

**References**

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'

**Related options**: [`linear_interpolation`](#linear-interpolation), [`em_factor`](#em-factor)

_Last reviewed 2026-05-04 by macroforecast author._

### `linear_interpolation`  --  operational

Linear interpolation between adjacent observations.

Smooths over isolated missing observations. Not appropriate for leading / trailing missings.

**When to use**

Interior missing observations in well-behaved series.

**References**

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'
* Chow & Lin (1971) 'Best Linear Unbiased Interpolation, Distribution, and Extrapolation of Time Series by Related Series', RES 53(4).

**Related options**: [`forward_fill`](#forward-fill), [`em_factor`](#em-factor)

_Last reviewed 2026-05-04 by macroforecast author._

### `none_propagate`  --  operational

Pass NaN through; downstream layers handle it.

Useful when the recipe wants L3 / L4 to see the missing pattern (e.g., for missingness-as-feature studies).

**When to use**

Studies that treat missingness as informative; or panels with no missings.

**References**

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'

**Related options**: [`em_factor`](#em-factor), [`mean`](#mean), [`forward_fill`](#forward-fill)

_Last reviewed 2026-05-04 by macroforecast author._
