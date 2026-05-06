# `outlier_action`

[Back to L2](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``outlier_action`` on sub-layer ``l2_c`` (layer ``l2``).

## Sub-layer

**l2_c**

## Axis metadata

- Default: `'flag_as_nan'`
- Sweepable: True
- Status: operational

## Operational status summary

- Operational: 3 option(s)
- Future: 1 option(s)

## Options

### `flag_as_nan`  --  operational

Replace flagged outliers with NaN; let L2.D imputation fill them.

Default for the McCracken-Ng pipeline. Outliers become missing values, then EM-factor imputation in L2.D recovers a smoothed value from the cross-series factor structure.

**When to use**

Default. Pairs with em_factor / em_multivariate imputation.

**References**

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'
* McCracken & Ng (2016) 'FRED-MD: A Monthly Database for Macroeconomic Research', JBES 34(4). (doi:10.1080/07350015.2015.1086655)

**Related options**: [`replace_with_median`](#replace-with-median), [`replace_with_cap_value`](#replace-with-cap-value), [`keep_with_indicator`](#keep-with-indicator)

_Last reviewed 2026-05-04 by macroforecast author._

### `replace_with_median`  --  operational

Replace flagged outliers with the per-series median.

Simpler than imputation; useful when L2.D is set to ``none_propagate``.

Configures the ``outlier_action`` axis on ``l2_c`` (layer ``l2``); the ``replace_with_median`` value is materialised in the recipe's ``fixed_axes`` block under that sub-layer.

**When to use**

Studies that want a deterministic, no-imputation outlier handler.

**References**

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'

**Related options**: [`flag_as_nan`](#flag-as-nan), [`replace_with_cap_value`](#replace-with-cap-value)

_Last reviewed 2026-05-04 by macroforecast author._

### `replace_with_cap_value`  --  operational

Replace outliers with the cap value (winsorize-style cap).

Caps at the threshold rather than the median. Pairs with the winsorize / iqr policies.

**When to use**

Bounded-output studies; portfolios with hard limits on extreme values.

**References**

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'

**Related options**: [`flag_as_nan`](#flag-as-nan), [`replace_with_median`](#replace-with-median)

_Last reviewed 2026-05-04 by macroforecast author._

### `keep_with_indicator`  --  future

_(no schema description for `keep_with_indicator`)_

> TBD: option doc not yet authored for this value. The encyclopedia falls back to the bare schema description above. PRs adding a full ``OptionDoc`` entry under ``macroforecast/scaffold/option_docs/l2.py`` are welcome.
