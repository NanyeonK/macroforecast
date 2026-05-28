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

- Operational: 4 option(s)
- Future: 0 option(s)

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

### `keep_with_indicator`  --  operational

Keep the outlier value but add a binary indicator column.

Leaves the flagged observation's value unchanged and instead appends a new binary column named ``{col}__outlier_flag`` that equals 1 for flagged rows and 0 otherwise. Downstream layers can use the flag column as a covariate to let the model decide whether and how to discount the outlier.

The ``{col}__outlier_flag`` semantics are:

* Column name pattern: original column name + ``__outlier_flag`` suffix.
* Value: 1 (flagged by the chosen outlier_policy), 0 (clean).
* The indicator is added to the panel before L2.D imputation and is visible to all downstream feature engineering and model estimation steps.

This action is appropriate when extreme observations may carry genuine signal and the model should not lose the original value, but the researcher wants to flag the observation's provenance.

**When to use**

Studies where outliers may be genuine signals (financial crises, policy events); missingness-as-feature studies.

**When NOT to use**

When outliers are data errors that should be removed or replaced -- use flag_as_nan, replace_with_median, or replace_with_cap_value instead.

**References**

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'

**Related options**: [`flag_as_nan`](#flag-as-nan), [`replace_with_median`](#replace-with-median), [`replace_with_cap_value`](#replace-with-cap-value)

_Last reviewed 2026-05-04 by macroforecast author._
