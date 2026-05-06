# `outlier_policy`

[Back to L2](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``outlier_policy`` on sub-layer ``l2_c`` (layer ``l2``).

## Sub-layer

**l2_c**

## Axis metadata

- Default: `'mccracken_ng_iqr'`
- Sweepable: True
- Status: operational

## Operational status summary

- Operational: 4 option(s)
- Future: 0 option(s)

## Options

### `mccracken_ng_iqr`  --  operational

McCracken-Ng's published IQR-multiple outlier rule.

Flags any observation more than ``leaf_config.outlier_iqr_threshold`` (default 10) IQRs from the per-series median. The 10×IQR threshold is the published McCracken-Ng default and matches their replication scripts.

Pairs with an L2.C ``outlier_action`` to specify what happens to flagged observations (replace with NaN by default, then L2.D imputation fills them).

**When to use**

Default for FRED-based studies. Canonical replication path.

**References**

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'
* McCracken & Ng (2016) 'FRED-MD: A Monthly Database for Macroeconomic Research', JBES 34(4). (doi:10.1080/07350015.2015.1086655)

**Related options**: [`winsorize`](#winsorize), [`zscore_threshold`](#zscore-threshold), [`none`](#none), [`outlier_action`](#outlier-action)

_Last reviewed 2026-05-04 by macroforecast author._

### `winsorize`  --  operational

Cap observations at user-supplied quantile thresholds.

Truncates each series at ``leaf_config.winsorize_lower_quantile`` (default 0.01) and ``leaf_config.winsorize_upper_quantile`` (default 0.99). Less aggressive than the McCracken-Ng IQR rule and preserves more of the tail.

**When to use**

Studies that want a bounded but non-NaN outlier handler; alternative-rule comparisons.

**References**

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'
* Tukey (1977) 'Exploratory Data Analysis', Addison-Wesley.

**Related options**: [`mccracken_ng_iqr`](#mccracken-ng-iqr), [`zscore_threshold`](#zscore-threshold)

_Last reviewed 2026-05-04 by macroforecast author._

### `zscore_threshold`  --  operational

Flag observations beyond a z-score threshold.

Computes the rolling z-score per series and flags ``|z|`` > ``leaf_config.zscore_threshold_value`` (default 3.0). Simpler than IQR but assumes approximately Gaussian residuals.

**When to use**

Approximately-Gaussian series; quick sanity-check sweeps.

**References**

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'

**Related options**: [`mccracken_ng_iqr`](#mccracken-ng-iqr), [`winsorize`](#winsorize)

_Last reviewed 2026-05-04 by macroforecast author._

### `none`  --  operational

Skip outlier handling.

Pass series through unchanged. Useful when L1 already cleaned outliers (``raw_outlier_policy``) or when the study wants to compare against a no-cleaning baseline.

**When to use**

Custom panels already cleaned upstream; no-cleaning ablations.

**References**

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'

**Related options**: [`mccracken_ng_iqr`](#mccracken-ng-iqr), [`winsorize`](#winsorize), [`zscore_threshold`](#zscore-threshold)

_Last reviewed 2026-05-04 by macroforecast author._
