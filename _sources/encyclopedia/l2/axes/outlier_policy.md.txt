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

See [mccracken_ng_iqr function page](../outlier_policy/mccracken_ng_iqr.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.iqr_outlier_clean``.

### `winsorize`  --  operational

Cap observations at user-supplied quantile thresholds.

See [winsorize function page](../outlier_policy/winsorize.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.winsorize_clean``.

### `zscore_threshold`  --  operational

Flag observations beyond a z-score threshold.

See [zscore_threshold function page](../outlier_policy/zscore_threshold.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.zscore_outlier_clean``.

### `none`  --  operational

Skip outlier handling.

Pass series through unchanged. Useful when L1 already cleaned outliers (``raw_outlier_policy``) or when the study wants to compare against a no-cleaning baseline.

**When to use**

Custom panels already cleaned upstream; no-cleaning ablations.

**References**

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'

**Related options**: [`mccracken_ng_iqr`](#mccracken-ng-iqr), [`winsorize`](#winsorize), [`zscore_threshold`](#zscore-threshold)

_Last reviewed 2026-05-04 by macroforecast author._
