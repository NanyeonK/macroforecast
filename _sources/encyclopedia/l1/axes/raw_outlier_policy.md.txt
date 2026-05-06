# `raw_outlier_policy`

[Back to L1](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``raw_outlier_policy`` on sub-layer ``l1_c`` (layer ``l1``).

## Sub-layer

**l1_c**

## Axis metadata

- Default: `'preserve_raw_outliers'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 6 option(s)
- Future: 0 option(s)

## Options

### `preserve_raw_outliers`  --  operational

Pass raw outliers through to L2.C.

Default; relies on L2.C McCracken-Ng IQR detection and the configured ``outlier_action`` to handle extreme values.

**When to use**

Default; the canonical workflow.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`winsorize_raw`](#winsorize-raw), [`iqr_clip_raw`](#iqr-clip-raw), [`mad_clip_raw`](#mad-clip-raw), [`zscore_clip_raw`](#zscore-clip-raw), [`set_raw_outliers_to_missing`](#set-raw-outliers-to-missing)

_Last reviewed 2026-05-05 by macroforecast author._

### `winsorize_raw`  --  operational

Winsorise raw series at quantile cutpoints (default p1 / p99).

Caps extreme values at the specified quantile before t-coding. Preserves observation count but compresses tails.

**When to use**

Heavy-tailed financial / macro series where extreme observations would dominate downstream estimates.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`preserve_raw_outliers`](#preserve-raw-outliers), [`iqr_clip_raw`](#iqr-clip-raw)

_Last reviewed 2026-05-05 by macroforecast author._

### `iqr_clip_raw`  --  operational

Clip raw observations beyond k×IQR thresholds.

Clips values outside ``Q1 - k·IQR``, ``Q3 + k·IQR`` (k typically 1.5 or 3). Robust to non-Gaussian distributions.

**When to use**

Robust outlier handling on non-normal series.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`winsorize_raw`](#winsorize-raw), [`mad_clip_raw`](#mad-clip-raw), [`zscore_clip_raw`](#zscore-clip-raw)

_Last reviewed 2026-05-05 by macroforecast author._

### `mad_clip_raw`  --  operational

Clip raw observations beyond k×MAD thresholds.

Median Absolute Deviation -based clipping; even more robust than IQR. Default k = 3 maps to roughly 3σ for normal data.

**When to use**

Highly non-Gaussian series with sparse outliers.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`iqr_clip_raw`](#iqr-clip-raw), [`zscore_clip_raw`](#zscore-clip-raw)

_Last reviewed 2026-05-05 by macroforecast author._

### `zscore_clip_raw`  --  operational

Clip raw observations beyond k standard deviations.

Standard z-score rule (typically k = 3). Cheapest option but assumes approximate normality.

**When to use**

Approximately Gaussian series; quick baseline.

**When NOT to use**

Heavy-tailed series -- use ``iqr_clip_raw`` or ``mad_clip_raw``.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`iqr_clip_raw`](#iqr-clip-raw), [`mad_clip_raw`](#mad-clip-raw)

_Last reviewed 2026-05-05 by macroforecast author._

### `set_raw_outliers_to_missing`  --  operational

Set raw outliers to NaN and defer to L2.D imputation.

Replaces flagged outliers with NaN. The L2.D imputation method then fills the resulting gaps; preserves observation count for downstream stages.

**When to use**

Pipelines where outliers should be re-imputed coherently with other missing data.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`preserve_raw_outliers`](#preserve-raw-outliers), [`winsorize_raw`](#winsorize-raw)

_Last reviewed 2026-05-05 by macroforecast author._
