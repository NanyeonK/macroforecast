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

Default; relies on L2.C McCracken-Ng IQR detection and the configured ``outlier_action`` to handle extreme values. See also: L2 ``outlier_policy`` / ``outlier_action`` (same surface, different stage: raw vs post-tcode).

**When to use**

Default; the canonical workflow.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`winsorize_raw`](#winsorize-raw), [`iqr_clip_raw`](#iqr-clip-raw), [`mad_clip_raw`](#mad-clip-raw), [`zscore_clip_raw`](#zscore-clip-raw), [`set_raw_outliers_to_missing`](#set-raw-outliers-to-missing)

_Last reviewed 2026-05-05 by macroforecast author._

### `winsorize_raw`  --  operational

Winsorise raw series at quantile cutpoints (default p1 / p99).

Caps extreme values at the specified quantile before t-coding. Preserves observation count but compresses tails. Configured via ``leaf_config.winsorize_quantiles`` (default [0.01, 0.99]). Compare: L2 ``outlier_policy=winsorize`` operates on the post-tcode panel.

**When to use**

Heavy-tailed financial / macro series where extreme observations would dominate downstream estimates.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`preserve_raw_outliers`](#preserve-raw-outliers), [`iqr_clip_raw`](#iqr-clip-raw)

**Parameters**

| name | type | default | constraint | description |
|---|---|---|---|---|
| `winsorize_quantiles` | `list[float, float]` | `'[0.01, 0.99]'` | 0 <= low < high <= 1; both elements required. | Lower and upper quantile clip thresholds. Defaults to symmetric 1%/99% winsorization. Values outside [low, high] quantile bounds are clipped to the bound value. |

_Last reviewed 2026-05-05 by macroforecast author._

### `iqr_clip_raw`  --  operational

Clip raw observations beyond k×IQR thresholds.

Clips values outside ``Q1 - k·IQR``, ``Q3 + k·IQR`` (k default 10.0, matching McCracken-Ng). Robust to non-Gaussian distributions. Configured via ``leaf_config.outlier_iqr_threshold``. Compare: L2 ``outlier_policy=mccracken_ng_iqr`` uses the same k but on the post-tcode panel.

**When to use**

Robust outlier handling on non-normal series.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`winsorize_raw`](#winsorize-raw), [`mad_clip_raw`](#mad-clip-raw), [`zscore_clip_raw`](#zscore-clip-raw)

**Parameters**

| name | type | default | constraint | description |
|---|---|---|---|---|
| `outlier_iqr_threshold` | `float` | `'10.0'` | >0 | IQR multiplier above which raw observations are clipped. McCracken-Ng default is 10.0. Observations satisfying |x - median| > k * IQR are clipped to the band boundary. |

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

Standard z-score rule (typically k = 3). Cheapest option but assumes approximate normality. Configured via ``leaf_config.zscore_threshold_value``.

**When to use**

Approximately Gaussian series; quick baseline.

**When NOT to use**

Heavy-tailed series -- use ``iqr_clip_raw`` or ``mad_clip_raw``.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`iqr_clip_raw`](#iqr-clip-raw), [`mad_clip_raw`](#mad-clip-raw)

**Parameters**

| name | type | default | constraint | description |
|---|---|---|---|---|
| `zscore_threshold_value` | `float` | `'3.0'` | >0 | Z-score threshold; observations with |z| > threshold are clipped to the threshold boundary. z is computed as (x - mean) / std over the series. |

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
