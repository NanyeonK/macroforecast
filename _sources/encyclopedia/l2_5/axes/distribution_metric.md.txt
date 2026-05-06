# `distribution_metric`

[Back to L2.5](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``distribution_metric`` on sub-layer ``L2_5_B_distribution_shift`` (layer ``l2_5``).

## Sub-layer

**L2_5_B_distribution_shift**

## Axis metadata

- Default: `['mean_change', 'sd_change', 'ks_statistic']`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 5 option(s)
- Future: 0 option(s)

## Options

### `ks_statistic`  --  operational

Kolmogorov-Smirnov statistic between raw and cleaned distributions.

L2.5.B distribution metric ``ks_statistic`` (multi-select axis).

This option configures the ``distribution_metric`` axis on the ``L2_5_B_distribution_shift`` sub-layer of L2.5; output is emitted under ``manifest.diagnostics/l2_5/L2_5_B_distribution_shift/`` alongside the other selected views.

**When to use**

Detecting that cleaning has moved the distribution non-trivially; KS > 0.1 is a typical 'investigate' threshold.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`mean_change`](#mean-change), [`sd_change`](#sd-change), [`skew_change`](#skew-change), [`kurtosis_change`](#kurtosis-change)

_Last reviewed 2026-05-05 by macroforecast author._

### `kurtosis_change`  --  operational

Δ-excess-kurtosis before vs after cleaning.

L2.5.B distribution metric ``kurtosis_change`` (multi-select axis).

This option configures the ``distribution_metric`` axis on the ``L2_5_B_distribution_shift`` sub-layer of L2.5; output is emitted under ``manifest.diagnostics/l2_5/L2_5_B_distribution_shift/`` alongside the other selected views.

**When to use**

Detecting heavy-tail removal; large negative changes confirm successful tail trimming.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`ks_statistic`](#ks-statistic), [`mean_change`](#mean-change), [`sd_change`](#sd-change), [`skew_change`](#skew-change)

_Last reviewed 2026-05-05 by macroforecast author._

### `mean_change`  --  operational

Δ-mean before vs after cleaning.

L2.5.B distribution metric ``mean_change`` (multi-select axis).

This option configures the ``distribution_metric`` axis on the ``L2_5_B_distribution_shift`` sub-layer of L2.5; output is emitted under ``manifest.diagnostics/l2_5/L2_5_B_distribution_shift/`` alongside the other selected views.

**When to use**

Spotting bias introduced by the imputation method (EM-factor can introduce systematic shifts).

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`ks_statistic`](#ks-statistic), [`sd_change`](#sd-change), [`skew_change`](#skew-change), [`kurtosis_change`](#kurtosis-change)

_Last reviewed 2026-05-05 by macroforecast author._

### `sd_change`  --  operational

Δ-standard-deviation before vs after cleaning.

L2.5.B distribution metric ``sd_change`` (multi-select axis).

This option configures the ``distribution_metric`` axis on the ``L2_5_B_distribution_shift`` sub-layer of L2.5; output is emitted under ``manifest.diagnostics/l2_5/L2_5_B_distribution_shift/`` alongside the other selected views.

**When to use**

Detecting variance compression from winsorisation / outlier-replacement.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`ks_statistic`](#ks-statistic), [`mean_change`](#mean-change), [`skew_change`](#skew-change), [`kurtosis_change`](#kurtosis-change)

_Last reviewed 2026-05-05 by macroforecast author._

### `skew_change`  --  operational

Δ-skewness before vs after cleaning.

L2.5.B distribution metric ``skew_change`` (multi-select axis).

This option configures the ``distribution_metric`` axis on the ``L2_5_B_distribution_shift`` sub-layer of L2.5; output is emitted under ``manifest.diagnostics/l2_5/L2_5_B_distribution_shift/`` alongside the other selected views.

**When to use**

Quantifying tail-trimming asymmetry; large skew changes indicate one-sided outlier treatment.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`ks_statistic`](#ks-statistic), [`mean_change`](#mean-change), [`sd_change`](#sd-change), [`kurtosis_change`](#kurtosis-change)

_Last reviewed 2026-05-05 by macroforecast author._
