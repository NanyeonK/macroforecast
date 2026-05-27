# `missing_view`

[Back to L1.5](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``missing_view`` on sub-layer ``L1_5_D_missing_outlier_audit`` (layer ``l1_5``).

## Sub-layer

**L1_5_D_missing_outlier_audit**

## Axis metadata

- Default: `'multi'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 4 option(s)
- Future: 0 option(s)

## Options

### `heatmap`  --  operational

Visualisation of missing pattern over time.

L1.5.D missing-data visualisation ``heatmap``.

This option configures the ``missing_view`` axis on the ``L1_5_D_missing_outlier_audit`` sub-layer of L1.5; output is emitted under ``manifest.diagnostics/l1_5/L1_5_D_missing_outlier_audit/`` alongside the other selected views.

**When to use**

Detecting block-missingness (e.g. all 1980s missing) vs scattered NaNs that influences the choice of imputation method.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`per_series_count`](#per-series-count), [`longest_gap`](#longest-gap), [`multi`](#multi)

_Last reviewed 2026-05-05 by macroforecast author._

### `longest_gap`  --  operational

Maximum consecutive-missing run per series.

L1.5.D missing-data visualisation ``longest_gap``.

This option configures the ``missing_view`` axis on the ``L1_5_D_missing_outlier_audit`` sub-layer of L1.5; output is emitted under ``manifest.diagnostics/l1_5/L1_5_D_missing_outlier_audit/`` alongside the other selected views.

**When to use**

Critical for forward-fill imputation -- long runs distort the imputed values; values > 12 (monthly) typically rule out forward-fill.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`per_series_count`](#per-series-count), [`heatmap`](#heatmap), [`multi`](#multi)

_Last reviewed 2026-05-05 by macroforecast author._

### `multi`  --  operational

Produce all three missingness views.

L1.5.D missing-data visualisation ``multi``.

This option configures the ``missing_view`` axis on the ``L1_5_D_missing_outlier_audit`` sub-layer of L1.5; output is emitted under ``manifest.diagnostics/l1_5/L1_5_D_missing_outlier_audit/`` alongside the other selected views.

**When to use**

Comprehensive missingness audit; recommended default for first-time runs.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`per_series_count`](#per-series-count), [`heatmap`](#heatmap), [`longest_gap`](#longest-gap)

_Last reviewed 2026-05-05 by macroforecast author._

### `per_series_count`  --  operational

Bar chart of NaN count per series.

L1.5.D missing-data visualisation ``per_series_count``.

This option configures the ``missing_view`` axis on the ``L1_5_D_missing_outlier_audit`` sub-layer of L1.5; output is emitted under ``manifest.diagnostics/l1_5/L1_5_D_missing_outlier_audit/`` alongside the other selected views.

**When to use**

Quick view of where L2.D imputation will work hardest; outliers in this chart flag candidates for dropping.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`heatmap`](#heatmap), [`longest_gap`](#longest-gap), [`multi`](#multi)

_Last reviewed 2026-05-05 by macroforecast author._
