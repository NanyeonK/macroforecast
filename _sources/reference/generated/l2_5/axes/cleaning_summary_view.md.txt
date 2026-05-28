# `cleaning_summary_view`

[Back to L2.5](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``cleaning_summary_view`` on sub-layer ``L2_5_D_cleaning_effect_summary`` (layer ``l2_5``).

## Sub-layer

**L2_5_D_cleaning_effect_summary**

## Axis metadata

- Default: `'multi'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 4 option(s)
- Future: 0 option(s)

## Options

### `multi`  --  operational

Render all three counts together.

L2.5.D cleaning effect view ``multi``.

This option configures the ``cleaning_summary_view`` axis on the ``L2_5_D_cleaning_effect_summary`` sub-layer of L2.5; output is emitted under ``manifest.diagnostics/l2_5/L2_5_D_cleaning_effect_summary/`` alongside the other selected views.

**When to use**

Default; full cleaning footprint summary.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`n_imputed_per_series`](#n-imputed-per-series), [`n_outliers_flagged`](#n-outliers-flagged), [`n_truncated_obs`](#n-truncated-obs)

_Last reviewed 2026-05-05 by macroforecast author._

### `n_imputed_per_series`  --  operational

Count of imputed cells per series.

L2.5.D cleaning effect view ``n_imputed_per_series``.

This option configures the ``cleaning_summary_view`` axis on the ``L2_5_D_cleaning_effect_summary`` sub-layer of L2.5; output is emitted under ``manifest.diagnostics/l2_5/L2_5_D_cleaning_effect_summary/`` alongside the other selected views.

**When to use**

Auditing imputation footprint; series with > 30% imputed cells warrant inspection.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`n_outliers_flagged`](#n-outliers-flagged), [`n_truncated_obs`](#n-truncated-obs), [`multi`](#multi)

_Last reviewed 2026-05-05 by macroforecast author._

### `n_outliers_flagged`  --  operational

Count of outlier-flagged cells per series.

L2.5.D cleaning effect view ``n_outliers_flagged``.

This option configures the ``cleaning_summary_view`` axis on the ``L2_5_D_cleaning_effect_summary`` sub-layer of L2.5; output is emitted under ``manifest.diagnostics/l2_5/L2_5_D_cleaning_effect_summary/`` alongside the other selected views.

**When to use**

Auditing outlier-handler aggressiveness; very high counts may indicate threshold mis-calibration.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`n_imputed_per_series`](#n-imputed-per-series), [`n_truncated_obs`](#n-truncated-obs), [`multi`](#multi)

_Last reviewed 2026-05-05 by macroforecast author._

### `n_truncated_obs`  --  operational

Count of observations dropped by L2.E frame-edge handling.

L2.5.D cleaning effect view ``n_truncated_obs``.

This option configures the ``cleaning_summary_view`` axis on the ``L2_5_D_cleaning_effect_summary`` sub-layer of L2.5; output is emitted under ``manifest.diagnostics/l2_5/L2_5_D_cleaning_effect_summary/`` alongside the other selected views.

**When to use**

Auditing edge truncation effects on the available sample size.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`n_imputed_per_series`](#n-imputed-per-series), [`n_outliers_flagged`](#n-outliers-flagged), [`multi`](#multi)

_Last reviewed 2026-05-05 by macroforecast author._
