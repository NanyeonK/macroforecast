# `outlier_view`

[Back to L1.5](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``outlier_view`` on sub-layer ``L1_5_D_missing_outlier_audit`` (layer ``l1_5``).

## Sub-layer

**L1_5_D_missing_outlier_audit**

## Axis metadata

- Default: `'iqr_flag'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 4 option(s)
- Future: 0 option(s)

## Options

### `iqr_flag`  --  operational

IQR-rule outlier flag per series (Tukey 1977).

L1.5.D outlier visualisation ``iqr_flag``.

This option configures the ``outlier_view`` axis on the ``L1_5_D_missing_outlier_audit`` sub-layer of L1.5; output is emitted under ``manifest.diagnostics/l1_5/L1_5_D_missing_outlier_audit/`` alongside the other selected views.

**When to use**

Robust to non-Gaussian distributions; flags values outside ``[Q1 - 1.5·IQR, Q3 + 1.5·IQR]``.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'
* Tukey (1977) 'Exploratory Data Analysis', Addison-Wesley.

**Related options**: [`zscore_flag`](#zscore-flag), [`multi`](#multi), [`none`](#none)

_Last reviewed 2026-05-05 by macroforecast author._

### `multi`  --  operational

Produce both IQR and z-score outlier flags.

L1.5.D outlier visualisation ``multi``.

This option configures the ``outlier_view`` axis on the ``L1_5_D_missing_outlier_audit`` sub-layer of L1.5; output is emitted under ``manifest.diagnostics/l1_5/L1_5_D_missing_outlier_audit/`` alongside the other selected views.

**When to use**

Cross-checking outlier counts across criteria; agreement strengthens the flag.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`iqr_flag`](#iqr-flag), [`zscore_flag`](#zscore-flag), [`none`](#none)

_Last reviewed 2026-05-05 by macroforecast author._

### `none`  --  operational

Skip outlier flagging.

L1.5.D outlier visualisation ``none``.

This option configures the ``outlier_view`` axis on the ``L1_5_D_missing_outlier_audit`` sub-layer of L1.5; output is emitted under ``manifest.diagnostics/l1_5/L1_5_D_missing_outlier_audit/`` alongside the other selected views.

**When to use**

Pre-cleaned panels where L2.C will not run; reducing diagnostic surface.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`iqr_flag`](#iqr-flag), [`zscore_flag`](#zscore-flag), [`multi`](#multi)

_Last reviewed 2026-05-05 by macroforecast author._

### `zscore_flag`  --  operational

``|z-score|`` > 3 outlier flag per series.

L1.5.D outlier visualisation ``zscore_flag``.

This option configures the ``outlier_view`` axis on the ``L1_5_D_missing_outlier_audit`` sub-layer of L1.5; output is emitted under ``manifest.diagnostics/l1_5/L1_5_D_missing_outlier_audit/`` alongside the other selected views.

**When to use**

Cheaper than IQR; assumes approximate normality. The 3σ threshold maps to ~0.3% tail probability under normality.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`iqr_flag`](#iqr-flag), [`multi`](#multi), [`none`](#none)

_Last reviewed 2026-05-05 by macroforecast author._
