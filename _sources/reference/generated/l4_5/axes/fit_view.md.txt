# `fit_view`

[Back to L4.5](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``fit_view`` on sub-layer ``L4_5_A_in_sample_fit`` (layer ``l4_5``).

## Sub-layer

**L4_5_A_in_sample_fit**

## Axis metadata

- Default: `'multi'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 5 option(s)
- Future: 0 option(s)

## Options

### `fitted_vs_actual`  --  operational

In-sample fitted vs actual scatter / time-series.

L4.5.A fit view ``fitted_vs_actual``.

This option configures the ``fit_view`` axis on the ``L4_5_A_in_sample_fit`` sub-layer of L4.5; output is emitted under ``manifest.diagnostics/l4_5/L4_5_A_in_sample_fit/`` alongside the other selected views.

**When to use**

Default fit visualisation; the most intuitive sanity check.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`residual_acf`](#residual-acf), [`residual_qq`](#residual-qq), [`residual_time`](#residual-time), [`multi`](#multi)

_Last reviewed 2026-05-05 by macroforecast author._

### `multi`  --  operational

Render all four fit views together.

L4.5.A fit view ``multi``.

This option configures the ``fit_view`` axis on the ``L4_5_A_in_sample_fit`` sub-layer of L4.5; output is emitted under ``manifest.diagnostics/l4_5/L4_5_A_in_sample_fit/`` alongside the other selected views.

**When to use**

Comprehensive in-sample audit.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`fitted_vs_actual`](#fitted-vs-actual), [`residual_acf`](#residual-acf), [`residual_qq`](#residual-qq), [`residual_time`](#residual-time)

_Last reviewed 2026-05-05 by macroforecast author._

### `residual_acf`  --  operational

ACF plot of in-sample residuals.

L4.5.A fit view ``residual_acf``.

This option configures the ``fit_view`` axis on the ``L4_5_A_in_sample_fit`` sub-layer of L4.5; output is emitted under ``manifest.diagnostics/l4_5/L4_5_A_in_sample_fit/`` alongside the other selected views.

**When to use**

Detecting residual autocorrelation -- significant ACF flags model misspecification.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`fitted_vs_actual`](#fitted-vs-actual), [`residual_qq`](#residual-qq), [`residual_time`](#residual-time), [`multi`](#multi)

_Last reviewed 2026-05-05 by macroforecast author._

### `residual_qq`  --  operational

QQ plot of in-sample residuals against the normal distribution.

L4.5.A fit view ``residual_qq``.

This option configures the ``fit_view`` axis on the ``L4_5_A_in_sample_fit`` sub-layer of L4.5; output is emitted under ``manifest.diagnostics/l4_5/L4_5_A_in_sample_fit/`` alongside the other selected views.

**When to use**

Validating Gaussianity assumption; deviations in the tails motivate robust losses or interval forecasts.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`fitted_vs_actual`](#fitted-vs-actual), [`residual_acf`](#residual-acf), [`residual_time`](#residual-time), [`multi`](#multi)

_Last reviewed 2026-05-05 by macroforecast author._

### `residual_time`  --  operational

Residual time-series plot.

L4.5.A fit view ``residual_time``.

This option configures the ``fit_view`` axis on the ``L4_5_A_in_sample_fit`` sub-layer of L4.5; output is emitted under ``manifest.diagnostics/l4_5/L4_5_A_in_sample_fit/`` alongside the other selected views.

**When to use**

Spotting heteroscedasticity / structural breaks; clusters of large residuals flag regime shifts.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`fitted_vs_actual`](#fitted-vs-actual), [`residual_acf`](#residual-acf), [`residual_qq`](#residual-qq), [`multi`](#multi)

_Last reviewed 2026-05-05 by macroforecast author._
