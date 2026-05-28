# `dfm_diagnostics`

[Back to L3.5](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``dfm_diagnostics`` on sub-layer ``L3_5_B_factor_block_inspection`` (layer ``l3_5``).

## Sub-layer

**L3_5_B_factor_block_inspection**

## Axis metadata

- Default: `'multi'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 4 option(s)
- Future: 0 option(s)

## Options

### `factor_var_stability`  --  operational

Plot of DFM factor-VAR coefficient stability over time.

L3.5.B DFM diagnostic ``factor_var_stability``.

This option configures the ``dfm_diagnostics`` axis on the ``L3_5_B_factor_block_inspection`` sub-layer of L3.5; output is emitted under ``manifest.diagnostics/l3_5/L3_5_B_factor_block_inspection/`` alongside the other selected views.

**When to use**

Detecting non-stationarity in the factor dynamics; rolling-window estimates flag breaks.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'
* Mariano & Murasawa (2003) 'A new coincident index of business cycles based on monthly and quarterly series', JAE 18(4): 427-443.

**Related options**: [`idiosyncratic_acf`](#idiosyncratic-acf), [`multi`](#multi), [`none`](#none)

_Last reviewed 2026-05-05 by macroforecast author._

### `idiosyncratic_acf`  --  operational

Autocorrelation of DFM idiosyncratic residuals.

L3.5.B DFM diagnostic ``idiosyncratic_acf``.

This option configures the ``dfm_diagnostics`` axis on the ``L3_5_B_factor_block_inspection`` sub-layer of L3.5; output is emitted under ``manifest.diagnostics/l3_5/L3_5_B_factor_block_inspection/`` alongside the other selected views.

**When to use**

Validating the idiosyncratic-AR(1) assumption; large residual ACF at lags > 1 indicates misspecification.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'
* Mariano & Murasawa (2003) 'A new coincident index of business cycles based on monthly and quarterly series', JAE 18(4): 427-443.

**Related options**: [`factor_var_stability`](#factor-var-stability), [`multi`](#multi), [`none`](#none)

_Last reviewed 2026-05-05 by macroforecast author._

### `multi`  --  operational

Render both DFM diagnostics together.

L3.5.B DFM diagnostic ``multi``.

This option configures the ``dfm_diagnostics`` axis on the ``L3_5_B_factor_block_inspection`` sub-layer of L3.5; output is emitted under ``manifest.diagnostics/l3_5/L3_5_B_factor_block_inspection/`` alongside the other selected views.

**When to use**

Comprehensive DFM validation; recommended after any DFM fit.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'
* Mariano & Murasawa (2003) 'A new coincident index of business cycles based on monthly and quarterly series', JAE 18(4): 427-443.

**Related options**: [`factor_var_stability`](#factor-var-stability), [`idiosyncratic_acf`](#idiosyncratic-acf), [`none`](#none)

_Last reviewed 2026-05-05 by macroforecast author._

### `none`  --  operational

Skip DFM-specific diagnostics.

L3.5.B DFM diagnostic ``none``.

This option configures the ``dfm_diagnostics`` axis on the ``L3_5_B_factor_block_inspection`` sub-layer of L3.5; output is emitted under ``manifest.diagnostics/l3_5/L3_5_B_factor_block_inspection/`` alongside the other selected views.

**When to use**

Pipelines without DFM blocks (PCA-only or no-factor pipelines).

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'
* Mariano & Murasawa (2003) 'A new coincident index of business cycles based on monthly and quarterly series', JAE 18(4): 427-443.

**Related options**: [`factor_var_stability`](#factor-var-stability), [`idiosyncratic_acf`](#idiosyncratic-acf), [`multi`](#multi)

_Last reviewed 2026-05-05 by macroforecast author._
