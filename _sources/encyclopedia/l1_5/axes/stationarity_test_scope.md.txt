# `stationarity_test_scope`

[Back to L1.5](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``stationarity_test_scope`` on sub-layer ``L1_5_C_stationarity_tests`` (layer ``l1_5``).

## Sub-layer

**L1_5_C_stationarity_tests**

## Axis metadata

- Default: `'target_and_predictors'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 3 option(s)
- Future: 0 option(s)

## Options

### `predictors_only`  --  operational

Run stationarity tests on predictor columns only.

Restricts L1.5.C tests to the ``predictors_only`` column subset.

This option configures the ``stationarity_test_scope`` axis on the ``L1_5_C_stationarity_tests`` sub-layer of L1.5; output is emitted under ``manifest.diagnostics/l1_5/L1_5_C_stationarity_tests/`` alongside the other selected views.

**When to use**

When the target is known stationary (growth rates, returns) and only the predictors are uncertain.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`target_only`](#target-only), [`target_and_predictors`](#target-and-predictors)

_Last reviewed 2026-05-05 by macroforecast author._

### `target_and_predictors`  --  operational

Run stationarity tests on every column.

Restricts L1.5.C tests to the ``target_and_predictors`` column subset.

This option configures the ``stationarity_test_scope`` axis on the ``L1_5_C_stationarity_tests`` sub-layer of L1.5; output is emitted under ``manifest.diagnostics/l1_5/L1_5_C_stationarity_tests/`` alongside the other selected views.

**When to use**

Default; comprehensive audit that catches surprises in either target or predictors.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`predictors_only`](#predictors-only), [`target_only`](#target-only)

_Last reviewed 2026-05-05 by macroforecast author._

### `target_only`  --  operational

Run stationarity tests on the target column only.

Restricts L1.5.C tests to the ``target_only`` column subset.

This option configures the ``stationarity_test_scope`` axis on the ``L1_5_C_stationarity_tests`` sub-layer of L1.5; output is emitted under ``manifest.diagnostics/l1_5/L1_5_C_stationarity_tests/`` alongside the other selected views.

**When to use**

Confirming the L2-applied tcode rendered the target stationary.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`predictors_only`](#predictors-only), [`target_and_predictors`](#target-and-predictors)

_Last reviewed 2026-05-05 by macroforecast author._
