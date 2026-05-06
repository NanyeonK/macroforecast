# `window_view`

[Back to L4.5](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``window_view`` on sub-layer ``L4_5_C_window_stability`` (layer ``l4_5``).

## Sub-layer

**L4_5_C_window_stability**

## Axis metadata

- Default: `'multi'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 5 option(s)
- Future: 0 option(s)

## Options

### `first_vs_last_window_forecast`  --  operational

First vs last training-window forecast overlay.

L4.5.C window view ``first_vs_last_window_forecast``.

This option configures the ``window_view`` axis on the ``L4_5_C_window_stability`` sub-layer of L4.5; output is emitted under ``manifest.diagnostics/l4_5/L4_5_C_window_stability/`` alongside the other selected views.

**When to use**

Quick window-instability check; large divergence flags non-stationarity.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`rolling_train_loss`](#rolling-train-loss), [`parameter_stability`](#parameter-stability), [`rolling_coef`](#rolling-coef), [`multi`](#multi)

_Last reviewed 2026-05-05 by macroforecast author._

### `multi`  --  operational

Render every window-stability view.

L4.5.C window view ``multi``.

This option configures the ``window_view`` axis on the ``L4_5_C_window_stability`` sub-layer of L4.5; output is emitted under ``manifest.diagnostics/l4_5/L4_5_C_window_stability/`` alongside the other selected views.

**When to use**

Comprehensive stability audit.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`rolling_train_loss`](#rolling-train-loss), [`parameter_stability`](#parameter-stability), [`rolling_coef`](#rolling-coef), [`first_vs_last_window_forecast`](#first-vs-last-window-forecast)

_Last reviewed 2026-05-05 by macroforecast author._

### `parameter_stability`  --  operational

Parameter (coefficient / depth) stability across windows.

L4.5.C window view ``parameter_stability``.

This option configures the ``window_view`` axis on the ``L4_5_C_window_stability`` sub-layer of L4.5; output is emitted under ``manifest.diagnostics/l4_5/L4_5_C_window_stability/`` alongside the other selected views.

**When to use**

Spotting structural instability in the fitted estimator.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`rolling_train_loss`](#rolling-train-loss), [`rolling_coef`](#rolling-coef), [`first_vs_last_window_forecast`](#first-vs-last-window-forecast), [`multi`](#multi)

_Last reviewed 2026-05-05 by macroforecast author._

### `rolling_coef`  --  operational

Coefficient values across rolling windows.

L4.5.C window view ``rolling_coef``.

This option configures the ``window_view`` axis on the ``L4_5_C_window_stability`` sub-layer of L4.5; output is emitted under ``manifest.diagnostics/l4_5/L4_5_C_window_stability/`` alongside the other selected views.

**When to use**

Linear-model coefficient drift detection; pair with the L7 ``mrf_gtvp`` for non-linear analogue.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`rolling_train_loss`](#rolling-train-loss), [`parameter_stability`](#parameter-stability), [`first_vs_last_window_forecast`](#first-vs-last-window-forecast), [`multi`](#multi)

_Last reviewed 2026-05-05 by macroforecast author._

### `rolling_train_loss`  --  operational

Training loss across rolling windows.

L4.5.C window view ``rolling_train_loss``.

This option configures the ``window_view`` axis on the ``L4_5_C_window_stability`` sub-layer of L4.5; output is emitted under ``manifest.diagnostics/l4_5/L4_5_C_window_stability/`` alongside the other selected views.

**When to use**

Detecting training instability; rising loss across windows flags drift.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`parameter_stability`](#parameter-stability), [`rolling_coef`](#rolling-coef), [`first_vs_last_window_forecast`](#first-vs-last-window-forecast), [`multi`](#multi)

_Last reviewed 2026-05-05 by macroforecast author._
