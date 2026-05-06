# `forecast_scale_view`

[Back to L4.5](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``forecast_scale_view`` on sub-layer ``L4_5_B_forecast_scale_view`` (layer ``l4_5``).

## Sub-layer

**L4_5_B_forecast_scale_view**

## Axis metadata

- Default: `'both_overlay'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 3 option(s)
- Future: 0 option(s)

## Options

### `back_transformed_only`  --  operational

Plot forecasts back-transformed to the target's level.

L4.5.B forecast scale view ``back_transformed_only``.

This option configures the ``forecast_scale_view`` axis on the ``L4_5_B_forecast_scale_view`` sub-layer of L4.5; output is emitted under ``manifest.diagnostics/l4_5/L4_5_B_forecast_scale_view/`` alongside the other selected views.

**When to use**

Default reporting view; matches the audience's mental model of the target.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`transformed_only`](#transformed-only), [`both_overlay`](#both-overlay)

_Last reviewed 2026-05-05 by macroforecast author._

### `both_overlay`  --  operational

Overlay both scales in side-by-side panels.

L4.5.B forecast scale view ``both_overlay``.

This option configures the ``forecast_scale_view`` axis on the ``L4_5_B_forecast_scale_view`` sub-layer of L4.5; output is emitted under ``manifest.diagnostics/l4_5/L4_5_B_forecast_scale_view/`` alongside the other selected views.

**When to use**

Comparing transformed vs level forecasts; useful when transformation effects matter.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`transformed_only`](#transformed-only), [`back_transformed_only`](#back-transformed-only)

_Last reviewed 2026-05-05 by macroforecast author._

### `transformed_only`  --  operational

Plot forecasts in the transformed (model-internal) scale.

L4.5.B forecast scale view ``transformed_only``.

This option configures the ``forecast_scale_view`` axis on the ``L4_5_B_forecast_scale_view`` sub-layer of L4.5; output is emitted under ``manifest.diagnostics/l4_5/L4_5_B_forecast_scale_view/`` alongside the other selected views.

**When to use**

Inspecting model-native predictions; useful for diagnosing fit issues at the estimator's scale.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`back_transformed_only`](#back-transformed-only), [`both_overlay`](#both-overlay)

_Last reviewed 2026-05-05 by macroforecast author._
