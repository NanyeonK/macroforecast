# `back_transform_method`

[Back to L4.5](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``back_transform_method`` on sub-layer ``L4_5_B_forecast_scale_view`` (layer ``l4_5``).

## Sub-layer

**L4_5_B_forecast_scale_view**

## Axis metadata

- Default: `'auto'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 2 option(s)
- Future: 0 option(s)

## Options

### `auto`  --  operational

Use the inverse of L2.B / L3 transforms automatically.

L4.5.B back-transform method ``auto``.

This option configures the ``back_transform_method`` axis on the ``L4_5_B_forecast_scale_view`` sub-layer of L4.5; output is emitted under ``manifest.diagnostics/l4_5/L4_5_B_forecast_scale_view/`` alongside the other selected views.

**When to use**

Default; works for the standard log_diff / pct_change pipeline.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`manual_function`](#manual-function)

_Last reviewed 2026-05-05 by macroforecast author._

### `manual_function`  --  operational

Use a user-registered inverse function.

L4.5.B back-transform method ``manual_function``.

This option configures the ``back_transform_method`` axis on the ``L4_5_B_forecast_scale_view`` sub-layer of L4.5; output is emitted under ``manifest.diagnostics/l4_5/L4_5_B_forecast_scale_view/`` alongside the other selected views.

**When to use**

Custom target transforms registered via ``macroforecast.custom.register_target_transformer``.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`auto`](#auto)

_Last reviewed 2026-05-05 by macroforecast author._
