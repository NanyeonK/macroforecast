# `weights_over_time_method`

[Back to L4.5](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``weights_over_time_method`` on sub-layer ``L4_5_E_ensemble_diagnostics`` (layer ``l4_5``).

## Sub-layer

**L4_5_E_ensemble_diagnostics**

## Axis metadata

- Default: `'stacked_area'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 3 option(s)
- Future: 0 option(s)

## Options

### `heatmap`  --  operational

Heatmap of weights (member × time).

L4.5.E weights-over-time rendering ``heatmap``.

This option configures the ``weights_over_time_method`` axis on the ``L4_5_E_ensemble_diagnostics`` sub-layer of L4.5; output is emitted under ``manifest.diagnostics/l4_5/L4_5_E_ensemble_diagnostics/`` alongside the other selected views.

**When to use**

Many-member ensembles (> 20) where line / area plots become unreadable.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`line_plot`](#line-plot), [`stacked_area`](#stacked-area)

_Last reviewed 2026-05-05 by macroforecast author._

### `line_plot`  --  operational

Line plot of weights per member over time.

L4.5.E weights-over-time rendering ``line_plot``.

This option configures the ``weights_over_time_method`` axis on the ``L4_5_E_ensemble_diagnostics`` sub-layer of L4.5; output is emitted under ``manifest.diagnostics/l4_5/L4_5_E_ensemble_diagnostics/`` alongside the other selected views.

**When to use**

Default reporting view; readable up to ~10 ensemble members.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`stacked_area`](#stacked-area), [`heatmap`](#heatmap)

_Last reviewed 2026-05-05 by macroforecast author._

### `stacked_area`  --  operational

Stacked-area plot summing to 1.

L4.5.E weights-over-time rendering ``stacked_area``.

This option configures the ``weights_over_time_method`` axis on the ``L4_5_E_ensemble_diagnostics`` sub-layer of L4.5; output is emitted under ``manifest.diagnostics/l4_5/L4_5_E_ensemble_diagnostics/`` alongside the other selected views.

**When to use**

Emphasising member share; ideal for showcasing weight redistribution events.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`line_plot`](#line-plot), [`heatmap`](#heatmap)

_Last reviewed 2026-05-05 by macroforecast author._
