# `marx_view`

[Back to L3.5](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``marx_view`` on sub-layer ``L3_5_D_lag_block_inspection`` (layer ``l3_5``).

## Sub-layer

**L3_5_D_lag_block_inspection**

## Axis metadata

- Default: `'weight_decay_visualization'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 2 option(s)
- Future: 0 option(s)

## Options

### `none`  --  operational

Skip MARX visualisations.

L3.5.D MARX view ``none``.

This option configures the ``marx_view`` axis on the ``L3_5_D_lag_block_inspection`` sub-layer of L3.5; output is emitted under ``manifest.diagnostics/l3_5/L3_5_D_lag_block_inspection/`` alongside the other selected views.

**When to use**

Pipelines without MARX blocks.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'
* Coulombe (2024) 'The Macroeconomic Random Forest', Journal of Applied Econometrics 39(7): 1190-1209.

**Related options**: [`weight_decay_visualization`](#weight-decay-visualization)

_Last reviewed 2026-05-05 by macroforecast author._

### `weight_decay_visualization`  --  operational

Plot MARX weight decay across windows.

L3.5.D MARX view ``weight_decay_visualization``.

This option configures the ``marx_view`` axis on the ``L3_5_D_lag_block_inspection`` sub-layer of L3.5; output is emitted under ``manifest.diagnostics/l3_5/L3_5_D_lag_block_inspection/`` alongside the other selected views.

**When to use**

MARX-specific block diagnostic; shows the decay shape of the multi-scale moving averages.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'
* Coulombe (2024) 'The Macroeconomic Random Forest', Journal of Applied Econometrics 39(7): 1190-1209.

**Related options**: [`none`](#none)

_Last reviewed 2026-05-05 by macroforecast author._
