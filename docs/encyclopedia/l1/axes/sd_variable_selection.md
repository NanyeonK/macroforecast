# `sd_variable_selection`

[Back to L1](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``sd_variable_selection`` on sub-layer ``l1_d`` (layer ``l1``).

## Sub-layer

**l1_d**

## Axis metadata

- Default: `'all_sd_variables'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 2 option(s)
- Future: 0 option(s)

## Options

### `all_sd_variables`  --  operational

Auto-select every variable in ``fred_sd_variable_group``.

Default; uses the full set defined by the active variable group.

Configures the ``sd_variable_selection`` axis on ``l1_d`` (layer ``l1``); the ``all_sd_variables`` value is materialised in the recipe's ``fixed_axes`` block under that sub-layer.

**When to use**

Default broad-spectrum study. Selecting ``all_sd_variables`` on ``l1.sd_variable_selection`` activates this branch of the layer's runtime.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`selected_sd_variables`](#selected-sd-variables)

_Last reviewed 2026-05-05 by macroforecast author._

### `selected_sd_variables`  --  operational

Use the explicit per-series list in leaf_config.

Reads ``leaf_config.selected_sd_variables`` -- a subset of the active variable group.

**When to use**

Targeted studies that focus on specific FRED-SD series.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`all_sd_variables`](#all-sd-variables)

_Last reviewed 2026-05-05 by macroforecast author._
