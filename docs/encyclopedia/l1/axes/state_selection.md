# `state_selection`

[Back to L1](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``state_selection`` on sub-layer ``l1_d`` (layer ``l1``).

## Sub-layer

**l1_d**

## Axis metadata

- Default: `'all_states'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 2 option(s)
- Future: 0 option(s)

## Options

### `all_states`  --  operational

Auto-select every state in ``fred_sd_state_group``.

Skips per-state cherry-picking; uses the full set defined by the active state group.

**When to use**

Default; state-group already does the filtering.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`selected_states`](#selected-states)

_Last reviewed 2026-05-05 by macroforecast author._

### `selected_states`  --  operational

Use the explicit per-state list in leaf_config.

Reads ``leaf_config.selected_states`` -- a subset of the active state-group, allowing fine-grained control.

**When to use**

Custom regional studies that need a non-standard state subset.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`all_states`](#all-states)

_Last reviewed 2026-05-05 by macroforecast author._
