# `information_set_type`

[Back to L1](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``information_set_type`` on sub-layer ``l1_a`` (layer ``l1``).

## Sub-layer

**l1_a**

## Axis metadata

- Default: `'final_revised_data'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 2 option(s)
- Future: 0 option(s)

## Options

### `final_revised_data`  --  operational

Each origin sees fully revised data; standard pseudo-OOS protocol.

At every walk-forward origin, the model has access to the *current* revised values for every observation up to that origin. This is the standard pseudo-out-of-sample protocol used by McCracken-Ng and most published forecasting comparisons.

Pros: simple, comparable across studies, no real-time data dep. Cons: optimistic about real-time forecast performance because later revisions correct early-vintage measurement error.

**When to use**

Default for any benchmark study; comparable to published work.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`pseudo_oos_on_revised_data`](#pseudo-oos-on-revised-data), [`vintage_policy`](#vintage-policy)

_Last reviewed 2026-05-04 by macroforecast author._

### `pseudo_oos_on_revised_data`  --  operational

Pseudo-OOS with revised data -- equivalent to final_revised_data for v1.0.

Synonym for ``final_revised_data`` in v1.0 (no ALFRED vintage tracking yet). Both options produce identical forecasts; the axis is exposed so future versions can route real-time vintages without breaking existing recipes.

**When to use**

When the recipe wants to make the pseudo-OOS protocol explicit (e.g., for clarity in published replication scripts).

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`final_revised_data`](#final-revised-data)

_Last reviewed 2026-05-04 by macroforecast author._
