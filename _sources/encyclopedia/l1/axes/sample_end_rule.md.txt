# `sample_end_rule`

[Back to L1](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``sample_end_rule`` on sub-layer ``l1_e`` (layer ``l1``).

## Sub-layer

**l1_e**

## Axis metadata

- Default: `'latest_available'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 2 option(s)
- Future: 0 option(s)

## Options

### `latest_available`  --  operational

End at the panel's last date.

Default. Uses the most recent observation in the bundled vintage.

Configures the ``sample_end_rule`` axis on ``l1_e`` (layer ``l1``); the ``latest_available`` value is materialised in the recipe's ``fixed_axes`` block under that sub-layer.

**When to use**

Default. Studies that want to use the full available history.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`fixed_date`](#fixed-date)

_Last reviewed 2026-05-04 by macroforecast author._

### `fixed_date`  --  operational

Pin the end date in leaf_config (e.g., 2019-12-31).

Requires ``leaf_config.sample_end_date``. Useful for pseudo-out-of-sample evaluation where the recipe wants to exclude the COVID window or stop at a paper's reported sample.

**When to use**

Pre-COVID benchmark studies; matching a paper's reported sample window.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`latest_available`](#latest-available)

_Last reviewed 2026-05-04 by macroforecast author._
