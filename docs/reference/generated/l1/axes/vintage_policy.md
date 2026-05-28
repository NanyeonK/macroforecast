# `vintage_policy`

[Back to L1](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``vintage_policy`` on sub-layer ``l1_a`` (layer ``l1``).

## Sub-layer

**l1_a**

## Axis metadata

- Default: `'current_vintage'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 1 option(s)
- Future: 0 option(s)

## Options

### `current_vintage`  --  operational

Use the latest available FRED-MD, FRED-QD, or FRED-SD vintage exposed by the public data loaders.

This is the only supported vintage policy in the current simplified data API. Real-time vintage correction is not part of the simplified data layer and can be redesigned later as a separate wrapper or evaluation feature.

**When to use**

Default for pseudo-out-of-sample studies using revised data.

**When NOT to use**

Do not use this option when a study requires a real-time data revision design; that workflow is not currently implemented in the simplified data layer.

**References**

* macroforecast design Part 1, L1: data definition pins source, target, geography, and horizon choices.

_Last reviewed 2026-05-28 by macroforecast author._
