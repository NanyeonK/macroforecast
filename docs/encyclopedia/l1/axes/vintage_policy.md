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
- Future: 1 option(s)

## Options

### `current_vintage`  --  operational

Use the latest available vintage of the dataset.

Loads the most recent FRED-MD/QD/SD snapshot bundled with the package. No real-time vintage tracking; revisions that happened after the snapshot date are not reflected.

This is the only operational option in v1.0. Real-time vintages (ALFRED-style) are tracked as a future axis -- see GitHub issues #XXX.

**When to use**

Default for any pseudo-out-of-sample study using revised data.

**When NOT to use**

Real-time forecasting evaluations -- those need ALFRED vintages.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`information_set_type`](#information-set-type)

_Last reviewed 2026-05-04 by macroforecast author._

### `real_time_alfred`  --  future

_(no schema description for `real_time_alfred`)_

> TBD: option doc not yet authored for this value. The encyclopedia falls back to the bare schema description above. PRs adding a full ``OptionDoc`` entry under ``macroforecast/scaffold/option_docs/l1.py`` are welcome.
