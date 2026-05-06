# `monthly_to_quarterly_rule`

[Back to L2](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``monthly_to_quarterly_rule`` on sub-layer ``l2_a`` (layer ``l2``).

## Sub-layer

**l2_a**

## Axis metadata

- Default: `'quarterly_average'`
- Sweepable: True
- Status: operational

## Operational status summary

- Operational: 3 option(s)
- Future: 0 option(s)

## Options

### `quarterly_average`  --  operational

Aggregate to quarterly via mean of the three monthly observations.

Standard NIPA aggregation for stocks / averages.

Configures the ``monthly_to_quarterly_rule`` axis on ``l2_a`` (layer ``l2``); the ``quarterly_average`` value is materialised in the recipe's ``fixed_axes`` block under that sub-layer.

**When to use**

Default. Stock variables (interest rates, prices, employment levels).

**References**

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'

**Related options**: [`quarterly_endpoint`](#quarterly-endpoint), [`quarterly_sum`](#quarterly-sum)

_Last reviewed 2026-05-04 by macroforecast author._

### `quarterly_endpoint`  --  operational

Aggregate via the end-of-quarter observation.

Use for series that snap to a quarter-end (e.g., balance-sheet data).

Configures the ``monthly_to_quarterly_rule`` axis on ``l2_a`` (layer ``l2``); the ``quarterly_endpoint`` value is materialised in the recipe's ``fixed_axes`` block under that sub-layer.

**When to use**

End-of-period stocks (M2 month-end, balance-sheet series).

**References**

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'

**Related options**: [`quarterly_average`](#quarterly-average), [`quarterly_sum`](#quarterly-sum)

_Last reviewed 2026-05-04 by macroforecast author._

### `quarterly_sum`  --  operational

Aggregate via the sum of the three monthly observations.

Standard for flow variables (production, sales, payroll growth).

Configures the ``monthly_to_quarterly_rule`` axis on ``l2_a`` (layer ``l2``); the ``quarterly_sum`` value is materialised in the recipe's ``fixed_axes`` block under that sub-layer.

**When to use**

Flow variables; cumulative-quantity series.

**References**

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'

**Related options**: [`quarterly_average`](#quarterly-average), [`quarterly_endpoint`](#quarterly-endpoint)

_Last reviewed 2026-05-04 by macroforecast author._
