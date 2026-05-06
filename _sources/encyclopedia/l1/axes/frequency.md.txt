# `frequency`

[Back to L1](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``frequency`` on sub-layer ``l1_a`` (layer ``l1``).

## Sub-layer

**l1_a**

## Axis metadata

- Default: `'derived'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 2 option(s)
- Future: 0 option(s)

## Options

### `monthly`  --  operational

Sample at monthly cadence; pairs with FRED-MD / monthly custom panels.

Sets the canonical sampling frequency to monthly. Affects horizon resolution (1 = one month ahead), L2 frequency-alignment rules (only applicable when datasets mix), and the ``standard_md`` horizon set.

The default is ``derived``: macroforecast infers the frequency from ``dataset`` (fred_md → monthly, fred_qd → quarterly). Setting frequency explicitly is required for custom panels.

**When to use**

Custom panels with monthly observations; explicit override of the FRED-MD default for clarity.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`quarterly`](#quarterly), [`dataset`](#dataset), [`horizon_set`](#horizon-set)

_Last reviewed 2026-05-04 by macroforecast author._

### `quarterly`  --  operational

Sample at quarterly cadence; pairs with FRED-QD / quarterly custom panels.

Sets the sampling frequency to quarterly. Activates the ``standard_qd`` horizon set (h ∈ {1, 2, 4, 8} quarters) and monthly→quarterly aggregation rules in L2.A when the panel mixes frequencies.

**When to use**

GDP / NIPA-style targets; quarterly custom panels; FRED-QD-based studies.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`monthly`](#monthly), [`dataset`](#dataset), [`horizon_set`](#horizon-set)

_Last reviewed 2026-05-04 by macroforecast author._
