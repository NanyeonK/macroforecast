# `sample_start_rule`

[Back to L1](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``sample_start_rule`` on sub-layer ``l1_e`` (layer ``l1``).

## Sub-layer

**l1_e**

## Axis metadata

- Default: `'max_balanced'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 3 option(s)
- Future: 0 option(s)

## Options

### `earliest_available`  --  operational

Start at the panel's earliest date; tolerates leading missing values.

Keeps every row; lets the L1.C ``raw_missing_policy`` and L2 imputation handle leading NaNs. Useful when the L2 EM-factor imputer can recover early observations and dropping them would lose informative history.

**When to use**

Studies that want maximum sample length and trust L2 imputation to handle leading NaNs.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`max_balanced`](#max-balanced), [`fixed_date`](#fixed-date)

_Last reviewed 2026-05-04 by macroforecast author._

### `fixed_date`  --  operational

Pin the start date in leaf_config (e.g., 1985-01-01).

Requires ``leaf_config.sample_start_date`` (ISO date). The L1 loader trims to that date verbatim. Useful for replication scripts that need an exact sample window matching a published paper.

**When to use**

Replication scripts; ablation studies over alternative start dates.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`max_balanced`](#max-balanced), [`earliest_available`](#earliest-available)

_Last reviewed 2026-05-04 by macroforecast author._

### `max_balanced`  --  operational

Start at the first date where every requested series is observed.

Computes the latest first-observation date across every column in the panel and trims earlier rows. Guarantees a balanced panel without imputing leading missing values.

Default for studies that mix series with different start dates (common on FRED-MD because some series only begin in the 1980s).

**When to use**

Default for FRED-MD/QD studies with mixed start dates.

**When NOT to use**

Custom panels where every series shares the same start date (use ``earliest_available`` to keep all rows).

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`earliest_available`](#earliest-available), [`fixed_date`](#fixed-date)

_Last reviewed 2026-05-04 by macroforecast author._
