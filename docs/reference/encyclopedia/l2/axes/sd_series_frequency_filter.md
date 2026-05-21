# `sd_series_frequency_filter`

[Back to L2](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``sd_series_frequency_filter`` on sub-layer ``l2_a`` (layer ``l2``).

## Sub-layer

**l2_a**

## Axis metadata

- Default: `'both'`
- Sweepable: True
- Status: operational

## Operational status summary

- Operational: 3 option(s)
- Future: 0 option(s)

## Options

### `monthly_only`  --  operational

Drop quarterly series; retain monthly FRED-SD only.

Filter applied before any L2.A frequency-alignment rule. Useful when the user wants a strictly monthly FRED-SD panel and would prefer dropping the quarterly variables to keeping them and accepting the alignment rule's interpolation.

**When to use**

Strict monthly panels; avoiding quarterly-to-monthly interpolation.

**When NOT to use**

When quarterly variables are central to the analysis.

**References**

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'

**Related options**: [`quarterly_only`](#quarterly-only), [`both`](#both)

_Last reviewed 2026-05-04 by macroforecast author._

### `quarterly_only`  --  operational

Drop monthly series; retain quarterly FRED-SD only.

Inverse of ``monthly_only``: keeps quarterly variables and drops monthly variables. Used when comparing to a quarterly benchmark (e.g. real GDP) and monthly variables would inflate the panel without contributing forecast skill.

**When to use**

Quarterly-target studies; FRED-QD style analyses on FRED-SD data.

**References**

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'

**Related options**: [`monthly_only`](#monthly-only), [`both`](#both)

_Last reviewed 2026-05-04 by macroforecast author._

### `both`  --  operational

Keep both monthly and quarterly FRED-SD series.

Default; defers to the L2.A frequency-alignment rules (``monthly_to_quarterly_rule`` / ``quarterly_to_monthly_rule``) to render the mixed-frequency panel into a single grid.

**When to use**

Default for FRED-SD recipes; mixed-frequency panels.

**References**

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'

**Related options**: [`monthly_only`](#monthly-only), [`quarterly_only`](#quarterly-only)

_Last reviewed 2026-05-04 by macroforecast author._
