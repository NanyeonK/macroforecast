# `transform_policy`

[Back to L2](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``transform_policy`` on sub-layer ``l2_b`` (layer ``l2``).

## Sub-layer

**l2_b**

## Axis metadata

- Default: `'apply_official_tcode'`
- Sweepable: True
- Status: operational

## Operational status summary

- Operational: 3 option(s)
- Future: 0 option(s)

## Options

### `apply_official_tcode`  --  operational

Apply McCracken-Ng's series-by-series stationarity transforms.

Each FRED-MD/QD series ships with a transformation code (1-7) mapping to a stationarity transform. ``apply_official_tcode`` runs the canonical mapping per series:

* 1 = level
* 2 = first difference
* 3 = second difference
* 4 = log
* 5 = first difference of log (≈ growth rate)
* 6 = second difference of log
* 7 = log diff of (1 + growth rate)

Applied per-origin within walk-forward to avoid look-ahead.

**When to use**

Default for FRED-based studies. Canonical replication path.

**References**

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'
* McCracken & Ng (2016) 'FRED-MD: A Monthly Database for Macroeconomic Research', JBES 34(4). (doi:10.1080/07350015.2015.1086655)

**Related options**: [`no_transform`](#no-transform), [`custom_tcode`](#custom-tcode), [`transform_scope`](#transform-scope)

_Last reviewed 2026-05-04 by macroforecast author._

### `no_transform`  --  operational

Skip transforms; pass raw levels through.

Useful for tree-based / ranking models that don't need stationarity, or for studies that apply alternative transforms in L3 (Hodrick-Prescott filter, Hamilton (2018) detrender, etc.).

**When to use**

Tree / forest models; alternative-transform studies; custom panels with already-transformed data.

**References**

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'

**Related options**: [`apply_official_tcode`](#apply-official-tcode), [`custom_tcode`](#custom-tcode)

_Last reviewed 2026-05-04 by macroforecast author._

### `custom_tcode`  --  operational

User-supplied per-series t-code map.

Requires ``leaf_config.custom_tcode_map: {series_name: int}``. Macrocast applies the same 1-7 mapping as ``apply_official_tcode`` but reads codes from the user's dict instead of the bundled FRED metadata. Useful for custom panels where the user wants the McCracken-Ng transform vocabulary.

**When to use**

Custom panels with user-defined stationarity codes.

**References**

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'

**Related options**: [`apply_official_tcode`](#apply-official-tcode), [`no_transform`](#no-transform)

_Last reviewed 2026-05-04 by macroforecast author._
