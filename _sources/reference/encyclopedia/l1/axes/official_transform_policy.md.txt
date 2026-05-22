# `official_transform_policy`

[Back to L1](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``official_transform_policy`` on sub-layer ``l1_c`` (layer ``l1``).

## Sub-layer

**l1_c**

## Axis metadata

- Default: `'apply_official_tcode'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 2 option(s)
- Future: 0 option(s)

## Options

### `apply_official_tcode`  --  operational

Apply McCracken-Ng's series-by-series stationarity transforms.

Each FRED-MD/QD series ships with a transformation code (t-code) 1-7 that maps to a stationarity transform: 1=level, 2=Δlevel, 5=Δlog, 6=Δ²log, etc. ``apply_official_tcode`` runs the canonical transform per series so downstream estimators see stationary inputs.

This is the canonical preprocessing path for the McCracken-Ng benchmark family. Every published replication on FRED-MD/QD uses it.

**When to use**

Default for FRED-MD/QD studies. Canonical replication path.

**When NOT to use**

Studies that want to compare alternative transform schemes (use ``keep_official_raw_scale`` and apply transforms in L2 manually).

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'
* McCracken & Ng (2016) 'FRED-MD: A Monthly Database for Macroeconomic Research', Journal of Business & Economic Statistics 34(4). (doi:10.1080/07350015.2015.1086655)

**Related options**: [`keep_official_raw_scale`](#keep-official-raw-scale), [`official_transform_scope`](#official-transform-scope)

_Last reviewed 2026-05-04 by macroforecast author._

### `keep_official_raw_scale`  --  operational

Skip the canonical t-codes; keep raw level data.

Series stay on their native scale (levels, ratios, indices). Useful for tree-based models that don't need stationarity, or for studies that apply alternative transforms in L2 / L3.

**When to use**

Tree / forest models that don't require stationarity; alternative-transform studies.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`apply_official_tcode`](#apply-official-tcode)

_Last reviewed 2026-05-04 by macroforecast author._
