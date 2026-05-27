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

See [apply_official_tcode function page](../transform_policy/apply_official_tcode.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.apply_tcode_transform``.

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
