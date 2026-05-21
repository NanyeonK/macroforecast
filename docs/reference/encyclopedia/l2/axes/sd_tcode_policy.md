# `sd_tcode_policy`

[Back to L2](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``sd_tcode_policy`` on sub-layer ``l2_b`` (layer ``l2``).

## Sub-layer

**l2_b**

## Axis metadata

- Default: `'none'`
- Sweepable: True
- Status: operational

## Operational status summary

- Operational: 3 option(s)
- Future: 0 option(s)

## Options

### `none`  --  operational

Default: no FRED-SD-specific t-code applied.

FRED-SD does not publish official transformation codes. The default ``none`` policy leaves FRED-SD source values as published and applies whatever ``transform_policy`` the user selected (default ``apply_official_tcode`` only operates on FRED-MD/QD columns; FRED-SD columns pass through). Use this option whenever the study does not depend on a particular FRED-SD stationarity transform.

**When to use**

Default for FRED-SD recipes; canonical baseline.

**References**

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'

**Related options**: [`inferred`](#inferred), [`empirical`](#empirical)

_Last reviewed 2026-05-04 by macroforecast author._

### `inferred`  --  operational

Apply the inferred SD t-code map (national-analog research layer).

Opt-in: applies the package-shipped inferred t-code map for FRED-SD columns. The map is derived by taking the FRED-MD/QD national analog of each FRED-SD variable and inheriting that analog's published t-code. The manifest records ``data_reports.sd_inferred_tcodes`` with ``official: false``, the map version, and the allowed review statuses.

**When to use**

Studies that want a published (but non-official) t-code path; replication of national-analog research.

**References**

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'

**Related options**: [`none`](#none), [`empirical`](#empirical)

_Last reviewed 2026-05-04 by macroforecast author._

### `empirical`  --  operational

Apply the empirical stationarity-audit t-code map.

Opt-in: applies an empirical t-code map derived from a stationarity audit of the FRED-SD panel. Two ``unit`` modes:

* ``variable_global`` -- one t-code per FRED-SD variable,   shared across states
* ``state_series`` -- one t-code per (variable, state) pair;   requires ``leaf_config.sd_tcode_code_map`` and   ``sd_tcode_audit_uri``

The manifest records ``official: false`` plus the audit URI and chosen unit.

**When to use**

Stationarity-audit driven research; per-state t-code policies.

**References**

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'

**Related options**: [`none`](#none), [`inferred`](#inferred)

_Last reviewed 2026-05-04 by macroforecast author._
