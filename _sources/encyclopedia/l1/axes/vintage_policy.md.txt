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

Real-time forecasting evaluations -- those need ALFRED vintages (future feature; see real_time_alfred).

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`real_time_alfred`](#real-time-alfred), [`information_set_type`](#information-set-type)

_Last reviewed 2026-05-16 by macroforecast author._

### `real_time_alfred`  --  future

Real-time ALFRED vintage policy (not yet implemented).

ALFRED (Archival FRED) is the St. Louis Fed's real-time data archive. It stores historical vintages of every FRED series, allowing researchers to reconstruct the information set that was actually available at any past date -- before subsequent data revisions occurred.

Future macroforecast support will pull the historical-as-of vintage for each forecast origin from the ALFRED API, enabling true real-time replication studies where the model never sees data that was not yet released at the forecast origin.

**Current behavior**: selecting ``real_time_alfred`` raises a hard ``ValueError`` at recipe validation with the message ``'real_time_alfred is not yet implemented; future feature. Use current_vintage (default).'`` (Cycle 14 K-4). No partial execution occurs.

**When to use**

Future. For now, use ``current_vintage`` and document the data-revision context via ``data_revision_tag`` in manifest provenance (Cycle 14 K-3 auto-captures ``fred-md@YYYY-MM``).

**When NOT to use**

Any current recipe -- this option is hard-rejected at validation in all released versions up to and including v0.9.x.

**References**

* Federal Reserve Bank of St. Louis, 'ALFRED: Archival Federal Reserve Economic Data' -- real-time vintage archive of FRED series. <https://alfred.stlouisfed.org/>
* Croushore & Stark (2001) 'A real-time data set for macroeconomists', Journal of Econometrics 105(1). (doi:10.1016/S0304-4076(01)00072-0)

**Related options**: [`current_vintage`](#current-vintage)

_Last reviewed 2026-05-16 by macroforecast author._
