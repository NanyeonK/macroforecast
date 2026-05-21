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

- Operational: 2 option(s)
- Future: 0 option(s)

## Options

### `current_vintage`  --  operational

Use the latest available vintage of the dataset.

Loads the most recent FRED-MD/QD/SD snapshot bundled with the package. No real-time vintage tracking; revisions that happened after the snapshot date are not reflected.

This is the only operational option in v1.0. Real-time vintages (ALFRED-style) are tracked as a future axis -- see GitHub issues #XXX.

**When to use**

Default for any pseudo-out-of-sample study using revised data.

**When NOT to use**

Real-time forecasting evaluations -- those need ALFRED vintages; use real_time_alfred (operational since Cycle 50).

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`real_time_alfred`](#real-time-alfred), [`information_set_type`](#information-set-type)

_Last reviewed 2026-05-16 by macroforecast author._

### `real_time_alfred`  --  operational

Real-time ALFRED vintage policy: uses historical vintage snapshots at each forecast origin.

ALFRED (Archival FRED) is the St. Louis Fed's real-time data archive. It stores historical vintages of every FRED series, allowing researchers to reconstruct the information set that was actually available at any past date -- before subsequent data revisions occurred. Croushore & Stark (2001) established the methodological case for real-time evaluation.

Operational since Cycle 50 (2026-05-22). Two modes are supported:

* ``alfred_mode=local`` (default): loads pre-downloaded vintage snapshots from ``leaf_config.alfred_snapshot_dir``. The directory must contain per-origin Parquet or CSV files named by date (e.g., ``1999-01-01.parquet``). No network access at runtime.

* ``alfred_mode=api``: queries the ALFRED REST API at each origin using ``leaf_config.alfred_api_key`` or the ``FRED_API_KEY`` environment variable. Requires network access.

At each walk-forward origin the runtime selects the vintage whose release date is the latest date not exceeding the origin date, so the model never sees data that was not yet published.

**When to use**

Real-time forecasting evaluations; replication of published studies that used ALFRED vintages; studies that quantify the effect of data revisions on forecast accuracy.

**When NOT to use**

Standard pseudo-OOS benchmarks using revised data -- use ``current_vintage`` (the default) for those.

**References**

* Federal Reserve Bank of St. Louis, 'ALFRED: Archival Federal Reserve Economic Data' -- real-time vintage archive of FRED series. <https://alfred.stlouisfed.org/>
* Croushore & Stark (2001) 'A real-time data set for macroeconomists', Journal of Econometrics 105(1). (doi:10.1016/S0304-4076(01)00072-0)
* Stark & Croushore (2002) 'Forecasting with a real-time data set for macroeconomists', Journal of Macroeconomics 24(4). (doi:10.1016/S0164-0704(02)00041-0)

**Related options**: [`current_vintage`](#current-vintage), [`information_set_type`](#information-set-type)

**Parameters**

| name | type | default | constraint | description |
|---|---|---|---|---|
| `alfred_mode` | `str` | `'local'` | One of: 'local', 'api'. | Controls how ALFRED vintage data is accessed. 'local' loads pre-downloaded per-origin snapshots from ``alfred_snapshot_dir``. 'api' queries the ALFRED REST API at runtime. |
| `alfred_snapshot_dir` | `str | Path` | — | Required when alfred_mode='local'. Must be a directory containing per-date vintage files named YYYY-MM-DD.parquet or YYYY-MM-DD.csv. | Directory of pre-downloaded ALFRED vintage snapshots. Each file represents the FRED panel as-published on that date. The runtime selects the latest file whose date does not exceed the forecast origin. |
| `alfred_api_key` | `str` | `None` | Required when alfred_mode='api' and FRED_API_KEY env var is not set. | FRED/ALFRED API key. Alternatively, set the FRED_API_KEY environment variable. Obtain a free key at https://fred.stlouisfed.org/docs/api/api_key.html. |

_Last reviewed 2026-05-22 by macroforecast author._
