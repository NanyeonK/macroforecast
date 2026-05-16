# `release_lag_rule`

[Back to L1](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``release_lag_rule`` on sub-layer ``l1_c`` (layer ``l1``).

## Sub-layer

**l1_c**

## Axis metadata

- Default: `'ignore_release_lag'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 3 option(s)
- Future: 0 option(s)

## Options

### `ignore_release_lag`  --  operational

Treat every observation as available at its calendar period.

Pseudo-real-time mode: ignores the release-lag distinction; every variable is assumed to be available the moment the period closes.

**When to use**

Backtests where real-time vintage data is unavailable.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`fixed_lag_all_series`](#fixed-lag-all-series), [`series_specific_lag`](#series-specific-lag)

_Last reviewed 2026-05-05 by macroforecast author._

### `fixed_lag_all_series`  --  operational

Apply a single release lag to every series.

All series shift by ``leaf_config.fixed_lag_periods`` periods. Approximates real-time availability without per-series detail.

**When to use**

Coarse real-time approximations.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`series_specific_lag`](#series-specific-lag)

**Parameters**

| name | type | default | constraint | description |
|---|---|---|---|---|
| `fixed_lag_periods` | `int` | — | >=0; optional; defaults to 0 if not set. | Uniform release lag in periods applied to every predictor series. A value of 1 means each series is available one period after the period it was observed. |

_Last reviewed 2026-05-05 by macroforecast author._

### `series_specific_lag`  --  operational

Use per-series release lags from leaf_config.

Honours the published release-lag table in ``leaf_config.release_lag_per_series``. Most accurate option for true real-time studies.

**When to use**

Real-time / nowcasting studies that respect publication delays.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`fixed_lag_all_series`](#fixed-lag-all-series)

**Parameters**

| name | type | default | constraint | description |
|---|---|---|---|---|
| `release_lag_per_series` | `dict[str, int]` | — | Required when release_lag_rule=series_specific_lag; non-empty dict. | Per-series release lag in periods. Maps series name to a non-negative integer. Series not present in the dict are treated as zero-lag (available immediately). |

_Last reviewed 2026-05-05 by macroforecast author._
