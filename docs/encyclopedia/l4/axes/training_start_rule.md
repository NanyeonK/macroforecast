# `training_start_rule`

[Back to L4](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``training_start_rule`` on sub-layer ``L4_C_training_window`` (layer ``l4``).

## Sub-layer

**L4_C_training_window**

## Axis metadata

- Default: `'expanding'`
- Sweepable: True
- Status: operational

## Operational status summary

- Operational: 3 option(s)
- Future: 0 option(s)

## Options

### `expanding`  --  operational

Expanding window: training data grows by one observation per origin.

Standard pseudo-OOS protocol. Each origin sees all data from t=0 up to that origin.

**When to use**

Default. Comparable across publications.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'

_Last reviewed 2026-05-04 by macroforecast author._

### `rolling`  --  operational

Rolling window of fixed size (params.rolling_window).

Drops early observations; useful for non-stationary series where parameter drift matters.

**When to use**

Non-stationary series; structural-change studies.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'

_Last reviewed 2026-05-04 by macroforecast author._

### `fixed`  --  operational

Fixed window with start/end pinned in leaf_config.

Useful for ablation studies where every origin should see the same training sample.

**When to use**

Replication of papers with fixed training windows.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'

_Last reviewed 2026-05-04 by macroforecast author._
