# `forecast_strategy`

[Back to L4](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``forecast_strategy`` on sub-layer ``L4_B_forecast_strategy`` (layer ``l4``).

## Sub-layer

**L4_B_forecast_strategy**

## Axis metadata

- Default: `'direct'`
- Sweepable: True
- Status: operational

## Operational status summary

- Operational: 3 option(s)
- Future: 0 option(s)

## Options

### `direct`  --  operational

One model per horizon (h=1, h=6, h=12, ...).

Fits a separate model for each horizon h, using y_{t+h} as the target. The standard horse-race protocol: simple to implement, no error compounding, more compute.

**When to use**

Default for most studies. Comparable across publications.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'

_Last reviewed 2026-05-04 by macroforecast author._

### `iterated`  --  operational

Fit h=1 model; apply recursively for h>1.

Trains a single model on (y_t, X_t) → y_{t+1}, then iterates the prediction h times. Faster (one fit per cell) but errors compound.

**When to use**

Speed-sensitive sweeps; replication of papers using iterated VAR.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'

_Last reviewed 2026-05-04 by macroforecast author._

### `path_average`  --  operational

Forecast the cumulative-average target over horizon h.

Pairs with the L3 ``cumulative_average`` target-construction op. Useful for studies forecasting the *average* growth rate over horizon h rather than the level.

**When to use**

Cumulative-growth forecasting (e.g., Stock-Watson 2002).

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'

_Last reviewed 2026-05-04 by macroforecast author._
