# `regime_estimation_temporal_rule`

[Back to L1](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``regime_estimation_temporal_rule`` on sub-layer ``l1_g`` (layer ``l1``).

## Sub-layer

**l1_g** (gated)

## Axis metadata

- Default: `'expanding_window_per_origin'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 3 option(s)
- Future: 0 option(s)

## Options

### `expanding_window_per_origin`  --  operational

Re-estimate regimes on every expanding window.

Default for ``estimated_*`` regime methods. Avoids look-ahead by re-fitting the regime model on data through each origin date.

**When to use**

Default; OOS-safe regime estimation.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`rolling_window_per_origin`](#rolling-window-per-origin), [`block_recompute`](#block-recompute)

_Last reviewed 2026-05-05 by macroforecast author._

### `rolling_window_per_origin`  --  operational

Re-estimate regimes on a fixed-length rolling window.

Uses the most-recent ``params.window`` observations only. Useful when regime structure drifts over time.

**When to use**

Drifting / non-stationary regime structure.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`expanding_window_per_origin`](#expanding-window-per-origin), [`block_recompute`](#block-recompute)

_Last reviewed 2026-05-05 by macroforecast author._

### `block_recompute`  --  operational

Re-estimate every leaf_config.regime_recompute_interval origins.

Cheap approximation to per-origin re-fits. Caches the regime classification between recompute boundaries.

**When to use**

Long sweeps where per-origin regime re-fits are infeasible.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`expanding_window_per_origin`](#expanding-window-per-origin), [`rolling_window_per_origin`](#rolling-window-per-origin)

_Last reviewed 2026-05-05 by macroforecast author._
