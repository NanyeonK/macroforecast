# `imputation_temporal_rule`

[Back to L2](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``imputation_temporal_rule`` on sub-layer ``l2_d`` (layer ``l2``).

## Sub-layer

**l2_d**

## Axis metadata

- Default: `'expanding_window_per_origin'`
- Sweepable: True
- Status: operational

## Operational status summary

- Operational: 3 option(s)
- Future: 0 option(s)

## Options

### `expanding_window_per_origin`  --  operational

Re-fit the imputation model on every expanding window.

Default temporal_rule: at each OOS origin, the imputation model is fit on all data from the sample start through the origin date. Avoids look-ahead while ensuring the model has access to maximum data at each step.

**When to use**

Default; OOS-safe imputation. Selecting ``expanding_window_per_origin`` on ``l2.imputation_temporal_rule`` activates this branch of the layer's runtime.

**When NOT to use**

When per-origin re-fits are too expensive -- consider ``block_recompute``.

**References**

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'

**Related options**: [`rolling_window_per_origin`](#rolling-window-per-origin), [`block_recompute`](#block-recompute)

_Last reviewed 2026-05-04 by macroforecast author._

### `rolling_window_per_origin`  --  operational

Re-fit the imputation model on a fixed-length rolling window.

Fits the imputation model on the most-recent ``params.window`` observations only. Useful when the underlying covariance structure is non-stationary and old data should not influence current imputations.

**When to use**

Non-stationary panels where covariance drifts.

**When NOT to use**

When the panel is stationary -- expanding window uses more information.

**References**

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'

**Related options**: [`expanding_window_per_origin`](#expanding-window-per-origin), [`block_recompute`](#block-recompute)

_Last reviewed 2026-05-04 by macroforecast author._

### `block_recompute`  --  operational

Re-fit the imputation model every N origins.

Fits the imputation model once every ``leaf_config.imputation_recompute_interval`` origins; intermediate origins reuse the cached fit. Cheap approximation to ``expanding_window_per_origin``.

**When to use**

Long sweeps where per-origin re-fits are computationally infeasible.

**When NOT to use**

When precise OOS-safe imputation is critical.

**References**

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'

**Related options**: [`expanding_window_per_origin`](#expanding-window-per-origin), [`rolling_window_per_origin`](#rolling-window-per-origin)

_Last reviewed 2026-05-04 by macroforecast author._
