# `refit_policy`

[Back to L4](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``refit_policy`` on sub-layer ``L4_C_training_window`` (layer ``l4``).

## Sub-layer

**L4_C_training_window**

## Axis metadata

- Default: `'every_origin'`
- Sweepable: True
- Status: operational

## Operational status summary

- Operational: 3 option(s)
- Future: 0 option(s)

## Options

### `every_origin`  --  operational

Re-fit the model at every walk-forward origin.

Most expensive but most accurate -- the model's coefficients update with every new observation.

**When to use**

Default. Standard walk-forward protocol.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'

_Last reviewed 2026-05-04 by macroforecast author._

### `every_n_origins`  --  operational

Re-fit every n origins (caps refit cost).

Requires ``leaf_config.refit_interval``. Saves wall-clock when fits are slow but introduces stale-coefficient bias.

**When to use**

Long sweeps with slow estimators (e.g., LSTM / xgboost on large panels).

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'

_Last reviewed 2026-05-04 by macroforecast author._

### `single_fit`  --  operational

Fit once on the full sample; use the same coefficients at every origin.

Equivalent to in-sample evaluation. Useful for parameter-stability studies but does not test out-of-sample performance.

**When to use**

In-sample studies; coefficient-stability pins.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'

_Last reviewed 2026-05-04 by macroforecast author._
