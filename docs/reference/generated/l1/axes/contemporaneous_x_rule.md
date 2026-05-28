# `contemporaneous_x_rule`

[Back to L1](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``contemporaneous_x_rule`` on sub-layer ``l1_c`` (layer ``l1``).

## Sub-layer

**l1_c**

## Axis metadata

- Default: `'allow_same_period_predictors'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 2 option(s)
- Future: 0 option(s)

## Options

### `allow_same_period_predictors`  --  operational

Permit predictors observed in the same period as the target.

Default; predictor ``x_t`` and target ``y_t`` are both available at time t. Used for nowcasting where contemporaneous information is exploited.

**When to use**

Default; standard fitting / nowcasting flow.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`forbid_same_period_predictors`](#forbid-same-period-predictors)

_Last reviewed 2026-05-05 by macroforecast author._

### `forbid_same_period_predictors`  --  operational

Require predictors to be at least one period stale.

Forces predictors to be lagged ``y_t`` is forecast from ``x_{t-1}, x_{t-2}, ...``. Cleanest causal interpretation.

**When to use**

Pure forecasting setups where contemporaneous information would create look-ahead.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`allow_same_period_predictors`](#allow-same-period-predictors)

_Last reviewed 2026-05-05 by macroforecast author._
