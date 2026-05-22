# `official_transform_scope`

[Back to L1](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``official_transform_scope`` on sub-layer ``l1_c`` (layer ``l1``).

## Sub-layer

**l1_c**

## Axis metadata

- Default: `'target_and_predictors'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 4 option(s)
- Future: 0 option(s)

## Options

### `target_only`  --  operational

Apply official t-codes only to the target column.

Restricts McCracken-Ng tcode application to ``y``. Predictors flow through untransformed.

**When to use**

When predictors are already pre-transformed.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`predictors_only`](#predictors-only), [`target_and_predictors`](#target-and-predictors), [`none`](#none)

_Last reviewed 2026-05-05 by macroforecast author._

### `predictors_only`  --  operational

Apply official t-codes only to predictor columns.

Used when the user supplies an externally-transformed target.

Configures the ``official_transform_scope`` axis on ``l1_c`` (layer ``l1``); the ``predictors_only`` value is materialised in the recipe's ``fixed_axes`` block under that sub-layer.

**When to use**

When the target is pre-engineered (e.g. growth rate).

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`target_only`](#target-only), [`target_and_predictors`](#target-and-predictors)

_Last reviewed 2026-05-05 by macroforecast author._

### `target_and_predictors`  --  operational

Apply official t-codes to both target and predictors.

Default; canonical McCracken-Ng workflow.

Configures the ``official_transform_scope`` axis on ``l1_c`` (layer ``l1``); the ``target_and_predictors`` value is materialised in the recipe's ``fixed_axes`` block under that sub-layer.

**When to use**

Default; FRED-MD / -QD recipes.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`target_only`](#target-only), [`predictors_only`](#predictors-only), [`none`](#none)

_Last reviewed 2026-05-05 by macroforecast author._

### `none`  --  operational

Skip official t-codes entirely.

Disables L1's official tcode application. Used together with ``transform_policy = no_transform`` or ``custom_tcode``.

**When to use**

Custom panels with bespoke transforms.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`target_and_predictors`](#target-and-predictors)

_Last reviewed 2026-05-05 by macroforecast author._
