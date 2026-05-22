# `transform_scope`

[Back to L2](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``transform_scope`` on sub-layer ``l2_b`` (layer ``l2``).

## Sub-layer

**l2_b**

## Axis metadata

- Default: `'derived'`
- Sweepable: True
- Status: operational

## Operational status summary

- Operational: 4 option(s)
- Future: 0 option(s)

## Options

### `target_and_predictors`  --  operational

Apply the rule to target and all predictors.

Default scope: every series in the panel passes through the stage. Maintains consistency between target and predictors (e.g. both differenced, both winsorised).

**When to use**

Default; matches McCracken-Ng's convention.

**References**

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'

**Related options**: [`predictors_only`](#predictors-only), [`target_only`](#target-only), [`not_applicable`](#not-applicable)

_Last reviewed 2026-05-04 by macroforecast author._

### `predictors_only`  --  operational

Apply only to predictors; leave the target untouched.

Used when the target's transform / cleaning policy is controlled separately (e.g. user already applied a tcode to the target via raw_panel).

**When to use**

When the target enters the pipeline already cleaned.

**References**

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'

**Related options**: [`target_and_predictors`](#target-and-predictors), [`target_only`](#target-only), [`not_applicable`](#not-applicable)

_Last reviewed 2026-05-04 by macroforecast author._

### `target_only`  --  operational

Apply only to the target.

Rare scope; used when predictors are pre-engineered and do not need this stage (e.g. PCA scores are already stationary).

**When to use**

Pre-engineered predictor panels.

**References**

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'

**Related options**: [`target_and_predictors`](#target-and-predictors), [`predictors_only`](#predictors-only), [`not_applicable`](#not-applicable)

_Last reviewed 2026-05-04 by macroforecast author._

### `not_applicable`  --  operational

Skip the stage entirely (gate inactive).

Used when an upstream stage already produced the desired form. Equivalent in effect to selecting the no-op option on the primary axis.

**When to use**

Pipelines that bypass this stage by construction.

**References**

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'

**Related options**: [`target_and_predictors`](#target-and-predictors), [`predictors_only`](#predictors-only), [`target_only`](#target-only)

_Last reviewed 2026-05-04 by macroforecast author._
