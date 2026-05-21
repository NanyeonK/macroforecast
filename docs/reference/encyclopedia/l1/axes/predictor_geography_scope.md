# `predictor_geography_scope`

[Back to L1](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``predictor_geography_scope`` on sub-layer ``l1_d`` (layer ``l1``).

## Sub-layer

**l1_d**

## Axis metadata

- Default: `'match_target'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 4 option(s)
- Future: 0 option(s)

## Options

### `match_target`  --  operational

Use the same geography scope as the target.

Default; predictor states match the L1.D ``target_geography_scope``. Ensures spatial coherence for state-level forecasts.

**When to use**

Default for state-level forecasts.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`all_states`](#all-states), [`selected_states`](#selected-states), [`national_only`](#national-only)

_Last reviewed 2026-05-05 by macroforecast author._

### `all_states`  --  operational

Use predictors from every state regardless of target geography.

All-50-states predictor block. Useful when cross-state spillovers matter and the target is a single state.

**When to use**

Spillover / cross-state interaction studies.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`match_target`](#match-target), [`selected_states`](#selected-states)

_Last reviewed 2026-05-05 by macroforecast author._

### `selected_states`  --  operational

Use predictors from a user-supplied state list.

Reads ``leaf_config.predictor_states`` and restricts the predictor block to that subset.

**When to use**

Custom regional studies (e.g. neighbouring states).

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`match_target`](#match-target), [`all_states`](#all-states)

**Parameters**

| name | type | default | constraint | description |
|---|---|---|---|---|
| `predictor_states` | `list[str]` | — | non-empty list required | Explicit predictor state list. Independent of target_states; permits cross-state-pair studies. |

_Last reviewed 2026-05-05 by macroforecast author._

### `national_only`  --  operational

Use only national-aggregate predictors.

Strips state-level predictors and keeps only national series. Reduces panel dimension when state-level features are noise.

**When to use**

When national variables alone explain target variation.

**When NOT to use**

State-level forecasts where regional predictors carry signal.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`all_states`](#all-states), [`match_target`](#match-target)

_Last reviewed 2026-05-05 by macroforecast author._
