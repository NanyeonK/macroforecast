# `target_geography_scope`

[Back to L1](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``target_geography_scope`` on sub-layer ``l1_d`` (layer ``l1``).

## Sub-layer

**l1_d**

## Axis metadata

- Default: `'all_states'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 3 option(s)
- Future: 0 option(s)

## Options

### `single_state`  --  operational

Single FRED-SD state target (e.g., California payrolls).

Selects one US state as the target. Requires ``leaf_config.target_state`` (two-letter postal code). Predictors default to ``match_target`` (same state).

**When to use**

State-level case studies (e.g., CA / TX / NY-specific forecasts).

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`all_states`](#all-states), [`selected_states`](#selected-states), [`predictor_geography_scope`](#predictor-geography-scope)

_Last reviewed 2026-05-04 by macroforecast author._

### `all_states`  --  operational

Forecast every state's series jointly (50+DC targets).

Treats every state series as a target. The L5 metrics table carries one row per (model, state, horizon, origin) and the L7 ``us_state_choropleth`` figure type maps importance scores to the geographic layout.

This is the standard FRED-SD configuration for cross-state comparison studies.

**When to use**

Geographic-importance studies; cross-state benchmark comparisons.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`single_state`](#single-state), [`selected_states`](#selected-states), [`fred_sd_state_group`](#fred-sd-state-group)

_Last reviewed 2026-05-04 by macroforecast author._

### `selected_states`  --  operational

Forecast a user-supplied subset of states.

Like ``all_states`` but restricted to ``leaf_config.target_states = [postal_codes...]`` or to a named ``fred_sd_state_group`` (census regions / divisions, BEA regions, etc.).

**When to use**

Region-specific studies (Northeast vs. Midwest), Census-division comparisons.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`all_states`](#all-states), [`fred_sd_state_group`](#fred-sd-state-group)

_Last reviewed 2026-05-04 by macroforecast author._
