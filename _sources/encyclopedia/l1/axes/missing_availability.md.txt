# `missing_availability`

[Back to L1](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``missing_availability`` on sub-layer ``l1_c`` (layer ``l1``).

## Sub-layer

**l1_c**

## Axis metadata

- Default: `'zero_fill_leading_predictor_gaps'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 4 option(s)
- Future: 0 option(s)

## Options

### `require_complete_rows`  --  operational

Drop any row containing a missing value.

Strict listwise-deletion rule applied at L1 before L2 imputation. Useful when the recipe author prefers to lose rows rather than rely on imputation; produces a smaller, fully-observed panel.

**When to use**

Studies where imputation is methodologically inappropriate; sensitivity analyses against imputation effects.

**When NOT to use**

When the panel is sparsely observed -- listwise deletion can leave too few rows.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`keep_available_rows`](#keep-available-rows), [`impute_predictors_only`](#impute-predictors-only)

_Last reviewed 2026-05-05 by macroforecast author._

### `keep_available_rows`  --  operational

Keep every row that has the target observed.

Default; passes interior predictor NaNs through to L2.D for imputation. Ensures the maximum sample size while letting downstream imputation handle holes.

**When to use**

Default for FRED-MD / -QD recipes where L2.D EM imputation is the canonical workflow.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`require_complete_rows`](#require-complete-rows), [`impute_predictors_only`](#impute-predictors-only)

_Last reviewed 2026-05-05 by macroforecast author._

### `impute_predictors_only`  --  operational

Impute predictor missings at L1; never impute the target.

Restricts imputation to the predictor block at L1 stage and forbids any target imputation in subsequent layers. Avoids accidentally back-filling the target via L2.D.

**When to use**

Recipes where the target should be the ground-truth signal and never imputed.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`keep_available_rows`](#keep-available-rows)

**Parameters**

| name | type | default | constraint | description |
|---|---|---|---|---|
| `x_imputation` | `str` | — | required; one of ['bfill', 'ffill', 'mean', 'median']. | Imputation method applied to predictor missings at L1. Used only when missing_availability=impute_predictors_only. |

_Last reviewed 2026-05-05 by macroforecast author._

### `zero_fill_leading_predictor_gaps`  --  operational

Zero-fill leading predictor NaNs; preserve interior gaps.

Replaces leading NaNs (before the predictor's first observation) with zero so the panel has a uniform start date. Interior NaNs pass through to L2.D unchanged.

**When to use**

FRED-SD panels where some series start later but the user wants a balanced start date.

**When NOT to use**

When zero is a meaningful value for the predictor -- choose ``preserve_raw_missing`` instead.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`require_complete_rows`](#require-complete-rows)

_Last reviewed 2026-05-05 by macroforecast author._
