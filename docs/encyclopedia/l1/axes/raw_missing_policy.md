# `raw_missing_policy`

[Back to L1](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``raw_missing_policy`` on sub-layer ``l1_c`` (layer ``l1``).

## Sub-layer

**l1_c**

## Axis metadata

- Default: `'preserve_raw_missing'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 4 option(s)
- Future: 0 option(s)

## Options

### `preserve_raw_missing`  --  operational

Pass raw NaN values through unchanged.

Default; raw missingness flows into L2.D imputation. Required for the McCracken-Ng EM-factor imputation workflow. See also: L2 ``imputation_policy`` (same surface, different stage: raw vs post-tcode).

**When to use**

Default; required when L2.D will run EM-factor or similar global imputation.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`zero_fill_leading_predictor_missing_before_tcode`](#zero-fill-leading-predictor-missing-before-tcode), [`impute_raw_predictors`](#impute-raw-predictors), [`drop_raw_missing_rows`](#drop-raw-missing-rows)

_Last reviewed 2026-05-05 by macroforecast author._

### `zero_fill_leading_predictor_missing_before_tcode`  --  operational

Zero-fill leading predictor NaNs prior to t-code application.

Important for level-difference t-codes that fail when leading NaNs are interspersed with observed values. The zero-fill creates a clean prefix for differencing.

**When to use**

Tcode 1 / 2 / 5 / 6 pipelines where leading NaNs would propagate after differencing.

**When NOT to use**

When zero is a meaningful value for the predictor.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`preserve_raw_missing`](#preserve-raw-missing)

_Last reviewed 2026-05-05 by macroforecast author._

### `impute_raw_predictors`  --  operational

Impute raw predictor NaNs at L1 (before any L2 stage).

Runs a simple per-series imputation (mean / median / forward-fill) at L1. Useful when L2.D is disabled or when the user wants to pre-clean raw data before the t-code stage.

**When to use**

Pipelines that use ``no_transform`` t-codes and need cleaning at L1.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`preserve_raw_missing`](#preserve-raw-missing), [`drop_raw_missing_rows`](#drop-raw-missing-rows)

_Last reviewed 2026-05-05 by macroforecast author._

### `drop_raw_missing_rows`  --  operational

Drop rows containing any raw missing predictor.

Aggressive listwise deletion at the raw stage. Reduces panel size before any cleaning runs.

**When to use**

Sensitivity analyses; sanity checks against imputation effects.

**When NOT to use**

When the panel is small -- you'll lose a lot of rows.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`preserve_raw_missing`](#preserve-raw-missing)

_Last reviewed 2026-05-05 by macroforecast author._
