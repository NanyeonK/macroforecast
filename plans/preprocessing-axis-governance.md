# Macrocast Preprocessing Axis Governance

Status: reboot-stage architecture priority
Date: 2026-04-14
Priority: immediate next architecture lock after Stage 0 completion

## Core diagnosis

The repo has already moved beyond bare skeleton status.
The direction is now broadly correct:
- raw dataset adapters exist for FRED-MD / FRED-QD / FRED-SD
- Stage 0 grammar exists
- recipe/run contract exists
- the package is already shaped like a real benchmarking framework

The next risk is not lack of functionality.
The next risk is interpretive slippage in benchmarking.

The most important current issue is preprocessing axis governance.

If preprocessing is not governed explicitly, later benchmarking results will become difficult to interpret:
- did performance move because the model changed?
- or because preprocessing changed?
- or because data representation changed before preprocessing?

That ambiguity is fatal for a package whose core mission is fair tool comparison.

## Main rule

Preprocessing must not remain a hidden default pipeline.
It must become explicit study language.

In particular, the package must stop treating the whole preprocessing stack as one undifferentiated operation.

## Architectural distinction to lock

`t-code` is not merely a generic cleaning step.
It is closer to a data representation choice or statistical object-definition choice.

Therefore the package should explicitly separate:
- target-side representation choice
- x-side representation choice
- additional preprocessing choice
- the order in which those are applied

## Required axes to introduce

### 1. `target_transform_policy`

Allowed conceptual values:
- `raw_level`
- `tcode_transformed`
- `custom_target_transform`

Meaning:
- what target representation enters the forecasting study

### 2. `x_transform_policy`

Allowed conceptual values:
- `raw_level`
- `dataset_tcode_transformed`
- `custom_x_transform`

Meaning:
- what predictor representation enters the forecasting study

### 3. `preprocess_order`

Allowed conceptual values:
- `none`
- `tcode_only`
- `extra_only`
- `tcode_then_extra`
- `extra_then_tcode`

Meaning:
- the order relationship between representation transform and additional preprocessing

### 4. `extra_preprocess_recipe`

Allowed conceptual values in current planning vocabulary:
- `none`
- `outlier_to_nan`
- `em_impute`
- `outlier_to_nan_plus_em`
- `scaling`
- other explicit custom recipes

Meaning:
- additional preprocessing beyond raw/t-code representation choice

## Required semantic distinction

At minimum, macrocast must explicitly distinguish these two cases:

1. `tcode_then_extra`
- apply t-code transform first
- then apply additional preprocessing

2. `extra_then_no_tcode` or equivalent raw-representation path
- remain at raw/nonstationary representation
- apply additional preprocessing without first switching into t-code-transformed representation

These are not the same preprocessing family.
They are different design choices.

In particular:
- outlier removal on raw nonstationary levels
- outlier removal on stationary transformed data
are not interchangeable operations

Likewise:
- EM imputation on raw level data
- EM imputation on t-code-transformed data
should not be silently collapsed into one pipeline default

## Governance rules for benchmarking

### Rule 1. Preprocessing fixed by default
In benchmark/baseline model comparison, preprocessing should be treated as a fixed axis by default.

### Rule 2. Preprocessing sweep must be explicit
Only when the researcher is studying preprocessing itself should preprocessing become a sweep axis.

Recommended explicit marker:
- `preprocessing_sweep = true`
or equivalent study-mode signal

### Rule 3. Do not co-sweep model and preprocessing in ordinary baseline comparison
For standard model benchmarking, do not change model and preprocessing simultaneously.

### Rule 4. If multiple axes move, label it honestly
If both model and preprocessing move together, the package should label the study as:
- ablation study
- factorial study
or another explicit controlled-variation family

It should not be presented as an ordinary fair baseline model comparison.

### Rule 5. Train-only fit is mandatory
All preprocessing fit operations must obey train-only rules.

Disallowed:
- full-sample fit before split
- imputation or scaling informed by future observations

## Recommended package-facing representation

The package should carry these fields explicitly in recipes and manifests:
- `target_transform_policy`
- `x_transform_policy`
- `preprocess_order`
- `extra_preprocess_recipe`

These should not be hidden behind a single undocumented `transform()` default.

## Implication for `MacroFrame.transform()`

The current package direction should avoid allowing `MacroFrame.transform()` to carry too many meanings at once.

Two acceptable architecture directions exist.

### Option A. Keep `apply_tcodes()` pure and explicit
Preferred interpretation:
- `apply_tcodes()` remains a pure transformation from one representation to another
- a separate wrapper such as `prepare_fredmd_canonical()` performs a package-standard sequence like:
  - tcode
  - trim
  - outlier-to-NaN
  - EM imputation

This keeps representation choice separate from extra preprocessing policy.

### Option B. Keep one high-level transform entry point, but make recipe semantics explicit
If a higher-level transform wrapper is retained, then recipes and manifests must explicitly store:
- `target_transform_policy`
- `x_transform_policy`
- `preprocess_order`
- `extra_preprocess_recipe`

The key rule is the same either way:
- no hidden preprocessing defaults that erase design meaning

## Preferred direction

For package clarity, Option A is cleaner.

Why:
- better separation of representation vs extra preprocessing
- more interpretable manifests
- easier benchmarking governance
- smaller risk that one convenience method becomes semantically overloaded

## Immediate next architecture priority

Before adding more model families or expanding benchmarking functionality, macrocast should:

1. lock preprocessing as fixed-vs-sweep governed language
2. separate t-code from generic extra preprocessing
3. make `tcode_then_extra` versus raw-level-extra explicit path semantics
4. enforce train-only preprocessing fit rules in future execution contracts

## What this means for later layers

### Stage 0 / design layer
Stage 0 should ultimately recognize preprocessing governance as part of study language, not hidden implementation detail.

### Recipe layer
Recipes must carry explicit preprocessing semantics.

### Manifest/output layer
Run artifacts should record the exact preprocessing representation and order used.

### Benchmark interpretation
Reported gains should be attributable to:
- model change
- preprocessing change
- or both explicitly, never implicitly

## Bottom line

The next important package step is not adding more models.
It is making preprocessing semantics explicit enough that benchmarking results stay interpretable.

That means:
- fixed preprocessing by default
- explicit preprocessing sweep only when intended
- t-code treated as a representation axis
- preprocessing order treated as a first-class design choice
