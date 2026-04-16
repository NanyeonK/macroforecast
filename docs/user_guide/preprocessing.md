# Preprocessing contract

## Purpose

macrocast separates preprocessing into explicit, governed axes rather than hiding it behind one undocumented pipeline.

## Governance fields (mandatory)

### `representation_policy`
Controls the data representation entering the study.
- `raw_only` — no representation transform (operational)
- `tcode_only` — t-code transform only (planned)
- `custom_transform_only` — user-defined (registry_only)

### `tcode_application_scope`
Controls which series receive t-code transforms.
- `apply_tcode_to_none` — no t-code applied (operational)
- `apply_tcode_to_target` — t-code on target only (planned)
- `apply_tcode_to_X` — t-code on predictors only (planned)
- `apply_tcode_to_both` — t-code on both (planned)

### `preprocessing_axis_role`
Controls whether preprocessing is fixed or intentionally varied.
- `fixed_preprocessing` — preprocessing is fixed for fair comparison (operational)
- `swept_preprocessing` — preprocessing is intentionally varied (planned)
- `ablation_preprocessing` — preprocessing is part of ablation study (planned)

### Cross-validation rules
- `raw_only` representation requires `tcode_application_scope=apply_tcode_to_none`
- `tcode_only` representation requires consistent tcode_application_scope
- Model sweep + preprocessing sweep simultaneously is rejected

## Preprocessing axes (24 total)

### Target-side
- `target_transform_policy`, `target_missing_policy`, `target_outlier_policy`
- `target_transform`, `target_normalization`, `target_domain`

### X-side
- `x_transform_policy`, `x_missing_policy`, `x_outlier_policy`
- `scaling_policy`, `scaling_scope`
- `dimensionality_reduction_policy`, `feature_selection_policy`
- `additional_preprocessing`, `x_lag_creation`, `feature_grouping`

### Execution semantics
- `tcode_policy`, `preprocess_order`, `preprocess_fit_scope`
- `inverse_transform_policy`, `recipe_mode`

## Current operational preprocessing paths

1. **raw_only**: no transforms, no extra preprocessing
2. **Train-only EM impute + standard scaling**: `tcode_policy=extra_preprocess_without_tcode`, `x_missing_policy=em_impute`, `scaling_policy=standard`
3. **Train-only EM impute + robust scaling**: same with `scaling_policy=robust`
4. **Train-only EM impute + minmax scaling**: same with `scaling_policy=minmax`
5. **PCA dimensionality reduction**: `dimensionality_reduction_policy=pca`
