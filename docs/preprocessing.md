# Preprocessing contract

## Purpose

macrocast no longer treats preprocessing as one hidden default pipeline.
The package now separates:
- representation / transform choices
- extra preprocessing choices
- execution semantics

## Current public surface

The preprocessing layer now exposes:
- `PreprocessContract`
- `build_preprocess_contract()`
- `check_preprocess_governance()`
- `is_operational_preprocess_contract()`
- `preprocess_summary()`
- `preprocess_to_dict()`

## Representation / transform axes

The contract records:
- `target_transform_policy`
- `x_transform_policy`
- `tcode_policy`

Current t-code policy vocabulary:
- `raw_only`
- `tcode_only`
- `tcode_then_extra_preprocess`
- `extra_preprocess_without_tcode`
- `extra_then_tcode`
- `custom_transform_pipeline`

These are intentionally distinct design choices.
macrocast does not treat them as the same path.

## Extra preprocessing axes

The contract also records:
- `target_missing_policy`
- `x_missing_policy`
- `target_outlier_policy`
- `x_outlier_policy`
- `scaling_policy`
- `dimensionality_reduction_policy`
- `feature_selection_policy`

## Execution semantics

The contract records:
- `preprocess_order`
- `preprocess_fit_scope`
- `inverse_transform_policy`
- `evaluation_scale`

This is the minimum needed to keep model effects separate from preprocessing effects in later benchmarking interpretation.

## Current executable subset

The current runtime is intentionally honest and still narrow.
Operational contracts are:
- explicit raw-only contract:
  - raw target representation
  - raw x representation
  - no t-code transform
  - no extra preprocessing
  - no inverse transform
  - raw-level evaluation scale
- train-only raw-panel extra-preprocess path:
  - `tcode_policy = extra_preprocess_without_tcode`
  - `x_missing_policy = em_impute`
  - `scaling_policy = standard` or `robust`
  - `preprocess_order = extra_only`
  - `preprocess_fit_scope = train_only`
  - `evaluation_scale = raw_level`

Other preprocessing choices are already representable in package grammar, but not yet executable.
That distinction is explicit through registry/compiler status rather than hidden behavior.


## Stage 2 governance additions

The preprocessing contract now also records explicit governance fields:
- `representation_policy`
- `preprocessing_axis_role`
- `tcode_application_scope`
- `target_transform`
- `target_normalization`
- `target_domain`
- `scaling_scope`
- `additional_preprocessing`
- `x_lag_creation`
- `feature_grouping`
- `recipe_mode`

These default conservatively so legacy recipes still compile unchanged.

## Expanded operational Stage 2 runtime slice

Current train-only raw-panel runtime additionally supports:
- X missing: `mean_impute`, `median_impute`, `ffill`, `interpolate_linear`, `em_impute`
- X outlier: `winsorize`, `iqr_clip`, `zscore_clip`
- scaling: `standard`, `robust`, `minmax`
- dimensionality reduction: `pca`, `static_factor`
- feature selection: `correlation_filter`, `lasso_select`

Still not executable in current slice:
- simultaneous dimensionality reduction + feature selection
- non-columnwise scaling scopes
- extra filters under `additional_preprocessing`
- nontrivial `x_lag_creation`
- nontrivial `feature_grouping`
