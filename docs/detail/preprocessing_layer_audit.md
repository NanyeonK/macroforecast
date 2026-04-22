# Preprocessing Layer Audit

This audit records the current preprocessing contract after the `Experiment` MVP pass.

The practical question is whether the simple API can safely expose preprocessing sweeps. Current answer: not yet.

## Canonical Layer 2 Role

Layer 2 is the researcher preprocessing and feature-representation layer. Layer 1 produces the
baseline official or raw feature frame: dataset/source/frequency, information
set, target/horizon/sample window, release-lag availability, official
transform/T-code policy, raw-source missing/outlier repair before T-codes, and
the eligible variable universe. Layer 2 starts after that point.

The purpose of Layer 2 is to support research designs that ask how forecasts
change when the researcher changes the representation handed to the forecast
generator. It owns optional transformations of predictors and the target, feature engineering,
feature-block selection, predictor family, feature builders, dimensionality
reduction, factor-count decisions, target-scale handling, preprocessing order,
and leakage discipline. It does not own dataset identity, official data
availability, model family, benchmark family, model training/tuning protocol,
scoring metrics, or statistical tests.

This means the canonical Layer 2 question is not "what preprocessing happens by
default?" but "what feature representation `Z` should the researcher construct
from Layer 1 outputs `H`, `X`, and target history before Layer 3 fits a model?"

## Layer 2 Decision Space

Layer 2 currently has the original preprocessing contract axes plus migrated
feature-representation bridge axes. The canonical decision groups are:

| Group | Axes | What the group decides |
|-------|------|------------------------|
| Research feature representation | `feature_builder`, `predictor_family`, `data_richness_mode`, `factor_count`, `feature_block_set`, `target_lag_block`, `x_lag_feature_block`, `factor_feature_block`, `level_feature_block`, `rotation_feature_block`, `temporal_feature_block`, `feature_block_combination` | Which feature matrix `Z` is constructed from Layer 1 outputs before forecasting. Current runtime uses the migrated compatibility names; the explicit feature-block axes are registry-only. |
| X additional preprocessing | `x_missing_policy`, `x_outlier_policy`, `scaling_policy`, `scaling_scope`, `additional_preprocessing`, `x_lag_creation` | How predictor columns are imputed, clipped, scaled, filtered, or lag-augmented after Layer 1. |
| X representation and selection | `dimensionality_reduction_policy`, `feature_selection_policy`, `feature_grouping` | Whether the predictor panel is reduced to factors/components, screened to a subset, or grouped before modeling. |
| Target-side preprocessing | `horizon_target_construction`, `target_transform`, `target_normalization`, `target_domain`, `target_missing_policy`, `target_outlier_policy`, `inverse_transform_policy`, `evaluation_scale`, `target_transformer` | How the target is constructed, transformed, normalized, inverted, and evaluated. |
| Preprocessing order and leakage discipline | `preprocess_order`, `preprocess_fit_scope`, `separation_rule` | Whether extra preprocessing is applied before/after official transforms, and whether each step is fit on train-only data. |
| Custom extension hooks | `custom_preprocessor`, `target_transformer` | Researcher-supplied predictor-side and target-side preprocessing protocols when built-ins are insufficient. |
| Legacy representation bridge | `target_transform_policy`, `x_transform_policy`, `tcode_policy`, `representation_policy`, `tcode_application_scope` | Compatibility fields that still help the runtime `PreprocessContract` represent raw vs official T-code frames. They are not the canonical place to choose official transforms in new recipes. |

The natural full Layer 2 profile is
`dataset_tcode_then_train_only_extra`: Layer 1 applies official FRED-MD/QD
transforms/T-codes, then Layer 2 applies researcher-selected imputation,
scaling, filtering, dimensionality reduction, feature selection, or custom
preprocessing under train-only fit discipline. That profile is the current
generic preprocessing support surface, not the full research feature-block grammar.

Built-in Layer 2 choices should stay aligned with macro-forecasting research:
official dataset transforms, X-side imputation, scaling, filtering, PCA/static
factor extraction, feature screening, fixed lag construction, level add-backs,
lag rotations, local temporal factors, and custom hooks for researcher
extensions. Named papers such as Goulet Coulombe et al. (2021) should be
represented as presets over general feature-block primitives, not as layer names.

## Current Implementation Surface

The table below is the implementation status, not the boundary definition. It
records what the current runtime can execute today.

| Axis | Executable values today | Notes |
|------|-------------------------|-------|
| `feature_builder` | `autoreg_lagged_target`, `factors_plus_AR`, `raw_feature_panel`, `raw_X_only`, `factor_pca` | Currently used by compiler/runtime dispatch; semantically this chooses feature representation. `sequence_tensor` is future. |
| `feature_block_set` and feature-block primitive axes | none | The explicit grammar is defined as registry-only: `target_lag_block`, `x_lag_feature_block`, `factor_feature_block`, `level_feature_block`, `rotation_feature_block`, `temporal_feature_block`, and `feature_block_combination`. |
| `predictor_family` | `target_lags_only`, `all_macro_vars`, `category_based`, `factor_only`, `handpicked_set` | Canonical Layer 2 owner; runtime support is constrained by `feature_builder` compatibility guards. |
| `data_richness_mode` | `target_lags_only`, `factor_plus_lags`, `full_high_dimensional_X`, `selected_sparse_X` | Canonical Layer 2 owner; `mixed_mode` remains registry-only. |
| `factor_count` | `fixed`, `cv_select`, `BaiNg_rule` | Canonical Layer 2 owner for factor representation dimensions. `variance_explained_rule` and `model_specific` remain registry-only. |
| `x_missing_policy` | `none`, `drop`, `drop_rows`, `drop_columns`, `drop_if_above_threshold`, `missing_indicator`, `em_impute`, `mean_impute`, `median_impute`, `ffill`, `interpolate_linear` | Executes in the raw-panel extra-preprocess path. `drop` and `drop_rows` are pass-through aliases because predictor/target row coordination happens upstream. |
| `x_outlier_policy` | `none`, `winsorize`, `trim`, `iqr_clip`, `mad_clip`, `zscore_clip`, `outlier_to_missing` | Operates on post-frame X_train/X_pred. Raw-source outlier handling belongs to Layer 1. |
| `scaling_policy` | `none`, `standard`, `robust`, `minmax`, `demean_only`, `unit_variance_only` | Fitted on X_train and applied to X_pred. |
| `scaling_scope` | `columnwise`, `global_train_only` | Other scopes are blocked by governance. |
| `additional_preprocessing` | `none`, `hp_filter` | Moving average, EMA, and bandpass are registry-only. |
| `x_lag_creation` | `no_x_lags`, `fixed_x_lags` | CV-selected and variable/category-specific lags are not wired. |
| `dimensionality_reduction_policy` | `none`, `pca`, `static_factor` | Cannot be combined with feature selection. |
| `feature_selection_policy` | `none`, `correlation_filter`, `lasso_select` | Cannot be combined with dimensionality reduction. |
| `feature_grouping` | `none` | Non-`none` grouping is blocked in governance. |
| `horizon_target_construction` | `future_target_level_t_plus_h`, `future_diff`, `future_logdiff`, `average_growth_1_to_h`, `average_difference_1_to_h`, `average_log_growth_1_to_h` | Path-average target constructions have Layer 2 protocol metadata but remain registry-only until multi-step target execution is wired in Layer 3. |
| `target_transform` | `level`, `difference`, `log`, `log_difference`, `growth_rate` | Applied to the target series before model execution, with limited inverse/evaluation semantics. |
| `target_normalization` | `none` | Z-score variants are helper-tested but registry-only until normalization is fit inside each training window. |
| `target_domain` | `unconstrained` | Domain constraints are not implemented. |
| `target_missing_policy` | `none` | Target-side missing algorithms are not supported in the operational contract. |
| `target_outlier_policy` | `none` | Target-side outlier algorithms are not supported in the operational contract. |
| `inverse_transform_policy` | `none` | Inverse-transform policy needs a separate target/evaluation contract. |
| `evaluation_scale` | `raw_level`, `original_scale` | `transformed_scale` and `both` remain representable but are registry-only until inverse/evaluation semantics are finalized. |
| `preprocess_order` | `none`, derived `tcode_only`, `extra_only`, `tcode_then_extra` | `tcode_then_extra` is executable for supported raw-panel extra preprocessing after Layer 1 official t-codes. |
| `preprocess_fit_scope` | `not_applicable`, `train_only` | Extra preprocessing requires `train_only` today. |
| `separation_rule` | `strict_separation` | Non-strict helper modes are registry-only until wired into the main execution loop as a general dispatcher. |
| `custom_preprocessor` | fixed registered plugin name or `none` | Predictor-side function must return transformed X_train/X_test and must not transform the target. |
| `target_transformer` | fixed registered plugin name or `none` | Executable under target-transformer constraints; raw-scale evaluation only. |

Current runtime profiles:

- `dataset_tcode_only`: executable default. Layer 1 chooses official transforms;
  compiler derives a runtime bridge contract with no extra Layer 2 preprocessing.
- `raw_only`: executable non-default path with no official T-code and no extra
  preprocessing.
- `raw_train_only_extra`: executable for raw-panel style feature builders using
  train-only X-side extra preprocessing.
- `dataset_tcode_then_train_only_extra`: executable for raw-panel style feature
  builders when Layer 1 applies dataset t-codes first and Layer 2 applies
  supported train-only X-side extra preprocessing.


The main practical point: **Layer 2 is the research support layer for
additional preprocessing, and the current built-in implementation supports that
only for raw-panel style feature builders.** The default official T-code path is
executable, and supported extra preprocessing can now be attached after the
Layer 1 official T-code step through the derived
`tcode_then_extra_preprocess` bridge contract.
This definition pass splits the migrated representation axes into general
feature-block primitives: target-lag blocks, transformed-X lag blocks, factor
blocks, level add-backs, lag rotations, local temporal factors, volatility
blocks, and custom blocks. The split is defined in
`layer2_feature_representation.md`; the implementation sequence is defined in
`layer2_revision_plan.md`. Runtime support remains a separate implementation
task.


## Full Closure Status

Layer 2 is closed for fixed full recipes under the current runtime scope:

- all Layer 2 axes have canonical ownership and honest registry status;
- `dataset_tcode_only`, `raw_only`, `raw_train_only_extra`, and
  `dataset_tcode_then_train_only_extra` compile and execute where their
  constraints are satisfied;
- representable-but-not-executable values remain in the grammar as
  `registry_only`, not `operational`;
- the simple API still does not expose preprocessing sweeps.

The closed full profile is therefore a fixed single-run or fixed controlled
recipe, not an arbitrary public sweep. Full recipes can represent broader
macro-forecasting research intentions, but unsupported target-side
normalization/inversion, non-strict separation rules, feature grouping, CV-selected X lags, and
dual-scale evaluation stay blocked until they receive runtime integration and
acceptance tests.

## Current Default

`macrocast-default-v1` uses the official dataset transformation path:

| Axis | Default |
|------|---------|
| `official_transform_policy` | `dataset_tcode` |
| `official_transform_scope` | `apply_tcode_to_both` |
| `target_transform_policy` | `tcode_transformed` |
| `x_transform_policy` | `dataset_tcode_transformed` |
| `tcode_policy` | `tcode_only` |
| `representation_policy` | `tcode_only` |
| `tcode_application_scope` | `apply_tcode_to_both` |
| `preprocess_order` | `tcode_only` |
| `preprocess_fit_scope` | `not_applicable` |
| extra preprocessing axes | `none` |

This path is executable. It applies FRED-MD/QD dataset t-codes before the forecasting runtime. FRED-SD inferred t-codes remain opt-in and non-official.

Layer ownership after the migration pass:

- Layer 1 owns the official-frame decision through `official_transform_policy`
  and `official_transform_scope`.
- Layer 2 keeps the legacy t-code fields as a runtime compatibility bridge
  until the `PreprocessContract` no longer needs them.
- Layer 2 still owns researcher-controlled extra preprocessing after the
  official frame exists.

Missing/outlier boundary after the migration pass:

- Layer 1 owns raw-source missing/outlier treatment when it happens before
  official dataset transforms or T-codes. This is the "clean raw, then T-code"
  order, now represented by `raw_missing_policy` and `raw_outlier_policy`.
- Layer 2 owns missing imputation and outlier handling after the official frame
  exists. This is the "T-code first, then impute/clip/select/scale" order.
- Both orders can be reasonable for detailed empirical work. The second order
  can mix raw-source defects with transform-induced missing values and model
  input artifacts, so full-mode provenance must record whether the action
  happened before or after the official transform step.
- Simple mode should keep the current default. Full recipes may expose this
  choice through the Layer 1 raw-source axes.

Current bridge status:

- New default recipes emit only the Layer 1 official-transform axes.
- The compiler derives `PreprocessContract` bridge fields from those Layer 1
  axes for the runtime.
- Execution reads `data_task_spec["official_transform_policy"]` and
  `data_task_spec["official_transform_scope"]` first; `PreprocessContract`
  t-code fields are fallback only for older compiled specs.

## Executable Contract Classes

The code currently recognizes four useful preprocessing classes.

### `tcode_only`

Status: executable and default.

Contract:

- `target_transform_policy='tcode_transformed'`
- `x_transform_policy='dataset_tcode_transformed'`
- `tcode_policy='tcode_only'`
- `representation_policy='tcode_only'`
- `tcode_application_scope='apply_tcode_to_both'`
- no missing, outlier, scaling, dimensionality reduction, or feature selection extras

Runtime path:

- `_apply_tcode_preprocessing`
- then forecasting uses transformed frame

This is the only built-in preprocessing path exposed through the simple default profile.

### `raw_only`

Status: executable but not the simple default.

Contract:

- `target_transform_policy='raw_level'`
- `x_transform_policy='raw_level'`
- `tcode_policy='raw_only'`
- `preprocess_order='none'`
- no extra preprocessing

This path skips dataset t-code transforms.

### `extra_preprocess_without_tcode`

Status: executable for raw-panel style feature builders, not the simple default.

Contract:

- raw target and X representation
- `tcode_policy='extra_preprocess_without_tcode'`
- `preprocess_order='extra_only'`
- `preprocess_fit_scope='train_only'`
- no target-side missing/outlier transformation

Runtime helpers exist for:

- `x_missing_policy`: `em_impute`, `mean_impute`, `median_impute`, `ffill`, `interpolate_linear`, `drop_columns`, `drop_if_above_threshold`, `missing_indicator`
- `x_outlier_policy`: `winsorize`, `trim`, `iqr_clip`, `mad_clip`, `zscore_clip`, `outlier_to_missing`
- `scaling_policy`: `standard`, `robust`, `minmax`, `demean_only`, `unit_variance_only`
- `dimensionality_reduction_policy`: `pca`, `static_factor`
- `feature_selection_policy`: `correlation_filter`, `lasso_select`
- `additional_preprocessing`: `hp_filter`
- `x_lag_creation`: `fixed_x_lags`

Important caveat: these helpers are wired through `_apply_raw_panel_preprocessing`, which is used by raw-panel style feature builders. They are not a drop-in extension of the default autoregressive t-code path.

### `tcode_then_extra_preprocess`

Status: executable for supported raw-panel feature-builder paths.

This is the natural contract researchers expect for "official FRED transform,
then scale/impute/select features." The public recipe should express the
official transform through Layer 1 axes:

- `official_transform_policy='dataset_tcode'`
- `official_transform_scope` in `apply_tcode_to_target`, `apply_tcode_to_X`, or
  `apply_tcode_to_both`

When a supported Layer 2 extra-preprocessing axis is non-neutral, the compiler
derives the runtime bridge:

- `tcode_policy='tcode_then_extra_preprocess'`
- `preprocess_order='tcode_then_extra'`
- `representation_policy='tcode_only'`

Runtime order:

1. Layer 1 applies official dataset t-codes to the selected frame.
2. Layer 2 builds supervised train/test slices from that official frame.
3. Layer 2 fits supported extra preprocessing on each training slice only and
   applies it to the prediction slice.

Constraints:

- Supported extra preprocessing is X-side only: X missing, X outlier, scaling,
  HP filter, fixed X lags, dimensionality reduction, or feature selection.
- Target-side missing/outlier handling remains non-executable.
- Target normalization beyond `none` remains registry-only until normalization
  is fit inside each training window.
- `inverse_transform_policy` must remain `none`.
- `evaluation_scale` must remain `raw_level` or `original_scale`.
- Legacy bridge fields remain compatibility fields. New recipes should set the
  Layer 1 official-transform axes and let the compiler derive the bridge.

These helpers operate after the selected official frame or raw-panel feature
frame has been handed to Layer 2. They should not be documented as raw-source
cleaning unless the recipe explicitly chooses a raw-only preprocessing path and
records that the action occurred before any official transform/T-code step.

## Registry Status Cleanup

The registry now marks representable-but-not-executable Layer 2 values as
`registry_only` instead of `operational`.

| Axis/value | Registry status | Runtime status |
|------------|-----------------|----------------|
| `target_missing_policy='em_impute'` | `registry_only` | not executable as a target-side preprocessing path |
| `x_lag_creation='cv_selected_x_lags'` | `registry_only` | execution supports `no_x_lags` and `fixed_x_lags` |
| `feature_grouping='fred_category_group'` | `registry_only` | governance blocks non-`none` feature grouping |
| `feature_grouping='lag_group'` | `registry_only` | governance blocks non-`none` feature grouping |
| `evaluation_scale='transformed_scale'` | `registry_only` | inverse/evaluation-scale semantics are not finalized |
| `evaluation_scale='both'` | `registry_only` | dual-scale reporting is not implemented |
| `target_normalization='zscore_train_only'` | `registry_only` | helper exists, but runtime does not fit normalization per training window |
| `target_normalization='robust_zscore'` | `registry_only` | helper exists, but runtime does not fit normalization per training window |
| `separation_rule='shared_transform_then_split'` | `registry_only` | helper exists, but main execution does not dispatch this rule |
| `separation_rule='X_only_transform'` | `registry_only` | helper exists, but main execution does not dispatch this rule |
| `separation_rule='target_only_transform'` | `registry_only` | helper exists, but main execution does not dispatch this rule |

These values stay in the grammar so full research designs remain representable,
but public docs should not describe them as executable choices until runtime
support exists.

## Custom Preprocessor

Fixed custom preprocessors are executable in the MVP.

Contract:

```python
fn(X_train, y_train, X_test, context) -> (X_train_new, X_test_new)
```

Rules:

- `y_train` is read-only context.
- The custom preprocessor must not transform the target.
- Fit preprocessing decisions on `X_train` only.
- Return one-row `X_test`.

This is the safest current extension point for custom preprocessing. It lets researchers add methods without touching registry files, while keeping leakage discipline local to the runtime split.

## Why Simple Preprocessing Sweeps Stay Blocked

The simple API now blocks preprocessing sweeps before execution.

Reasons:

1. Default preprocessing is still `dataset_tcode_only`.
2. `dataset_tcode_then_train_only_extra` is executable only as a fixed
   preprocessing contract, not as a public simple sweep.
3. Co-sweeping model and preprocessing is explicitly rejected by governance for
   ordinary baseline comparison.
4. Target-side normalization, inverse, and evaluation-scale semantics are not finalized.
5. Some full grammar values remain `registry_only`.

Therefore, the executable MVP is:

- default `tcode_only`
- fixed `dataset_tcode_then_train_only_extra` for supported raw-panel paths
- model sweeps
- fixed custom model
- fixed custom preprocessor
- fixed target transformer for the autoregressive path
- optional FRED-SD inferred t-codes

The blocked MVP surface is:

- built-in preprocessing sweeps
- model and preprocessing co-sweeps
- target-side preprocessing beyond the target-transformer protocol

## Required Work Before Opening Preprocessing Sweeps

1. Decide the public preprocessing profiles:
   - `dataset_tcode_only`
   - `raw_train_only_extra`
   - `dataset_tcode_then_train_only_extra`

2. Keep registry statuses aligned with runtime:
   - keep representable-but-not-executable values as `registry_only`
   - promote values only with execution tests

3. Define sweep governance:
   - preprocessing-only sweep with fixed model
   - model-only sweep
   - explicit advanced grid for model x preprocessing, if allowed

4. Add remaining acceptance tests:
   - fixed extra preprocessing with `raw_feature_panel`
   - preprocessing-only sweep once governance allows it
   - model x preprocessing co-sweep blocked unless explicitly advanced

## Recommendation

Do not expose built-in preprocessing sweeps in the simple docs yet.

Next implementation target should be the `PreprocessContract` bridge cleanup:
keep Layer 1 official-transform axes as the public source of truth, keep legacy
Layer 2 t-code fields accepted for compatibility, and progressively remove
runtime dependence on those bridge fields after compiled manifests and tests no
longer need them.
