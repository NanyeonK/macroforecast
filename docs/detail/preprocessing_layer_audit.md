# Preprocessing Layer Audit

This audit records the current preprocessing contract after the `Experiment` MVP pass.

The practical question is whether the simple API can safely expose preprocessing sweeps. Current answer: not yet.

## Layer 2 Scope

Layer 2 decides researcher-controlled transformations after Layer 1 has
produced the selected official or raw feature frame. It does not decide dataset
identity, source adapter, frequency, information set, target identity, horizons,
sample window, release-lag availability, raw-source missing/outlier repair
before T-codes, official T-code scope, model family, benchmark family, metrics,
or statistical tests.

Layer 2 has 26 registry axes today, but they are not all equally public or
equally executable. They fall into these decision groups:

| Group | Axes | What the group decides | Current status |
|-------|------|------------------------|----------------|
| Representation / T-code bridge | `target_transform_policy`, `x_transform_policy`, `tcode_policy`, `representation_policy`, `tcode_application_scope`, `preprocess_order` | Whether the runtime sees raw values, official T-code values, or extra preprocessing order. | `raw_only`, `tcode_only`, and `extra_preprocess_without_tcode` are executable. Official dataset transforms should be set by Layer 1 and derived into these fields. `tcode_then_extra_preprocess` and `extra_then_tcode` are not executable. |
| Fit and leakage discipline | `preprocess_fit_scope`, `separation_rule` | Whether preprocessing is fitted on training data only and how train/test separation is enforced. | `not_applicable` and `train_only` are executable in the main contract. `expanding_train_only` and `rolling_train_only` are registry-only. `separation_rule` has a tested helper, but it is not wired into the main execution loop as a general dispatcher. |
| X missing / outlier / scale / lag treatment | `x_missing_policy`, `x_outlier_policy`, `scaling_policy`, `scaling_scope`, `additional_preprocessing`, `x_lag_creation` | Post-frame, model-input treatment of predictor columns. | Executable only in the raw-panel extra-preprocess path. Supported subsets are listed below. |
| X reduction and selection | `dimensionality_reduction_policy`, `feature_selection_policy`, `feature_grouping` | PCA/factor representation, supervised feature selection, and future grouped feature operations. | `pca`, `static_factor`, `correlation_filter`, and `lasso_select` are executable. Dimensionality reduction and feature selection cannot be combined. `feature_grouping` is blocked beyond `none`. |
| Target-side preprocessing | `target_transform`, `target_normalization`, `target_domain`, `target_missing_policy`, `target_outlier_policy`, `inverse_transform_policy`, `evaluation_scale`, `target_transformer` | Built-in or custom transformations of y and metric scale. | Built-in target transforms/normalization are narrow and do not provide a general inverse/evaluation-scale system. `target_missing_policy` and `target_outlier_policy` are effectively `none` only in the supported contract. `target_transformer` is the executable custom y-side path under documented constraints. |
| Plugin hooks | `custom_preprocessor`, `target_transformer` | User-supplied X-side or y-side extension points. | Fixed custom preprocessors are executable. Target transformers are executable for constrained feature-builder/model paths and raw-scale evaluation. |

## What Can Be Chosen Today

The following choices are meaningful in current runtime, assuming the compiled
recipe also satisfies the surrounding model/feature-builder constraints.

| Axis | Chooseable values now | Notes |
|------|-----------------------|-------|
| `target_transform_policy` | `raw_level`, derived `tcode_transformed` | New official T-code recipes should not set this directly; Layer 1 derives it. |
| `x_transform_policy` | `raw_level`, derived `dataset_tcode_transformed` | Same bridge rule as target side. |
| `tcode_policy` | `raw_only`, derived `tcode_only`, `extra_preprocess_without_tcode` | `tcode_then_extra_preprocess` is the expected future profile but not executable. |
| `representation_policy` | `raw_only`, derived `tcode_only` | `custom_transform_only` is registry-only. |
| `tcode_application_scope` | derived from Layer 1 or `apply_tcode_to_none` for raw paths | New recipes should use Layer 1 `official_transform_scope` for official transforms. |
| `preprocess_order` | `none`, derived `tcode_only`, `extra_only` | `tcode_then_extra` and `extra_then_tcode` are registry-only. |
| `preprocess_fit_scope` | `not_applicable`, `train_only` | Extra preprocessing requires `train_only` in current runtime. |
| `x_missing_policy` | `none`, `drop`, `drop_rows`, `drop_columns`, `drop_if_above_threshold`, `missing_indicator`, `em_impute`, `mean_impute`, `median_impute`, `ffill`, `interpolate_linear` | `drop` and `drop_rows` are pass-through aliases at this layer because X/y row coordination happens upstream. |
| `x_outlier_policy` | `none`, `winsorize`, `trim`, `iqr_clip`, `mad_clip`, `zscore_clip`, `outlier_to_missing` | Operates on post-frame X_train/X_pred. Raw-source outlier handling belongs to Layer 1. |
| `scaling_policy` | `none`, `standard`, `robust`, `minmax`, `demean_only`, `unit_variance_only` | Fitted on X_train and applied to X_pred. |
| `scaling_scope` | `columnwise`, `global_train_only` | Other scopes are blocked by governance. |
| `additional_preprocessing` | `none`, `hp_filter` | HP filter is executable; moving average, EMA, bandpass are registry-only. |
| `x_lag_creation` | `no_x_lags`, `fixed_x_lags` | CV-selected and variable/category-specific lags are not wired. |
| `dimensionality_reduction_policy` | `none`, `pca`, `static_factor` | Cannot be combined with feature selection. |
| `feature_selection_policy` | `none`, `correlation_filter`, `lasso_select` | Cannot be combined with dimensionality reduction. |
| `feature_grouping` | `none` | Non-`none` grouping is blocked in governance. |
| `target_transform` | `level`, `difference`, `log`, `log_difference`, `growth_rate` | Applied to the target series before model execution, with limited inverse/evaluation semantics. |
| `target_normalization` | `none`, `zscore_train_only`, `robust_zscore` | Current support is narrow; no general metric-scale system. |
| `target_domain` | `unconstrained` | Domain constraints are not implemented. |
| `target_missing_policy` | `none` | Target-side missing algorithms are not supported in the operational contract. |
| `target_outlier_policy` | `none` | Target-side outlier algorithms are not supported in the operational contract. |
| `inverse_transform_policy` | `none` | Inverse-transform policy needs a separate target/evaluation contract. |
| `evaluation_scale` | `raw_level`, `original_scale` | `transformed_scale` and `both` are overexposed by registry today. |
| `custom_preprocessor` | fixed registered plugin name or `none` | X-side function must return transformed X_train/X_test and not transform y. |
| `target_transformer` | fixed registered plugin name or `none` | Executable under target-transformer constraints; raw-scale evaluation only. |

The main practical point: **Layer 2 can choose post-frame preprocessing
methods, but only the raw-panel extra-preprocess profile executes those built-in
methods today.** The official T-code default path is executable, but extra
preprocessing cannot yet be attached after T-code.

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

The code currently recognizes three useful preprocessing classes.

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

These helpers operate after the selected official frame or raw-panel feature
frame has been handed to Layer 2. They should not be documented as raw-source
cleaning unless the recipe explicitly chooses a raw-only preprocessing path and
records that the action occurred before any official transform/T-code step.

## Not Supported Until Wired

### `tcode_then_extra_preprocess`

This is the natural contract researchers will expect for "official FRED transform, then scale/impute/select features." It is currently not executable.

Compiler result today:

```text
tcode_then_scaling -> not_supported
```

Reasons:

- `tcode_policy='tcode_then_extra_preprocess'` is `registry_only`.
- `preprocess_order='tcode_then_extra'` is `registry_only`.
- `_apply_tcode_preprocessing` only executes `tcode_policy='tcode_only'`; it raises for other t-code policies.
- inverse-transform/evaluation semantics for target-side t-code plus extra preprocessing are not finalized.

This is the main blocker for exposing built-in preprocessing sweeps in the simple API.

## Registry vs Runtime Mismatches

Some registry entries are marked `operational` but are not actually executable in the current runtime slice.

| Axis/value | Registry status | Runtime status |
|------------|-----------------|----------------|
| `target_missing_policy='em_impute'` | `operational` | not executable as a target-side preprocessing path |
| `dimensionality_reduction_policy='ipca'` | `operational` | execution supports `pca` and `static_factor`, not `ipca` |
| `x_lag_creation='cv_selected_x_lags'` | `operational` | execution supports `no_x_lags` and `fixed_x_lags` |
| `feature_grouping='fred_category_group'` | `operational` | governance blocks non-`none` feature grouping |
| `feature_grouping='lag_group'` | `operational` | governance blocks non-`none` feature grouping |

These should be demoted to `registry_only` or implemented before any public docs describe them as available.

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

1. Default preprocessing is `tcode_only`.
2. Extra built-in preprocessing cannot be attached to `tcode_only`.
3. The expected bridge, `tcode_then_extra_preprocess`, is not executable.
4. Co-sweeping model and preprocessing is explicitly rejected by governance for ordinary baseline comparison.
5. Some registry values overstate runtime support.

Therefore, the executable MVP is:

- default `tcode_only`
- model sweeps
- fixed custom model
- fixed custom preprocessor
- fixed target transformer for the autoregressive path
- optional FRED-SD inferred t-codes

The blocked MVP surface is:

- built-in preprocessing sweeps
- model and preprocessing co-sweeps
- target-side preprocessing beyond the target-transformer protocol
- t-code followed by train-only scaling/imputation/feature selection

## Required Work Before Opening Preprocessing Sweeps

1. Decide the public preprocessing profiles:
   - `dataset_tcode_only`
   - `raw_train_only_extra`
   - `dataset_tcode_then_train_only_extra`

2. Implement or demote registry values:
   - demote overstated `operational` values, or wire their runtime support.

3. Implement `tcode_then_extra_preprocess`:
   - apply dataset t-codes first
   - build supervised train/test slices afterward
   - fit extra preprocessing on each training slice only
   - record each step in manifest reports

4. Define sweep governance:
   - preprocessing-only sweep with fixed model
   - model-only sweep
   - explicit advanced grid for model x preprocessing, if allowed

5. Add acceptance tests:
   - fixed extra preprocessing with `raw_feature_panel`
   - blocked `tcode_then_extra_preprocess` until implemented
   - executable `tcode_then_extra_preprocess` after implementation
   - preprocessing-only sweep once governance allows it
   - model x preprocessing co-sweep blocked unless explicitly advanced

## Recommendation

Do not expose built-in preprocessing sweeps in the simple docs yet.

Next implementation target should be `dataset_tcode_then_train_only_extra`, because it matches the researcher expectation: start from official FRED-MD/QD transformations, then optionally sweep scaling, imputation, dimensionality reduction, or feature selection under train-only fit discipline.
