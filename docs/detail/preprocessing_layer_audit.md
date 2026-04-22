# Preprocessing Layer Audit

This audit records the current preprocessing contract after the `Experiment` MVP pass.

The practical question is whether the simple API can safely expose preprocessing sweeps. Current answer: not yet.

## Canonical Layer 2 Role

Layer 2 is the researcher additional-preprocessing layer. Layer 1 produces the
baseline official or raw feature frame: dataset/source/frequency, information
set, target/horizon/sample window, release-lag availability, official
transform/T-code policy, raw-source missing/outlier repair before T-codes, and
the eligible variable universe. Layer 2 starts after that point.

The purpose of Layer 2 is to support research designs that ask how forecasts
change when the researcher applies additional preprocessing before model
estimation. It owns optional transformations of X and y, feature engineering,
dimensionality reduction, target-scale handling, preprocessing order, and
leakage discipline. It does not own dataset identity, official data
availability, model family, benchmark family, scoring metrics, or statistical
tests.

This means the canonical Layer 2 question is not "what preprocessing happens by
default?" but "what additional preprocessing design does the researcher want to
study after Layer 1 has created the baseline frame?"

## Layer 2 Decision Space

Layer 2 has 26 registry axes today. The canonical decision groups are:

| Group | Axes | What the group decides |
|-------|------|------------------------|
| X additional preprocessing | `x_missing_policy`, `x_outlier_policy`, `scaling_policy`, `scaling_scope`, `additional_preprocessing`, `x_lag_creation` | How predictor columns are imputed, clipped, scaled, filtered, or lag-augmented after Layer 1. |
| X representation and selection | `dimensionality_reduction_policy`, `feature_selection_policy`, `feature_grouping` | Whether the predictor panel is reduced to factors/components, screened to a subset, or grouped before modeling. |
| Target-side preprocessing | `target_transform`, `target_normalization`, `target_domain`, `target_missing_policy`, `target_outlier_policy`, `inverse_transform_policy`, `evaluation_scale`, `target_transformer` | How y is transformed or normalized, how forecasts are inverted, and which scale metrics use. |
| Preprocessing order and leakage discipline | `preprocess_order`, `preprocess_fit_scope`, `separation_rule` | Whether extra preprocessing is applied before/after official transforms, and whether each step is fit on train-only data. |
| Custom extension hooks | `custom_preprocessor`, `target_transformer` | Researcher-supplied X-side and y-side preprocessing protocols when built-ins are insufficient. |
| Legacy representation bridge | `target_transform_policy`, `x_transform_policy`, `tcode_policy`, `representation_policy`, `tcode_application_scope` | Compatibility fields that still help the runtime `PreprocessContract` represent raw vs official T-code frames. They are not the canonical place to choose official transforms in new recipes. |

The natural full Layer 2 profile is
`dataset_tcode_then_train_only_extra`: Layer 1 applies official FRED-MD/QD
transforms/T-codes, then Layer 2 applies researcher-selected imputation,
scaling, filtering, dimensionality reduction, feature selection, or custom
preprocessing under train-only fit discipline. That profile is the target
contract for preprocessing research support.

## Current Implementation Surface

The table below is the implementation status, not the boundary definition. It
records what the current runtime can execute today.

| Axis | Executable values today | Notes |
|------|-------------------------|-------|
| `x_missing_policy` | `none`, `drop`, `drop_rows`, `drop_columns`, `drop_if_above_threshold`, `missing_indicator`, `em_impute`, `mean_impute`, `median_impute`, `ffill`, `interpolate_linear` | Executes in the raw-panel extra-preprocess path. `drop` and `drop_rows` are pass-through aliases because X/y row coordination happens upstream. |
| `x_outlier_policy` | `none`, `winsorize`, `trim`, `iqr_clip`, `mad_clip`, `zscore_clip`, `outlier_to_missing` | Operates on post-frame X_train/X_pred. Raw-source outlier handling belongs to Layer 1. |
| `scaling_policy` | `none`, `standard`, `robust`, `minmax`, `demean_only`, `unit_variance_only` | Fitted on X_train and applied to X_pred. |
| `scaling_scope` | `columnwise`, `global_train_only` | Other scopes are blocked by governance. |
| `additional_preprocessing` | `none`, `hp_filter` | Moving average, EMA, and bandpass are registry-only. |
| `x_lag_creation` | `no_x_lags`, `fixed_x_lags` | CV-selected and variable/category-specific lags are not wired. |
| `dimensionality_reduction_policy` | `none`, `pca`, `static_factor` | Cannot be combined with feature selection. |
| `feature_selection_policy` | `none`, `correlation_filter`, `lasso_select` | Cannot be combined with dimensionality reduction. |
| `feature_grouping` | `none` | Non-`none` grouping is blocked in governance. |
| `target_transform` | `level`, `difference`, `log`, `log_difference`, `growth_rate` | Applied to the target series before model execution, with limited inverse/evaluation semantics. |
| `target_normalization` | `none`, `zscore_train_only`, `robust_zscore` | Current support is narrow; no general metric-scale system yet. |
| `target_domain` | `unconstrained` | Domain constraints are not implemented. |
| `target_missing_policy` | `none` | Target-side missing algorithms are not supported in the operational contract. |
| `target_outlier_policy` | `none` | Target-side outlier algorithms are not supported in the operational contract. |
| `inverse_transform_policy` | `none` | Inverse-transform policy needs a separate target/evaluation contract. |
| `evaluation_scale` | `raw_level`, `original_scale` | `transformed_scale` and `both` are overexposed by registry today. |
| `preprocess_order` | `none`, derived `tcode_only`, `extra_only` | `tcode_then_extra` is the desired full profile but not executable yet. |
| `preprocess_fit_scope` | `not_applicable`, `train_only` | Extra preprocessing requires `train_only` today. |
| `separation_rule` | helper supports several values | The helper is tested, but not wired into the main execution loop as a general dispatcher. |
| `custom_preprocessor` | fixed registered plugin name or `none` | X-side function must return transformed X_train/X_test and must not transform y. |
| `target_transformer` | fixed registered plugin name or `none` | Executable under target-transformer constraints; raw-scale evaluation only. |

Current runtime profiles:

- `dataset_tcode_only`: executable default. Layer 1 chooses official transforms;
  compiler derives a runtime bridge contract with no extra Layer 2 preprocessing.
- `raw_only`: executable non-default path with no official T-code and no extra
  preprocessing.
- `raw_train_only_extra`: executable for raw-panel style feature builders using
  train-only X-side extra preprocessing.
- `dataset_tcode_then_train_only_extra`: canonical Layer 2 research target, but
  not executable yet.

The main practical point: **Layer 2 is the research support layer for
additional preprocessing, but current built-in extra preprocessing executes only
in the raw-panel `extra_preprocess_without_tcode` profile.** The default
official T-code path is executable, but extra preprocessing cannot yet be
attached after T-code.

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
