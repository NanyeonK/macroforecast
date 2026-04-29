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
and leakage discipline. It does not own dataset identity, FRED data
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
| Research feature representation | `feature_builder`, `predictor_family`, `data_richness_mode`, `factor_count`, `feature_block_set`, `target_lag_block`, `x_lag_feature_block`, `factor_feature_block`, `factor_rotation_order`, `level_feature_block`, `rotation_feature_block`, `temporal_feature_block`, `feature_block_combination`, `fred_sd_mixed_frequency_representation` | Which feature matrix `Z` is constructed from Layer 1 outputs before forecasting. Current runtime reads explicit Layer 2 blocks first and keeps migrated compatibility names as fallback/provenance; fixed target lags, fixed X lags, FRED-SD mixed-frequency panel shaping, level add-backs, deterministic temporal blocks, static PCA factors, PCA factor lags, supervised PLS factors, registered custom feature blocks, registered custom combiners, custom-block final-`Z` selection, MARX basis replacement, `marx_then_factor`, `factor_then_marx`, MARX append composition, MAF factor-score rotation, and `moving_average_rotation` execute in their supported slices. Unregistered custom blocks/combiners remain gated. |
| X additional preprocessing | `x_missing_policy`, `x_outlier_policy`, `scaling_policy`, `scaling_scope`, `additional_preprocessing`, `x_lag_creation` | How predictor columns are imputed, clipped, scaled, filtered, or lag-augmented after Layer 1. |
| X representation and selection | `dimensionality_reduction_policy`, `feature_selection_policy`, `feature_grouping` | Whether the predictor panel is reduced to factors/components, screened to a subset, or grouped before modeling. |
| Target-side preprocessing | `horizon_target_construction`, `target_transform`, `target_normalization`, `target_domain`, `target_missing_policy`, `target_outlier_policy`, `inverse_transform_policy`, `evaluation_scale`, `target_transformer` | How the target is constructed, transformed, normalized, inverted, and evaluated. |
| Preprocessing order and leakage discipline | `preprocess_order`, `preprocess_fit_scope`, `separation_rule` | Whether extra preprocessing is applied before/after official transforms, and whether each step is fit on train-only data. |
| Custom extension hooks | `custom_preprocessor`, `target_transformer` | Researcher-supplied predictor-side and target-side preprocessing protocols when built-ins are insufficient. |
| Legacy representation bridge | `target_transform_policy`, `x_transform_policy`, `tcode_policy`, `representation_policy`, `tcode_application_scope` | Compatibility fields that still help the runtime `PreprocessContract` represent raw vs official T-code frames. They are not the canonical place to choose official transforms in new recipes. |

The natural full Layer 2 profile is
`apply_official_tcode_then_train_only_extra`: Layer 1 applies official FRED-MD/QD
transforms/T-codes, then Layer 2 applies researcher-selected imputation,
scaling, filtering, dimensionality reduction, feature selection, or custom
preprocessing under train-only fit discipline. That profile is the current
generic preprocessing support surface, not the full research feature-block grammar.

Built-in Layer 2 choices should stay aligned with macro-forecasting research:
FRED-provided transforms, X-side imputation, scaling, filtering, PCA/static
factor extraction, feature screening, fixed lag construction, level add-backs,
lag rotations, local temporal factors, and custom hooks for researcher
extensions. Named papers such as Goulet Coulombe et al. (2021) should be
represented as presets over general feature-block primitives, not as layer names.

## Current Implementation Surface

The table below is the implementation status, not the boundary definition. It
records what the current runtime can execute today.

| Axis | Executable values today | Notes |
|------|-------------------------|-------|
| `feature_builder` | `target_lag_features`, `factors_plus_target_lags`, `raw_feature_panel`, `raw_predictors_only`, `pca_factor_features` | Compatibility/source bridge for old recipes. Runtime dispatch now derives the supported feature runtime from Layer 2 blocks first, with this value retained as fallback/provenance. `sequence_tensor` is future. |
| `fred_sd_mixed_frequency_representation` | `calendar_aligned_frame`, `drop_unknown_native_frequency`, `drop_non_target_native_frequency`, operational-narrow `native_frequency_block_payload`, operational-narrow `mixed_frequency_model_adapter` | FRED-SD input-panel shaping after Layer 1 frequency report/policy and generic frequency conversion, before FRED-SD inferred t-codes and feature construction. The advanced block/adapter routes require FRED-SD data, `feature_builder=raw_feature_panel`, direct forecasts, and a registered custom Layer 3 model or built-in `midas_almon`, `midasr`, or `midasr_nealmon` executor. |
| `target_lag_block` / `target_lag_selection` | `none`, `fixed_target_lags` / `none`, `fixed` | Fixed target-lag construction is executable from the explicit block first. It can run as the standalone target-lag runtime or concatenate into raw-panel/factor-panel direct `Z` after X-side block construction. Legacy target-lag fields remain fallback/provenance. IC, CV, horizon-specific, and custom lag selection remain registry-only. |
| `x_lag_feature_block` | `none`, `fixed_predictor_lags` | Fixed predictor lags are executable from the explicit block first with origin-aligned prediction lags; `x_lag_creation` remains a compatibility fallback. |
| `factor_feature_block` | `none`, `pca_static_factors`, `pca_factor_lags`, `supervised_factors`, registered `custom_factors` | Static PCA factors are executable from the explicit block first; PCA factor lags append lagged factor scores; supervised factors use a train-window PLS factor block; registered custom factor callables append or replace named feature blocks. Old factor builders and raw-panel `dimensionality_reduction_policy=pca` / `static_factor` remain compatibility fallbacks. Runtime writes factor fit-state/loadings provenance for the latest recursive window. |
| `level_feature_block` | `none`, `target_level_addback`, `x_level_addback`, `selected_level_addbacks`, `level_growth_pairs` | Level add-backs are executable for raw-panel feature runtimes. Target add-back appends observed `target_t` / `target_origin`; X-level add-back appends raw-level `H` predictor values preserved before FRED transforms/T-codes; selected subset add-back restricts those columns via `leaf_config.selected_level_addback_columns`; level-growth pairs record existing transformed predictor columns with raw-level counterparts from `leaf_config.level_growth_pair_columns`. |
| `temporal_feature_block` | `none`, `moving_average_features`, `rolling_moments`, `local_temporal_factors`, `volatility_features`, registered `custom_temporal_features` | Moving-average, rolling-moment, local-temporal-factor, and volatility temporal features are executable for raw-panel feature runtimes with trailing 3-period `{predictor}_ma3`, `{predictor}_mean3` / `{predictor}_var3`, deterministic `local_temporal_factor_mean3` / `local_temporal_factor_dispersion3`, and `{predictor}_vol3` features. Registered custom temporal callables execute under `custom_feature_block_callable_v1`. They can compose with fixed X lags, `moving_average_rotation`, and MARX append mode. |
| Other feature-block primitive axes | `rotation_feature_block=none`, `moving_average_rotation`, `marx_rotation`, `maf_rotation`, registered `custom_rotation` | Non-rotated feature representation is executable and records explicit no-rotation provenance when selected. `moving_average_rotation` is executable for raw-panel feature runtimes as deterministic trailing 3- and 6-period rotations of each active predictor and can compose with fixed X lags plus deterministic temporal append blocks. `marx_rotation` is executable for raw-panel feature runtimes when `leaf_config.marx_max_lag` is set; it builds `lag_polynomial_rotation_contract_v1` features, replaces the source X lag-polynomial basis, supports `marx_then_factor`, supports `factor_then_marx` via `factor_rotation_order=factor_then_rotation`, and supports append/concatenate composition with fixed X lags or deterministic temporal blocks through `feature_block_combination=append_to_base_predictors` / `concatenate_named_blocks`. `maf_rotation` is executable as a factor-score moving-average composer for `pca_static_factors`. Registered custom rotations execute under `custom_feature_block_callable_v1`. `feature_block_set=factor_blocks_only` records static-factor-only representation; unknown old bridge recipes may still use `feature_builder_compatibility_bridge` as compatibility provenance. |
| `predictor_family` | `target_lags_only`, `all_macro_vars`, `category_based`, `factor_only`, `explicit_variable_list` | Canonical Layer 2 owner; runtime support is constrained by the selected feature runtime and explicit block composer coverage. |
| `data_richness_mode` | `target_lags_only`, `factors_plus_target_lags`, `high_dimensional_predictors`, `selected_sparse_predictors` | Canonical Layer 2 owner; `mixed_feature_blocks` remains registry-only. |
| `factor_count` | `fixed`, `cv_select`, `BaiNg_rule` | Canonical Layer 2 owner for factor representation dimensions. `variance_explained_rule` and `model_specific` remain registry-only. |

For decomposition/attribution, these feature-representation axes use the
canonical component name `feature_representation`. Legacy decomposition plans
that request `feature_builder` are normalized to `feature_representation`.

| `x_missing_policy` | `none`, `drop`, `drop_rows`, `drop_columns`, `drop_if_above_threshold`, `missing_indicator`, `em_impute`, `mean_impute`, `median_impute`, `ffill`, `interpolate_linear` | Executes in the raw-panel extra-preprocess path. `drop` and `drop_rows` are pass-through aliases because predictor/target row coordination happens upstream. |
| `x_outlier_policy` | `none`, `winsorize`, `trim`, `iqr_clip`, `mad_clip`, `zscore_clip`, `outlier_to_missing` | Operates on post-frame X_train/X_pred. Raw-source outlier handling belongs to Layer 1. |
| `scaling_policy` | `none`, `standard`, `robust`, `minmax`, `demean_only`, `unit_variance_only` | Fitted on X_train and applied to X_pred. |
| `scaling_scope` | `columnwise`, `global_train_only` | Other scopes are blocked by governance. |
| `additional_preprocessing` | `none`, `hp_filter` | Moving average, EMA, and bandpass are registry-only. |
| `x_lag_creation` | `no_predictor_lags`, `fixed_predictor_lags` | CV-selected and variable/category-specific lags are not wired. |
| `dimensionality_reduction_policy` | `none`, `pca`, `static_factor` | Compatibility bridge for `factor_feature_block=pca_static_factors`; built-in factor blocks support explicit `select_before_factor` and `select_after_factor` semantics when feature selection is active. |
| `feature_selection_policy` | `none`, `correlation_filter`, `lasso_selection` | Applies to raw predictor blocks directly, and can also feed built-in factor blocks through the explicit `select_before_factor` or `select_after_factor` semantics. `select_after_factor` selects over final `Z` after factor scores plus supported target-lag and deterministic append columns. |
| `feature_grouping` | `none` | Non-`none` grouping is blocked in governance. |
| `horizon_target_construction` | `future_target_level_t_plus_h`, `future_diff`, `future_logdiff`, `average_growth_1_to_h`, `average_difference_1_to_h`, `average_log_growth_1_to_h` | Point and path-average target constructions are wired through the Layer 2 target-construction contract and runtime target builder. |
| `target_transform` | `level`, `difference`, `log`, `log_difference`, `growth_rate` | Applied to the target series before model execution, with limited inverse/evaluation semantics. |
| `target_normalization` | `none`, `zscore_train_only`, `robust_zscore`, `minmax`, `unit_variance` | Built-in target normalization is fit inside each training window and inverted for forecast/evaluation artifacts as requested. |
| `target_domain` | `unconstrained` | Domain constraints are not implemented. |
| `target_missing_policy` | `none` | Target-side missing algorithms are not supported in the operational contract. |
| `target_outlier_policy` | `none` | Target-side outlier algorithms are not supported in the operational contract. |
| `inverse_transform_policy` | `none`, `target_only`, `forecast_scale_only` | Built-in inverse paths are executable for supported target transforms. Custom inverse policies remain gated. |
| `evaluation_scale` | `raw_level`, `original_scale`, `transformed_scale`, `both` | Runtime writes model/transformed/original scale prediction columns and scale-specific metric summaries. |
| `preprocess_order` | `none`, derived `official_tcode_only`, `extra_only`, `official_tcode_then_extra` | `official_tcode_then_extra` is executable for supported raw-panel extra preprocessing after Layer 1 official t-codes. |
| `preprocess_fit_scope` | `not_applicable`, `train_only` | Extra preprocessing requires `train_only` today. |
| `separation_rule` | `strict_separation` | Non-strict helper modes are registry-only until wired into the main execution loop as a general dispatcher. |
| `custom_preprocessor` | fixed registered plugin name or `none` | Predictor-side function must return transformed X_train/X_test and must not transform the target. |
| `target_transformer` | fixed registered plugin name or `none` | Executable under target-transformer constraints; raw-scale evaluation only. |

Current runtime profiles:

- `official_tcode_only`: executable default. Layer 1 chooses official transforms;
  compiler derives a runtime bridge contract with no extra Layer 2 preprocessing.
- `raw_only`: executable non-default path with no official T-code and no extra
  preprocessing.
- `raw_train_only_extra`: executable for raw-panel style feature builders using
  train-only X-side extra preprocessing.
- `apply_official_tcode_then_train_only_extra`: executable for raw-panel style feature
  builders when Layer 1 applies dataset t-codes first and Layer 2 applies
  supported train-only X-side extra preprocessing.


The main practical point: **Layer 2 is the research support layer for
additional preprocessing, and the current built-in implementation supports that
only for raw-panel style feature builders.** The default official T-code path is
executable, and supported extra preprocessing can now be attached after the
Layer 1 official T-code step through the derived
`official_tcode_then_extra_preprocess` bridge contract.
This definition pass splits the migrated representation axes into general
feature-block primitives: target-lag blocks, transformed-X lag blocks, factor
blocks, level add-backs, lag rotations, local temporal factors, volatility
blocks, and custom blocks. The split is defined in
`layer2_feature_representation.md`; the implementation sequence is defined in
`layer2_revision_plan.md`. Runtime support now reads fixed target-lag, fixed
X-lag, and static PCA factor blocks before compatibility bridge fields. Fixed
target lags can compose with raw-panel X blocks and static PCA factor blocks;
other joint/custom composition beyond the supported runtime slices is still an
implementation task.


## Full Closure Status

Layer 2 is closed for fixed full recipes under the current runtime scope:

- all Layer 2 axes have canonical ownership and honest registry status;
- `official_tcode_only`, `raw_only`, `raw_train_only_extra`, and
  `apply_official_tcode_then_train_only_extra` compile and execute where their
  constraints are satisfied;
- representable-but-not-executable values remain in the grammar as
  `registry_only`, not `operational`;
- the simple API still does not expose preprocessing sweeps.

The closed full profile is therefore a fixed comparison-cell or fixed controlled
recipe, not an arbitrary public sweep. Full recipes can represent broader
macro-forecasting research intentions, but unsupported target-side
normalization/inversion, non-strict separation rules, feature grouping,
CV-selected X lags, and dual-scale evaluation stay blocked until they receive
runtime integration and acceptance tests.

## Current Default

`macrocast-default-v1` uses the FRED-provided transformation path:

| Axis | Default |
|------|---------|
| `official_transform_policy` | `apply_official_tcode` |
| `official_transform_scope` | `target_and_predictors` |
| `target_transform_policy` | `official_tcode_transformed` |
| `x_transform_policy` | `official_tcode_transformed` |
| `tcode_policy` | `official_tcode_only` |
| `representation_policy` | `official_tcode_only` |
| `tcode_application_scope` | `target_and_predictors` |
| `preprocess_order` | `official_tcode_only` |
| `preprocess_fit_scope` | `not_applicable` |
| extra preprocessing axes | `none` |

This path is executable. It applies FRED-MD/QD dataset t-codes before the forecasting runtime. FRED-SD inferred/empirical t-codes remain opt-in and non-official.

Layer ownership after the migration pass:

- Layer 1 owns the official-frame decision through `official_transform_policy`
  and `official_transform_scope`.
- Layer 2 keeps the legacy t-code fields as a runtime compatibility bridge
  until the `PreprocessContract` no longer needs them.
- Layer 2 still owns researcher-controlled extra preprocessing after the
  FRED frame exists.

Missing/outlier boundary after the migration pass:

- Layer 1 owns raw-source missing/outlier treatment when it happens before
  FRED-provided transforms or T-codes. This is the "clean raw, then T-code"
  order, now represented by `raw_missing_policy` and `raw_outlier_policy`.
- Layer 2 owns missing imputation and outlier handling after the FRED frame
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
- Compiled manifests also record `data_task_spec["official_transform_source"]`
  so full-mode audits can distinguish canonical Layer 1 axes from legacy Layer 2
  t-code bridge inputs. Runtime t-code reports repeat whether execution used
  `data_task_spec` or legacy `PreprocessContract` fallback fields.

## Executable Contract Classes

The code currently recognizes four useful preprocessing classes.

### `official_tcode_only`

Status: executable and default.

Contract:

- `target_transform_policy='official_tcode_transformed'`
- `x_transform_policy='official_tcode_transformed'`
- `tcode_policy='official_tcode_only'`
- `representation_policy='official_tcode_only'`
- `tcode_application_scope='target_and_predictors'`
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

### `extra_preprocess_only`

Status: executable for raw-panel style feature builders, not the simple default.

Contract:

- raw target and X representation
- `tcode_policy='extra_preprocess_only'`
- `preprocess_order='extra_only'`
- `preprocess_fit_scope='train_only'`
- no target-side missing/outlier transformation

Runtime helpers exist for:

- `x_missing_policy`: `em_impute`, `mean_impute`, `median_impute`, `ffill`, `interpolate_linear`, `drop_columns`, `drop_if_above_threshold`, `missing_indicator`
- `x_outlier_policy`: `winsorize`, `trim`, `iqr_clip`, `mad_clip`, `zscore_clip`, `outlier_to_missing`
- `scaling_policy`: `standard`, `robust`, `minmax`, `demean_only`, `unit_variance_only`
- `dimensionality_reduction_policy`: `pca`, `static_factor`
- `feature_selection_policy`: `correlation_filter`, `lasso_selection`
- `additional_preprocessing`: `hp_filter`
- `x_lag_creation`: `fixed_predictor_lags`

Important caveat: these helpers are wired through `_apply_raw_panel_preprocessing`, which is used by raw-panel style feature builders. They are not a drop-in extension of the default autoregressive t-code path.

### `official_tcode_then_extra_preprocess`

Status: executable for supported raw-panel feature-builder paths.

This is the natural contract researchers expect for "official FRED transform,
then scale/impute/select features." The public recipe should express the
official transform through Layer 1 axes:

- `official_transform_policy='apply_official_tcode'`
- `official_transform_scope` in `target_only`, `predictors_only`, or
  `target_and_predictors`

When a supported Layer 2 extra-preprocessing axis is non-neutral, the compiler
derives the runtime bridge:

- `tcode_policy='official_tcode_then_extra_preprocess'`
- `preprocess_order='official_tcode_then_extra'`
- `representation_policy='official_tcode_only'`

Runtime order:

1. Layer 1 applies FRED-provided t-codes to the selected frame.
2. Layer 2 builds supervised train/test slices from that FRED frame.
3. Layer 2 fits supported extra preprocessing on each training slice only and
   applies it to the prediction slice.

Constraints:

- Supported extra preprocessing is X-side only: X missing, X outlier, scaling,
  HP filter, fixed X lags, dimensionality reduction, or feature selection.
- Target-side missing/outlier handling remains non-executable.
- Built-in target normalization, target-only inverse paths, transformed-scale
  evaluation, and dual-scale evaluation are executable with per-window fit state.
- Legacy bridge fields remain compatibility fields. New recipes should set the
  Layer 1 official-transform axes and let the compiler derive the bridge.

These helpers operate after the selected FRED frame or raw-panel feature
frame has been handed to Layer 2. They should not be documented as raw-source
cleaning unless the recipe explicitly chooses a raw-only preprocessing path and
records that the action occurred before any official transform/T-code step.

## Registry Status Cleanup

The registry now marks representable-but-not-executable Layer 2 values as
`registry_only` instead of `operational`.

| Axis/value | Registry status | Runtime status |
|------------|-----------------|----------------|
| `target_missing_policy='em_impute'` | `registry_only` | not executable as a target-side preprocessing path |
| `x_lag_creation='cv_selected_predictor_lags'` | `registry_only` | execution supports `no_predictor_lags` and `fixed_predictor_lags` |
| `feature_grouping='fred_category_group'` | `registry_only` | governance blocks non-`none` feature grouping |
| `feature_grouping='lag_group'` | `registry_only` | governance blocks non-`none` feature grouping |
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

1. Default preprocessing is still `official_tcode_only`.
2. `apply_official_tcode_then_train_only_extra` is executable only as a fixed
   preprocessing contract, not as a public simple sweep.
3. Co-sweeping model and preprocessing is explicitly rejected by governance for
   ordinary baseline comparison.
4. Target-side built-in normalization, inverse, and evaluation-scale semantics are executable.
5. Some full grammar values remain `registry_only`.

Therefore, the executable MVP is:

- default `official_tcode_only`
- fixed `apply_official_tcode_then_train_only_extra` for supported raw-panel paths
- model sweeps
- fixed custom model
- fixed custom preprocessor
- fixed target transformer for the autoregressive path
- optional FRED-SD mixed-frequency panel shaping and inferred/empirical t-codes

The blocked MVP surface is:

- built-in preprocessing sweeps
- model and preprocessing co-sweeps
- target-side preprocessing beyond the target-transformer protocol

## Required Work Before Opening Preprocessing Sweeps

1. Decide the public preprocessing profiles:
   - `official_tcode_only`
   - `raw_train_only_extra`
   - `apply_official_tcode_then_train_only_extra`

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

Layer 2 cleanup is closed for supported fixed full/runtime slices. Layer 1
official-transform axes remain the public source of truth, legacy Layer 2
t-code fields remain accepted for compatibility, and supported Layer 2 runtime
paths now expose explicit block/runtime provenance.

Next implementation targets should be semantic feature-composer tasks rather
than bridge cleanup: broader factor/selection composition beyond static PCA,
custom rotations, custom callable contracts, target-side
normalization/evaluation-scale expansion, and public sweep governance. The
detailed Layer 2 x Layer 3 free-sweep contract is in
`layer2_layer3_sweep_contract.md`.
