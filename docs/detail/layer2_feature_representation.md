# Layer 2 Feature Representation

Date: 2026-04-22

Layer 2 is the research preprocessing and feature-representation layer. Its job
is to turn the Layer 1 official data frame into the model input matrices used by
Layer 3.

## Contract

Layer 1 provides:

- `H`: the available level/raw-style panel after source loading, sample
  restriction, information-set rules, release-lag handling, variable-universe
  filtering, and any raw-source missing/outlier repair selected before official
  transforms;
- `X`: the official transformed predictor frame when dataset/FRED-MD/QD
  transformation codes are applied;
- `Y`: the target series and horizon-aligned target construction inputs;
- provenance for availability, official transforms, and raw-source repair.

Layer 2 must produce, for each training window and forecast origin:

- `Z_train`: the feature matrix fit by Layer 3;
- `Z_pred`: the one-origin or batch prediction feature matrix;
- `feature_names`: stable names for every column in `Z`;
- `feature_block_metadata`: which blocks generated each feature;
- fit state for any imputer, scaler, selector, factor extractor, rotation, or
  custom preprocessor that was fit on the training window.

Any operation that estimates parameters from data must be recursive or
train-only for the relevant forecast origin. Full-sample transformations are
allowed only when a recipe is explicitly a fixed replication profile and the
provenance marks that choice.

## What Layer 2 Owns

Layer 2 owns four types of decisions.

| Group | Canonical questions | Existing bridge axes |
|---|---|---|
| Frame conditioning | How are post-official-frame missing values, outliers, scaling, filters, and target transforms handled? | `x_missing_policy`, `x_outlier_policy`, `scaling_policy`, `additional_preprocessing`, `target_transform`, `target_transformer` |
| Feature-block construction | Which blocks are built from `H`, `X`, and `Y` before forecasting? | `feature_builder`, `x_lag_creation`, `dimensionality_reduction_policy`, `feature_selection_policy` |
| Block composition | Which blocks are included in `Z`, and how are they concatenated or substituted? | `predictor_family`, `data_richness_mode`, `feature_grouping` |
| Representation dimensions and leakage discipline | How many factors/lags/features are used, and where are transforms fit? | `factor_count`, `preprocess_fit_scope`, `separation_rule` |

Layer 2 does not own model family, benchmark family, direct/iterated forecast
generation, validation split, hyperparameter search, scoring metrics,
statistical tests, or feature-importance interpretation.

## Feature-Block Grammar

The current runtime still executes through the coarse `feature_builder` bridge.
This pass defines the canonical feature-block grammar as registry-only axes so
recipes can name research intentions before runtime support is widened.

| Axis | Values | Meaning |
|---|---|---|
| `feature_block_set` | `legacy_feature_builder_bridge`, `target_lags_only`, `transformed_x`, `transformed_x_lags`, `factors_plus_target_lags`, `high_dimensional_x`, `selected_sparse_x`, `level_augmented_x`, `rotation_augmented_x`, `mixed_blocks`, `custom_blocks` | Top-level recipe for which blocks should form `Z`. |
| `target_lag_block` | `none`, `fixed_target_lags`, `ic_selected_target_lags`, `horizon_specific_target_lags`, `custom_target_lags` | Target-history features built from `Y`. |
| `x_lag_feature_block` | `none`, `fixed_x_lags`, `variable_specific_x_lags`, `category_specific_x_lags`, `cv_selected_x_lags`, `custom_x_lags` | Lagged predictor features built from `X`. |
| `factor_feature_block` | `none`, `pca_static_factors`, `pca_factor_lags`, `supervised_factors`, `custom_factors` | Reduced-rank/factor features built from `X`. |
| `level_feature_block` | `none`, `target_level_addback`, `x_level_addback`, `selected_level_addbacks`, `level_growth_pairs` | Level or level-growth add-back features built from `H` and `Y`. |
| `rotation_feature_block` | `none`, `marx_rotation`, `maf_rotation`, `moving_average_rotation`, `custom_rotation` | Rotated features such as moving-average rotations of `X` or factors. |
| `temporal_feature_block` | `none`, `moving_average_features`, `rolling_moments`, `local_temporal_factors`, `volatility_features`, `custom_temporal_features` | Local time-series features built within each training window. |
| `feature_block_combination` | `replace_with_blocks`, `append_to_base_x`, `append_to_target_lags`, `concatenate_named_blocks`, `custom_combiner` | How selected blocks are assembled into `Z`. |

All axes in this section are `registry_only` as of this definition pass. The
operational bridge remains `feature_builder` plus the existing preprocessing
contract.

## Mapping From Existing Bridge Names

The current coarse names map to the new language as follows:

| Current bridge | Feature-block interpretation |
|---|---|
| `feature_builder=autoreg_lagged_target` | target-lag block only; current runtime still lets Layer 3 select AR lag order through `y_lag_count`. |
| `feature_builder=raw_feature_panel` | transformed or raw predictor panel block, chosen after Layer 1 official-frame policy and `predictor_family`. |
| `feature_builder=raw_X_only` | predictor panel block without target-lag features. |
| `feature_builder=factor_pca` | factor feature block from the predictor panel. |
| `feature_builder=factors_plus_AR` | factor feature block plus target-lag block. |
| `data_richness_mode=target_lags_only` | `feature_block_set=target_lags_only`. |
| `data_richness_mode=factor_plus_lags` | `feature_block_set=factors_plus_target_lags`. |
| `data_richness_mode=full_high_dimensional_X` | `feature_block_set=high_dimensional_x`. |
| `data_richness_mode=selected_sparse_X` | `feature_block_set=selected_sparse_x`. |

This mapping is descriptive. Runtime code should not be changed to consume the
new axes until each block has train-window fit/apply tests and provenance.

## Boundary Cases

`y_lag_count` is split by meaning:

- if it selects AR/model order inside an estimator, it remains Layer 3;
- if it creates lagged target columns in `Z`, it belongs to
  `target_lag_block` in Layer 2.

`factor_ar_lags` is also split by meaning:

- factor or target-lag feature construction belongs to Layer 2;
- model-specific lag-order selection belongs to Layer 3.

`dimensionality_reduction_policy` and `factor_feature_block` are related but
not identical. The former is the current preprocessing contract switch used by
the runtime; the latter is the future feature-block grammar that will make
factor blocks composable with target lags, level add-backs, rotations, and
custom blocks.

## Implementation Order

A safe implementation order is:

1. Keep current `feature_builder` bridge operational.
2. Add feature-block provenance to compiled specs without changing runtime
   matrices.
3. Implement `target_lag_block` and `x_lag_feature_block` with train-window
   alignment tests.
4. Implement `factor_feature_block` with recursive factor fit/apply tests.
5. Implement `level_feature_block`, `rotation_feature_block`, and
   `temporal_feature_block` as optional blocks.
6. Move runtime dispatch from coarse `feature_builder` names to explicit block
   composition only after old recipes can be translated losslessly.
