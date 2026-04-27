# 4.3 Layer 2: Representation / Research Preprocessing

- Parent: [4. Detail (code): Full](../index.md)
- Previous: [4.2 Layer 1: Data Task](../layer1/index.md)
- Current: Layer 2
- Next: [4.4 Layer 3: Forecast Generator](../layer3/index.md)

Layer 2 owns representation construction after Layer 1 produces the official frame. It supports research preprocessing choices such as t-code handling, target construction, missing/outlier handling after the official frame, scaling, feature blocks, factor blocks, lag blocks, rotations, feature selection, and custom representation hooks.

## Decision order

| Group | Axes |
|---|---|
| FRED-SD mixed frequency | `fred_sd_mixed_frequency_representation` |
| Target construction | `horizon_target_construction`, `target_transform`, `target_normalization` |
| Transform and cleaning | `tcode_policy`, `x_missing_policy`, `x_outlier_policy`, `scaling_policy` |
| Feature blocks | `target_lag_block`, `x_lag_feature_block`, `factor_feature_block`, `level_feature_block`, `temporal_feature_block`, `rotation_feature_block` |
| Composition and selection | `feature_block_combination`, `feature_selection_policy`, `feature_selection_semantics` |
| Handoff | `evaluation_scale`, `feature_builder` compatibility bridge |

## Current naming migration

Layer 2 IDs now prefer researcher-facing predictor and target-lag language.
Old recipe values are still accepted through `registry_naming_v1`; compiled
recipes, Navigator paths, and generated YAML emit the canonical names below.

| Axis | Legacy value | Canonical value |
|---|---|---|
| `feature_builder` | `autoreg_lagged_target` | `target_lag_features` |
| `feature_builder` | `factors_plus_AR` | `factors_plus_target_lags` |
| `feature_builder` | `raw_X_only` | `raw_predictors_only` |
| `feature_builder` | `factor_pca` | `pca_factor_features` |
| `data_richness_mode` | `full_high_dimensional_X` | `high_dimensional_predictors` |
| `data_richness_mode` | `selected_sparse_X` | `selected_sparse_predictors` |
| `data_richness_mode` | `mixed_mode` | `mixed_feature_blocks` |
| `feature_block_set` | `legacy_feature_builder_bridge` | `feature_builder_compatibility_bridge` |
| `feature_block_set` | `transformed_x` | `transformed_predictors` |
| `feature_block_set` | `transformed_x_lags` | `transformed_predictor_lags` |
| `feature_block_set` | `high_dimensional_x` | `high_dimensional_predictors` |
| `feature_block_set` | `selected_sparse_x` | `selected_sparse_predictors` |
| `feature_block_set` | `level_augmented_x` | `level_augmented_predictors` |
| `feature_block_set` | `rotation_augmented_x` | `rotation_augmented_predictors` |
| `feature_block_set` | `mixed_blocks` | `mixed_feature_blocks` |
| `feature_block_set` | `custom_blocks` | `custom_feature_blocks` |
| `x_lag_creation` | `no_x_lags` | `no_predictor_lags` |
| `x_lag_creation` | `fixed_x_lags` | `fixed_predictor_lags` |
| `x_lag_creation` | `cv_selected_x_lags` | `cv_selected_predictor_lags` |
| `x_lag_creation` | `variable_specific_lags` | `variable_specific_predictor_lags` |
| `x_lag_creation` | `category_specific_lags` | `category_specific_predictor_lags` |
| `x_lag_feature_block` | `fixed_x_lags` | `fixed_predictor_lags` |
| `x_lag_feature_block` | `variable_specific_x_lags` | `variable_specific_predictor_lags` |
| `x_lag_feature_block` | `category_specific_x_lags` | `category_specific_predictor_lags` |
| `x_lag_feature_block` | `cv_selected_x_lags` | `cv_selected_predictor_lags` |
| `x_lag_feature_block` | `custom_x_lags` | `custom_predictor_lags` |
| `feature_block_combination` | `replace_with_blocks` | `replace_with_selected_blocks` |
| `feature_block_combination` | `append_to_base_x` | `append_to_base_predictors` |
| `feature_block_combination` | `custom_combiner` | `custom_feature_combiner` |
| `feature_selection_policy` | `lasso_select` | `lasso_selection` |
| `feature_selection_semantics` | `select_after_custom_blocks` | `select_after_custom_feature_blocks` |

## Layer contract

Input:
- Layer 1 official frame and target task.

Output:
- `layer2_representation_v1`;
- feature matrices and representation metadata consumed by Layer 3;
- auxiliary payloads for narrow advanced routes.

## Related reference

- [Layer 2 Feature Representation](../layer2_feature_representation.md)
- [Layer 2 Closure Ledger](../layer2_closure_ledger.md)
- [Layer 2 / Layer 3 Sweep Contract](../layer2_layer3_sweep_contract.md)
