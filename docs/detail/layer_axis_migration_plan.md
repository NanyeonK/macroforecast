# Layer Axis Migration Plan

Date: 2026-04-24

This ledger records the Layer 1/2/3/4/6 boundary cleanup. `current path`
describes where many old recipes may still place the axis. `canonical owner`
describes the registry layer after migration.

## Migrated In This Pass

| Axis | Old owner | Canonical owner | Reason |
|---|---|---|---|
| `benchmark_family` | 1_data_task | 3_training compatibility axis | Benchmarks generate forecasts. Canonically this is baseline generator role assignment, not a separate model species. |
| `forecast_type` | 1_data_task | 3_training | Direct vs iterated is forecast-generation logic. |
| `forecast_object` | 1_data_task | 3_training | Mean/median/quantile is model output contract. |
| `predictor_family` | 1_data_task / 3_training | 2_preprocessing | Predictor family chooses the feature representation before model fitting. |
| `min_train_size` | 1_data_task | 3_training | Minimum training window is training protocol. |
| `training_start_rule` | 1_data_task | 3_training | Training start is model training protocol, distinct from sample period. |
| `horizon_target_construction` | 1_data_task | 2_preprocessing | Diff/logdiff/level target construction is target representation. |
| `deterministic_components` | 1_data_task | 2_preprocessing | Trends, seasonals, and break dummies are feature construction. |
| `structural_break_segmentation` | 1_data_task | 2_preprocessing | Current implementation adds break dummy features. |
| `feature_builder` | 3_training | 2_preprocessing | Feature builders construct `Z`; Layer 3 only consumes `Z` to fit/predict. |
| `data_richness_mode` | 3_training | 2_preprocessing | Data richness describes representation breadth, not estimator behavior. |
| `factor_count` | 3_training | 2_preprocessing | Factor count is a representation dimension for factor blocks. |
| `oos_period` | 1_data_task | 4_evaluation | Recession/expansion-only selection is evaluation subset filtering. |
| `overlap_handling` | 1_data_task | 6_stat_tests | HAC/overlap handling is inference over dependent forecast errors. |
| `official_transform_policy` | split from Layer 2 t-code axes | 1_data_task | Official dataset transformations define the official frame, before researcher preprocessing. |
| `official_transform_scope` | split from `tcode_application_scope` | 1_data_task | Target/X official transform scope is part of official frame construction. |
| `source_adapter` | `dataset_source` | 1_data_task | Loader dispatch is an adapter choice; `dataset` remains the schema identity. |
| `target_structure` | `task` | 1_data_task | Layer 1 records target cautoregressive_diffusion_indexnality; Layer 0 owns runner grammar. |

## Still To Migrate

| Axis / concept | Current owner | Target owner | Note |
|---|---|---|---|
| `model_family` terminology | 3_training compatibility axis | 3_training `forecast_generator_family` concept | Keep the public axis for recipe/API compatibility. Docs should describe it as the candidate forecast generator family. |
| `benchmark_family` terminology | 3_training compatibility axis | 3_training baseline generator role assignment | Keep the public axis for recipe/API compatibility. Docs should describe it as forecast-generator role assignment, not a separate family type. |
| broad runtime discipline wording | split across 0/1/3 | explicit split | Layer 0 owns run control; Layer 1 owns data timing/information-set discipline; Layer 3 owns estimator training discipline only. |
| legacy official t-code bridge fields | 2_preprocessing | compatibility bridge | Keep accepting `target_transform_policy`, `x_transform_policy`, `tcode_policy=official_tcode_only`, `tcode_application_scope`, and `preprocess_order=official_tcode_only` while generated recipes move to Layer 1 official-transform axes. |
| `tcode_policy` values beyond official transform | 2_preprocessing | 2_preprocessing | Keep extra/custom transform pipelines in Layer 2. |
| `preprocess_order=official_tcode_only` | 2_preprocessing | compatibility bridge | Official-only order is represented by Layer 1 `official_transform_policy=apply_official_tcode`; extra orders remain Layer 2. |
| `y_lag_count` | 3_training | split in provenance | AR model-order selection remains Layer 3 for now; fixed target-lag feature construction is recorded with Layer 2 `target_lag_selection` / `target_lag_block` provenance and now executes through the explicit target-lag block path when available. |
| `factor_ar_lags` leaf/training config | 3_training config | legacy compatibility alias | New compiled specs split it into Layer 2 `target_lag_count` and factor-block `factor_lag_count`; runtime still accepts the legacy key for old recipes. |
| Layer 2 fields still emitted in `training_spec` | 3_training-shaped compatibility payload | 2_preprocessing / compatibility aliases | Cleanup passes moved `data_richness_mode`, `target_lag_selection`, `target_lag_count`, `custom_preprocessor`, `target_transformer`, `factor_count`, `fixed_factor_count`, `max_factors`, and `factor_ar_lags` meanings into `layer2_representation_spec` for newly compiled specs. |
| Layer 3 fields still mirrored in `data_task_spec` | 1_data_task-shaped compatibility payload | 3_training | Done for new compiled specs: `forecast_type`, `forecast_object`, `min_train_size`, and `training_start_rule` now emit in `training_spec`; runtime keeps `data_task_spec` fallback for old recipes. |
| Layer 2 fields still mirrored in `data_task_spec` | 1_data_task-shaped compatibility payload | 2_preprocessing | Done for the main feature/target representation fields: `horizon_target_construction`, `predictor_family`, `contemporaneous_x_rule`, `deterministic_components`, and `structural_break_segmentation` now emit in `layer2_representation_spec`; runtime keeps `data_task_spec` fallback for old recipes. |

## Feature-Block Grammar Introduced

This pass defines the Layer 2 feature-block grammar and starts retiring the old
bridge as the runtime owner. The new axes are canonical Layer 2 concepts.
Supported slices now read explicit blocks first and keep old bridge fields as
fallback/provenance until each joint block composer has train-window fit/apply
tests.

| Axis | Canonical owner | Current support |
|---|---|---|
| `target_lag_block`, `target_lag_selection`, `x_lag_feature_block` | 2_preprocessing | `none` and fixed-lag values operational from explicit blocks first; fixed target lags can compose with raw-panel X blocks, fixed X lags, and static PCA factor blocks |
| `factor_feature_block` | 2_preprocessing | `none` and `pca_static_factors` operational from explicit blocks first; `feature_selection_policy` can compose as `select_before_factor` or `select_after_factor` for the static-PCA slice |
| `level_feature_block` | 2_preprocessing | all built-in values operational in raw-panel feature runtimes: `none`, `target_level_addback`, `x_level_addback`, `selected_level_addbacks`, and `level_growth_pairs` |
| `temporal_feature_block` | 2_preprocessing | `none`, `moving_average_features`, `rolling_moments`, `local_temporal_factors`, and `volatility_features` operational in raw-panel feature runtimes; these deterministic append blocks can compose with fixed X lags and `moving_average_rotation`; supported static PCA factor composition is open through explicit composers; registered `custom_temporal_features` are executable under `custom_feature_block_callable_v1`; custom-block final-`Z` selection is executable through `feature_selection_semantics=select_after_custom_feature_blocks`. |
| `feature_block_set` | 2_preprocessing | `target_lags_only`, `transformed_predictors`, `transformed_predictor_lags`, `factor_blocks_only`, `factors_plus_target_lags`, `high_dimensional_predictors`, `selected_sparse_predictors`, `level_augmented_predictors`, `rotation_augmented_predictors`, `mixed_feature_blocks`, and `custom_feature_blocks` are supported as provenance/runtime-dispatch inputs where their component blocks are executable; registered custom combiners execute through `custom_feature_combiner_v1`. |
| `rotation_feature_block` | 2_preprocessing | `none`, `moving_average_rotation`, `marx_rotation`, and `maf_rotation` operational in raw-panel feature runtimes where their component contracts are satisfied; `moving_average_rotation` composes with fixed X lags and deterministic temporal append blocks; MARX replaces the X lag-polynomial basis, supports named-block append/concatenate with fixed X-lag and temporal blocks, supports `marx_then_factor` with static PCA factors, and supports `factor_then_marx` through `factor_rotation_order=factor_then_rotation`; MAF rotation is a factor-score moving-average composer for static PCA factors; unregistered custom rotations remain gated. |
| `feature_block_combination` | 2_preprocessing | `replace_with_selected_blocks`, `append_to_base_predictors`, `append_to_target_lags`, and `concatenate_named_blocks` are operational with compiler/runtime pruning; `custom_feature_combiner` is executable when a registered `custom_feature_combiner` is supplied. |

## Compatibility Policy

- Do not break old recipes in this pass.
- Registry `layer` is canonical for docs/governance.
- Compiler continues to accept migrated axes at old recipe paths.
- Compiler now records Layer 1 `official_transform_policy` and
  `official_transform_scope` in `data_task_spec`, deriving them from legacy
  Layer 2 t-code fields when old recipes omit the new canonical axes.
- Generated default recipes no longer need to place official t-code bridge
  fields in `2_preprocessing`; the compiler derives the runtime
  `PreprocessContract` bridge from Layer 1 official-transform axes.
- Runtime official dataset transformation now reads `data_task_spec` first.
  Legacy `PreprocessContract.tcode_*` fields remain only as fallback for older
  compiled specs.
- Compiled manifests include `data_task_spec["official_transform_source"]`;
  runtime t-code reports include the matching source/fallback marker. This keeps
  the bridge visible without making `PreprocessContract` the owner of the
  official-transform choice.
- `dataset_source` remains accepted as a legacy recipe alias for
  `source_adapter`. New compiled specs and manifests write
  `data_task_spec["source_adapter"]`; execution falls back to old
  `data_task_spec["dataset_source"]` only for previously compiled specs.
- `task` remains accepted as a legacy recipe alias for `target_structure`.
  New compiled specs and manifests write `data_task_spec["target_structure"]`.
- Generated recipes should be updated gradually after tests lock the canonical layers.
- Runtime dispatch reads explicit Layer 2 feature blocks first and keeps
  `feature_builder`, `predictor_family`, `data_richness_mode`, and
  `factor_count` from compiled specs as compatibility/provenance fields. The
  registry layer records their canonical ownership as Layer 2. Explicit Layer 2
  block recipes can omit the legacy `feature_builder` input; the compiler
  derives the bridge internally for Stage 0 and old runtime dispatch.
- Layer 2 cleanup is closed for supported fixed full/runtime slices. Generic
  `Z` unification is a Layer 2 representation-handoff task, while Layer 3 stays
  a forecast-generator consumer. The current cleanup frontier is making that
  Layer 3 boundary explicit. The first implementation pass now writes
  target-lag and custom-hook Layer 2 choices into
  `layer2_representation_spec`; old `training_spec` readers remain as
  compatibility fallbacks. Factor-count, factor-lag, target construction,
  input-panel, and deterministic-block runtime migration are now also Layer 2
  metadata-first. Remaining implementation debt is adding the Layer 3
  capability matrix and removing remaining compatibility bridges.
