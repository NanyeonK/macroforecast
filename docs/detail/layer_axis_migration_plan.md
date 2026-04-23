# Layer Axis Migration Plan

Date: 2026-04-22

This ledger records the Layer 1/2/3/4/6 boundary cleanup. `current path`
describes where many old recipes may still place the axis. `canonical owner`
describes the registry layer after migration.

## Migrated In This Pass

| Axis | Old owner | Canonical owner | Reason |
|---|---|---|---|
| `benchmark_family` | 1_data_task | 3_training | Benchmarks generate forecasts; they are model/baseline choices. |
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
| `target_structure` | `task` | 1_data_task | Layer 1 records target cardinality; Layer 0 owns runner grammar. |

## Still To Migrate

| Axis / concept | Current owner | Target owner | Note |
|---|---|---|---|
| legacy official t-code bridge fields | 2_preprocessing | compatibility bridge | Keep accepting `target_transform_policy`, `x_transform_policy`, `tcode_policy=tcode_only`, `tcode_application_scope`, and `preprocess_order=tcode_only` while generated recipes move to Layer 1 official-transform axes. |
| `tcode_policy` values beyond official transform | 2_preprocessing | 2_preprocessing | Keep extra/custom transform pipelines in Layer 2. |
| `preprocess_order=tcode_only` | 2_preprocessing | compatibility bridge | Official-only order is represented by Layer 1 `official_transform_policy=dataset_tcode`; extra orders remain Layer 2. |
| `y_lag_count` | 3_training | split in provenance | AR model-order selection remains Layer 3 for now; fixed target-lag feature construction is recorded with Layer 2 `target_lag_selection` / `target_lag_block` provenance and lowered to the legacy runtime bridge. |
| `factor_ar_lags` leaf/training config | 3_training config | split in provenance | Legacy runtime key remains accepted; target-lag feature count next to factor blocks is recorded as Layer 2 `target_lag_count` provenance. |

## Feature-Block Grammar Introduced

This pass defines the Layer 2 feature-block grammar and starts runtime support
through bridge lowering. The new axes are canonical Layer 2 concepts, while
most runtime execution still flows through the old `feature_builder` bridge
until each block has train-window fit/apply tests and provenance.

| Axis | Canonical owner | Current support |
|---|---|---|
| `target_lag_block`, `target_lag_selection`, `x_lag_feature_block` | 2_preprocessing | `none` and fixed-lag values operational through separate bridge lowering; target-plus-X block composition remains gated |
| `factor_feature_block` | 2_preprocessing | `none` and `pca_static_factors` operational through factor bridge lowering; factor/selection composition remains gated |
| `level_feature_block` | 2_preprocessing | all built-in values operational through raw-panel bridge lowering: `none`, `target_level_addback`, `x_level_addback`, `selected_level_addbacks`, and `level_growth_pairs` |
| `temporal_feature_block` | 2_preprocessing | `none`, `moving_average_features`, `rolling_moments`, `local_temporal_factors`, and `volatility_features` operational through raw-panel bridge lowering; X-lag/factor composition and custom temporal blocks remain gated |
| `feature_block_set` | 2_preprocessing | registry-only |
| `rotation_feature_block` | 2_preprocessing | registry-only |
| `feature_block_combination` | 2_preprocessing | registry-only |

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
- `dataset_source` remains accepted as a legacy recipe alias for
  `source_adapter`. New compiled specs and manifests write
  `data_task_spec["source_adapter"]`; execution falls back to old
  `data_task_spec["dataset_source"]` only for previously compiled specs.
- `task` remains accepted as a legacy recipe alias for `target_structure`.
  New compiled specs and manifests write `data_task_spec["target_structure"]`.
- Generated recipes should be updated gradually after tests lock the canonical layers.
- Legacy execution still dispatches on `feature_builder`, `predictor_family`,
  `data_richness_mode`, and `factor_count` from compiled specs. That is a
  runtime compatibility shape; the registry layer now records their canonical
  ownership as Layer 2.
