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
| `predictor_family` | 1_data_task | 3_training | Predictor recipe is model-input construction. |
| `min_train_size` | 1_data_task | 3_training | Minimum training window is training protocol. |
| `training_start_rule` | 1_data_task | 3_training | Training start is model training protocol, distinct from sample period. |
| `horizon_target_construction` | 1_data_task | 2_preprocessing | Diff/logdiff/level target construction is target representation. |
| `deterministic_components` | 1_data_task | 2_preprocessing | Trends, seasonals, and break dummies are feature construction. |
| `structural_break_segmentation` | 1_data_task | 2_preprocessing | Current implementation adds break dummy features. |
| `oos_period` | 1_data_task | 4_evaluation | Recession/expansion-only selection is evaluation subset filtering. |
| `overlap_handling` | 1_data_task | 6_stat_tests | HAC/overlap handling is inference over dependent forecast errors. |
| `official_transform_policy` | split from Layer 2 t-code axes | 1_data_task | Official dataset transformations define the official frame, before researcher preprocessing. |
| `official_transform_scope` | split from `tcode_application_scope` | 1_data_task | Target/X official transform scope is part of official frame construction. |

## Still To Migrate

| Axis / concept | Current owner | Target owner | Note |
|---|---|---|---|
| legacy official t-code bridge fields | 2_preprocessing | compatibility bridge | Keep accepting `target_transform_policy`, `x_transform_policy`, `tcode_policy=tcode_only`, `tcode_application_scope`, and `preprocess_order=tcode_only` while generated recipes move to Layer 1 official-transform axes. |
| `tcode_policy` values beyond official transform | 2_preprocessing | 2_preprocessing | Keep extra/custom transform pipelines in Layer 2. |
| `preprocess_order=tcode_only` | 2_preprocessing | compatibility bridge | Official-only order is represented by Layer 1 `official_transform_policy=dataset_tcode`; extra orders remain Layer 2. |
| `dataset_source` naming | 1_data_task | 1_data_task | Rename to `source_adapter` after compatibility plan. |

## Compatibility Policy

- Do not break old recipes in this pass.
- Registry `layer` is canonical for docs/governance.
- Compiler continues to accept migrated axes at old recipe paths.
- Compiler now records Layer 1 `official_transform_policy` and
  `official_transform_scope` in `data_task_spec`, deriving them from legacy
  Layer 2 t-code fields when old recipes omit the new canonical axes.
- Generated recipes should be updated gradually after tests lock the canonical layers.
