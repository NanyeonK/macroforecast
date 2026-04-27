# 4.5 Layer 4: Evaluation

- Parent: [4. Detail (code): Full](../index.md)
- Previous: [4.4 Layer 3: Forecast Generator](../layer3/index.md)
- Current: Layer 4
- Next: [4.6 Layer 5: Output / Provenance](../layer5/index.md)

Layer 4 owns evaluation of forecast artifacts. It chooses metric families, benchmark comparison scope, aggregation, ranking, reporting style, regime use, decomposition, and OOS period.

## Decision order

| Group | Axes |
|---|---|
| Metrics | `primary_metric`, `point_metrics`, `density_metrics`, `direction_metrics`, `relative_metrics`, `economic_metrics` |
| Benchmark comparison | `benchmark_window`, `benchmark_scope` |
| Aggregation | `agg_time`, `agg_horizon`, `agg_target` |
| Reporting | `ranking`, `report_style` |
| Regimes and decomposition | `regime_definition`, `regime_use`, `regime_metrics`, `decomposition_target`, `decomposition_order` |
| Evaluation window | `oos_period` |

## Naming migration

Layer 4 canonical values use lower-snake researcher names. Older mixed-case
metric IDs still compile through `registry_naming_v1`.

| Axis | Legacy value | Canonical value |
|---|---:|---:|
| `point_metrics` | `MSE` / `MSFE` / `RMSE` / `MAE` / `MAPE` | `mse` / `msfe` / `rmse` / `mae` / `mape` |
| `point_metrics` | `sMAPE` / `MASE` / `RMSSE` / `MedAE` | `smape` / `mase` / `rmsse` / `median_absolute_error` |
| `point_metrics` | `Huber_loss` / `QLIKE` / `TheilU` | `huber_loss` / `qlike` / `theil_u` |
| `relative_metrics` | `relative_MSFE` / `relative_RMSE` / `relative_MAE` | `relative_msfe` / `relative_rmse` / `relative_mae` |
| `relative_metrics` | `oos_R2` / `CSFE_difference` | `oos_r2` / `csfe_difference` |
| `density_metrics` | `CRPS` / `NLL` / `PIT_based_metric` | `crps` / `nll` / `pit_based_metric` |
| `direction_metrics` | `F1` / `AUC` / `Brier_score` | `f1` / `auc` / `brier_score` |
| `agg_time` | `full_oos_average` | `full_out_of_sample_average` |
| `agg_target` | `scale_adjusted_weight` / `economic_priority_weight` | `scale_adjusted_weighting` / `economic_priority_weighting` |
| `ranking` | `benchmark_beat_freq` / `MCS_inclusion_priority` | `benchmark_beat_frequency` / `mcs_inclusion_priority` |
| `regime_definition` | `NBER_recession` / `Markov_switching_regime` | `nber_recession` / `markov_switching_regime` |
| `regime_use` | `eval_only` | `evaluation_only` |
| `decomposition_target` | `cv_scheme_effect` | `validation_scheme_effect` |
| `decomposition_order` | `Shapley_style_effect_decomp` | `shapley_style_effect_decomp` |

## Layer contract

Input:
- forecast payloads and predictions from Layer 3;
- provenance needed to compare against benchmarks.

Output:
- metrics;
- evaluation summaries;
- ranking and decomposition artifacts where selected.

## Related reference

- [Artifacts and Manifest](../artifacts_and_manifest.md)
- [Layer Contract Ledger](../layer_contract_ledger.md)
