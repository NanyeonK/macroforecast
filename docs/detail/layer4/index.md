# 4.5 Layer 4: Evaluation

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
