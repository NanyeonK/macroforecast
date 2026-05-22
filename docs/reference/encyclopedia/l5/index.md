# Layer L5  --  Evaluation

[Back to encyclopedia](../index.md) | [Browse layers](../browse_by_layer.md) | [Browse all axes](../browse_by_axis.md)

- Layer ID: `l5`
- Category: `consumption`
- Sub-layers: 5
- Axes: 18
- Options across axes: 30

## Sub-layers

| Sub-layer | Name | Gate | Axes |
|---|---|---|---|
| `L5_A_metric_specification` | Metric specification | always | [primary_metric](axes/primary_metric.md), [point_metrics](axes/point_metrics.md), [density_metrics](axes/density_metrics.md), [direction_metrics](axes/direction_metrics.md), [relative_metrics](axes/relative_metrics.md) |
| `L5_B_benchmark_comparison` | Benchmark comparison | always | [benchmark_window](axes/benchmark_window.md), [benchmark_scope](axes/benchmark_scope.md) |
| `L5_C_aggregation` | Aggregation | always | [agg_time](axes/agg_time.md), [agg_horizon](axes/agg_horizon.md), [agg_target](axes/agg_target.md), [agg_state](axes/agg_state.md) |
| `L5_D_sample_slicing_decomposition` | Sample slicing & decomposition | always | [oos_period](axes/oos_period.md), [regime_use](axes/regime_use.md), [regime_metrics](axes/regime_metrics.md), [decomposition_target](axes/decomposition_target.md), [decomposition_order](axes/decomposition_order.md) |
| `L5_E_ranking_reporting` | Ranking & reporting | always | [ranking](axes/ranking.md), [report_style](axes/report_style.md) |

```{toctree}
:hidden:
:maxdepth: 1

axes/primary_metric
axes/point_metrics
axes/density_metrics
axes/direction_metrics
axes/relative_metrics
axes/benchmark_window
axes/benchmark_scope
axes/agg_time
axes/agg_horizon
axes/agg_target
axes/agg_state
axes/oos_period
axes/regime_use
axes/regime_metrics
axes/decomposition_target
axes/decomposition_order
axes/ranking
axes/report_style
```
