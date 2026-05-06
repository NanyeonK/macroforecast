# Layer 5: Evaluation

- Parent: [Detail: Layer Contracts](../index.md)
- Previous: [Layer 4](../layer4/index.md)
- Current: Layer 5
- Next: [Layer 6](../layer6/index.md)

Layer 5 computes forecast accuracy, benchmark-relative metrics, aggregation, slicing, decomposition, ranking, and reporting artifacts. It is descriptive evaluation; inference belongs to Layer 6.

## Contract

Inputs:

- `l4_forecasts_v1`;
- `l4_model_artifacts_v1`;
- `l1_data_definition_v1`;
- optional `l1_regime_metadata_v1`;
- `l3_metadata_v1`.

Output:

- `l5_evaluation_v1`.

## Sub-Layers

| Slot | Purpose |
|---|---|
| L5.A | metric specification |
| L5.B | benchmark comparison |
| L5.C | aggregation |
| L5.D | sample slicing and decomposition |
| L5.E | ranking and reporting |

## Main Axes

- metrics: `primary_metric`, `point_metrics`, `density_metrics`, `direction_metrics`, `relative_metrics`;
- benchmark: `benchmark_window`, `benchmark_scope`;
- aggregation: `agg_time`, `agg_horizon`, `agg_target`, `agg_state`;
- slicing: `oos_period`, `regime_use`, `regime_metrics`;
- decomposition: `decomposition_target`, `decomposition_order`;
- output shape: `ranking`, `report_style`.

## Gates

- Relative metrics require an L4 benchmark.
- Density metrics require quantile or density forecasts.
- `agg_target` requires multi-target data.
- `agg_state` and `by_state` decomposition require FRED-SD.
- Regime metrics and `by_regime` decomposition require an active L1 regime.
- `ranking: mcs_inclusion` requires active L6 MCS.

## Example

```yaml
5_evaluation:
  fixed_axes:
    primary_metric: relative_mse
    point_metrics: [mse, mae]
    relative_metrics: [relative_mse, r2_oos]
    benchmark_scope: per_target_horizon
    agg_horizon: per_horizon_separate
    ranking: by_relative_metric
    report_style: per_target_horizon_panel
```

## See encyclopedia

For the full per-axis × per-option catalogue (every value with its OptionDoc summary, when-to-use / when-NOT, references), see [`encyclopedia/l5/`](../../encyclopedia/l5/index.md).
