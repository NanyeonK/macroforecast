# Layer 6: Statistical Tests

- Parent: [Detail: Layer Contracts](../index.md)
- Previous: [Layer 5](../layer5/index.md)
- Current: Layer 6
- Next: [Layer 7](../layer7/index.md)

Layer 6 runs inferential tests on forecasts and evaluation artifacts. It is default off and requires `enabled: true`. Individual sub-layers also require their own `enabled: true`.

## Contract

Inputs:

- `l4_forecasts_v1`;
- `l4_model_artifacts_v1`;
- `l5_evaluation_v1`;
- `l1_data_definition_v1`;
- optional `l1_regime_metadata_v1`.

Output:

- `l6_tests_v1`.

## Globals

- `test_scope`;
- `dependence_correction`;
- `overlap_handling`.

## Sub-Layers

| Sub-layer | Purpose |
|---|---|
| L6.A | equal predictive ability tests |
| L6.B | nested model tests |
| L6.C | conditional predictive ability and instability |
| L6.D | multiple model tests |
| L6.E | density and interval tests |
| L6.F | direction tests |
| L6.G | residual diagnostics |

## Gates

- `enabled: false` disables all sub-layers.
- Benchmark-only pair strategies require an L4 benchmark.
- Density/interval tests require quantile or density forecasts.
- CPA regime conditioning requires an active L1 regime.
- `overlap_handling: none` is rejected for overlapping horizons.
- Bootstrap replications below 100 are hard errors; below 500 are warnings.

## Example

```yaml
6_statistical_tests:
  enabled: true
  test_scope: per_target_horizon
  dependence_correction: newey_west
  sub_layers:
    L6_A_equal_predictive:
      enabled: true
      fixed_axes:
        equal_predictive_test: dm_diebold_mariano
        model_pair_strategy: vs_benchmark_only
    L6_D_multiple_model:
      enabled: true
      fixed_axes:
        multiple_model_test: mcs_hansen
        mcs_alpha: 0.10
```
