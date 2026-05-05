# Recipe Layers

Current layer-contract recipes use numeric YAML keys that map directly to layer IDs.

## Canonical Keys

| Layer ID | YAML key | Required in minimal path |
|---|---|---|
| `l0` | `0_meta` | optional; defaults apply |
| `l1` | `1_data` | yes |
| `l2` | `2_preprocessing` | yes |
| `l3` | `3_feature_engineering` | yes |
| `l4` | `4_forecasting_model` | yes |
| `l5` | `5_evaluation` | yes |
| `l6` | `6_statistical_tests` | no; default off |
| `l7` | `7_interpretation` | no; default off |
| `l8` | `8_output` | yes |
| `l1_5` | `1_5_data_summary` | no; default off |
| `l2_5` | `2_5_pre_post_preprocessing` | no; default off |
| `l3_5` | `3_5_feature_diagnostics` | no; default off |
| `l4_5` | `4_5_generator_diagnostics` | no; default off |

## Shape Rules

- List layers use `fixed_axes` plus optional `leaf_config`.
- Graph layers use `nodes` and `sinks`.
- Diagnostics require `enabled: true` to produce a diagnostic sink.
- L8 derives default saved objects from active upstream layers unless `saved_objects` is explicit.

## Minimal Skeleton

```yaml
1_data:
  fixed_axes: {}

2_preprocessing:
  fixed_axes: {}

3_feature_engineering:
  nodes: []
  sinks: {}

4_forecasting_model:
  nodes: []
  sinks: {}

5_evaluation:
  fixed_axes: {}

8_output:
  fixed_axes: {}
```

The skeleton above shows structure only. A runnable recipe must provide valid L3 and L4 DAG nodes and required sinks.
