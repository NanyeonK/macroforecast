# 4.4 Layer 3: Forecast Generator

- Parent: [4. Detail (code): Full](../index.md)
- Previous: [4.3 Layer 2: Representation / Research Preprocessing](../layer2/index.md)
- Current: Layer 3
- Next: [4.5 Layer 4: Evaluation](../layer4/index.md)

Layer 3 owns forecast generation. It consumes Layer 2 representation contracts and chooses model, benchmark, forecast type/object, future-X path policy, training windows, and tuning behavior.

## Decision order

| Group | Axes |
|---|---|
| Generator family | `model_family`, `midasr_weight_family`, `benchmark_family` |
| Forecast object | `forecast_type`, `forecast_object` |
| Future X / recursive paths | `exogenous_x_path_policy`, `recursive_x_model_family` |
| Runtime framework | `framework`, `outer_window`, `refit_policy` |
| Training window | `min_train_size`, `training_start_rule`, `y_lag_count` |
| Tuning | `search_algorithm`, `tuning_objective`, `tuning_budget`, `validation_location`, `validation_size_rule` |

## Current naming migration

Layer 3 IDs now prefer explicit model and benchmark names over glued
abbreviations. Old recipe values are still accepted through
`registry_naming_v1`; compiled recipes, Navigator paths, and generated YAML
emit the canonical names below.

| Axis | Legacy value | Canonical value |
|---|---|---|
| `model_family` | `bayesianridge` | `bayesian_ridge` |
| `model_family` | `adaptivelasso` | `adaptive_lasso` |
| `model_family` | `randomforest` | `random_forest` |
| `model_family` | `extratrees` | `extra_trees` |
| `model_family` | `gbm` | `gradient_boosting` |
| `benchmark_family` | `ar_bic` | `autoregressive_bic` |
| `benchmark_family` | `ar_fixed_p` | `autoregressive_fixed_lag` |
| `benchmark_family` | `ardi` | `autoregressive_diffusion_index` |
| `benchmark_family` | `factor_model` | `factor_model_benchmark` |
| `benchmark_family` | `multi_benchmark_suite` | `benchmark_suite` |

## Layer contract

Input:
- `layer2_representation_v1`;
- Layer 0 runtime policy;
- Layer 1 task metadata.

Output:
- forecast payloads;
- predictions;
- training/tuning metadata;
- model and benchmark artifacts.

## Related reference

- [Layer 3 Training Audit](../layer3_training_audit.md)
- [Raw Panel Iterated Contract](../raw_panel_iterated_contract.md)
- [Custom Extensions](../custom_extensions.md)
