# 4.4 Layer 3: Forecast Generator

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
