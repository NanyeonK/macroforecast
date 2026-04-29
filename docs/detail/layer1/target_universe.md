# 4.1.3 Target And Variable Universe

- Parent: [4.1 Layer 1: Data Task](index.md)
- Current group: target and variable universe

Layer 1 identifies the target series inside the FRED data frame. It does not choose forecast type, target transformation, target normalization, or the model feature representation.

| Axis | Choices | Default / rule |
|---|---|---|
| `target_structure` | `single_target`, `multi_target` | Simple default `single_target`; Full should write the structure that matches `leaf_config.target` or `leaf_config.targets`. |
| `variable_universe` | `all_variables`, `core_variables`, `category_variables`, `target_specific_variables`, `explicit_variable_list` | Default `all_variables`; non-default values filter the source column universe before Layer 2 representation choices. |

Leaf config:

- `target`: required for `single_target`.
- `targets`: required for `multi_target`.
- `horizons`: required forecast horizons.
- `sample_start_date` / `sample_end_date`: sample period.
- `variable_universe_columns`: required for `explicit_variable_list`.
- `variable_universe_category_columns` and `variable_universe_category`: required for `category_variables`.
- `target_specific_columns`: required for `target_specific_variables`.

Layer 0 connection:

- `one_target_*` Study Scope values require `target_structure=single_target`.
- `multiple_targets_*` Study Scope values require `target_structure=multi_target`.

Layer 2 boundary:

- `horizon_target_construction`, `target_transform_policy`, `target_transform`, `target_normalization`, and target missing/outlier policies are Layer 2 representation decisions.

