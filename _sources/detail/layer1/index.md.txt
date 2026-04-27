# 4.2 Layer 1: Data Task

- Parent: [4. Detail (code): Full](../index.md)
- Previous: [4.1 Layer 0: Study Setup](../layer0/index.md)
- Current: Layer 1
- Next: [4.3 Layer 2: Representation / Research Preprocessing](../layer2/index.md)

Layer 1 owns the official data task. It decides which data source is used, which target structure is being forecast, what information is available, and how raw source-level missing/outlier and official-transform policies are handled before representation construction.

## Decision order

| Group | Axes |
|---|---|
| Source and frame | `dataset`, `source_adapter`, `frequency`, `information_set_type` |
| FRED-SD source selection | `fred_sd_frequency_policy`, `fred_sd_state_group`, `fred_sd_variable_group` |
| Target and universe | `target_structure`, `variable_universe` |
| Availability and timing | `missing_availability`, `release_lag_rule`, `contemporaneous_x_rule` |
| Raw source cleaning | `raw_missing_policy`, `raw_outlier_policy` |
| Official transforms | `official_transform_policy`, `official_transform_scope` |

## Layer contract

Input:
- source request and target request.

Output:
- `layer1_official_frame_v1`;
- source availability contract;
- data reports for availability, release lag, missing policy, and FRED-SD source metadata when relevant.

## Naming migration

Layer 1 now uses canonical names that describe the research decision instead
of internal implementation vocabulary. Old recipe IDs remain valid through
`registry_naming_v1`; new recipes and docs should use the canonical IDs below.

| Axis | Legacy ID | Canonical ID |
|---|---|---|
| `information_set_type` | `revised` | `final_revised_data` |
| `information_set_type` | `pseudo_oos_revised` | `pseudo_oos_on_revised_data` |
| `target_structure` | `single_target_point_forecast` | `single_target` |
| `target_structure` | `multi_target_point_forecast` | `multi_target` |
| `official_transform_policy` | `dataset_tcode` | `apply_official_tcode` |
| `official_transform_policy` | `raw_official_frame` | `keep_official_raw_scale` |
| `official_transform_scope` | `apply_tcode_to_target` | `target_only` |
| `official_transform_scope` | `apply_tcode_to_X` | `predictors_only` |
| `official_transform_scope` | `apply_tcode_to_both` | `target_and_predictors` |
| `official_transform_scope` | `apply_tcode_to_none` | `none` |
| `contemporaneous_x_rule` | `allow_contemporaneous` | `allow_same_period_predictors` |
| `contemporaneous_x_rule` | `forbid_contemporaneous` | `forbid_same_period_predictors` |
| `variable_universe` | `preselected_core` | `core_variables` |
| `variable_universe` | `category_subset` | `category_variables` |
| `variable_universe` | `target_specific_subset` | `target_specific_variables` |
| `variable_universe` | `handpicked_set` | `explicit_variable_list` |
| `missing_availability` | `complete_case_only` | `require_complete_rows` |
| `missing_availability` | `available_case` | `keep_available_rows` |
| `missing_availability` | `x_impute_only` | `impute_predictors_only` |
| `missing_availability` | `zero_fill_before_start` | `zero_fill_leading_predictor_gaps` |
| `raw_missing_policy` | `zero_fill_leading_x_before_tcode` | `zero_fill_leading_predictor_missing_before_tcode` |
| `raw_missing_policy` | `x_impute_raw` | `impute_raw_predictors` |
| `raw_missing_policy` | `drop_rows_with_raw_missing` | `drop_raw_missing_rows` |
| `raw_outlier_policy` | `raw_outlier_to_missing` | `set_raw_outliers_to_missing` |

## Related reference

- [Layer 1 Data Task Audit](../layer1_data_task_audit.md)
- [Data Source and Frame](../../user_guide/data/source.md)
- [Target Structure](../../user_guide/data/target_structure.md)
- [Data Handling Policies](../../user_guide/data/policies.md)
