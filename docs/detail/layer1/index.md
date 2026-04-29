# 4.1 Layer 1: Data Task

- Parent: [4. Detail (code): Full](../index.md)
- Previous: [4.0 Layer 0: Study Scope](../layer0/index.md)
- Current: Layer 1
- Next: [4.2 Layer 2: Representation / Research Preprocessing](../layer2/index.md)

Layer 1 owns the official data task. It decides which data source is used, which target structure is being forecast, what information is available, and how raw source-level missing/outlier and official-transform policies are handled before representation construction.

## Simple vs Full

Simple asks for the data question directly: `dataset`, `target`, `start`, `end`, and `horizons`. Standalone `fred_sd` also needs `frequency`. Optional Simple helpers expose FRED-SD state/variable selection, FRED-SD frequency policy, and FRED-SD t-code policies, but the ordinary path keeps Layer 1 mostly defaulted.

Full exposes the complete Layer 1 official-frame contract. The live registry has **17 Layer 1 axes**. The Navigator primary tree shows 15 user-facing axes; `state_selection` / `sd_variable_selection` are lower source-load selectors used by explicit FRED-SD selector helpers and group resolution.

## Decision order

Read Layer 1 in runtime order:

| Step | Group | Axes |
|---|---|---|
| 4.1.1 | [Source and frame](source_frame.md) | `dataset`, `frequency`, `information_set_type` |
| 4.1.2 | [FRED-SD source selection](fred_sd_source_selection.md) | `fred_sd_frequency_policy`, `fred_sd_state_group`, `fred_sd_variable_group`, hidden `state_selection`, hidden `sd_variable_selection` |
| 4.1.3 | [Target and variable universe](target_universe.md) | `target_structure`, `variable_universe`; target IDs, horizons, and sample dates live in `leaf_config` |
| 4.1.4 | [Raw source cleaning](raw_source_cleaning.md) | `raw_missing_policy`, `raw_outlier_policy` before official transforms/T-codes |
| 4.1.5 | [Official transforms](official_transforms.md) | `official_transform_policy`, `official_transform_scope` |
| 4.1.6 | [Availability and timing](availability_timing.md) | `missing_availability`, `release_lag_rule`, `contemporaneous_x_rule` after the official frame exists |

## Defaults and Required Choices

| Axis | Simple default | Full rule |
|---|---|---|
| `dataset` | required user choice | required; choose `custom_csv` or `custom_parquet` for user-supplied files |
| `frequency` | inferred for FRED-MD/QD/composites; required for standalone FRED-SD | MD/composites are constrained; standalone FRED-SD must choose monthly or quarterly |
| `information_set_type` | `final_revised_data` | write explicitly in Full recipes |
| `fred_sd_frequency_policy` | `report_only` | defaulted; non-default values require a dataset containing FRED-SD |
| `fred_sd_state_group` | `all_states` | defaulted; non-default values require FRED-SD |
| `fred_sd_variable_group` | `all_sd_variables` | defaulted; non-default values require FRED-SD |
| `state_selection` | `all_states` | hidden selector; `selected_states` requires `leaf_config.sd_states` |
| `sd_variable_selection` | `all_sd_variables` | hidden selector; `selected_sd_variables` requires `leaf_config.sd_variables` |
| `target_structure` | `single_target` | write `single_target` with `leaf_config.target` or `multi_target` with `leaf_config.targets` |
| `variable_universe` | `all_variables` | defaulted unless the research design filters source columns |
| `raw_missing_policy` | `preserve_raw_missing` | defaulted; non-default values act before official transforms/T-codes |
| `raw_outlier_policy` | `preserve_raw_outliers` | defaulted; non-default values act before official transforms/T-codes |
| `official_transform_policy` | `apply_official_tcode` | default profile uses official t-codes; `keep_official_raw_scale` preserves raw scale |
| `official_transform_scope` | `target_and_predictors` | default profile transforms both target and predictors |
| `missing_availability` | `zero_fill_leading_predictor_gaps` | defaulted after the official frame exists |
| `release_lag_rule` | `ignore_release_lag` | defaulted; `series_specific_lag` requires `leaf_config.release_lag_per_series` |
| `contemporaneous_x_rule` | `forbid_same_period_predictors` | defaulted realistic forecasting rule; `allow_same_period_predictors` is an oracle benchmark |

## Layer contract

Input:
- source request and target request.

Output:
- `layer1_official_frame_v1`;
- source availability contract;
- data reports for availability, release lag, missing policy, and FRED-SD source metadata when relevant.

## Canonical names

Layer 1 is canonical-only. Recipes should use the axis IDs in the decision-order table and the values documented on each axis page. Removed aliases for source dispatch, information-set regime, and target shape are rejected during registry validation, so generated YAML, docs, and manifests stay on one vocabulary.

## Related reference

- [Layer 1 Data Task Audit](../layer1_data_task_audit.md)
- [Data Source and Frame](../../user_guide/data/source.md)
- [Target Structure](../../user_guide/data/target_structure.md)
- [Data Handling Policies](../../user_guide/data/policies.md)

```{toctree}
:maxdepth: 1

source_frame
fred_sd_source_selection
target_universe
raw_source_cleaning
official_transforms
availability_timing
```
