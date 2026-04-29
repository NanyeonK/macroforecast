# 4.1 Layer 1: Data Task

- Parent: [4. Detail (code): Full](../index.md)
- Previous: [4.0 Layer 0: Study Scope](../layer0/index.md)
- Current: Layer 1
- Next: [4.2 Layer 2: Representation / Research Preprocessing](../layer2/index.md)

Layer 1 owns the official data task. It decides which data source is used, which target structure is being forecast, what information is available, and how raw source-level missing/outlier and official-transform policies are handled before representation construction.

## Simple vs Full

Simple asks for the data question directly: `dataset`, `target`, `start`, `end`, and `horizons`. Standalone `fred_sd` also needs `frequency`. Custom files are optional source modifiers: Simple can expose `custom_source_policy`, `custom_source_format`, `custom_source_schema`, and `custom_source_path` when the user wants to replace or append to the official source panel. Optional Simple helpers expose FRED-SD state/variable selection, FRED-SD frequency policy, and FRED-SD t-code policies, but the ordinary path keeps Layer 1 mostly defaulted.

Full exposes the complete Layer 1 official-frame contract. The live registry has **20 Layer 1 axes**. The Navigator primary tree shows 18 axes, but those axes are not all the same depth: some are primary decisions, some are derived/required follow-ups, some are conditional custom/FRED-SD sub-decisions, and many are defaulted policy controls. `state_selection` / `sd_variable_selection` are lower source-load selectors used by explicit FRED-SD selector helpers and group resolution.

## Hierarchy

Layer 1 should be read as a hierarchy, not a flat checklist.

| Level | Group | Axes | Rule |
|---|---|---|---|
| Primary decision | Source identity | `dataset`, `custom_source_policy`, `custom_source_format`, `custom_source_schema`, `frequency` | `dataset` is the official source-panel choice. Custom source axes decide whether to use official data only, custom data only, or official plus custom data. `frequency` is derived for MD/QD/composites and required only for standalone FRED-SD. |
| Primary policy | Information regime | `information_set_type`, `release_lag_rule`, `contemporaneous_x_rule` | Defines what information is available at each forecast origin. |
| Conditional subgroup | FRED-SD source scope | `fred_sd_frequency_policy`, `fred_sd_state_group`, `fred_sd_variable_group` | Active only when `dataset` includes FRED-SD. |
| Contract-derived | Target request | `target_structure` | Constrained by Layer 0 `study_scope`; target IDs, target lists, horizons, and dates live in `leaf_config`. |
| Secondary policy | Source universe | `variable_universe` | Limits eligible raw source columns before Layer 2 builds representations. |
| Secondary policy | Raw source quality | `raw_missing_policy`, `raw_outlier_policy` | Handles defects present in raw source data before official transforms/T-codes. |
| Secondary policy | Official frame policy | `official_transform_policy`, `official_transform_scope`, `missing_availability` | Closes the official Layer 1 frame before Layer 2 research preprocessing begins. |

## Decision order

Read Layer 1 in runtime order. The table below is ordered, but the hierarchy above explains which rows are parent decisions and which are subordinate controls.

| Step | Group | Axes |
|---|---|---|
| 4.1.1 | [Source and frame](source_frame.md) | `dataset`, `custom_source_policy`, `custom_source_format`, `custom_source_schema`, `frequency`, `information_set_type` |
| 4.1.2 | [FRED-SD source selection](fred_sd_source_selection.md) | `fred_sd_frequency_policy`, `fred_sd_state_group`, `fred_sd_variable_group`, hidden `state_selection`, hidden `sd_variable_selection` |
| 4.1.3 | [Target and variable universe](target_universe.md) | `target_structure`, `variable_universe`; target IDs, horizons, and sample dates live in `leaf_config` |
| 4.1.4 | [Raw source cleaning](raw_source_cleaning.md) | `raw_missing_policy`, `raw_outlier_policy` before official transforms/T-codes |
| 4.1.5 | [Official transforms](official_transforms.md) | `official_transform_policy`, `official_transform_scope` |
| 4.1.6 | [Availability and timing](availability_timing.md) | `missing_availability`, `release_lag_rule`, `contemporaneous_x_rule` after the official frame exists |

## Defaults and Required Choices

| Axis | Simple default | Full rule |
|---|---|---|
| `dataset` | required user choice | required; official source panel only: `fred_md`, `fred_qd`, `fred_sd`, `fred_md+fred_sd`, or `fred_qd+fred_sd` |
| `custom_source_policy` | `official_only` | choose `custom_panel_only` or `official_plus_custom` when using a custom file |
| `custom_source_format` | `none` | required as `csv` or `parquet` when `custom_source_policy` is not `official_only` |
| `custom_source_schema` | `none` | required as `fred_md`, `fred_qd`, or `fred_sd` when `custom_source_policy` is not `official_only` |
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
