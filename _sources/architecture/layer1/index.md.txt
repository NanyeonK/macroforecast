# Layer 1: Data Source, Target y, Predictor x

- Parent: [4. Detail (code): Full](../index.md)
- Previous: [4.0 Layer 0: Study Scope](../layer0/index.md)
- Current: Layer 1
- Next: [4.2 Layer 2: Representation / Research Preprocessing](../layer2/index.md)

Layer 1 owns the source-frame contract for a macro forecasting study. It decides
which source data define the panel, the analysis frequency, what information is
available at each forecast origin, what target y is being forecast, and which
predictor x columns are eligible before Layer 2 builds research representations.

## Simple vs Full

Simple asks for the data question directly: `Data Source Mode`, analysis
frequency when needed, target y, sample dates, and horizons. FRED-only studies
choose a FRED source panel; custom-only studies provide a custom file path and
frequency without choosing a FRED panel. Optional Simple helpers expose
FRED-SD state/variable selection and FRED-SD frequency evidence policies, but
the ordinary path keeps Layer 1 mostly defaulted.

Full exposes the complete Layer 1 source-frame contract. The live registry
keeps hidden compatibility axes for older custom-source recipes, but the
Navigator primary tree shows only user-facing decisions. Those axes are not all
the same depth: some are primary decisions, some are derived/required
follow-ups, some are conditional FRED-SD sub-decisions, and many are defaulted
policy controls. `state_selection` / `sd_variable_selection` are lower
source-load selectors used by explicit FRED-SD selector helpers and group
resolution.

## Hierarchy

Layer 1 should be read as a hierarchy, not a flat checklist.

| Level | Group | Axes | Rule |
|---|---|---|---|
| Primary decision | Data Source Mode / Frequency | `custom_source_policy`, `dataset`, `frequency` | First choose FRED-only, custom-only, or FRED-plus-custom data. Then close the analysis frequency. `dataset` is a FRED source-panel choice, not a custom-only choice. |
| Primary policy | Forecast-Time Information | `information_set_type`, `release_lag_rule`, `contemporaneous_x_rule` | Defines the data revision/vintage regime, publication lag, and same-period x availability at each forecast origin. For custom-only data, only same-period x availability is exposed by default. |
| Contract-derived | Target (y) Definition | `target_structure` | Constrained by Layer 0 `study_scope`; y IDs, horizons, and dates live in `leaf_config`. |
| Secondary policy | Predictor (x) Definition | `variable_universe` | Limits eligible FRED-MD/QD x columns before Layer 2 builds representations. Custom-only x columns are defined by the custom file; standalone FRED-SD x columns are defined by state and series scope. |
| Conditional subgroup | FRED-SD Predictor Scope | `fred_sd_frequency_policy`, `fred_sd_state_group`, `state_selection`, `fred_sd_variable_group`, `sd_variable_selection` | Active only when the FRED source panel includes FRED-SD. |
| Secondary policy | Raw source quality | `raw_missing_policy`, `raw_outlier_policy` | Handles defects present in raw source data before FRED transforms/T-codes. |
| Secondary policy | Official transform / frame availability | `official_transform_policy`, `official_transform_scope`, `missing_availability` | Applies FRED-MD/QD official transform codes when available and closes source-frame availability gaps before Layer 2 begins. |

## Decision order

Read Layer 1 in runtime order. The table below is ordered, but the hierarchy above explains which rows are parent decisions and which are subordinate controls.

| Step | Group | Axes |
|---|---|---|
| 4.1.1 | [Data source mode / frequency](source_frame.md) | `custom_source_policy`, `dataset`, `frequency`; custom paths live in `leaf_config.custom_source_path` |
| 4.1.2 | [Forecast-time information](availability_timing.md) | `information_set_type`, `release_lag_rule`, `contemporaneous_x_rule` |
| 4.1.3 | [Target (y) and predictor (x) definitions](target_universe.md) | `target_structure`, `variable_universe`; target IDs, horizons, sample dates, and x column lists live in `leaf_config` |
| 4.1.4 | [FRED-SD predictor scope](fred_sd_source_selection.md) | `fred_sd_frequency_policy`, `fred_sd_state_group`, `state_selection`, `fred_sd_variable_group`, `sd_variable_selection` |
| 4.1.5 | [Raw source cleaning](raw_source_cleaning.md) | `raw_missing_policy`, `raw_outlier_policy` before FRED transforms/T-codes |
| 4.1.6 | [Official transforms](official_transforms.md) | `official_transform_policy`, `official_transform_scope` |
| 4.1.7 | [Frame availability](frame_availability.md) | `missing_availability` after the source frame exists |

## Defaults and Required Choices

| Axis | Simple default | Full rule |
|---|---|---|
| `custom_source_policy` | `official_only` | first source choice; choose FRED-only, custom-only, or FRED-plus-custom data |
| `dataset` | required only when FRED data is used | FRED source panel only: `fred_md`, `fred_qd`, `fred_sd`, `fred_md+fred_sd`, or `fred_qd+fred_sd`; custom-only should not expose this as a user choice |
| `custom_source_path` | none | required in `leaf_config` when `custom_source_policy` is not `official_only`; parser/schema are inferred |
| `frequency` | inferred for FRED-MD/QD/composites; required for standalone FRED-SD and custom-only | analysis frequency: monthly or quarterly |
| `information_set_type` | `final_revised_data` | data revision/vintage regime; hidden by default for custom-only data |
| `release_lag_rule` | `ignore_release_lag` | publication-lag rule; hidden by default for custom-only data; `series_specific_lag` requires `leaf_config.release_lag_per_series` |
| `contemporaneous_x_rule` | `forbid_same_period_predictors` | same-period predictor rule; `allow_same_period_predictors` is an oracle benchmark |
| `fred_sd_frequency_policy` | `report_only` | defaulted; non-default values require a dataset containing FRED-SD |
| `fred_sd_state_group` | `all_states` | defaulted; non-default values require FRED-SD |
| `fred_sd_variable_group` | `all_sd_variables` | defaulted; non-default values require FRED-SD |
| `state_selection` | `all_states` | conditional FRED-SD State List selector; `selected_states` requires `leaf_config.sd_states` |
| `sd_variable_selection` | `all_sd_variables` | conditional FRED-SD Series List selector; `selected_sd_variables` requires `leaf_config.sd_variables` |
| `target_structure` | `single_target` | write `single_target` with `leaf_config.target` or `multi_target` with `leaf_config.targets`; docs call these Single Target and Multiple Targets |
| `variable_universe` | `all_variables` | FRED-MD/QD predictor metadata axis; hidden by default when no FRED-MD/QD source is selected |
| `raw_missing_policy` | `preserve_raw_missing` | defaulted; non-default values act before FRED transforms/T-codes |
| `raw_outlier_policy` | `preserve_raw_outliers` | defaulted; non-default values act before FRED transforms/T-codes |
| `official_transform_policy` | `apply_official_tcode` | FRED-MD/QD official t-code axis; hidden by default when no FRED-MD/QD source is selected |
| `official_transform_scope` | `target_and_predictors` | FRED-MD/QD official t-code scope; hidden by default when no FRED-MD/QD source is selected |
| `missing_availability` | `zero_fill_leading_predictor_gaps` | defaulted after the Layer 1 source frame exists |

## Layer contract

Input:
- source request, custom-source request when relevant, target y request, and candidate predictor x request.

Output:
- `layer1_official_frame_v1`, the compatibility artifact name for the Layer 1 source-frame handoff;
- source availability contract;
- data reports for availability, release lag, missing policy, and FRED-SD source metadata when relevant.

## Canonical names

Layer 1 is canonical-only. Recipes should use the axis IDs in the decision-order table and the values documented on each axis page. Removed aliases for source dispatch, information-set regime, and target shape are rejected during registry validation, so generated YAML, docs, and manifests stay on one vocabulary.

## Related reference

- [Data source and frame](../../for_recipe_authors/data/source.md)
- [Target and predictor definition](../../for_recipe_authors/data/target_structure.md)
- [Data handling policies](../../for_recipe_authors/data_policies.md)

```{toctree}
:maxdepth: 1

source_frame
availability_timing
target_universe
fred_sd_source_selection
raw_source_cleaning
official_transforms
frame_availability
```
