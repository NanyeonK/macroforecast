# FRED-SD Predictor Scope

- Parent: [4.1 Layer 1: Data Source, Target y, Predictor x](index.md)
- Current group: FRED-SD predictor scope

This conditional group is shown only when the selected FRED source panel
includes FRED-SD. It restricts which state-level source columns can become
candidate predictors x and records the native-frequency evidence needed by
Layer 2.

These axes do not choose a mixed-frequency model. They close the source
selection contract: which states, which FRED-SD workbook series, and how
strictly the selected set must agree on native frequency.

If the source mode is custom-only, or if the FRED panel is `fred_md` /
`fred_qd` without FRED-SD, these axes are hidden by default. Imported recipes
with stale non-default FRED-SD choices keep the axes visible as incompatible
choices so the user can remove or revise them.

| Axis | Choices | Default / rule |
|---|---|---|
| `fred_sd_frequency_policy` | `report_only`, `allow_mixed_frequency`, `reject_mixed_known_frequency`, `require_single_known_frequency` | Default `report_only`. Non-default values require a source panel containing FRED-SD. |
| `fred_sd_state_group` | Census regions/divisions, `contiguous_48_plus_dc`, `custom_state_group`, `all_states` | State Scope. Default `all_states`. Group values resolve to explicit source-load selectors. |
| `state_selection` | `all_states`, `selected_states` | State List. Default `all_states`. `selected_states` requires `leaf_config.sd_states`. |
| `fred_sd_variable_group` | `all_sd_variables`, economic/t-code-review groups, `custom_sd_variable_group` | Series Scope. Default `all_sd_variables`. Group values resolve to explicit source-load selectors. |
| `sd_variable_selection` | `all_sd_variables`, `selected_sd_variables` | Series List. Default `all_sd_variables`. `selected_sd_variables` requires `leaf_config.sd_variables`. |

State Scope vs State List:

- `fred_sd_state_group` is the user-friendly group choice. It can select all
  states, census regions/divisions, the contiguous-state set, or a custom group.
- `state_selection` is the lower explicit-list switch. It exists so custom
  state groups and recipe imports can say whether `leaf_config.sd_states` must
  be read.

Series Scope vs Series List:

- `fred_sd_variable_group` is the user-friendly workbook-series group choice.
  It can select all FRED-SD series, predefined economic groups, predefined
  t-code-review groups, or a custom group.
- `sd_variable_selection` is the lower explicit-list switch. It exists so
  custom series groups and recipe imports can say whether
  `leaf_config.sd_variables` must be read.

Layer 1 output:

- selected states;
- selected workbook series;
- source sheets and series metadata;
- native-frequency report for selected FRED-SD series.

Layer 2 boundary:

- `fred_sd_mixed_frequency_representation` chooses calendar alignment,
  dropping policies, native-frequency block payloads, or mixed-frequency model
  adapters after Layer 1 has loaded and reported the source columns.
- MIDAS or other model-side mixed-frequency behavior is Layer 3 training logic.

Selector YAML:

```yaml
path:
  1_data_task:
    fixed_axes:
      dataset: fred_sd
      frequency: monthly
      target_structure: single_target
      fred_sd_state_group: custom_state_group
      fred_sd_variable_group: custom_sd_variable_group
      state_selection: selected_states
      sd_variable_selection: selected_sd_variables
    leaf_config:
      target: UR_CA
      horizons: [1]
      sd_states: [CA, NY, TX]
      sd_variables: [UR, PAYEMS]
```
