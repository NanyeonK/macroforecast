# 4.1.4 FRED-SD Predictor Scope

- Parent: [4.1 Layer 1: Data Task](index.md)
- Current group: FRED-SD predictor scope

This conditional group is active only when the FRED source panel includes
FRED-SD. It restricts which state-level source columns can become candidate
predictors x and records the native-frequency evidence needed by Layer 2.

These axes do not choose a mixed-frequency model. They close the source
selection contract: which states, which FRED-SD workbook variables, and how
strictly the selected set must agree on native frequency.

| Axis | Choices | Default / rule |
|---|---|---|
| `fred_sd_frequency_policy` | `report_only`, `allow_mixed_frequency`, `reject_mixed_known_frequency`, `require_single_known_frequency` | Default `report_only`. Non-default values require a source panel containing FRED-SD. |
| `fred_sd_state_group` | Census regions/divisions, `contiguous_48_plus_dc`, `custom_state_group`, `all_states` | Default `all_states`. Group values resolve to explicit source-load selectors. |
| `fred_sd_variable_group` | `all_sd_variables`, economic/t-code-review groups, `custom_sd_variable_group` | Default `all_sd_variables`. Group values resolve to explicit source-load selectors. |
| `state_selection` | `all_states`, `selected_states` | Hidden lower selector; `selected_states` requires `leaf_config.sd_states`. |
| `sd_variable_selection` | `all_sd_variables`, `selected_sd_variables` | Hidden lower selector; `selected_sd_variables` requires `leaf_config.sd_variables`. |

Layer 1 output:

- selected states;
- selected workbook variables;
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
    leaf_config:
      target: UR_CA
      horizons: [1]
      sd_states: [CA, NY, TX]
      sd_variables: [UR, PAYEMS]
```
