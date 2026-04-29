# 4.1.2 FRED-SD Source Selection

- Parent: [4.1 Layer 1: Data Task](index.md)
- Current group: FRED-SD source selection

This group is active only when `dataset` contains `fred_sd`. It selects which state-level source columns enter the official frame and how strict the native-frequency gate should be.

| Axis | Choices | Default / rule |
|---|---|---|
| `fred_sd_frequency_policy` | `report_only`, `allow_mixed_frequency`, `reject_mixed_known_frequency`, `require_single_known_frequency` | Default `report_only`. Non-default values require FRED-SD. |
| `fred_sd_state_group` | Census regions/divisions, `contiguous_48_plus_dc`, `custom_state_group`, `all_states` | Default `all_states`. Group values resolve to explicit source-load selectors. |
| `fred_sd_variable_group` | `all_sd_variables`, economic/t-code-review groups, `custom_sd_variable_group` | Default `all_sd_variables`. Group values resolve to explicit source-load selectors. |
| `state_selection` | `all_states`, `selected_states` | Hidden lower selector; `selected_states` requires `leaf_config.sd_states`. |
| `sd_variable_selection` | `all_sd_variables`, `selected_sd_variables` | Hidden lower selector; `selected_sd_variables` requires `leaf_config.sd_variables`. |

Layer 1 owns FRED-SD source evidence: selected states, selected variables, source sheets, native-frequency report, and series metadata. Layer 2 owns mixed-frequency representation choices such as calendar alignment, dropping unknown native-frequency series, native-frequency payloads, and MIDAS adapter payloads.

Selector YAML:

```yaml
path:
  1_data_task:
    fixed_axes:
      dataset: fred_sd
      frequency: monthly
      information_set_type: final_revised_data
      state_selection: selected_states
      sd_variable_selection: selected_sd_variables
    leaf_config:
      target: UR_CA
      horizons: [1]
      sd_states: [CA, NY, TX]
      sd_variables: [UR, PAYEMS]
```

