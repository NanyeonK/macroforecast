# Data Layer

Layer 1 defines the forecasting dataset: source, target, predictor universe, geography, sample window, horizons, and regimes.

Core source axes:

| Axis | Common values |
| --- | --- |
| `custom_source_policy` | `official_only`, `custom_panel_only`, `official_plus_custom` |
| `dataset` | `fred_md`, `fred_qd`, `fred_sd`, `fred_md+fred_sd`, `fred_qd+fred_sd` |
| `frequency` | `monthly`, `quarterly` |
| `information_set_type` | `final_revised_data`, `pseudo_oos_on_revised_data` |
| `vintage_policy` | `current_vintage`; `real_time_alfred` is schema-present but future. |

Target and horizon axes:

| Axis | Common values |
| --- | --- |
| `target_structure` | `single_target`, `multi_target` |
| `variable_universe` | `all_variables`, `core_variables`, `category_variables`, `target_specific_variables`, `explicit_variable_list` |
| `horizon_set` | `single`, `custom_list`, `range_up_to_h`, `standard_md`, `standard_qd` |
| `sample_start_rule` | `earliest_available`, `fixed_date`, `max_balanced` |
| `sample_end_rule` | `latest_available`, `fixed_date` |

Use `leaf_config.target` for `single_target`, `leaf_config.targets` for `multi_target`, `leaf_config.target_horizons` for `single` or `custom_list`, and `leaf_config.max_horizon` for `range_up_to_h`.

FRED-SD geography is controlled by `target_geography_scope`, `predictor_geography_scope`, `fred_sd_state_group`, `state_selection`, `fred_sd_variable_group`, and `sd_variable_selection`.
