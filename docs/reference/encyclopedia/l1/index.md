# Layer L1  --  Data

[Back to encyclopedia](../index.md) | [Browse layers](../browse_by_layer.md) | [Browse all axes](../browse_by_axis.md)

- Layer ID: `l1`
- Category: `construction`
- Sub-layers: 7
- Axes: 26
- Options across axes: 108

## Sub-layers

| Sub-layer | Name | Gate | Axes |
|---|---|---|---|
| `l1_a` | Source selection | always | [custom_source_policy](axes/custom_source_policy.md), [dataset](axes/dataset.md), [frequency](axes/frequency.md), [information_set_type](axes/information_set_type.md), [vintage_policy](axes/vintage_policy.md), [fred_sd_frequency_policy](axes/fred_sd_frequency_policy.md) |
| `l1_b` | Target definition | always | [target_structure](axes/target_structure.md) |
| `l1_c` | Predictor universe | always | [variable_universe](axes/variable_universe.md), [missing_availability](axes/missing_availability.md), [raw_missing_policy](axes/raw_missing_policy.md), [raw_outlier_policy](axes/raw_outlier_policy.md), [release_lag_rule](axes/release_lag_rule.md), [contemporaneous_x_rule](axes/contemporaneous_x_rule.md), [official_transform_policy](axes/official_transform_policy.md), [official_transform_scope](axes/official_transform_scope.md) |
| `l1_d` | Geography scope | always | [target_geography_scope](axes/target_geography_scope.md), [predictor_geography_scope](axes/predictor_geography_scope.md), [fred_sd_state_group](axes/fred_sd_state_group.md), [state_selection](axes/state_selection.md), [fred_sd_variable_group](axes/fred_sd_variable_group.md), [sd_variable_selection](axes/sd_variable_selection.md) |
| `l1_e` | Sample window | always | [sample_start_rule](axes/sample_start_rule.md), [sample_end_rule](axes/sample_end_rule.md) |
| `l1_f` | Horizon set | always | [horizon_set](axes/horizon_set.md) |
| `l1_g` | Regime definition | always | [regime_definition](axes/regime_definition.md), [regime_estimation_temporal_rule](axes/regime_estimation_temporal_rule.md) |

```{toctree}
:hidden:
:maxdepth: 1

axes/custom_source_policy
axes/dataset
axes/frequency
axes/information_set_type
axes/vintage_policy
axes/fred_sd_frequency_policy
axes/target_structure
axes/variable_universe
axes/missing_availability
axes/raw_missing_policy
axes/raw_outlier_policy
axes/release_lag_rule
axes/contemporaneous_x_rule
axes/official_transform_policy
axes/official_transform_scope
axes/target_geography_scope
axes/predictor_geography_scope
axes/fred_sd_state_group
axes/state_selection
axes/fred_sd_variable_group
axes/sd_variable_selection
axes/sample_start_rule
axes/sample_end_rule
axes/horizon_set
axes/regime_definition
axes/regime_estimation_temporal_rule
```
