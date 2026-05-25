# Layer L2  --  Preprocessing

[Back to encyclopedia](../index.md) | [Browse layers](../browse_by_layer.md) | [Browse all axes](../browse_by_axis.md)

- Layer ID: `l2`
- Category: `construction`
- Sub-layers: 5
- Axes: 11
- Options across axes: 42

## Sub-layers

| Sub-layer | Name | Gate | Axes |
|---|---|---|---|
| `l2_a` | Mixed frequency alignment | always | [sd_series_frequency_filter](axes/sd_series_frequency_filter.md), [mixed_frequency_representation](axes/mixed_frequency_representation.md), [quarterly_to_monthly_policy](axes/quarterly_to_monthly_policy.md), [monthly_to_quarterly_policy](axes/monthly_to_quarterly_policy.md) |
| `l2_b` | Transform | always | [transform_policy](axes/transform_policy.md), [sd_tcode_policy](axes/sd_tcode_policy.md) |
| `l2_c` | Outlier handling | always | [outlier_policy](axes/outlier_policy.md), [outlier_action](axes/outlier_action.md) |
| `l2_d` | Imputation | always | [imputation_policy](axes/imputation_policy.md), [imputation_temporal_rule](axes/imputation_temporal_rule.md) |
| `l2_e` | Frame edge | always | [frame_edge_policy](axes/frame_edge_policy.md) |

```{toctree}
:hidden:
:maxdepth: 1

axes/sd_series_frequency_filter
axes/mixed_frequency_representation
axes/quarterly_to_monthly_policy
axes/monthly_to_quarterly_policy
axes/transform_policy
axes/sd_tcode_policy
axes/outlier_policy
axes/outlier_action
axes/imputation_policy
axes/imputation_temporal_rule
axes/frame_edge_policy
```
