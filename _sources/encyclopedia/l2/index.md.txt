# Layer L2  --  Preprocessing

[Back to encyclopedia](../index.md) | [Browse layers](../browse_by_layer.md) | [Browse all axes](../browse_by_axis.md)

- Layer ID: `l2`
- Category: `construction`
- Sub-layers: 5
- Axes: 13
- Options across axes: 50

## Sub-layers

| Sub-layer | Name | Gate | Axes |
|---|---|---|---|
| `l2_a` | FRED-SD frequency alignment | always | [sd_series_frequency_filter](axes/sd_series_frequency_filter.md), [quarterly_to_monthly_rule](axes/quarterly_to_monthly_rule.md), [monthly_to_quarterly_rule](axes/monthly_to_quarterly_rule.md) |
| `l2_b` | Transform | always | [transform_policy](axes/transform_policy.md), [transform_scope](axes/transform_scope.md) |
| `l2_c` | Outlier handling | always | [outlier_policy](axes/outlier_policy.md), [outlier_action](axes/outlier_action.md), [outlier_scope](axes/outlier_scope.md) |
| `l2_d` | Imputation | always | [imputation_policy](axes/imputation_policy.md), [imputation_temporal_rule](axes/imputation_temporal_rule.md), [imputation_scope](axes/imputation_scope.md) |
| `l2_e` | Frame edge | always | [frame_edge_policy](axes/frame_edge_policy.md), [frame_edge_scope](axes/frame_edge_scope.md) |

```{toctree}
:hidden:
:maxdepth: 1

axes/sd_series_frequency_filter
axes/quarterly_to_monthly_rule
axes/monthly_to_quarterly_rule
axes/transform_policy
axes/transform_scope
axes/outlier_policy
axes/outlier_action
axes/outlier_scope
axes/imputation_policy
axes/imputation_temporal_rule
axes/imputation_scope
axes/frame_edge_policy
axes/frame_edge_scope
```
