# Layer L4  --  Forecasting model

[Back to encyclopedia](../index.md) | [Browse layers](../browse_by_layer.md) | [Browse all axes](../browse_by_axis.md)

- Layer ID: `l4`
- Category: `construction`
- Sub-layers: 4
- Axes: 5
- Options across axes: 55

## Sub-layers

| Sub-layer | Name | Gate | Axes |
|---|---|---|---|
| `L4_A_model_selection` | Model selection | always | [family](axes/family.md) |
| `L4_B_forecast_strategy` | Forecast strategy | always | [forecast_strategy](axes/forecast_strategy.md) |
| `L4_C_training_window` | Training window | always | [training_start_rule](axes/training_start_rule.md), [refit_policy](axes/refit_policy.md) |
| `L4_D_tuning` | Tuning | always | [search_algorithm](axes/search_algorithm.md) |

```{toctree}
:hidden:
:maxdepth: 1

axes/family
axes/forecast_strategy
axes/training_start_rule
axes/refit_policy
axes/search_algorithm
```
