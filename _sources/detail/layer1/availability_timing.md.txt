# 4.1.2 Forecast-Time Information

- Parent: [4.1 Layer 1: Data Source, Target y, Predictor x](index.md)
- Current group: Forecast-Time Information

These policies define what information is available when a forecast is made.
They separate three concepts that are often mixed in macro forecasting code:

- data revision/vintage regime: which version of the data is used;
- publication lag: when each predictor x observation becomes usable;
- same-period predictor access: whether x at the target date is allowed.

Frame availability is separate. `missing_availability` runs after the Layer 1
source frame exists and is documented in [4.1.7 Frame Availability](frame_availability.md).

| Axis | Choices | Default / rule |
|---|---|---|
| `information_set_type` | `final_revised_data`, `pseudo_oos_on_revised_data` | Data revision/vintage regime. Default uses final revised data. |
| `release_lag_rule` | `ignore_release_lag`, `fixed_lag_all_series`, `series_specific_lag` | Publication lag rule. Default ignores publication lag; `series_specific_lag` requires `leaf_config.release_lag_per_series`. |
| `contemporaneous_x_rule` | `forbid_same_period_predictors`, `allow_same_period_predictors` | Same-period x rule. Default forbids target-date predictors; `allow_same_period_predictors` is an oracle/data-leak benchmark. |

Boundary rule:

- `information_set_type` is about revisions/vintages. It answers whether the
  frame uses final revised data or a pseudo-real-time restriction on revised
  data.
- `release_lag_rule` is about publication timing. It answers whether an
  observation dated at month/quarter t is already known at the forecast origin.
- `contemporaneous_x_rule` controls whether `x_{t+h}` can be used when
  forecasting `y_{t+h}`. Allowing it is an oracle benchmark unless the research
  design is explicitly nowcasting with contemporaneous indicators.
- `missing_availability` is not a forecast-time information axis. It is a
  source-frame availability policy after source loading, raw-source quality
  handling, and official transforms.

YAML:

```yaml
path:
  1_data_task:
    fixed_axes:
      information_set_type: final_revised_data
      release_lag_rule: ignore_release_lag
      contemporaneous_x_rule: forbid_same_period_predictors
```
