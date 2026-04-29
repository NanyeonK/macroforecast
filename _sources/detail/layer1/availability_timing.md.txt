# 4.1.6 Availability And Timing

- Parent: [4.1 Layer 1: Data Task](index.md)
- Current group: availability and timing

These policies act after the official frame exists and before Layer 2 builds the representation.

| Axis | Choices | Default / rule |
|---|---|---|
| `missing_availability` | `zero_fill_leading_predictor_gaps`, `require_complete_rows`, `keep_available_rows`, `impute_predictors_only` | Default `zero_fill_leading_predictor_gaps`; `impute_predictors_only` requires `leaf_config.x_imputation`. |
| `release_lag_rule` | `ignore_release_lag`, `fixed_lag_all_series`, `series_specific_lag` | Default `ignore_release_lag`; `series_specific_lag` requires `leaf_config.release_lag_per_series`. |
| `contemporaneous_x_rule` | `forbid_same_period_predictors`, `allow_same_period_predictors` | Default `forbid_same_period_predictors`; `allow_same_period_predictors` is an oracle/data-leak benchmark. |

Boundary rule:

- `missing_availability` is official-frame availability, not raw-source repair.
- `release_lag_rule` simulates publication lag by shifting predictors.
- `contemporaneous_x_rule` controls whether `X_{t+h}` can be used when forecasting `y_{t+h}`.

YAML:

```yaml
path:
  1_data_task:
    fixed_axes:
      missing_availability: zero_fill_leading_predictor_gaps
      release_lag_rule: ignore_release_lag
      contemporaneous_x_rule: forbid_same_period_predictors
```

