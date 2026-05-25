# Data Policies

Data-policy axes live in Layer 1 and Layer 2.

Layer 1 policies decide what the recipe means before preprocessing:

| Axis | Valid values |
| --- | --- |
| `missing_availability` | `require_complete_rows`, `keep_available_rows`, `impute_predictors_only`, `zero_fill_leading_predictor_gaps` |
| `release_lag_rule` | `ignore_release_lag`, `fixed_lag_all_series`, `series_specific_lag` |
| `contemporaneous_x_rule` | `allow_same_period_predictors`, `forbid_same_period_predictors` |

The current default for same-period predictors is `allow_same_period_predictors`.

Layer 2 applies cleaning after L1: transform, outlier handling, imputation, and frame-edge trimming.
