# Data Policies

Data-policy axes live in Layer 1 and Layer 2.

Layer 1 policies decide what the recipe means before preprocessing:

| Axis | Valid values |
| --- | --- |
| `missing_availability` | `require_complete_rows`, `keep_available_rows`, `impute_predictors_only`, `zero_fill_leading_predictor_gaps` |
| `raw_missing_policy` | `preserve_raw_missing`, `zero_fill_leading_predictor_missing_before_tcode`, `impute_raw_predictors`, `drop_raw_missing_rows` |
| `raw_outlier_policy` | `preserve_raw_outliers`, `winsorize_raw`, `iqr_clip_raw`, `mad_clip_raw`, `zscore_clip_raw`, `set_raw_outliers_to_missing` |
| `release_lag_rule` | `ignore_release_lag`, `fixed_lag_all_series`, `series_specific_lag` |
| `contemporaneous_x_rule` | `allow_same_period_predictors`, `forbid_same_period_predictors` |
| `official_transform_policy` | `apply_official_tcode`, `keep_official_raw_scale` |
| `official_transform_scope` | `target_only`, `predictors_only`, `target_and_predictors`, `none` |

The current default for same-period predictors is `allow_same_period_predictors`.

Layer 2 applies cleaning after L1: transform, outlier handling, imputation, and frame-edge trimming.
