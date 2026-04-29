# 4.1.4 Raw Source Cleaning

- Parent: [4.1 Layer 1: Data Task](index.md)
- Current group: raw source cleaning

These policies act on defects already present in the raw source panel before official transforms or FRED t-codes are applied.

| Axis | Choices | Default / rule |
|---|---|---|
| `raw_missing_policy` | `preserve_raw_missing`, `zero_fill_leading_predictor_missing_before_tcode`, `impute_raw_predictors`, `drop_raw_missing_rows` | Default `preserve_raw_missing`; `impute_raw_predictors` requires `leaf_config.raw_x_imputation`. |
| `raw_outlier_policy` | `preserve_raw_outliers`, `winsorize_raw`, `iqr_clip_raw`, `mad_clip_raw`, `zscore_clip_raw`, `set_raw_outliers_to_missing` | Default `preserve_raw_outliers`; `leaf_config.raw_outlier_columns` can restrict the affected columns. |

Boundary rule:

- Layer 1 raw cleaning changes the source panel before official transforms/T-codes. That means the transform output can change.
- Layer 2 missing/outlier policies act after the official frame exists. They can mix raw-source defects with transform-induced missing values and model-input preprocessing artifacts.

Full researchers can choose either phase, but provenance must record which phase was used.

