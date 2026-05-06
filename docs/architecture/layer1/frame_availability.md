# Frame Availability

- Parent: [4.1 Layer 1: Data Source, Target y, Predictor x](index.md)
- Current group: Official transform / frame availability

`missing_availability` decides how Layer 1 closes availability gaps after the
source frame exists. It is a source-frame policy, not a forecast-time
information policy and not raw-source repair.

Runtime order:

1. Load FRED data, custom data, or FRED-plus-custom data.
2. Apply raw-source missing/outlier policies when non-default values are set.
3. Apply official FRED transform codes when available and enabled.
4. Apply `missing_availability` to the resulting Layer 1 source frame.
5. Hand the source frame to Layer 2 representation and research preprocessing.

| Axis | Choices | Default / rule |
|---|---|---|
| `missing_availability` | `zero_fill_leading_predictor_gaps`, `require_complete_rows`, `keep_available_rows`, `impute_predictors_only` | Default `zero_fill_leading_predictor_gaps`; `impute_predictors_only` requires `leaf_config.x_imputation`. |

## Value catalog

| Value | Meaning |
|---|---|
| `zero_fill_leading_predictor_gaps` | Default. Fill predictor leading gaps after source-frame construction; target gaps remain guarded by target availability checks. |
| `require_complete_rows` | Require complete rows for the source frame. Use when the study intentionally avoids any frame-level missingness. |
| `keep_available_rows` | Keep rows that are usable under the current target/predictor availability contract. |
| `impute_predictors_only` | Impute predictor x gaps only. Requires `leaf_config.x_imputation` in {`mean`, `median`, `ffill`, `bfill`}; target y is not imputed here. |

Boundary rule:

- Raw missing values already present in loaded source files belong to
  [4.1.5 Raw Source Cleaning](raw_source_cleaning.md).
- Publication timing belongs to
  [4.1.2 Forecast-Time Information](availability_timing.md).
- Researcher-chosen missing-data strategies after representation construction
  belong to Layer 2.
- Target y imputation is not done by this axis. Missing target values remain a
  supervised-learning contract issue.

YAML:

```yaml
path:
  1_data_task:
    fixed_axes:
      missing_availability: zero_fill_leading_predictor_gaps
```

Predictor-only imputation example:

```yaml
path:
  1_data_task:
    fixed_axes:
      missing_availability: impute_predictors_only
    leaf_config:
      x_imputation: ffill
```
