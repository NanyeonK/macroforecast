# Frame Availability

- Parent: [L1 — Data Source, Target y, Predictor x](index.md)
- Current group: Frame availability

`missing_availability` decides how L1 closes availability gaps after the
source frame exists. It is a source-frame policy, not a forecast-time
information policy and not raw-source repair.

Runtime order:

1. Load FRED data, custom data, or FRED-plus-custom data.
2. Apply `missing_availability` to the resulting L1 source frame.
3. Hand the source frame to preprocessing representation and research preprocessing.

preprocessing `transform_policy` applies official FRED transform codes. Raw-source
cleaning (missing/outlier handling before transforms) is now also an L2
decision via `imputation_policy` and `outlier_policy`.

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

- Raw missing values already present in loaded source files are handled by
  preprocessing `imputation_policy` and `outlier_policy`.
- Publication timing belongs to
  [4.1.2 Forecast-Time Information](availability_timing.md).
- Researcher-chosen missing-data strategies after representation construction
  belong to preprocessing.
- Target y imputation is not done by this axis. Missing target values remain a
  supervised-learning contract issue.

YAML:

```yaml
path:
  data:
    fixed_axes:
      missing_availability: zero_fill_leading_predictor_gaps
```

Predictor-only imputation example:

```yaml
path:
  data:
    fixed_axes:
      missing_availability: impute_predictors_only
    leaf_config:
      x_imputation: ffill
```
