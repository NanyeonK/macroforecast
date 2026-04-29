# 4.1.5 Official Transforms

- Parent: [4.1 Layer 1: Data Task](index.md)
- Current group: official transforms

This group decides whether official dataset transform codes are applied while constructing the official frame.

| Axis | Choices | Default / rule |
|---|---|---|
| `official_transform_policy` | `apply_official_tcode`, `keep_official_raw_scale` | Simple default `apply_official_tcode`. |
| `official_transform_scope` | `target_only`, `predictors_only`, `target_and_predictors`, `none` | Simple default `target_and_predictors`. |

Layer boundary:

- Layer 1 owns official FRED-MD/FRED-QD t-code application to the source frame.
- FRED-SD has no official t-codes. Inferred or empirical FRED-SD t-code policies are Layer 2 representation policies because they are researcher-chosen transformations, not official source definitions.
- Extra scaling, imputation, factor extraction, lag construction, and target representation choices are Layer 2.

YAML:

```yaml
path:
  1_data_task:
    fixed_axes:
      official_transform_policy: apply_official_tcode
      official_transform_scope: target_and_predictors
```

