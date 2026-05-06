# Official Transforms

- Parent: [Layer 1: Data Source, Target y, Predictor x](index.md)
- Current group: official transforms

This group decides whether FRED-provided transform codes are applied while
constructing the source frame for FRED-MD and FRED-QD routes.

Visibility rule:

- FRED-MD and FRED-QD expose this group because their source workbooks ship
  with official transform-code metadata.
- Custom-only studies hide this group by default. The custom file is already a
  user-defined source frame, so transform decisions belong either in the custom
  file itself or in Layer 2 research preprocessing.
- Standalone FRED-SD hides this group by default. FRED-SD has no official
  transform-code column comparable to FRED-MD/QD.
- Composite `fred_md+fred_sd` and `fred_qd+fred_sd` routes expose this group
  for the FRED-MD/QD portion of the frame. FRED-SD-specific inferred or
  empirical transformations remain Layer 2 decisions.

| Axis | Choices | Default / rule |
|---|---|---|
| `official_transform_policy` | `apply_official_tcode`, `keep_official_raw_scale` | Simple default `apply_official_tcode`. |
| `official_transform_scope` | `target_only`, `predictors_only`, `target_and_predictors`, `none` | Simple default `target_and_predictors`. |

Layer boundary:

- Layer 1 owns official FRED-MD/FRED-QD t-code application to the source frame.
- FRED-SD has no official t-codes. Inferred or empirical FRED-SD t-code policies are Layer 2 representation policies because they are researcher-chosen transformations, not official source definitions.
- Extra scaling, imputation, factor extraction, lag construction, and target representation choices are Layer 2.
- If an imported recipe sets this axis while no FRED-MD/QD source is selected,
  the Navigator keeps the non-default choice visible as an incompatibility so
  the user can fix the recipe rather than silently dropping it.

YAML:

```yaml
path:
  1_data_task:
    fixed_axes:
      official_transform_policy: apply_official_tcode
      official_transform_scope: target_and_predictors
```
