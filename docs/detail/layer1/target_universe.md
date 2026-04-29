# 4.1.3 Target (y) And Predictor (x) Definitions

- Parent: [4.1 Layer 1: Data Source, Target y, Predictor x](index.md)
- Current group: Target (y) and predictor (x) definitions

This group names the forecasting object and the eligible raw predictor columns.
Layer 1 does not decide how y is transformed, how horizon targets are built, how
x is lagged, or which feature representation reaches a model. Those decisions
start in Layer 2.

## Target (y) Definition

`target_structure` says whether the study forecasts one y series or multiple y
series. Layer 0 `study_scope` constrains this axis.

| Value | Required payload | Meaning |
|---|---|---|
| `single_target` | `leaf_config.target` | One y series. Required by one-target Study Scope values. |
| `multi_target` | `leaf_config.targets` | Two or more y series. Required by multiple-target Study Scope values. |

Target payload:

- `target`: required for `single_target`.
- `targets`: required for `multi_target`.
- `horizons`: required forecast horizons.
- `sample_start_date` / `sample_end_date`: optional sample-period bounds.

Layer 2 boundary:

- `horizon_target_construction` decides whether y is level, difference,
  log-difference, direct average, path-average growth, or another supported
  horizon target representation.
- `target_transform`, `target_normalization`, target missing policies, and
  target outlier policies are representation decisions, not source-frame
  identity.

## Predictor (x) Definition

`variable_universe` filters the source columns that are eligible as candidate
predictors x before Layer 2 builds lags, factors, feature blocks, rotations, or
custom representations.

| Value | Required payload | Meaning |
|---|---|---|
| `all_variables` | none | Use all eligible non-date source columns except the target y columns. |
| `core_variables` | package metadata | Use the package's core macro subset when metadata are available. |
| `category_variables` | `leaf_config.variable_universe_category` and category mapping | Use columns in a named category. |
| `target_specific_variables` | `leaf_config.target_specific_columns` | Use a different x list for each y. |
| `explicit_variable_list` | `leaf_config.variable_universe_columns` | Use an explicit list of x columns. |

Custom data rule:

- Custom files do not automatically have FRED category metadata.
- For custom-only data, the safest choices are `all_variables` or
  `explicit_variable_list`.
- `category_variables` and `target_specific_variables` are valid only when the
  recipe supplies the needed mappings in `leaf_config`.

FRED-SD rule:

- FRED-SD state and workbook-variable filters are handled by
  [4.1.4 FRED-SD Predictor Scope](fred_sd_source_selection.md).
- `variable_universe` still applies after the selected FRED-SD source columns
  are loaded, so it is a general x-universe filter rather than a FRED-SD source
  selector.

Example:

```yaml
path:
  1_data_task:
    fixed_axes:
      target_structure: single_target
      variable_universe: explicit_variable_list
    leaf_config:
      target: INDPRO
      horizons: [1, 3, 6]
      sample_start_date: "1980-01"
      sample_end_date: "2019-12"
      variable_universe_columns: [RPI, UNRATE, CPIAUCSL, FEDFUNDS]
```

Multi-target example:

```yaml
path:
  1_data_task:
    fixed_axes:
      target_structure: multi_target
      variable_universe: target_specific_variables
    leaf_config:
      targets: [INDPRO, UNRATE]
      horizons: [1, 3]
      target_specific_columns:
        INDPRO: [RPI, UNRATE, CPIAUCSL]
        UNRATE: [PAYEMS, CLAIMSx, INDPRO]
```
