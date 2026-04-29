# Target (y) And Predictor (x) Definition (1.2)

Layer 1 declares which y series is forecast and which source columns are
eligible as candidate predictors x. It does not choose y transformations,
horizon target construction, x lags, factors, or model-ready feature blocks.

## 1.2.1 `target_structure`

**Target (y) Definition** says whether the recipe has one y series or multiple y
series.

| Value | Required payload | Meaning |
|---|---|---|
| `single_target` | `leaf_config.target` | One y series. |
| `multi_target` | `leaf_config.targets` | Multiple y series. |

Layer 0 connection:

- `one_target_one_method` and `one_target_compare_methods` require
  `target_structure=single_target`.
- `multiple_targets_one_method` and `multiple_targets_compare_methods` require
  `target_structure=multi_target`.

## 1.2.2 `variable_universe`

**Predictor (x) Universe** decides which source columns are eligible as x before
Layer 2 representation construction.

| Value | Required payload | Meaning |
|---|---|---|
| `all_variables` | none | Use all eligible non-date, non-target source columns. |
| `core_variables` | package metadata | Use the package's core macro subset. |
| `category_variables` | category mapping payload | Use columns mapped to a category. |
| `target_specific_variables` | `leaf_config.target_specific_columns` | Use different x sets by target y. |
| `explicit_variable_list` | `leaf_config.variable_universe_columns` | Use an explicit x column list. |

Custom data note:

- Custom-only files have no automatic FRED category metadata.
- Use `all_variables` or `explicit_variable_list` unless the recipe provides
  category or target-specific mappings.

Recipe usage:

```yaml
path:
  1_data_task:
    fixed_axes:
      target_structure: single_target
      variable_universe: explicit_variable_list
    leaf_config:
      target: INDPRO
      horizons: [1, 3, 6]
      variable_universe_columns: [RPI, UNRATE, CPIAUCSL]
```

Multi-target usage:

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

## Boundary

These are not Layer 1 target/predictor-definition choices:

- `horizon_target_construction`: Layer 2 target representation.
- `target_transform` and `target_normalization`: Layer 2 target preprocessing.
- `target_lag_block`, `x_lag_feature_block`, `factor_feature_block`, and
  feature-block combinations: Layer 2 representation construction.
- `forecast_type` and `forecast_object`: Layer 3 forecast-generation contract.
- `model_family` and `benchmark_family`: Layer 3 training choices.
