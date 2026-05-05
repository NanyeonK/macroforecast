# 4.1.3 Target (y) And Predictor (x) Definitions

- Parent: [4.1 Layer 1: Data Source, Target y, Predictor x](index.md)
- Current group: Target (y) and predictor (x) definitions

This group names the forecasting target y and, for FRED-backed routes, the
eligible raw predictor x columns. Layer 1 does not decide how y is transformed,
how horizon targets are built, how x is lagged, or which representation reaches
a model. Those decisions start in Layer 2.

FRED column dictionaries are not maintained in this Layer 1 page. Use
[5. FRED-Dataset](../../for_researchers/fred_datasets/index.md) for the current FRED-MD,
FRED-QD, and FRED-SD column definitions before writing explicit y/x lists.

## Target (y) Definition

`target_structure` has two user-facing choices: `Single Target` and
`Multiple Targets`. The canonical recipe values remain `single_target` and
`multi_target`.

| Value | Required payload | Meaning |
|---|---|---|
| `single_target` | `leaf_config.target` | Single Target: one y series. Required by one-target Study Scope values. |
| `multi_target` | `leaf_config.targets` | Multiple Targets: two or more y series. Required by multiple-target Study Scope values. |

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

## FRED-MD/QD Predictor (x) Universe

`variable_universe` is a FRED-MD/QD metadata axis. It filters FRED-MD/FRED-QD
source columns that are eligible as candidate predictors x before Layer 2
builds lags, factors, feature blocks, rotations, or custom representations.
The current all-column dictionaries live in
[5.1 FRED-MD](../../for_researchers/fred_datasets/fred_md.md) and
[5.2 FRED-QD](../../for_researchers/fred_datasets/fred_qd.md).

For `custom_source_policy: custom_panel_only` or standalone `dataset: fred_sd`,
this axis is hidden by default. Custom-only x columns are defined by the custom
file itself. FRED-SD x columns are defined by state scope and series scope in
[4.1.4 FRED-SD Predictor Scope](fred_sd_source_selection.md), with the current
generated column dictionary in
[5.3 FRED-SD](../../for_researchers/fred_datasets/fred_sd.md).

| Value | Required payload | Meaning for FRED-MD/QD |
|---|---|---|
| `all_variables` | none | Use all eligible non-date source columns except target y. For FRED-MD/QD this means the whole selected FRED panel after source loading and official transform policy. |
| `core_variables` | none | Use the package core macro subset: `INDPRO`, `PAYEMS`, `CPIAUCSL`, `FEDFUNDS`, `GS10`, `M2SL`, `UNRATE`, when those columns exist in the selected panel. |
| `category_variables` | `leaf_config.variable_universe_category_columns` and `leaf_config.variable_universe_category` | Use a named category from a user-supplied category map. The map should be built from the [FRED-MD](../../for_researchers/fred_datasets/fred_md.md) or [FRED-QD](../../for_researchers/fred_datasets/fred_qd.md) all-column table, or another documented study taxonomy. |
| `target_specific_variables` | `leaf_config.target_specific_columns` | Use a different x list for each y. Write these lists after inspecting the selected dataset's all-column table in [5. FRED-Dataset](../../for_researchers/fred_datasets/index.md). |
| `explicit_variable_list` | `leaf_config.variable_universe_columns` | Use one explicit x list for all target y series. Write the list after inspecting the selected dataset's all-column table in [5. FRED-Dataset](../../for_researchers/fred_datasets/index.md). |

Current package behavior:

- `all_variables` uses the loaded panel columns directly.
- `core_variables` uses the fixed package subset listed above.
- `category_variables` does not currently infer built-in FRED category maps at
  runtime. It requires `leaf_config.variable_universe_category_columns`.
- `target_specific_variables` and `explicit_variable_list` are user-authored
  lists. They should be written against the column names visible in the
  selected FRED-MD/QD panel. Use [5. FRED-Dataset](../../for_researchers/fred_datasets/index.md)
  as the current column reference.

Category map example:

```yaml
path:
  1_data_task:
    fixed_axes:
      variable_universe: category_variables
    leaf_config:
      variable_universe_category: labor
      variable_universe_category_columns:
        labor: [PAYEMS, UNRATE]
        prices: [CPIAUCSL]
        policy: [FEDFUNDS, GS10]
```

Explicit list example:

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

Target-specific example:

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

FRED-SD rule:

- FRED-SD state and series filters are handled by
  [4.1.4 FRED-SD Predictor Scope](fred_sd_source_selection.md).
- Standalone `fred_sd` should use State Scope / State List and Series Scope /
  Series List, not `variable_universe`.
- Composite `fred_md+fred_sd` or `fred_qd+fred_sd` routes use
  `variable_universe` for the FRED-MD/QD portion and the FRED-SD scope axes for
  the state-level portion.
