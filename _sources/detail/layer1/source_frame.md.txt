# 4.1.1 Source And Frame

- Parent: [4.1 Layer 1: Data Task](index.md)
- Current group: Source and frame

This group starts the Layer 1 hierarchy. `dataset` now means the official
FRED-family source panel. Custom files are not official panels and are not
selected as `dataset` values. If the study uses a user-supplied file, choose
that in `custom_source_policy`, then declare the file format and the schema it
conforms to.

`frequency` is not the same level as `dataset`: it is derived for FRED-MD,
FRED-QD, and composite official panels. It is only a required follow-up when
the official source does not imply a single calendar frequency.

| Axis | Choices | Default / rule |
|---|---|---|
| `dataset` | `fred_md`, `fred_qd`, `fred_sd`, `fred_md+fred_sd`, `fred_qd+fred_sd` | Required user choice. This is the official source-panel family. |
| `custom_source_policy` | `official_only`, `custom_panel_only`, `official_plus_custom` | Default is `official_only`. This is the user-facing choice: official data only, my data only, or official data plus my data. |
| `custom_source_format` | `none`, `csv`, `parquet` | Default is `none`. Must be `csv` or `parquet` when a custom source is selected. |
| `custom_source_schema` | `none`, `fred_md`, `fred_qd`, `fred_sd` | Default is `none`. Must be a FRED-family schema when a custom source is selected. |
| `frequency` | `monthly`, `quarterly` | Derived for FRED-MD/QD/composites. Required for standalone FRED-SD. |
| `information_set_type` | `final_revised_data`, `pseudo_oos_on_revised_data` | Simple default is `final_revised_data`; Full recipes should write it explicitly. |

Contracts:

- `fred_md` and `fred_md+fred_sd` are monthly.
- `fred_qd` and `fred_qd+fred_sd` are quarterly.
- standalone `fred_sd` must choose monthly or quarterly because the source contains mixed native frequencies.
- User-supplied files are configured with `custom_source_policy` and
  `custom_source_format`, not as `dataset` values.
- `custom_source_policy: official_only` means official data only. It requires
  `custom_source_format: none`, `custom_source_schema: none`, and no
  `leaf_config.custom_source_path`.
- `custom_source_policy: custom_panel_only` means my data only. It loads a
  custom file instead of one selected official panel. It supports single
  official panels only: `fred_md`, `fred_qd`, or `fred_sd`. The custom schema
  must match the selected panel.
- `custom_source_policy: official_plus_custom` loads the selected official
  panel and appends custom columns after frequency alignment. It can be used
  with single or composite official panels.
- custom sources require `custom_source_schema` in `fred_md`, `fred_qd`, or
  `fred_sd`, plus `leaf_config.custom_source_path`.
- Custom CSV/Parquet schema contract:
  - first column, or Parquet index, is a parseable date index;
  - remaining columns are numeric series;
  - column names should match the selected FRED-family schema when the custom
    file is replacing an official panel;
  - appended custom columns may use new names, but duplicate names are renamed
    with a `__custom` suffix at runtime.
- `pseudo_oos_on_revised_data` uses revised data but applies pseudo-OOS availability discipline.

> **Warning:** Custom files are user-supplied source data. The Navigator can
> enforce the enum contract, but it cannot prove that the file's date index,
> frequency, column naming, vintage discipline, or T-code handling is correct.
> Check this page before running custom-source recipes.

Custom file shape by schema and frequency:

| `custom_source_schema` | Expected file shape | Layer 1 frequency behavior |
|---|---|---|
| `fred_md` | Monthly date index, national macro series columns. Column names should be FRED-style series IDs such as `INDPRO` or `CPIAUCSL`. | Native monthly. If appended to a quarterly route, Layer 1 aggregates monthly rows to quarterly means. |
| `fred_qd` | Quarterly date index, national macro series columns. Column names should be FRED-style series IDs such as `GDPC1`. | Native quarterly. If appended to a monthly route, Layer 1 linearly interpolates quarterly rows to monthly. |
| `fred_sd` | State-level panel with a parseable date index and state-level series columns. Current v1 custom-source runtime treats this as monthly (`state_monthly`). | Monthly by contract. If the selected route is quarterly, Layer 1 aggregates monthly state rows to quarterly means. |

Additional custom-source fields:

| Field | Where | Required when | Meaning |
|---|---|---|---|
| `custom_source_format` | `fixed_axes` | `custom_source_policy != official_only` | File parser: `csv` or `parquet`. |
| `custom_source_schema` | `fixed_axes` | `custom_source_policy != official_only` | FRED-family schema the file follows. |
| `custom_source_path` | `leaf_config` | `custom_source_policy != official_only` | Local file path. This is not an enum axis because it is study-specific. |

Minimal YAML:

```yaml
path:
  1_data_task:
    fixed_axes:
      dataset: fred_md
      custom_source_policy: official_only
      custom_source_format: none
      custom_source_schema: none
      information_set_type: final_revised_data
    leaf_config:
      target: INDPRO
      horizons: [1, 3, 6]
      sample_start_date: "1980-01"
      sample_end_date: "2019-12"
```

Replace FRED-MD with a custom CSV:

```yaml
path:
  1_data_task:
    fixed_axes:
      dataset: fred_md
      custom_source_policy: custom_panel_only
      custom_source_format: csv
      custom_source_schema: fred_md
      frequency: monthly
      information_set_type: final_revised_data
    leaf_config:
      target: INDPRO
      horizons: [1]
      custom_source_path: /path/to/panel.csv
```

Append custom state-level columns to FRED-MD:

```yaml
path:
  1_data_task:
    fixed_axes:
      dataset: fred_md
      custom_source_policy: official_plus_custom
      custom_source_format: parquet
      custom_source_schema: fred_sd
      frequency: monthly
      information_set_type: final_revised_data
    leaf_config:
      target: INDPRO
      horizons: [1, 3, 6]
      custom_source_path: /path/to/custom_state_panel.parquet
```
