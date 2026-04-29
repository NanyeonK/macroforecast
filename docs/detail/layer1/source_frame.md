# 4.1.1 Source And Frame

- Parent: [4.1 Layer 1: Data Task](index.md)
- Current group: Source and frame

This group starts the Layer 1 hierarchy. Choose the data source mode first:
FRED data only, custom data only, or FRED data plus custom data. Then close the
analysis frequency. A FRED source-panel choice is shown only when the selected
mode uses FRED data.

Custom files are not FRED panels. If the study uses a custom file, provide
`leaf_config.custom_source_path`. The parser is inferred from the file
extension. The schema is inferred from the file frequency and source mode; the
user should not have to choose a fake FRED-MD/FRED-QD route for custom-only data.

`frequency` is the final calendar frequency of the Layer 1 source frame. It is
derived for FRED-MD, FRED-QD, and composite FRED panels. It is required for
standalone FRED-SD and custom-only data.

| Axis | Choices | Default / rule |
|---|---|---|
| `custom_source_policy` | `official_only`, `custom_panel_only`, `official_plus_custom` | Default is `official_only`. First source choice: FRED data only, custom data only, or FRED data plus custom data. |
| `dataset` | `fred_md`, `fred_qd`, `fred_sd`, `fred_md+fred_sd`, `fred_qd+fred_sd` | FRED source-panel choice. Active only when source mode uses FRED data. |
| `frequency` | `monthly`, `quarterly` | Derived for FRED-MD/QD/composites. Required for standalone FRED-SD and custom-only data. |

Contracts:

- `fred_md` and `fred_md+fred_sd` are monthly.
- `fred_qd` and `fred_qd+fred_sd` are quarterly.
- standalone `fred_sd` must choose monthly or quarterly because the source contains mixed native frequencies.
- `custom_source_policy: official_only` means FRED data only. It requires no
  `leaf_config.custom_source_path`; `dataset` selects the FRED loader.
- `custom_source_policy: custom_panel_only` means custom data only. It loads a
  custom file as the source panel. The UI should ask for file path and
  frequency, not for a FRED source panel.
- `custom_source_policy: official_plus_custom` loads the selected FRED
  panel and appends custom columns. It can be used with single or composite
  FRED panel routes.
- User-supplied files are configured with `custom_source_policy` plus
  `leaf_config.custom_source_path`, not as `dataset` values.
- custom sources require `leaf_config.custom_source_path`. The compiler infers
  `csv` from `.csv` and `parquet` from `.parquet`/`.pq`. Legacy recipes may
  still set `fixed_axes.custom_source_format` when a path has no extension.
- Custom CSV/Parquet schema contract:
  - first column, or Parquet index, is a parseable date index;
  - remaining columns are numeric series;
  - for `custom_panel_only`, the target y column must exist in the file;
  - predictor x columns are all non-target numeric columns unless
    `variable_universe` narrows them later;
  - appended custom columns may use new names, but duplicate names are renamed
    with a `__custom` suffix at runtime.
- In practical terms, a custom file is acceptable if its date index, column
  shape, and row frequency already match the selected Layer 1 frequency.
  Monthly routes expect monthly rows. Quarterly routes expect quarterly rows.
- `pseudo_oos_on_revised_data` uses revised data but applies pseudo-OOS availability discipline.

> **Warning:** Custom files are user-supplied source data. The Navigator can
> enforce the route contract, but it cannot prove that the file's date index,
> frequency, column naming, vintage discipline, or T-code handling is correct.
> Check this page before running custom-source recipes.

Custom file shape by selected Layer 1 frequency:

| Selected `frequency` | Expected file shape | Internal loader label |
|---|---|---|
| `monthly` | Monthly date index, numeric series columns. Column names may be FRED IDs or study-specific IDs. | Generic monthly custom panel. |
| `quarterly` | Quarterly date index, numeric series columns. Column names may be FRED IDs or study-specific IDs. | Generic quarterly custom panel. |

Additional custom-source fields:

| Field | Where | Required when | Meaning |
|---|---|---|---|
| `custom_source_path` | `leaf_config` | `custom_source_policy != official_only` | Local file path. This is not an enum axis because it is study-specific. |

Compatibility fields:

| Field | Status | Meaning |
|---|---|---|
| `custom_source_format` | hidden legacy fixed axis | Optional override for extensionless files; otherwise inferred from `custom_source_path`. |
| `custom_source_schema` | hidden legacy fixed axis | Optional override for older recipes; otherwise inferred from source mode and `frequency`. |

Minimal YAML:

```yaml
path:
  1_data_task:
    fixed_axes:
      dataset: fred_md
      custom_source_policy: official_only
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
      custom_source_policy: custom_panel_only
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
      frequency: monthly
      information_set_type: final_revised_data
    leaf_config:
      target: INDPRO
      horizons: [1, 3, 6]
      custom_source_path: /path/to/custom_state_panel.parquet
```
