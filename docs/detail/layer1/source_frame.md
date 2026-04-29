# 4.1.1 Source And Frame

- Parent: [4.1 Layer 1: Data Task](index.md)
- Current group: Source and frame

This group starts the Layer 1 hierarchy. `dataset` now means the official
FRED-family source panel. Custom files are not official panels and are not
selected as `dataset` values. If the study uses a user-supplied file, choose
that in `custom_source_mode`, then declare the file format and the schema it
conforms to.

`frequency` is not the same level as `dataset`: it is derived for FRED-MD,
FRED-QD, and composite official panels. It is only a required follow-up when
the official source does not imply a single calendar frequency.

| Axis | Choices | Default / rule |
|---|---|---|
| `dataset` | `fred_md`, `fred_qd`, `fred_sd`, `fred_md+fred_sd`, `fred_qd+fred_sd` | Required user choice. This is the official source-panel family. |
| `custom_source_mode` | `no_custom_source`, `replace_official_panel`, `append_to_official_panel` | Default is `no_custom_source`. Custom use requires `custom_source_format`, `leaf_config.custom_dataset_schema`, and `leaf_config.custom_data_path`. |
| `custom_source_format` | `none`, `csv`, `parquet` | Default is `none`. Must be `csv` or `parquet` when a custom source is selected. |
| `frequency` | `monthly`, `quarterly` | Derived for FRED-MD/QD/composites. Required for standalone FRED-SD. |
| `information_set_type` | `final_revised_data`, `pseudo_oos_on_revised_data` | Simple default is `final_revised_data`; Full recipes should write it explicitly. |

Contracts:

- `fred_md` and `fred_md+fred_sd` are monthly.
- `fred_qd` and `fred_qd+fred_sd` are quarterly.
- standalone `fred_sd` must choose monthly or quarterly because the source contains mixed native frequencies.
- User-supplied files are configured with `custom_source_mode` and
  `custom_source_format`, not as `dataset` values.
- `custom_source_mode: replace_official_panel` replaces one selected official
  panel with a custom file. It supports single official panels only:
  `fred_md`, `fred_qd`, or `fred_sd`. The custom schema must match the selected
  panel.
- `custom_source_mode: append_to_official_panel` loads the selected official
  panel and appends custom columns after frequency alignment. It can be used
  with single or composite official panels.
- custom sources require `leaf_config.custom_dataset_schema` in `fred_md`,
  `fred_qd`, or `fred_sd`, plus `leaf_config.custom_data_path`.
- Custom CSV/Parquet schema contract:
  - first column, or Parquet index, is a parseable date index;
  - remaining columns are numeric series;
  - column names should match the selected FRED-family schema when the custom
    file is replacing an official panel;
  - appended custom columns may use new names, but duplicate names are renamed
    with a `__custom` suffix at runtime.
- `pseudo_oos_on_revised_data` uses revised data but applies pseudo-OOS availability discipline.

Minimal YAML:

```yaml
path:
  1_data_task:
    fixed_axes:
      dataset: fred_md
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
      custom_source_mode: replace_official_panel
      custom_source_format: csv
      frequency: monthly
      information_set_type: final_revised_data
    leaf_config:
      target: INDPRO
      horizons: [1]
      custom_dataset_schema: fred_md
      custom_data_path: /path/to/panel.csv
```

Append custom state-level columns to FRED-MD:

```yaml
path:
  1_data_task:
    fixed_axes:
      dataset: fred_md
      custom_source_mode: append_to_official_panel
      custom_source_format: parquet
      frequency: monthly
      information_set_type: final_revised_data
    leaf_config:
      target: INDPRO
      horizons: [1, 3, 6]
      custom_dataset_schema: fred_sd
      custom_data_path: /path/to/custom_state_panel.parquet
```
