# 4.1.1 Source And Frame

- Parent: [4.1 Layer 1: Data Task](index.md)
- Current group: Source and frame

This group starts the Layer 1 hierarchy. `dataset` is the only source-choice axis; custom files are selected directly as `custom_csv` or `custom_parquet`. `frequency` is not the same level as `dataset`: it is derived for FRED-MD/FRED-QD/composites and is only a required follow-up when the source does not imply a single calendar frequency.

| Axis | Choices | Default / rule |
|---|---|---|
| `dataset` | `fred_md`, `fred_qd`, `fred_sd`, `fred_md+fred_sd`, `fred_qd+fred_sd`, `custom_csv`, `custom_parquet` | Required user choice. Custom values require `leaf_config.custom_dataset_schema` and `leaf_config.custom_data_path`. |
| `frequency` | `monthly`, `quarterly` | Derived for FRED-MD/QD/composites. Required for standalone FRED-SD. |
| `information_set_type` | `final_revised_data`, `pseudo_oos_on_revised_data` | Simple default is `final_revised_data`; Full recipes should write it explicitly. |

Contracts:

- `fred_md` and `fred_md+fred_sd` are monthly.
- `fred_qd` and `fred_qd+fred_sd` are quarterly.
- standalone `fred_sd` must choose monthly or quarterly because the source contains mixed native frequencies.
- `custom_csv` and `custom_parquet` are dataset choices for user-supplied files.
- custom datasets require `leaf_config.custom_dataset_schema` in `fred_md`, `fred_qd`, or `fred_sd`, plus `leaf_config.custom_data_path`.
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

Custom CSV YAML:

```yaml
path:
  1_data_task:
    fixed_axes:
      dataset: custom_csv
      frequency: monthly
      information_set_type: final_revised_data
    leaf_config:
      target: INDPRO
      horizons: [1]
      custom_dataset_schema: fred_md
      custom_data_path: /path/to/panel.csv
```
