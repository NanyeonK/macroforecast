# 4.1.1 Source And Frame

- Parent: [4.1 Layer 1: Data Task](index.md)
- Current group: Source and frame

This group selects the official data source, loader, frequency, and information-set regime.

| Axis | Choices | Default / rule |
|---|---|---|
| `dataset` | `fred_md`, `fred_qd`, `fred_sd`, `fred_md+fred_sd`, `fred_qd+fred_sd` | Required user choice. |
| `source_adapter` | `fred_md`, `fred_qd`, `fred_sd`, `custom_csv`, `custom_parquet` | Omit for canonical FRED loaders; it defaults to the selected dataset. |
| `frequency` | `monthly`, `quarterly` | Derived for FRED-MD/QD/composites. Required for standalone FRED-SD. |
| `information_set_type` | `final_revised_data`, `pseudo_oos_on_revised_data` | Simple default is `final_revised_data`; Full recipes should write it explicitly. |

Contracts:

- `fred_md` and `fred_md+fred_sd` are monthly.
- `fred_qd` and `fred_qd+fred_sd` are quarterly.
- standalone `fred_sd` must choose monthly or quarterly because the source contains mixed native frequencies.
- `custom_csv` and `custom_parquet` require `leaf_config.custom_data_path`.
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

