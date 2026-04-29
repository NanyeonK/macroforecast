# Source, Frequency, And Forecast-Time Information (1.1)

Layer 1 starts with the source frame. The first choice is **Data Source Mode**:
use FRED data only, custom data only, or FRED data plus custom columns. After
that, close the analysis frequency and forecast-time information contract.

| Section | Axis / payload | User-facing name | Role |
|---|---|---|---|
| 1.1.1 | `custom_source_policy` | Data Source Mode | FRED data only, custom data only, or FRED plus custom data |
| 1.1.2 | `dataset` | FRED Source Panel | Which FRED panel to load; hidden for custom-only studies |
| 1.1.3 | `custom_source_path` | Custom file path | Study-specific file path in `leaf_config`; not a Navigator axis |
| 1.1.4 | `frequency` | Analysis Frequency | Monthly or quarterly Layer 1 frame |
| 1.1.5 | `information_set_type` | Data Revision / Vintage Regime | Final revised data or pseudo-OOS on revised data |

**At a glance.**

- `custom_source_policy = official_only` is the default.
- `dataset` is required only when the source mode uses FRED data.
- `custom_panel_only` should not expose a FRED Source Panel choice; it needs
  `frequency` and `leaf_config.custom_source_path`.
- `official_plus_custom` loads a FRED Source Panel and appends custom columns
  that already match the selected frequency.
- `information_set_type` controls revision/vintage status, not publication lag.
  Publication lag is [Data Handling Policies](policies.md).

`data_domain` was removed because every built-in FRED source already implies a
macro domain through source metadata.

---

## 1.1.1 `custom_source_policy`

**Data Source Mode** decides what defines the Layer 1 panel.

| Value | Label | Meaning |
|---|---|---|
| `official_only` | FRED Data Only (Default) | Load one FRED Source Panel. |
| `custom_panel_only` | Custom Data Only | Load a custom file as the source panel. |
| `official_plus_custom` | FRED Data + Custom Data | Load a FRED Source Panel and append custom columns. |

Custom modes require:

```yaml
path:
  1_data_task:
    leaf_config:
      custom_source_path: /path/to/panel.csv
```

## 1.1.2 `dataset`

**FRED Source Panel** is active only when `custom_source_policy` is
`official_only` or `official_plus_custom`.

| Value | Loader | Frequency rule |
|---|---|---|
| `fred_md` | `macrocast.raw.load_fred_md` | monthly |
| `fred_qd` | `macrocast.raw.load_fred_qd` | quarterly |
| `fred_sd` | `macrocast.raw.load_fred_sd` | user chooses monthly or quarterly |
| `fred_md+fred_sd` | composite loader | monthly |
| `fred_qd+fred_sd` | composite loader | quarterly |

Do not use `dataset` as a fake schema selector for custom-only data. Custom
files are accepted by file shape: parseable dates, numeric series columns, and
rows at the selected `frequency`.

## 1.1.3 Custom File Path Contract

`custom_source_path` is a `leaf_config` payload because it is study-specific,
not a reusable enum axis.

Parser inference:

- `.csv` -> `macrocast.raw.load_custom_csv`
- `.parquet` or `.pq` -> `macrocast.raw.load_custom_parquet`

File shape:

- first CSV column, first Parquet column, or Parquet index must be parseable as
  dates;
- remaining columns must be numeric series;
- monthly custom data must have monthly rows;
- quarterly custom data must have quarterly rows;
- for custom-only data, the target y column must exist in the file;
- candidate predictor x columns are non-target numeric columns unless
  `variable_universe` narrows them.

The package can validate file parseability and enum choices. It cannot certify
economic meaning, release timing, vintage discipline, or whether a T-code row
was stripped or applied correctly. Treat the file path and file hash as
provenance.

## 1.1.4 `frequency`

**Analysis Frequency** is the final calendar frequency of the Layer 1 source
frame.

| Value | Meaning |
|---|---|
| `monthly` | Monthly source frame. Required for FRED-MD and MD+SD; allowed for FRED-SD and custom-only. |
| `quarterly` | Quarterly source frame. Required for FRED-QD and QD+SD; allowed for FRED-SD and custom-only. |

Runtime conversion is available for FRED-SD monthly/quarterly conversion and is
recorded in provenance. Custom files must already be at the selected frequency.

## 1.1.5 `information_set_type`

**Data Revision / Vintage Regime** decides which version of the data is allowed
at each forecast origin.

| Value | Label | Meaning |
|---|---|---|
| `final_revised_data` | Final Revised Data (Default) | Use the latest revised values. |
| `pseudo_oos_on_revised_data` | Pseudo-OOS on Revised Data | Use revised data with pseudo out-of-sample masking. |

This is not the publication-lag rule. Use `release_lag_rule` when the question
is "when was this series published and usable?"

## Recipe Examples

FRED-MD:

```yaml
path:
  1_data_task:
    fixed_axes:
      custom_source_policy: official_only
      dataset: fred_md
      information_set_type: final_revised_data
    leaf_config:
      target: INDPRO
      horizons: [1, 3, 6]
```

Custom-only monthly CSV:

```yaml
path:
  1_data_task:
    fixed_axes:
      custom_source_policy: custom_panel_only
      frequency: monthly
      information_set_type: final_revised_data
    leaf_config:
      target: INDPRO
      horizons: [1, 3]
      custom_source_path: /path/to/custom_monthly_panel.csv
```

FRED-QD plus custom quarterly columns:

```yaml
path:
  1_data_task:
    fixed_axes:
      custom_source_policy: official_plus_custom
      dataset: fred_qd
      frequency: quarterly
      information_set_type: final_revised_data
    leaf_config:
      target: GDPC1
      horizons: [1, 2, 4]
      custom_source_path: /path/to/custom_quarterly_columns.parquet
```

## Takeaways

- Choose Data Source Mode first.
- Choose FRED Source Panel only when the source mode uses FRED data.
- Choose Analysis Frequency explicitly for standalone FRED-SD and custom-only
  data.
- Treat Data Revision / Vintage Regime and Publication Lag Rule as different
  decisions.
