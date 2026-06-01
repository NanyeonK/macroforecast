# FRED-QD + FRED-SD

[Back to FRED Datasets](index.md)

`load_fred_qd_sd()` combines the quarterly national FRED-QD panel with selected
state-level FRED-SD series. This is the recommended combined loader when the
target, evaluation date, or main research design is quarterly.

## What This Loader Is

| Item | Value |
| --- | --- |
| Loader | `macroforecast.data.load_fred_qd_sd()` |
| Metadata dataset | `"fred_qd+fred_sd"` |
| Default output frequency | `"quarterly"` |
| National source | FRED-QD |
| State source | FRED-SD |
| Recommended use | Quarterly state analysis with national macro controls. |
| Discouraged use | Monthly target analysis; use [FRED-MD + FRED-SD](fred_md_sd.md) instead. |

## Function

```python
macroforecast.data.load_fred_qd_sd(
    vintage: str | None = None,
    *,
    force: bool = False,
    cache_root: str | pathlib.Path | None = None,
    local_fred_qd_source: str | pathlib.Path | None = None,
    local_fred_sd_source: str | pathlib.Path | None = None,
    states: list[str] | None = None,
    variables: list[str] | None = None,
    frequency: str = "quarterly",
    quarterly_to_monthly: str = "repeat_within_quarter",
    monthly_to_quarterly: str = "quarterly_average",
) -> DataBundle
```

## Inputs

| Name | Default | Meaning |
| --- | --- | --- |
| `vintage` | `None` | Shared vintage label in `YYYY-MM` form. |
| `force` | `False` | Re-download or re-copy both raw sources. |
| `cache_root` | `None` | Shared raw-file cache root. |
| `local_fred_qd_source` | `None` | Local FRED-QD CSV instead of online source. |
| `local_fred_sd_source` | `None` | Local FRED-SD workbook or CSV instead of online source. |
| `states` | `None` | FRED-SD state subset, such as `["CA", "TX"]`. |
| `variables` | `None` | FRED-SD variable subset, such as `["UR", "NQGSP"]`. |
| `frequency` | `"quarterly"` | `"quarterly"`, `"monthly"`, or `"native"`. |
| `quarterly_to_monthly` | `"repeat_within_quarter"` | Rule used only when `frequency="monthly"`. |
| `monthly_to_quarterly` | `"quarterly_average"` | Rule used when a selected FRED-SD series is monthly but the output panel is quarterly. |

## Output

The function returns `DataBundle(panel, metadata)`.

| Field | Value |
| --- | --- |
| `bundle.panel` | Combined wide pandas panel. |
| `bundle.metadata["dataset"]` | `"fred_qd+fred_sd"` |
| `bundle.metadata["frequency"]` | Requested output frequency. |
| `bundle.metadata["combined_sources"]` | Metadata snapshots from FRED-QD and FRED-SD. |
| `bundle.metadata["source_by_column"]` | Source dataset for each column. |
| `bundle.metadata["native_frequency_by_column"]` | Inferred source frequency before alignment. |
| `bundle.metadata["native_frequency_counts"]` | Count of monthly, quarterly, weekly, and unknown native columns before alignment. |
| `bundle.metadata["date_anchor_by_column"]` | FRED-SD date-anchor pattern for state columns when available. |
| `bundle.metadata["date_anchor_counts"]` | Count of FRED-SD date-anchor patterns when available. |
| `bundle.metadata["output_frequency_by_column"]` | Frequency represented in the returned panel. |
| `bundle.metadata["frequency_conversion_warnings"]` | Conversion records for source columns that changed frequency. |
| `bundle.metadata["alignment"]` | Target frequency and alignment rules. |
| `bundle.metadata["transform_codes"]` | FRED-QD official t-codes for national columns. |

FRED-SD has no official t-code map. State-level transformation codes must be
supplied during preprocessing.

## Default Quarterly Use

```python
import macroforecast as mf

bundle = mf.data.load_fred_qd_sd(
    states=["CA", "TX"],
    variables=["UR", "NQGSP"],
    frequency="quarterly",
    monthly_to_quarterly="quarterly_average",
)
```

This is the intended path for quarterly state analysis. Quarterly FRED-SD
columns remain quarterly. Monthly FRED-SD columns are aligned to quarterly
output using `monthly_to_quarterly`.

## FRED-SD Metadata Preserved In The Combined Panel

FRED-SD contains state series with different native frequencies. The QD+SD
loader aligns the returned panel to quarterly output by default, but it keeps
the FRED-SD source-frequency and date-anchor metadata so the user can audit
what was changed.

| Metadata | Meaning |
| --- | --- |
| `native_frequency_by_column` | Source frequency before combination, such as `"monthly"` or `"quarterly"`. |
| `native_frequency_counts` | Count of native source frequencies in the combined panel. |
| `date_anchor_by_column` | FRED-SD date-anchor pattern, such as `"month_start"`, `"quarter_start"`, or `"monthly_weekday_anchor"`. |
| `date_anchor_counts` | Count of FRED-SD date-anchor patterns in selected state columns. |
| `panel.attrs["macrocast_reports"]["fred_sd_series_metadata"]` | Per-state/per-variable observed span, native frequency, and date-anchor report. |

This is especially important when monthly state indicators are used for a
quarterly target. The loader records the monthly-to-quarterly aggregation rule
in `metadata["frequency_conversion_warnings"]`. For FRED-SD variables that are
already quarterly, such as `OTOT` and `STHPI` according to the official
FRED-SD correction, no monthly-to-quarterly aggregation is needed.

## Monthly-To-Quarterly Rules

| Rule | Meaning |
| --- | --- |
| `"quarterly_average"` | Average monthly observations inside each quarter. This is the default. |
| `"quarterly_endpoint"` | Use the last available monthly observation inside each quarter. |
| `"quarterly_sum"` | Sum monthly observations inside each quarter. |

When a monthly FRED-SD series is converted to quarterly output, the loader emits
`UserWarning` and records the affected variables in
`metadata["frequency_conversion_warnings"]`.

## Discouraged Monthly Use

```python
bundle = mf.data.load_fred_qd_sd(
    states=["CA"],
    variables=["UR", "NQGSP"],
    frequency="monthly",
    quarterly_to_monthly="repeat_within_quarter",
)
```

This is allowed but not recommended. If the target is monthly, FRED-MD is the
natural national panel. The returned metadata includes a parse note explaining
that monthly alignment of FRED-QD is supported but discouraged.

## Weekly Or Other FRED-SD Frequencies

The default combined output is quarterly. The current official FRED-SD workbook
has monthly and quarterly state series, plus columns whose observed dates do
not support a reliable monthly or quarterly classification. The loader does
not silently aggregate weekly or unknown-frequency columns into a combined
quarterly panel; it raises `ValueError` instead. Use `frequency="native"` to
inspect the mixed native panel first, or call `mf.data.align_frequency()`
explicitly when you want to decide how a non-quarterly state series should
enter a quarterly design.

`ICLAIMS` is treated as `native_frequency="monthly"` with
`date_anchor="monthly_weekday_anchor"` because the workbook has one observation
per month on a weekday/Saturday-style anchor rather than first-of-month dates.
When it enters the default quarterly QD+SD panel, the selected
`monthly_to_quarterly` rule is recorded in metadata.

## Official URLs

| Source | URL |
| --- | --- |
| FRED-QD official page | <https://www.stlouisfed.org/research/economists/mccracken/fred-databases> |
| FRED-QD current CSV | <https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/quarterly/current.csv> |
| FRED-QD article | <https://www.stlouisfed.org/publications/review/2021/01/14/fred-qd-a-quarterly-database-for-macroeconomic-research> |
| FRED-SD official page | <https://www.stlouisfed.org/research/economists/owyang/fred-sd> |
