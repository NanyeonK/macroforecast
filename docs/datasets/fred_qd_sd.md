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

The default combined output is quarterly. If a selected FRED-SD column is
detected as weekly, the current combined loader does not silently aggregate it
to quarterly. Use `frequency="native"` to inspect the mixed native panel first,
or wait for the dedicated FRED-SD frequency policy audit before relying on
weekly state series in a quarterly combined panel.

## Official URLs

| Source | URL |
| --- | --- |
| FRED-QD official page | <https://www.stlouisfed.org/research/economists/mccracken/fred-databases> |
| FRED-QD current CSV | <https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/quarterly/current.csv> |
| FRED-QD article | <https://www.stlouisfed.org/publications/review/2021/01/14/fred-qd-a-quarterly-database-for-macroeconomic-research> |
| FRED-SD official page | <https://www.stlouisfed.org/research/economists/owyang/fred-sd> |
