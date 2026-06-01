# FRED-MD + FRED-SD

[Back to FRED Datasets](index.md)

`load_fred_md_sd()` combines the monthly national FRED-MD panel with selected
state-level FRED-SD series. This is the recommended combined loader when the
target, evaluation date, or main research design is monthly.

## What This Loader Is

| Item | Value |
| --- | --- |
| Loader | `macroforecast.data.load_fred_md_sd()` |
| Metadata dataset | `"fred_md+fred_sd"` |
| Default output frequency | `"monthly"` |
| National source | FRED-MD |
| State source | FRED-SD |
| Recommended use | Monthly state analysis with national macro controls. |
| Discouraged use | Quarterly target analysis; use [FRED-QD + FRED-SD](fred_qd_sd.md) instead. |

## Function

```python
macroforecast.data.load_fred_md_sd(
    vintage: str | None = None,
    *,
    force: bool = False,
    cache_root: str | pathlib.Path | None = None,
    local_fred_md_source: str | pathlib.Path | None = None,
    local_fred_sd_source: str | pathlib.Path | None = None,
    states: list[str] | None = None,
    variables: list[str] | None = None,
    frequency: str = "monthly",
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
| `local_fred_md_source` | `None` | Local FRED-MD CSV instead of online source. |
| `local_fred_sd_source` | `None` | Local FRED-SD workbook or CSV instead of online source. |
| `states` | `None` | FRED-SD state subset, such as `["CA", "TX"]`. |
| `variables` | `None` | FRED-SD variable subset, such as `["UR", "ICLAIMS"]`. |
| `frequency` | `"monthly"` | `"monthly"`, `"quarterly"`, or `"native"`. |
| `quarterly_to_monthly` | `"repeat_within_quarter"` | Rule used when a selected FRED-SD series is quarterly but the output panel is monthly. |
| `monthly_to_quarterly` | `"quarterly_average"` | Rule used only when `frequency="quarterly"`. |

## Output

The function returns `DataBundle(panel, metadata)`.

| Field | Value |
| --- | --- |
| `bundle.panel` | Combined wide pandas panel. |
| `bundle.metadata["dataset"]` | `"fred_md+fred_sd"` |
| `bundle.metadata["frequency"]` | Requested output frequency. |
| `bundle.metadata["combined_sources"]` | Metadata snapshots from FRED-MD and FRED-SD. |
| `bundle.metadata["source_by_column"]` | Source dataset for each column. |
| `bundle.metadata["native_frequency_by_column"]` | Inferred source frequency before alignment. |
| `bundle.metadata["output_frequency_by_column"]` | Frequency represented in the returned panel. |
| `bundle.metadata["frequency_conversion_warnings"]` | Conversion records for source columns that changed frequency. |
| `bundle.metadata["alignment"]` | Target frequency and alignment rules. |
| `bundle.metadata["transform_codes"]` | FRED-MD official t-codes for national columns. |

FRED-SD has no official t-code map. State-level transformation codes must be
supplied during preprocessing.

## Default Monthly Use

```python
import macroforecast as mf

bundle = mf.data.load_fred_md_sd(
    states=["CA", "TX"],
    variables=["UR", "ICLAIMS", "NQGSP"],
    frequency="monthly",
)
```

This is the intended path for monthly state analysis. Monthly FRED-SD columns
remain monthly. Quarterly FRED-SD columns are aligned to monthly output using
`quarterly_to_monthly`.

## Quarterly-To-Monthly Rules

| Rule | Meaning |
| --- | --- |
| `"repeat_within_quarter"` | Assign the quarterly value to each month in the quarter. This is the default. |
| `"quarter_end_ffill"` | Place the value at quarter end and forward-fill monthly dates. |
| `"linear_interpolation"` | Interpolate monthly values between quarterly observations. |

When a quarterly FRED-SD series is converted to monthly output, the loader emits
`UserWarning` and records the affected variables in
`metadata["frequency_conversion_warnings"]`.

## Discouraged Quarterly Use

```python
bundle = mf.data.load_fred_md_sd(
    states=["CA"],
    variables=["UR", "NQGSP"],
    frequency="quarterly",
    monthly_to_quarterly="quarterly_average",
)
```

This is allowed but not recommended. If the target is quarterly, FRED-QD is the
natural national panel. The returned metadata includes a parse note explaining
that quarterly alignment of FRED-MD is supported but discouraged.

## Weekly Or Other FRED-SD Frequencies

The default combined output is monthly. If a selected FRED-SD column is detected
as weekly, the current combined loader does not silently aggregate it to
monthly. Use `frequency="native"` to inspect the mixed native panel first, or
wait for the dedicated FRED-SD frequency policy audit before relying on weekly
state series in a monthly combined panel.

## Official URLs

| Source | URL |
| --- | --- |
| FRED-MD official page | <https://www.stlouisfed.org/research/economists/mccracken/fred-databases> |
| FRED-MD current CSV | <https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/current.csv> |
| FRED-SD official page | <https://www.stlouisfed.org/research/economists/owyang/fred-sd> |
