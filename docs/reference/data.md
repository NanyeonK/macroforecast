# macroforecast.data

[Back to reference](index.md)

`macroforecast.data` is the data entry point for the package. It loads official
or user-supplied data, normalizes it to one pandas panel contract, and attaches
source metadata. It also creates run-level data specifications and combines
national FRED-MD/FRED-QD data with state-level FRED-SD panels.

This module does not apply stationarity transforms, outlier rules, imputation,
feature engineering, model fitting, or evaluation. Those steps happen later.
The main output is always a `DataBundle` or `DataSpec`.

The usual flow is:

```python
import macroforecast as mf

bundle = mf.data.load_fred_md()

data_spec = mf.data.spec(
    bundle,
    target="INDPRO",
    horizons=[1, 3, 6, 12],
    start="1960-01",
    end="2024-12",
    predictors="all",
)
```

## Canonical Panel

Every public loader returns a `DataBundle`.

```python
panel = bundle.panel
metadata = bundle.metadata
```

`DataBundle` also supports tuple unpacking:

```python
panel, metadata = mf.data.load_fred_md()
```

### Panel Contract

| Property | Required Value |
| --- | --- |
| Type | `pandas.DataFrame` |
| Index | `pandas.DatetimeIndex` |
| Index name | `"date"` |
| Sort order | ascending date order |
| Duplicate dates | not allowed |
| Columns | variable IDs |
| Values | numeric values or `NaN` |

Metadata is explicit on `DataBundle.metadata`. The panel also carries
`panel.attrs["macroforecast_metadata"]` for pandas-native handoff. FRED-MD and
FRED-QD transform codes are attached to
`panel.attrs["macroforecast_transform_codes"]`; preprocessing is responsible
for using them.

### Metadata Contract

Every loader writes a metadata dictionary with these common keys.

| Key | Type | Meaning |
| --- | --- | --- |
| `dataset` | `str` | Dataset identifier such as `fred_md`, `fred_qd`, `fred_sd`, `fred_md+fred_sd`, or `fred_qd+fred_sd`. |
| `source_family` | `str` | Source family. Official loaders use `fred-md`, `fred-qd`, `fred-sd`, or `combined`. |
| `frequency` | `str` | Loader-level frequency label: `monthly`, `quarterly`, `state_monthly`, `mixed`, or the chosen combined frequency. |
| `version_mode` | `str` | `current`, `vintage`, or `mixed` for combined inputs with different modes. |
| `vintage` | `str` or `None` | Requested vintage label in `YYYY-MM` form, or `None` for current data. |
| `data_through` | `str` or `None` | Last date present in the loaded panel, formatted as `YYYY-MM`. |
| `support_tier` | `str` | `stable` for official loaders, `provisional` for user-supplied files. |
| `parse_notes` | `tuple[str, ...]` | Loader notes, including discouraged frequency alignments for combined datasets. |
| `artifact` | `dict` or `None` | Raw-file provenance for single-source loads; combined bundles use `None`. |
| `transform_codes` | `dict[str, int]` | Official FRED-MD/FRED-QD t-codes when available. FRED-SD has no official t-code map. |

Combined bundles add:

| Key | Type | Meaning |
| --- | --- | --- |
| `combined_sources` | `list[dict]` | Full metadata dictionaries from the source bundles. |
| `source_by_column` | `dict[str, str]` | Source dataset for each output column. |
| `frequency_conversion_warnings` | `list[dict]` | Records of monthly-to-quarterly or quarterly-to-monthly conversions. |
| `alignment` | `dict` | Chosen target frequency, alignment rules, and source-level alignment summaries. |

## DataBundle

```python
macroforecast.data.DataBundle(
    panel: pandas.DataFrame,
    metadata: dict,
)
```

### Output

| Field | Type | Meaning |
| --- | --- | --- |
| `panel` | `pandas.DataFrame` | Canonical date-indexed data panel. |
| `metadata` | `dict` | Source, vintage, artifact, frequency, and transform-code metadata. |

### Methods

| Method | Input | Output | Meaning |
| --- | --- | --- | --- |
| `attach(stage, values)` | `stage: str`, `values: Mapping` | `DataBundle` | Return a new bundle with one metadata stage added. |

Preprocessing outputs can use the same metadata-attachment pattern.

## DataSpec

```python
macroforecast.data.DataSpec(
    panel: pandas.DataFrame,
    metadata: dict,
    target: str | None,
    targets: tuple[str, ...],
    horizons: tuple[int, ...],
    start: str | None = None,
    end: str | None = None,
    predictors: "all" | tuple[str, ...] = "all",
)
```

`DataSpec` is the output of `spec(...)`. It keeps the canonical panel and
metadata together with the target, horizons, sample window, and predictor
selection for a run.

## load_fred_md

Load FRED-MD and return `DataBundle`.

```python
macroforecast.data.load_fred_md(
    vintage: str | None = None,
    *,
    force: bool = False,
    cache_root: str | pathlib.Path | None = None,
    local_source: str | pathlib.Path | None = None,
    local_zip_source: str | pathlib.Path | None = None,
) -> DataBundle
```

### Input

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `vintage` | <code>str &#124; None</code> | `None` | Vintage in `YYYY-MM` form. `None` loads current. |
| `force` | `bool` | `False` | Re-download or re-copy even if cache exists. |
| `cache_root` | path-like or `None` | `None` | Raw cache root. |
| `local_source` | path-like or `None` | `None` | Local CSV source instead of download. |
| `local_zip_source` | path-like or `None` | `None` | Local historical zip source for vintage files. |

### Output

Returns `DataBundle` with a monthly FRED-MD panel and metadata. The official
CSV transform row is parsed into `metadata["transform_codes"]` and
`panel.attrs["macroforecast_transform_codes"]`.

See [FRED-MD](fred_md.md) for dataset-specific details.

## load_fred_qd

Load FRED-QD and return `DataBundle`.

```python
macroforecast.data.load_fred_qd(
    vintage: str | None = None,
    *,
    force: bool = False,
    cache_root: str | pathlib.Path | None = None,
    local_source: str | pathlib.Path | None = None,
) -> DataBundle
```

Returns a quarterly canonical panel. The official CSV transform row is parsed
into `metadata["transform_codes"]` and
`panel.attrs["macroforecast_transform_codes"]`.

See [FRED-QD](fred_qd.md) for dataset-specific details.

## load_fred_sd

Load FRED-SD and return `DataBundle`.

```python
macroforecast.data.load_fred_sd(
    vintage: str | None = None,
    *,
    force: bool = False,
    cache_root: str | pathlib.Path | None = None,
    local_source: str | pathlib.Path | None = None,
    states: list[str] | None = None,
    variables: list[str] | None = None,
) -> DataBundle
```

### Input

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `states` | <code>list[str] &#124; None</code> | `None` | Optional state subset. |
| `variables` | <code>list[str] &#124; None</code> | `None` | Optional FRED-SD variable subset. |

FRED-SD columns are wide variable-state IDs such as `UR_CA`. The loader also
adds `panel.attrs["macrocast_reports"]["fred_sd_series_metadata"]`, which
records each column's state, FRED-SD variable, observed date range, non-missing
count, and native frequency inferred from the official series workbook.

See [FRED-SD](fred_sd.md) for monthly/quarterly series details and t-code
limitations.

## load_fred_md_sd

Load FRED-MD and FRED-SD, align them to one panel, and return `DataBundle`.

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

### Purpose

Use this when the outcome or main state panel is monthly and national
macroeconomic controls should come from FRED-MD. This is the recommended
combined dataset for monthly state analysis.

### Input

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `vintage` | <code>str &#124; None</code> | `None` | Vintage label shared across FRED-MD and FRED-SD. |
| `force` | `bool` | `False` | Re-download or re-copy raw sources. |
| `cache_root` | path-like or `None` | `None` | Raw cache root used by both loaders. |
| `local_fred_md_source` | path-like or `None` | `None` | Local FRED-MD CSV source. |
| `local_fred_sd_source` | path-like or `None` | `None` | Local FRED-SD workbook or CSV source. |
| `states` | <code>list[str] &#124; None</code> | `None` | FRED-SD state subset. |
| `variables` | <code>list[str] &#124; None</code> | `None` | FRED-SD variable subset. |
| `frequency` | `str` | `"monthly"` | `"monthly"`, `"quarterly"`, or `"native"`. Quarterly is supported but not recommended for this loader. |
| `quarterly_to_monthly` | `str` | `"repeat_within_quarter"` | Rule used if an included FRED-SD series is quarterly and the target panel is monthly. |
| `monthly_to_quarterly` | `str` | `"quarterly_average"` | Rule used only when `frequency="quarterly"`. |

### Output

Returns a combined `DataBundle` with:

- `metadata["dataset"] == "fred_md+fred_sd"`
- `metadata["source_family"] == "combined"`
- `metadata["frequency"] == frequency`
- FRED-MD official t-codes in `metadata["transform_codes"]`
- FRED-SD series metadata preserved in `panel.attrs["macrocast_reports"]`
- any frequency conversions recorded in `metadata["frequency_conversion_warnings"]`

If a quarterly FRED-SD series is included in a monthly panel, the function
emits a `UserWarning` and records the conversion. The default
`quarterly_to_monthly="repeat_within_quarter"` assigns the quarterly value to
each month inside the quarter.

## load_fred_qd_sd

Load FRED-QD and FRED-SD, align them to one panel, and return `DataBundle`.

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

### Purpose

Use this when the target or outcome is quarterly and national controls should
come from FRED-QD. This is the recommended combined dataset for quarterly
state-level analysis.

### Input

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `vintage` | <code>str &#124; None</code> | `None` | Vintage label shared across FRED-QD and FRED-SD. |
| `force` | `bool` | `False` | Re-download or re-copy raw sources. |
| `cache_root` | path-like or `None` | `None` | Raw cache root used by both loaders. |
| `local_fred_qd_source` | path-like or `None` | `None` | Local FRED-QD CSV source. |
| `local_fred_sd_source` | path-like or `None` | `None` | Local FRED-SD workbook or CSV source. |
| `states` | <code>list[str] &#124; None</code> | `None` | FRED-SD state subset. |
| `variables` | <code>list[str] &#124; None</code> | `None` | FRED-SD variable subset. |
| `frequency` | `str` | `"quarterly"` | `"quarterly"`, `"monthly"`, or `"native"`. Monthly is supported but not recommended for this loader. |
| `quarterly_to_monthly` | `str` | `"repeat_within_quarter"` | Rule used only when `frequency="monthly"`. |
| `monthly_to_quarterly` | `str` | `"quarterly_average"` | Rule used if an included FRED-SD series is monthly and the target panel is quarterly. |

### Output

Returns a combined `DataBundle` with:

- `metadata["dataset"] == "fred_qd+fred_sd"`
- `metadata["source_family"] == "combined"`
- `metadata["frequency"] == frequency`
- FRED-QD official t-codes in `metadata["transform_codes"]`
- FRED-SD series metadata preserved in `panel.attrs["macrocast_reports"]`
- any frequency conversions recorded in `metadata["frequency_conversion_warnings"]`

If a monthly FRED-SD series is included in a quarterly panel, the function
emits a `UserWarning` and records the conversion. The default
`monthly_to_quarterly="quarterly_average"` averages monthly observations inside
each quarter.

## combine

Combine already-loaded `DataBundle` objects into one canonical panel.

```python
macroforecast.data.combine(
    *bundles,
    dataset: str | None = None,
    frequency: str = "native",
    quarterly_to_monthly: str = "repeat_within_quarter",
    monthly_to_quarterly: str = "quarterly_average",
) -> DataBundle
```

### Input

| Name | Type | Default | Choices |
| --- | --- | --- | --- |
| `*bundles` | `DataBundle` | required | Two or more bundles to concatenate by date index. |
| `dataset` | `str` or `None` | joined source names | Output dataset label. |
| `frequency` | `str` | `"native"` | `"native"`, `"monthly"`, or `"quarterly"`. |
| `quarterly_to_monthly` | `str` | `"repeat_within_quarter"` | `"repeat_within_quarter"`, `"quarter_end_ffill"`, `"linear_interpolation"`. |
| `monthly_to_quarterly` | `str` | `"quarterly_average"` | `"quarterly_average"`, `"quarterly_endpoint"`, `"quarterly_sum"`. |

### Frequency Conversion Rules

| Direction | Rule | Meaning |
| --- | --- | --- |
| quarterly to monthly | `repeat_within_quarter` | Assign the quarterly value to each month in that quarter. |
| quarterly to monthly | `quarter_end_ffill` | Place the quarterly value at quarter end and forward-fill after it is observed. |
| quarterly to monthly | `linear_interpolation` | Interpolate between observed quarter-end values on the monthly grid. |
| monthly to quarterly | `quarterly_average` | Average monthly observations in the quarter. |
| monthly to quarterly | `quarterly_endpoint` | Use the last monthly observation in the quarter. |
| monthly to quarterly | `quarterly_sum` | Sum monthly observations in the quarter. |

Weekly data are not supported in combined official-data bundles. If a source
contains a native weekly column, `combine()` raises `ValueError`.

### Output

Returns `DataBundle`. The panel is a column-wise concatenation after frequency
alignment. Duplicate output column names raise `ValueError`.

### Frequency Conversion Warnings

When `combine()` changes a source column's native frequency, it emits
`UserWarning` and records the same information in
`metadata["frequency_conversion_warnings"]`.

Each record has:

| Key | Type | Meaning |
| --- | --- | --- |
| `dataset` | `str` | Source dataset whose columns were converted. |
| `from_frequency` | `str` | Native frequency before alignment. |
| `to_frequency` | `str` | Combined panel frequency. |
| `rule` | `str` | Alignment rule used. |
| `variables` | `list[str]` | Variable-level names, e.g. `["NQGSP"]` for `NQGSP_CA`. |
| `columns` | `list[str]` | Exact converted panel columns. |
| `n_columns` | `int` | Number of converted columns. |

Example warning:

```text
fred_sd monthly variables were aligned to quarterly using quarterly_average:
UR, ICLAIMS (102 columns).
```

## load_custom_csv

Load a user CSV and normalize it to the canonical panel contract.

```python
macroforecast.data.load_custom_csv(
    path,
    *,
    date: str | None = None,
    date_col: str | int | None = None,
    columns: Iterable[str] | None = None,
    series_columns: Iterable[str] | None = None,
    rename: Mapping[str, str] | None = None,
    dataset: str = "custom",
    frequency: str = "unknown",
    metadata: Mapping[str, object] | None = None,
    transform_codes: Mapping[str, int] | None = None,
) -> DataBundle
```

### Input

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `path` | path-like | required | CSV file path. |
| `date` | <code>str &#124; None</code> | `None` | Date column. If omitted, uses a DatetimeIndex or parses the first column. |
| `date_col` | <code>str &#124; int &#124; None</code> | `None` | Alias for `date`; integer values select the date column by zero-based position. |
| `columns` | iterable or `None` | `None` | Columns to keep before renaming. |
| `series_columns` | iterable or `None` | `None` | Alias for `columns`; use this name when thinking in panel series IDs. |
| `rename` | mapping or `None` | `None` | Column rename map. |
| `dataset` | `str` | `"custom"` | Metadata dataset label. |
| `frequency` | `str` | `"unknown"` | Metadata frequency label. |
| `metadata` | mapping or `None` | `None` | User metadata to attach. |
| `transform_codes` | mapping or `None` | `None` | Optional McCracken-Ng t-code map. Keys must match final loaded series columns after selection and renaming. |

### Output

Returns a `DataBundle`. The normalized panel is available as `bundle.panel` and
metadata as `bundle.metadata`. If `transform_codes` is provided, it is stored in
both `bundle.metadata["transform_codes"]` and
`bundle.panel.attrs["macroforecast_transform_codes"]`, so
`mf.preprocessing.reprocess(bundle)` can use the codes automatically.

Example:

```python
bundle = mf.data.load_custom_csv(
    "panel.csv",
    date_col="DATE",
    series_columns=["INDPRO", "spread"],
    frequency="monthly",
    transform_codes={"INDPRO": 5, "spread": 2},
)

processed = mf.preprocessing.reprocess(bundle)
```

## load_custom_parquet

Load a user Parquet file with the same normalization contract as
`load_custom_csv`.

```python
macroforecast.data.load_custom_parquet(
    path,
    *,
    date: str | None = None,
    date_col: str | int | None = None,
    columns: Iterable[str] | None = None,
    series_columns: Iterable[str] | None = None,
    rename: Mapping[str, str] | None = None,
    dataset: str = "custom",
    frequency: str = "unknown",
    metadata: Mapping[str, object] | None = None,
    transform_codes: Mapping[str, int] | None = None,
) -> DataBundle
```

`date_col`, `series_columns`, and `transform_codes` have the same meaning as in
`load_custom_csv`.

## as_panel

Normalize an existing pandas `DataFrame`.

```python
macroforecast.data.as_panel(
    frame,
    *,
    date: str | None = None,
    columns: Iterable[str] | None = None,
    rename: Mapping[str, str] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> pandas.DataFrame
```

`as_panel` returns a canonical panel. It raises if the date column is missing,
dates are duplicated, or any retained column cannot be represented as numeric
values or `NaN`.

## validate_panel

Validate the canonical panel contract.

```python
macroforecast.data.validate_panel(panel) -> None
```

Raises `TypeError` or `ValueError` when the panel is not canonical.

## panel_info

Return a compact panel summary.

```python
macroforecast.data.panel_info(bundle_or_panel) -> dict
```

Output keys include `n_rows`, `n_columns`, `start`, `end`, `columns`,
`missing_values`, and inferred `frequency`.

## metadata

Return explicit metadata from a `DataBundle`, `DataSpec`, `(panel, metadata)`
tuple, or `DataFrame`.

```python
macroforecast.data.metadata(obj) -> dict
```

## spec

Attach run-level data choices to a bundle or panel.

```python
macroforecast.data.spec(
    data,
    *,
    metadata: Mapping[str, object] | None = None,
    target: str | None = None,
    targets: Iterable[str] | None = None,
    horizons: Iterable[int] | int | None = None,
    start: str | None = None,
    end: str | None = None,
    predictors: "all" | Iterable[str] = "all",
) -> DataSpec
```

### Input

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `data` | `DataBundle`, `DataSpec`, `(panel, metadata)`, or `DataFrame` | required | Canonical data input. |
| `metadata` | mapping or `None` | `None` | Extra metadata to merge. |
| `target` | <code>str &#124; None</code> | `None` | Single target column. |
| `targets` | iterable or `None` | `None` | Multiple target columns. |
| `horizons` | iterable, int, or `None` | derived | Forecast horizons. |
| `start` | <code>str &#124; None</code> | `None` | Start date. Accepts `YYYY`, `YYYY-MM`, or `YYYY-MM-DD`. |
| `end` | <code>str &#124; None</code> | `None` | End date. Accepts `YYYY`, `YYYY-MM`, or `YYYY-MM-DD`. |
| `predictors` | <code>"all" &#124; iterable</code> | `"all"` | Predictor columns to keep. |

### Default Horizons

| Metadata frequency | Default horizons |
| --- | --- |
| `monthly` | `(1, 3, 6, 12)` |
| `quarterly` | `(1, 2, 4, 8)` |
| other or unknown | `(1,)` |

### Output

Returns `DataSpec`. Its metadata contains a `data_spec` entry with the chosen
target, targets, horizons, sample dates, predictors, and panel summary.

## Vintage Helpers

`list_vintages(dataset, start, end)` returns candidate vintage labels. The
selected vintage is passed to `load_fred_md`, `load_fred_qd`, or `load_fred_sd`
through `vintage=`.

## Official Source Pages

- FRED-MD and FRED-QD source page: <https://www.stlouisfed.org/research/economists/mccracken/fred-databases>
- FRED-SD source page: <https://www.stlouisfed.org/research/economists/owyang/fred-sd>
