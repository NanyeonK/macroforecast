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
| Empty panel | not allowed |
| Infinite values | not allowed |

Metadata is explicit on `DataBundle.metadata`. The panel also carries
`panel.attrs["macroforecast_metadata"]` for pandas-native handoff. FRED-MD and
FRED-QD transform codes are attached to
`panel.attrs["macroforecast_transform_codes"]`; preprocessing is responsible
for using them.

Panel normalization is strict by default. Invalid date values, non-numeric
cells that would be coerced to `NaN`, duplicate dates, empty panels, and
infinite values raise errors. When a caller deliberately sets `strict=False`,
lossy normalization is allowed but recorded in
`panel.attrs["macroforecast_panel_report"]` and
`metadata["panel"]` when the panel is returned inside a `DataBundle`.

`macroforecast_panel_report` contains:

| Key | Meaning |
| --- | --- |
| `contract` | Panel contract version, currently `macroforecast_panel_v1`. |
| `strict` | Whether lossy date/numeric coercion was rejected. |
| `input_rows`, `output_rows` | Row count before and after panel normalization. |
| `input_columns`, `output_columns` | Column names before and after selection/renaming. |
| `date_source` | Date source used: a column name or `"index"`. |
| `invalid_date_rows_dropped` | Number of invalid date rows dropped when `strict=False`. |
| `numeric_coercion` | Count and examples of non-numeric cells coerced to `NaN` when `strict=False`. |

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
| `native_frequency_by_column` | `dict[str, str]` | Original frequency for each output column before alignment. |
| `native_frequency_counts` | `dict[str, int]` | Count of columns by original frequency. |
| `output_frequency_by_column` | `dict[str, str]` | Frequency represented in the returned panel for each output column. |
| `output_frequency_counts` | `dict[str, int]` | Count of columns by returned-panel frequency. |
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

With `frequency="native"` or `frequency="mixed"`, no monthly/quarterly
conversion is applied. The returned panel keeps each source column on its
native observation dates and records `metadata["frequency"] == "mixed"`.
Quarterly columns therefore appear as sparse columns on the union date index
when they are combined with monthly columns. Downstream mixed-frequency models
should read `metadata["native_frequency_by_column"]` rather than infer
frequency from the overall index.

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

For mixed outputs, the key metadata fields are:

| Key | Meaning |
| --- | --- |
| `metadata["frequency"]` | `"mixed"`. |
| `metadata["native_frequency_by_column"]` | Native source frequency for each column. |
| `metadata["output_frequency_by_column"]` | Returned-panel frequency for each column; equal to native frequency in native mode. |
| `metadata["alignment"]["frequency"]` | `"native"` when no conversion was applied. |

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
    frequency_by_column: Mapping[str, str] | None = None,
    default_frequency: str | None = None,
    metadata: Mapping[str, object] | None = None,
    transform_codes: Mapping[str, int] | None = None,
    strict: bool = True,
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
| `frequency_by_column` | mapping or `None` | `None` | Optional final-column frequency map, e.g. `{"PAYEMS": "monthly", "GDPC1": "quarterly"}`. |
| `default_frequency` | `str` or `None` | `None` | Fill frequency for columns omitted from `frequency_by_column`. |
| `metadata` | mapping or `None` | `None` | User metadata to attach. |
| `transform_codes` | mapping or `None` | `None` | Optional McCracken-Ng t-code map. Keys must match final loaded series columns after selection and renaming. |
| `strict` | `bool` | `True` | Reject invalid date rows and non-numeric cells instead of silently coercing them. Set `False` only when you want a permissive load with a panel report. |

### Output

Returns a `DataBundle`. The normalized panel is available as `bundle.panel` and
metadata as `bundle.metadata`. If `transform_codes` is provided, it is stored in
both `bundle.metadata["transform_codes"]` and
`bundle.panel.attrs["macroforecast_transform_codes"]`, so
`mf.preprocessing.reprocess(bundle)` can use the codes automatically.

Custom loaders also store the strict-normalization report at
`bundle.metadata["panel"]`. With `strict=True`, malformed dates or non-numeric
cells raise `RawParseError` wrapping the underlying validation error. With
`strict=False`, those lossy operations are allowed and counted.

If `frequency_by_column` is provided, custom loaders call
`set_frequencies(...)` internally and write the same mixed-frequency metadata
contract used by official combined bundles. The keys must match final loaded
column names after selection and renaming.

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
    frequency_by_column: Mapping[str, str] | None = None,
    default_frequency: str | None = None,
    metadata: Mapping[str, object] | None = None,
    transform_codes: Mapping[str, int] | None = None,
    strict: bool = True,
) -> DataBundle
```

`date_col`, `series_columns`, `transform_codes`, and `strict` have the same
meaning as in `load_custom_csv`.

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
    strict: bool = True,
) -> pandas.DataFrame
```

`as_panel` returns a canonical panel. It raises if the date column is missing,
dates are duplicated, the output is empty, infinite values are present, or any
retained column cannot be represented as numeric values or `NaN`.

### Input

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `frame` | `pandas.DataFrame` | required | Raw or already canonical panel. |
| `date` | <code>str &#124; None</code> | `None` | Date column. If omitted and the index is not a `DatetimeIndex`, the first column is parsed as dates. |
| `columns` | iterable or `None` | `None` | Columns to keep before renaming. |
| `rename` | mapping or `None` | `None` | Rename retained columns after selection. |
| `metadata` | mapping or `None` | `None` | Metadata attached under `panel.attrs["macroforecast_metadata"]`. |
| `strict` | `bool` | `True` | Reject lossy date/numeric coercion. `False` permits it and records a panel report. |

### Output

Returns a `pandas.DataFrame` with `DatetimeIndex` named `"date"`, ascending
dates, numeric columns, and attrs containing `macroforecast_panel_report`.

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
`missing_values`, `frequency`, and `index_frequency`. If the input carries
metadata, `frequency` uses the metadata label such as `"mixed"` while
`index_frequency` reports the pandas-inferred date-index frequency. Combined
data also include compact native/output frequency counts.

## set_frequencies

Attach a column-level frequency contract to an existing panel or bundle.

```python
macroforecast.data.set_frequencies(
    data,
    frequency_by_column,
    *,
    default_frequency: str | None = None,
    output_frequency_by_column: Mapping[str, str] | None = None,
    frequency: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> DataBundle
```

### Input

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `data` | `DataBundle`, `DataSpec`, `(panel, metadata)`, or `DataFrame` | required | Canonical panel input. |
| `frequency_by_column` | mapping | required | Native frequency for each final panel column. |
| `default_frequency` | `str` or `None` | `None` | Fill omitted columns with one frequency. |
| `output_frequency_by_column` | mapping or `None` | `None` | Returned-panel frequency for each column; defaults to native frequency. |
| `frequency` | `str` or `None` | `None` | Overall metadata label. Defaults to the unique native frequency or `"mixed"`. |
| `metadata` | mapping or `None` | `None` | Extra metadata to merge before writing frequency fields. |

Allowed column frequencies are `monthly`, `quarterly`, `weekly`, `annual`,
`irregular`, and `unknown`, with short aliases such as `m`, `q`, and `w`.
For mixed-frequency DFM models, monthly and quarterly columns are the relevant
contract.

### Output

Returns a `DataBundle` with:

| Metadata key | Meaning |
| --- | --- |
| `frequency` | Overall label, usually `"mixed"` when multiple native frequencies are present. |
| `native_frequency_by_column` | Native frequency for each column. |
| `native_frequency_counts` | Counts by native frequency. |
| `output_frequency_by_column` | Frequency represented in the returned panel for each column. |
| `output_frequency_counts` | Counts by output frequency. |

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
| `predictors` | <code>"all" &#124; iterable</code> | `"all"` | Predictor columns to keep. `"all"` expands to all non-target columns. Explicit predictor lists may be empty for target-only or autoregressive designs, and may not include target columns. |

### Default Horizons

| Metadata frequency | Default horizons |
| --- | --- |
| `monthly` | `(1, 3, 6, 12)` |
| `quarterly` | `(1, 2, 4, 8)` |
| other or unknown | `(1,)` |

### Output

Returns `DataSpec`. Its metadata contains a `data_spec` entry with the chosen
target, targets, horizons, sample dates, expanded predictor list, and panel
summary. This expansion is deliberate: downstream model stages should consume a
concrete non-target predictor list, not infer from the full panel and risk
target leakage.

## Data Policy Helpers

These functions are direct Python replacements for the old data-policy axes.
They do not parse YAML and do not fit models.

### align_frequency

```python
macroforecast.data.align_frequency(
    data,
    *,
    method: str = "keep",
    quarterly_to_monthly: str = "repeat_within_quarter",
    weekly_to_monthly: str = "mean",
    monthly_to_quarterly: str = "quarterly_average",
    weekly_to_quarterly: str = "mean",
    chow_lin_indicator: str | Mapping[str, str] | None = None,
    chow_lin_aggregation: str = "mean",
    chow_lin_rho: float | None = None,
    chow_lin_rho_method: str = "fixed",
) -> DataBundle
```

Keeps, filters, or aligns a panel to a common data frequency. This belongs in
`macroforecast.data` because it changes the calendar and column-level frequency
contract before preprocessing or feature engineering.

| Input | Default | Choices |
| --- | --- | --- |
| `method` | `"keep"` | `"keep"`, `"monthly"`, `"quarterly"`, `"drop_non_monthly"`, `"drop_non_quarterly"` |
| `quarterly_to_monthly` | `"repeat_within_quarter"` | `"repeat_within_quarter"`, `"step_backward"`, `"step_forward"`, `"quarter_end_ffill"`, `"linear_interpolation"`, `"chow_lin"` |
| `weekly_to_monthly` | `"mean"` | `"mean"`, `"last"`, `"sum"` |
| `monthly_to_quarterly` | `"quarterly_average"` | `"quarterly_average"`, `"quarterly_endpoint"`, `"quarterly_sum"` |
| `weekly_to_quarterly` | `"mean"` | `"mean"`, `"last"`, `"sum"` |
| `chow_lin_indicator` | `None` | Indicator column name, or mapping from quarterly column to indicator column, used only when `quarterly_to_monthly="chow_lin"`. |
| `chow_lin_aggregation` | `"mean"` | `"mean"` or `"sum"`; the low-frequency aggregation to conserve. |
| `chow_lin_rho` | `None` | Fixed AR(1) residual correlation. If supplied, must be inside `(-1, 1)`. |
| `chow_lin_rho_method` | `"fixed"` | `"fixed"`, `"min_chi_squared"`, or `"max_likelihood"`. |

Output is a `DataBundle`. Metadata records `data_frequency_alignment`,
`native_frequency_by_column`, `output_frequency_by_column`, and frequency
counts. Frequency detection uses `native_frequency_by_column` first, then
FRED-SD series reports, then observed-date spacing.

```python
monthly = mf.data.align_frequency(
    mixed_bundle,
    method="monthly",
    quarterly_to_monthly="repeat_within_quarter",
)
```

For quarterly-to-monthly alignment, `step_backward` is accepted as an alias for
`repeat_within_quarter`; the latter is the clearer spelling. Use
`quarter_end_ffill` when values should only become available from the
quarter-end month forward.

Use `quarterly_to_monthly="chow_lin"` when a quarterly series should be
regression-disaggregated with a monthly indicator:

```python
monthly = mf.data.align_frequency(
    mixed_bundle,
    method="monthly",
    quarterly_to_monthly="chow_lin",
    chow_lin_indicator={"GDPC1": "INDPRO"},
    chow_lin_aggregation="mean",
)
```

This preserves the supplied quarterly observations when the output is
re-aggregated by the declared `chow_lin_aggregation`. The function records the
indicator and rho choices in `metadata["data_frequency_alignment"]`.

### chow_lin_disaggregate

```python
macroforecast.data.chow_lin_disaggregate(
    low_frequency,
    indicator,
    *,
    aggregation: str = "mean",
    rho: float | None = None,
    rho_method: str = "fixed",
) -> pandas.Series
```

Direct Chow-Lin quarterly-to-monthly style disaggregation. `low_frequency` is a
low-frequency `Series`, and `indicator` is a higher-frequency `Series` or a
single/first-column `DataFrame`. The returned series is indexed like the
indicator and conserves `low_frequency` under `aggregation="mean"` or
`aggregation="sum"`.

`rho_method="fixed"` uses `rho` when supplied and `0.0` otherwise.
`"min_chi_squared"` and `"max_likelihood"` estimate `rho` over a bounded grid.

### infer_frequencies

```python
macroforecast.data.infer_frequencies(data) -> tuple[dict[str, str], str]
macroforecast.data.frequency_hardening_issues(frequencies) -> list[dict]
```

`infer_frequencies()` returns `(frequency_by_column, source)`. The source is
`"native_frequency_by_column"`, `"fred_sd_series_metadata"`, or
`"observed_dates"`. `frequency_hardening_issues()` reports columns classified
as `unknown`, `irregular`, or `annual` before a caller aligns frequencies.

### availability_lag

```python
macroforecast.data.availability_lag(
    data,
    *,
    lags: int | Mapping[str, int] = 1,
    columns: Iterable[str] | None = None,
    drop_missing: bool = False,
) -> DataBundle
```

Positive lags delay predictor availability. `lags=1` means the value dated
`t-1` is the latest available value on row `t`. Pass a mapping for
column-specific release lags.

### same_period_predictors

```python
macroforecast.data.same_period_predictors(
    data_spec,
    *,
    policy: "allow" | "lag" | "drop" | "forbid" = "allow",
    lag: int = 1,
    columns: Iterable[str] | None = None,
    drop_missing: bool = False,
) -> DataSpec
```

`allow` records the choice, `lag` shifts selected predictors, `drop` removes
them from the active predictor set, and `forbid` raises if such predictors are
present. Targets are never shifted by this helper.

### define_regime

```python
macroforecast.data.define_regime(
    data,
    *,
    name: str = "regime",
    column: str | None = None,
    threshold: float | None = None,
    direction: "above" | "below" | "equal" | "not_equal" = "above",
    dates: Iterable[str | pandas.Timestamp] | None = None,
    values: Sequence[bool | int | float] | pandas.Series | None = None,
    append: bool = False,
    output_column: str | None = None,
) -> DataBundle
```

Exactly one regime source is required: threshold rule, explicit dates, or an
aligned vector/Series. The regime is stored in `metadata["regimes"]`; set
`append=True` to also add a numeric indicator column to the panel.

## Vintage Helpers

`list_vintages(dataset, start, end)` returns candidate vintage labels. The
selected vintage is passed to `load_fred_md`, `load_fred_qd`, or `load_fred_sd`
through `vintage=`.

## Official Source Pages

- FRED-MD and FRED-QD source page: <https://www.stlouisfed.org/research/economists/mccracken/fred-databases>
- FRED-SD source page: <https://www.stlouisfed.org/research/economists/owyang/fred-sd>
