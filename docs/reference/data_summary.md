# macroforecast.data_summary

[Back to reference](index.md)

`macroforecast.data_summary` describes one canonical pandas panel. It is for
checking the state of a dataset before or after preprocessing: sample coverage,
missingness, descriptive statistics, outlier flags, stationarity tests, and
optional correlations.

Use `macroforecast.data_analysis` when comparing two panels, such as raw data
versus a processed panel. Use `macroforecast.preprocessing.report()` when the
question is which preprocessing choices ran.

## Public Flow

```python
import macroforecast as mf

bundle = mf.data.load_fred_md()
summary = mf.data_summary.summarize_data(bundle)

summary.overview
summary.coverage
summary.univariate
summary.missing
```

## summarize_data

```python
macroforecast.data_summary.summarize_data(
    data,
    *,
    metrics: Sequence[str] | None = None,
    include_correlation: bool = False,
    correlation_method: str = "pearson",
    include_outliers: bool = False,
    outlier_method: str = "iqr",
    include_stationarity: bool = False,
    stationarity_test: str = "multi",
    stationarity_scope: str = "all",
) -> DataSummaryReport
```

### Input

| Name | Type | Default | Choices |
| --- | --- | --- | --- |
| `data` | `DataBundle`, `DataSpec`, `PreprocessedData`, `(panel, metadata)`, or `DataFrame` | required | Canonical date-indexed panel input. |
| `metrics` | sequence or `None` | `("mean", "sd", "min", "max", "n_obs", "n_missing")` | Any of `mean`, `sd`, `min`, `max`, `skew`, `kurtosis`, `n_obs`, `n_missing`. |
| `include_correlation` | `bool` | `False` | Whether to include a full numeric correlation matrix. |
| `correlation_method` | `str` | `"pearson"` | `"pearson"`, `"spearman"`, `"kendall"`. |
| `include_outliers` | `bool` | `False` | Whether to include `outlier_summary(...)`. |
| `outlier_method` | `str` | `"iqr"` | `"iqr"`, `"zscore"`, `"multi"`, `"both"`. |
| `include_stationarity` | `bool` | `False` | Whether to include `stationarity_tests(...)`. |
| `stationarity_test` | `str` | `"multi"` | `"adf"`, `"pp"`, `"kpss"`, `"multi"`, `"none"`. |
| `stationarity_scope` | `str` | `"all"` | `"all"`, `"target_and_predictors"`, `"target_only"`, `"predictors_only"`. |

### Output

Returns `DataSummaryReport`.

| Field | Type | Meaning |
| --- | --- | --- |
| `overview` | `dict` | Panel row/column count, date range, missing total, inferred frequency, metadata keys. |
| `coverage` | `pandas.DataFrame` | Per-series first/last observed date, `n_obs`, `n_missing`, missing rate. |
| `univariate` | `pandas.DataFrame` | Per-series descriptive statistics selected by `metrics`. |
| `missing` | `pandas.DataFrame` | Per-series missing count, missing rate, and longest missing run. |
| `correlation` | `pandas.DataFrame` or `None` | Correlation matrix when requested. |
| `outliers` | `pandas.DataFrame` or `None` | IQR and/or z-score outlier counts and rates when requested. |
| `stationarity` | `dict` or `None` | ADF/PP/KPSS results when requested. |
| `metadata` | `dict` | Input metadata plus a compact `data_summary` stage describing options and outputs. |

`DataSummaryReport.to_dict()` converts DataFrame fields into nested
dictionaries for serialization.

### Metadata

`summarize_data(...)` does not duplicate result tables inside metadata. It
keeps only run-level facts that are useful for provenance:

```python
summary.metadata["data_summary"]
```

| Key | Meaning |
| --- | --- |
| `metrics` | Univariate metrics requested. |
| `include_correlation`, `correlation_method` | Correlation option state. |
| `include_outliers`, `outlier_method` | Outlier option state. |
| `include_stationarity`, `stationarity_test`, `stationarity_scope` | Stationarity option state. |
| `panel` | Compact panel snapshot: rows, columns, start, end, missing count, inferred frequency. |
| `outputs` | Boolean flags for fields included in the report. |

## Helper Functions

| Function | Input | Output | Purpose |
| --- | --- | --- | --- |
| `panel_overview(data)` | canonical panel input | `dict` | Shape, dates, frequency, total missingness, metadata keys. |
| `sample_coverage(data)` | canonical panel input | `DataFrame` | Per-series sample coverage and missing rate. |
| `univariate_summary(data, metrics=...)` | canonical panel input | `DataFrame` | Per-series descriptive statistics. |
| `missing_summary(data)` | canonical panel input | `DataFrame` | Missing count/rate and longest missing run. |
| `correlation_matrix(data, method=...)` | canonical panel input | `DataFrame` | Numeric correlation matrix for one panel. |
| `outlier_summary(data, method=...)` | canonical panel input | `DataFrame` | Per-series IQR and/or z-score outlier counts and rates. |
| `stationarity_tests(data, test=..., scope=...)` | canonical panel input | `dict` | ADF, Phillips-Perron, KPSS, or all three. |
| `phillips_perron_test(values)` | sequence/array | `dict` | Native Phillips-Perron Z_tau unit-root test. |
| `mackinnon_pp_pvalue(z_tau, n=...)` | statistic/sample size | `float` | Approximate MacKinnon p-value used by `phillips_perron_test`. |

## outlier_summary

```python
macroforecast.data_summary.outlier_summary(
    data,
    *,
    method: str = "iqr",
    iqr_threshold: float = 10.0,
    zscore_threshold: float = 3.0,
) -> pandas.DataFrame
```

### Input

| Name | Default | Choices |
| --- | --- | --- |
| `method` | `"iqr"` | `"iqr"`, `"zscore"`, `"multi"`, `"both"` |
| `iqr_threshold` | `10.0` | Positive float. This matches the McCracken-Ng/FRED-MD outlier multiplier used by preprocessing defaults. |
| `zscore_threshold` | `3.0` | Positive float. |

### Output

Returns one row per numeric series. Columns include `n_obs`,
`iqr_outlier_count`, `iqr_outlier_rate`, `zscore_outlier_count`, and
`zscore_outlier_rate`, depending on `method`.

## stationarity_tests

```python
macroforecast.data_summary.stationarity_tests(
    data,
    *,
    test: str = "multi",
    scope: str = "all",
    target: str | None = None,
    targets: Sequence[str] | None = None,
    alpha: float = 0.05,
) -> dict
```

### Input

| Name | Default | Choices |
| --- | --- | --- |
| `test` | `"multi"` | `"adf"`, `"pp"`, `"kpss"`, `"multi"`, `"none"` |
| `scope` | `"all"` | `"all"`, `"target_and_predictors"`, `"target_only"`, `"predictors_only"` |
| `target`, `targets` | `None` | Optional target names. If `data` is a `DataSpec`, these default from the spec. |
| `alpha` | `0.05` | Significance level for rejection booleans. |

### Output

Returns a dictionary with `test`, `scope`, `alpha`, `n_series`, and
`by_series`. For each series, `by_series[column]` contains `n_obs` and one or
more of:

| Test | Key Outputs |
| --- | --- |
| `adf` | `statistic`, `p_value`, `reject_unit_root` |
| `pp` | `statistic`, `p_value`, `reject_unit_root`, `implementation`, `bandwidth_lags` when native |
| `kpss` | `statistic`, `p_value`, `reject_stationarity` |

`pp` uses `arch.unitroot.PhillipsPerron` when available and otherwise falls
back to macroforecast's native Newey-West/MacKinnon implementation.

## Boundary

| Question | Use |
| --- | --- |
| What does this single raw panel look like? | `mf.data_summary.summarize_data(raw)` |
| What does the processed panel look like? | `mf.data_summary.summarize_data(processed)` |
| What changed from raw to processed? | `mf.data_analysis.analyze_data(raw, processed.panel)` |
| Which preprocessing choices ran? | `mf.preprocessing.report(processed)` |
