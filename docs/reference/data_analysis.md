# macroforecast.data_analysis

[Back to reference](index.md)

## Purpose

`macroforecast.data_analysis` is the read-only inspection module for canonical
pandas panels. It covers two related tasks:

| Task | Main function | Input count | Use case |
| --- | --- | ---: | --- |
| Single-panel summary | `summarize_data(data)` | 1 | Inspect one raw, processed, or custom panel. |
| Raw-vs-processed comparison | `analyze_data(raw, clean)` | 2 | Inspect what changed after preprocessing. |

This module validates inputs, computes tables, and returns report objects. It
does not load data, transform values, impute missing observations, create
features, fit models, evaluate forecasts, or write files.

Inputs must satisfy the canonical panel contract used by
[`macroforecast.data`](data.md): `pandas.DataFrame`, `DatetimeIndex` named
`"date"`, ascending dates, no duplicate dates, no duplicate columns, numeric
values or `NaN`, no infinite values, and non-empty shape. `summarize_data(...)`
also accepts `DataBundle`, `DataSpec`, `(panel, metadata)`, and
`PreprocessedData`-style objects with `.panel` and `.metadata`.

## Public Functions

| Function | Kind | Output | Purpose |
| --- | --- | --- | --- |
| `summarize_data(data, ...)` | single-panel report | `DataSummaryReport` | Standard summary suite for one panel. |
| `panel_overview(data)` | single-panel helper | `dict` | Shape, dates, frequency, missingness, metadata keys. |
| `panel_snapshot(data)` | single-panel helper | `dict` | Compact rows/columns/dates/missingness/frequency snapshot. |
| `sample_coverage(data)` | single-panel helper | `DataFrame` | Per-series first/last valid dates, observation counts, missing rates. |
| `observation_counts(data)` | single-panel helper | `Series` | Per-series non-missing observation counts. |
| `missing_rates(data)` | single-panel helper | `Series` | Per-series missing rates. |
| `univariate_summary(data, ...)` | single-panel helper | `DataFrame` | Per-series numeric descriptive statistics. |
| `missing_summary(data)` | single-panel helper | `DataFrame` | Missing count, missing rate, longest missing run. |
| `correlation_matrix(data, ...)` | single-panel helper | `DataFrame` | Pairwise numeric correlation matrix. |
| `outlier_summary(data, ...)` | single-panel helper | `DataFrame` | IQR and/or z-score outlier counts and rates. |
| `stationarity_tests(data, ...)` | single-panel helper | `dict` | ADF, Phillips-Perron, KPSS, or all three. |
| `phillips_perron_test(values, ...)` | statistic helper | `dict` | Native PP fallback used when `arch` is unavailable. |
| `mackinnon_pp_pvalue(z_tau, ...)` | statistic helper | `float` | Approximate p-value helper for native PP. |
| `analyze_data(raw, clean, ...)` | before/after report | `DataAnalysisReport` | Standard comparison suite for raw and processed panels. |
| `compare_panels(raw, clean, ...)` | before/after helper | `dict` | Shape/date/column/index comparison plus changed-cell count. |
| `panel_snapshots(raw, clean)` | before/after helper | `dict` | Compact before/after snapshots. |
| `changed_cells(raw, clean, ...)` | before/after helper | `DataFrame` | Boolean changed-cell mask on common dates and columns. |
| `changed_cell_count(raw, clean, ...)` | before/after helper | `int` | Count changed common cells. |
| `changed_cell_summary(raw, clean, ...)` | before/after helper | `dict` | Changed-cell denominator, count, rate, and tolerance. |
| `missing_shift(raw, clean)` | before/after helper | `DataFrame` | Missing-count and missing-rate changes. |
| `distribution_shift(raw, clean, ...)` | before/after helper | `DataFrame` | Mean, scale, tail-shape, and KS-style shifts. |
| `correlation_shift(raw, clean, ...)` | before/after helper | `DataFrame` | Cleaned-minus-raw correlation differences. |
| `cleaning_effect_summary(...)` | metadata helper | `dict` | Normalize preprocessing metadata and counters. |

## Public Flow

```python
import macroforecast as mf

bundle = mf.data.load_fred_md()
summary = mf.data_analysis.summarize_data(
    bundle,
    include_outliers=True,
    include_stationarity=True,
)

spec = mf.data.spec(bundle, target="INDPRO", horizons=[1, 3, 6, 12])
processed = mf.preprocessing.reprocess(spec)

analysis = mf.data_analysis.analyze_data(
    spec.panel,
    processed.panel,
    include_correlation=True,
)
```

Example single-panel output:

```python
summary.overview
```

```python
{
    "n_rows": 4,
    "n_columns": 2,
    "start": "2020-01-01",
    "end": "2020-04-01",
    "missing_values": 1,
    "frequency": "monthly",
    "metadata_keys": ["dataset", "frequency"],
}
```

Example raw-vs-processed output:

```python
analysis.comparison
```

```python
{
    "raw_shape": (4, 3),
    "clean_shape": (4, 3),
    "raw_missing_total": 1,
    "clean_missing_total": 0,
    "common_columns": ["y", "x1", "x2"],
    "common_index_count": 4,
    "changed_cell_count": 2,
}
```

## summarize_data

Run the standard one-panel summary suite.

### Signature

```python
macroforecast.data_analysis.summarize_data(
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

| Name | Type | Default | Allowed values | Meaning |
| --- | --- | --- | --- | --- |
| `data` | `DataBundle`, `DataSpec`, `PreprocessedData`, `(panel, metadata)`, or `DataFrame` | required | canonical panel input | Panel to summarize. |
| `metrics` | sequence or `None` | default summary metrics | `mean`, `sd`, `min`, `max`, `skew`, `kurtosis`, `n_obs`, `n_missing` | Univariate statistics to compute. |
| `include_correlation` | `bool` | `False` | `True`, `False` | Include `correlation_matrix(...)`. |
| `correlation_method` | `str` | `"pearson"` | `"pearson"`, `"spearman"`, `"kendall"` | Correlation method when correlation is included. |
| `include_outliers` | `bool` | `False` | `True`, `False` | Include `outlier_summary(...)`. |
| `outlier_method` | `str` | `"iqr"` | `"iqr"`, `"zscore"`, `"multi"`, `"both"` | Outlier rule when outliers are included. |
| `include_stationarity` | `bool` | `False` | `True`, `False` | Include `stationarity_tests(...)`. |
| `stationarity_test` | `str` | `"multi"` | `"adf"`, `"pp"`, `"kpss"`, `"multi"`, `"none"` | Unit-root/stationarity test choice. |
| `stationarity_scope` | `str` | `"all"` | `"all"`, `"target_and_predictors"`, `"target_only"`, `"predictors_only"` | Columns to test. |

### Defaults

| Default | Value |
| --- | --- |
| Summary metrics | `DEFAULT_SUMMARY_METRICS = ("mean", "sd", "min", "max", "n_obs", "n_missing")` |
| Correlation included | `False` |
| Outlier summary included | `False` |
| Stationarity tests included | `False` |
| Metadata stage | `metadata["data_analysis"]` with `analysis_type="single_panel"` |

### Output

Returns `DataSummaryReport`.

| Field | Type | Meaning |
| --- | --- | --- |
| `overview` | `dict` | Panel row/column count, date range, missing total, inferred frequency, metadata keys. |
| `coverage` | `DataFrame` | Per-series first/last observed date, `n_obs`, `n_missing`, missing rate. |
| `univariate` | `DataFrame` | Per-series descriptive statistics selected by `metrics`. |
| `missing` | `DataFrame` | Per-series missing count, missing rate, longest missing run. |
| `correlation` | `DataFrame` or `None` | Numeric correlation matrix when requested. |
| `outliers` | `DataFrame` or `None` | IQR and/or z-score outlier counts and rates when requested. |
| `stationarity` | `dict` or `None` | ADF/PP/KPSS results when requested. |
| `metadata` | `dict` | Input metadata plus compact `data_analysis` run metadata. |

`DataSummaryReport.to_dict()` converts DataFrame fields into nested
dictionaries for serialization.

The returned `coverage`, `univariate`, `missing`, `correlation`, and `outliers`
tables carry `attrs["macroforecast_metadata"] == summary.metadata` when the
table is present.

### Metadata

`summarize_data(...)` stores run-level facts, not duplicate result tables:

```python
summary.metadata["data_analysis"]
```

| Key | Meaning |
| --- | --- |
| `analysis_type` | `"single_panel"`. |
| `metrics` | Univariate metrics requested. |
| `include_correlation`, `correlation_method` | Correlation option state. |
| `include_outliers`, `outlier_method` | Outlier option state. |
| `include_stationarity`, `stationarity_test`, `stationarity_scope` | Stationarity option state. |
| `panel` | Compact panel snapshot. |
| `input` | Source metadata snapshot and metadata-key list. |
| `outputs` | Boolean flags for report fields included. |

## Single-Panel Helpers

### panel_overview

```python
macroforecast.data_analysis.panel_overview(data) -> dict
```

Input is the same canonical one-panel input accepted by `summarize_data(...)`.
Output includes the full `panel_info(...)` dictionary plus `metadata_keys`.

### panel_snapshot

```python
macroforecast.data_analysis.panel_snapshot(data) -> dict
```

Returns a compact dictionary with `n_rows`, `n_columns`, `start`, `end`,
`missing_values`, and `frequency`.

### sample_coverage

```python
macroforecast.data_analysis.sample_coverage(data) -> pandas.DataFrame
```

Output columns:

| Column | Meaning |
| --- | --- |
| `first_valid` | First non-missing date for the series. |
| `last_valid` | Last non-missing date for the series. |
| `n_obs` | Non-missing observation count. |
| `n_missing` | Missing observation count. |
| `missing_rate` | `n_missing / n_panel_rows`. |

`observation_counts(data)` returns `sample_coverage(data)["n_obs"]`.
`missing_rates(data)` returns `sample_coverage(data)["missing_rate"]`.

### univariate_summary

```python
macroforecast.data_analysis.univariate_summary(
    data,
    *,
    metrics: Sequence[str] | None = None,
) -> pandas.DataFrame
```

| Input | Default | Allowed values |
| --- | --- | --- |
| `metrics` | default summary metrics | `mean`, `sd`, `min`, `max`, `skew`, `kurtosis`, `n_obs`, `n_missing` |

Returns one row per numeric column. Unknown metrics raise `ValueError`.

### missing_summary

```python
macroforecast.data_analysis.missing_summary(data) -> pandas.DataFrame
```

Returns `n_missing`, `missing_rate`, and `longest_missing_run` for each
series.

### correlation_matrix

```python
macroforecast.data_analysis.correlation_matrix(
    data,
    *,
    method: str = "pearson",
    min_periods: int = 1,
) -> pandas.DataFrame
```

| Input | Default | Allowed values |
| --- | --- | --- |
| `method` | `"pearson"` | `"pearson"`, `"spearman"`, `"kendall"` |
| `min_periods` | `1` | positive integer |

Invalid methods or `min_periods < 1` raise `ValueError`.

### outlier_summary

```python
macroforecast.data_analysis.outlier_summary(
    data,
    *,
    method: str = "iqr",
    iqr_threshold: float = 10.0,
    zscore_threshold: float = 3.0,
) -> pandas.DataFrame
```

| Input | Default | Allowed values |
| --- | --- | --- |
| `method` | `"iqr"` | `"iqr"`, `"zscore"`, `"multi"`, `"both"` |
| `iqr_threshold` | `10.0` | positive float |
| `zscore_threshold` | `3.0` | positive float |

The IQR default matches the McCracken-Ng/FRED-MD outlier multiplier used by
preprocessing defaults. The z-score path uses population standard deviation
(`ddof=0`) to match `macroforecast.preprocessing.zscore_outlier_clean(...)`.
Non-positive thresholds raise `ValueError`.

### stationarity_tests

```python
macroforecast.data_analysis.stationarity_tests(
    data,
    *,
    test: str = "multi",
    scope: str = "all",
    target: str | None = None,
    targets: Sequence[str] | None = None,
    alpha: float = 0.05,
) -> dict
```

| Input | Default | Allowed values |
| --- | --- | --- |
| `test` | `"multi"` | `"adf"`, `"pp"`, `"kpss"`, `"multi"`, `"none"` |
| `scope` | `"all"` | `"all"`, `"target_and_predictors"`, `"target_only"`, `"predictors_only"` |
| `target`, `targets` | `None` | target names in the panel |
| `alpha` | `0.05` | float strictly between `0` and `1` |

For `scope="target_only"` and `scope="predictors_only"`, target names must be
known from arguments or from a `DataSpec`. Missing target columns raise
`ValueError`.

Output dictionary:

| Key | Meaning |
| --- | --- |
| `test`, `scope`, `alpha` | Requested test settings. |
| `n_series` | Number of tested series. |
| `by_series` | Per-series test results. |

Per-test outputs:

| Test | Key outputs |
| --- | --- |
| `adf` | `statistic`, `p_value`, `reject_unit_root` |
| `pp` | `statistic`, `p_value`, `reject_unit_root`, `implementation`, `bandwidth_lags` when native |
| `kpss` | `statistic`, `p_value`, `reject_stationarity` |

`pp` uses `arch.unitroot.PhillipsPerron` when available. Otherwise it falls
back to macroforecast's native Newey-West/MacKinnon implementation.

### Phillips-Perron Helpers

```python
macroforecast.data_analysis.phillips_perron_test(values, *, alpha=0.05) -> dict
macroforecast.data_analysis.mackinnon_pp_pvalue(z_tau, *, n, regression="c") -> float
```

`phillips_perron_test(...)` drops non-finite values, requires at least eight
finite observations, and returns `status="insufficient_data"` or
`status="singular_design"` instead of raising for those data conditions.

`mackinnon_pp_pvalue(...)` approximates the MacKinnon p-value for the constant
case (`regression="c"`) using the internal critical-value table. For other
regression labels it falls back to a normal CDF approximation. Non-finite
statistics and non-positive sample sizes raise `ValueError`.

## analyze_data

Run the standard before/after data analysis suite.

### Signature

```python
macroforecast.data_analysis.analyze_data(
    raw,
    clean,
    *,
    distribution_metrics: Sequence[str] | None = None,
    include_correlation: bool = False,
    correlation_method: str = "pearson",
    sample: str = "common_index",
    cleaning_metadata: Mapping[str, object] | None = None,
    cleaning_log: Mapping[str, object] | None = None,
    transform_map_applied: Mapping[str, int] | None = None,
    n_imputed_cells: int | None = None,
    n_outliers_flagged: int | None = None,
    n_truncated_obs: int | None = None,
    column_metadata: Mapping[str, object] | None = None,
    tolerance: float = 0.0,
) -> DataAnalysisReport
```

### Input

| Name | Type | Default | Allowed values | Meaning |
| --- | --- | --- | --- | --- |
| `raw` | `DataFrame` | required | canonical panel | Before/preprocessing panel. |
| `clean` | `DataFrame` | required | canonical panel | After/preprocessing panel. |
| `distribution_metrics` | sequence or `None` | all defaults | `mean_change`, `sd_change`, `sd_ratio`, `skew_change`, `kurtosis_change`, `ks_statistic` | Distribution-shift columns to compute. |
| `include_correlation` | `bool` | `False` | `True`, `False` | Include cleaned-minus-raw correlations. |
| `correlation_method` | `str` | `"pearson"` | `"pearson"`, `"spearman"`, `"kendall"` | Correlation method. |
| `sample` | `str` | `"common_index"` | `"common_index"`, `"full"` | Date sample used by distribution and correlation shifts. |
| `cleaning_metadata` | mapping or `None` | auto from clean panel metadata | preprocessing metadata mapping | Source for effect counters and logs. |
| `cleaning_log` | mapping or `None` | from metadata when available | mapping | Optional explicit cleaning log. |
| `transform_map_applied` | mapping or `None` | from metadata when available | mapping from column to t-code | Optional explicit transform-code map. |
| `n_imputed_cells` | int or `None` | from metadata when available | non-negative count | Optional imputation counter. |
| `n_outliers_flagged` | int or `None` | from metadata when available | non-negative count | Optional outlier counter. |
| `n_truncated_obs` | int or `None` | from metadata when available | non-negative count | Optional truncation counter. |
| `column_metadata` | mapping or `None` | from metadata when available | mapping | Optional per-column preprocessing metadata. |
| `tolerance` | `float` | `0.0` | non-negative float | Absolute tolerance for changed-cell counting. |

### Defaults

| Default | Value |
| --- | --- |
| Distribution metrics | all six `DEFAULT_DISTRIBUTION_METRICS` values |
| Correlation included | `False` |
| Comparison sample | `"common_index"` |
| Changed-cell tolerance | `0.0` |
| Metadata stage | `metadata["data_analysis"]` with `analysis_type="raw_vs_processed"` |

### Output

Returns `DataAnalysisReport`.

| Field | Type | Meaning |
| --- | --- | --- |
| `comparison` | `dict` | Shape, date range, common columns/index, missing totals, changed-cell count. |
| `missing_shift` | `DataFrame` | Per-column raw/clean missing counts and rate changes. |
| `distribution_shift` | `DataFrame` | Per-column distribution changes for common numeric columns. |
| `correlation_shift` | `DataFrame` or `None` | Cleaned-minus-raw correlation matrix when requested. |
| `cleaning_effect_summary` | `dict` | Normalized preprocessing counters, transform map, cleaning log, column metadata. |
| `metadata` | `dict` | Input metadata plus compact `data_analysis` run metadata. |

`DataAnalysisReport.to_dict()` converts DataFrame fields into nested
dictionaries for serialization.

The returned `missing_shift`, `distribution_shift`, and `correlation_shift`
tables carry `attrs["macroforecast_metadata"] == analysis.metadata` when the
table is present.

### Metadata

```python
analysis.metadata["data_analysis"]
```

| Key | Meaning |
| --- | --- |
| `analysis_type` | `"raw_vs_processed"`. |
| `before` | Raw panel snapshot: rows, columns, start, end, missing count. |
| `after` | Processed panel snapshot with the same fields. |
| `common` | Common row/column counts and changed-cell count. |
| `options` | Distribution metrics, correlation option, sample, and tolerance. |
| `effects` | Compact preprocessing counters and metadata presence flags. |
| `metadata_keys` | Metadata keys detected on raw and processed panels. |

### Sample Choice

`distribution_shift(...)` and `correlation_shift(...)` default to
`sample="common_index"`. This avoids mixing distribution changes with dates
that only exist before or after preprocessing. Use `sample="full"` only when
the full available sample of each panel is the intended comparison.

`ks_statistic` is the two-sample KS statistic only; it does not compute a
p-value.

## Before/After Helpers

### compare_panels

```python
macroforecast.data_analysis.compare_panels(
    raw,
    clean,
    *,
    tolerance: float = 0.0,
) -> dict
```

Output keys include `raw_shape`, `clean_shape`, raw/clean index types, date
ranges, missing totals, `common_columns`, raw-only and clean-only columns,
common/raw-only/clean-only index counts, and `changed_cell_count`.

### panel_snapshots

```python
macroforecast.data_analysis.panel_snapshots(raw, clean) -> dict
```

Returns `{"before": ..., "after": ...}` using compact snapshots.

### changed_cells, changed_cell_count, changed_cell_summary

```python
macroforecast.data_analysis.changed_cells(raw, clean, *, tolerance=0.0) -> pandas.DataFrame
macroforecast.data_analysis.changed_cell_count(raw, clean, *, tolerance=0.0) -> int
macroforecast.data_analysis.changed_cell_summary(raw, clean, *, tolerance=0.0) -> dict
```

All three use common dates and common columns. Numeric cells whose absolute
difference is less than or equal to `tolerance` are treated as unchanged.
Negative tolerance raises `ValueError`.

### missing_shift

```python
macroforecast.data_analysis.missing_shift(raw, clean) -> pandas.DataFrame
```

Returns one row per unioned column with `column_status`, raw and clean sample
sizes, raw and clean missing counts, missing-count change, missing rates, and
missing-rate change.

### distribution_shift

```python
macroforecast.data_analysis.distribution_shift(
    raw,
    clean,
    *,
    metrics: Sequence[str] | None = None,
    sample: str = "common_index",
) -> pandas.DataFrame
```

Allowed metrics are `mean_change`, `sd_change`, `sd_ratio`, `skew_change`,
`kurtosis_change`, and `ks_statistic`. Unknown metrics raise `ValueError`.

### correlation_shift

```python
macroforecast.data_analysis.correlation_shift(
    raw,
    clean,
    *,
    method: str = "pearson",
    fill_value: float | None = None,
    sample: str = "common_index",
) -> pandas.DataFrame
```

Returns the cleaned-minus-raw correlation matrix for common numeric columns.
If fewer than two common numeric columns exist, returns an empty square
DataFrame indexed by the available common numeric columns.

### cleaning_effect_summary

```python
macroforecast.data_analysis.cleaning_effect_summary(
    *,
    cleaning_metadata: Mapping[str, object] | None = None,
    cleaning_log: Mapping[str, object] | None = None,
    transform_map_applied: Mapping[str, int] | None = None,
    n_imputed_cells: int | None = None,
    n_outliers_flagged: int | None = None,
    n_truncated_obs: int | None = None,
    column_metadata: Mapping[str, object] | None = None,
) -> dict
```

This helper normalizes preprocessing metadata into one compact dictionary. If
explicit counters are not supplied, it tries to derive them from preprocessing
step metadata.

## Boundaries

| Question | Use | Why |
| --- | --- | --- |
| What does this one panel look like? | `mf.data_analysis.summarize_data(panel)` | One input, level summary. |
| What changed from raw to processed? | `mf.data_analysis.analyze_data(raw, processed)` | Two inputs, before/after deltas. |
| Which preprocessing choices ran? | `mf.preprocessing.report(processed)` | Execution log rather than statistical summary. |
| Should this table be written to disk? | `mf.output` or `mf.reporting` | Output/rendering is separate from analysis. |

- `adf_test` -- Augmented Dickey-Fuller unit-root test for a single series (flat result).
- `kpss_test` -- KPSS stationarity test for a single series (flat result).
- `dfgls_test` -- Elliott-Rothenberg-Stock DF-GLS GLS-detrended unit-root test (urca::ur.ers).
- `zivot_andrews_test` -- Zivot-Andrews unit-root test with one endogenous structural break (urca::ur.za).

- `ndiffs` -- number of first differences for stationarity (KPSS/ADF/PP; forecast::ndiffs).
- `nsdiffs` -- number of seasonal differences via STL seasonal strength (forecast::nsdiffs).

- `acf` -- sample autocorrelation function with confidence bands (stats::acf / forecast::Acf).
- `pacf` -- sample partial autocorrelation function with confidence bands (stats::pacf / forecast::Pacf).

- `johansen_cointegration` -- Johansen cointegration test (trace + max-eigenvalue, rank selection, cointegrating vectors; urca::ca.jo).
- `newey_west` -- Newey-West HAC covariance for OLS coefficients with Bartlett kernel and coefficient table (sandwich::NeweyWest + lmtest::coeftest).
- `vcov_hc` -- heteroskedasticity-consistent (White HC0-HC3) covariance for OLS coefficients with coefficient table (sandwich::vcovHC + lmtest::coeftest).
- `breusch_pagan_test` -- Breusch-Pagan test for heteroskedasticity, Koenker studentized or classic variant (lmtest::bptest).
