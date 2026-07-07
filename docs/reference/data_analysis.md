# macroforecast.data_analysis

[Back to reference](index.md)

One-panel summaries and before/after preprocessing diagnostics.

## Public Symbols

| Symbol | Kind | Summary |
| --- | --- | --- |
| `DEFAULT_DISTRIBUTION_METRICS` | data | Built-in immutable sequence. |
| `DEFAULT_SUMMARY_METRICS` | data | Built-in immutable sequence. |
| `DataAnalysisReport` | class | Container returned by :func:`analyze_data`. |
| `DataSummaryReport` | class | Container returned by :func:`summarize_data`. |
| `changed_cell_count` | function | Return the number of changed common-index/common-column cells. |
| `changed_cell_summary` | function | Return changed-cell count and rate for the common sample. |
| `changed_cells` | function | Return a boolean mask of changed common-index/common-column cells. |
| `cleaning_effect_summary` | function | Normalize preprocessing metadata into a compact data analysis summary. |
| `compare_panels` | function | Compare raw and cleaned panels at panel, column, index, and cell level. |
| `correlation_matrix` | function | Return a numeric correlation matrix for one panel. |
| `correlation_shift` | function | Return cleaned-minus-raw correlation matrix for common numeric columns. |
| `mackinnon_pp_pvalue` | function | Approximate MacKinnon p-value for the Phillips-Perron Z_tau statistic. |
| `analyze_data` | function | Run the standard data analysis suite on raw and cleaned panels. |
| `distribution_shift` | function | Return per-series distribution changes from raw to cleaned data. |
| `missing_rates` | function | Return per-series missing rates. |
| `missing_summary` | function | Return per-series missing-count, missing-rate, and longest-gap summary. |
| `missing_shift` | function | Return per-column missing-count and missing-rate changes. |
| `observation_counts` | function | Return per-series non-missing observation counts. |
| `outlier_summary` | function | Return per-series outlier counts and rates for one panel. |
| `panel_overview` | function | Return panel-level shape, date range, frequency, and missingness. |
| `panel_snapshot` | function | Return a compact single-panel snapshot for reports and provenance. |
| `panel_snapshots` | function | Return compact before/after panel snapshots. |
| `adf_test` | function | Augmented Dickey-Fuller unit-root test for a single series. |
| `kpss_test` | function | KPSS stationarity test for a single series. |
| `acf` | function | Sample autocorrelation function (stats::acf / forecast::Acf). |
| `johansen_cointegration` | function | Johansen cointegration test (R urca::ca.jo / statsmodels coint_johansen). |
| `engle_granger` | function | Engle-Granger two-step residual-based cointegration test. |
| `phillips_ouliaris` | function | Phillips-Ouliaris residual-based cointegration test. |
| `variance_ratio` | function | Lo-MacKinlay variance-ratio test of the random-walk null. |
| `structural_stability` | function | OLS-CUSUM test for parameter stability of a regression. |
| `newey_west` | function | Newey-West HAC covariance for an OLS regression. |
| `vcov_hc` | function | Heteroskedasticity-consistent (White) covariance for an OLS regression. |
| `breusch_pagan_test` | function | Breusch-Pagan test for heteroskedasticity in an OLS regression. |
| `pacf` | function | Sample partial autocorrelation function (stats::pacf / forecast::Pacf). |
| `ndiffs` | function | Number of first differences to make a series stationary (forecast::ndiffs). |
| `nsdiffs` | function | Number of seasonal differences via seasonal strength (forecast::nsdiffs). |
| `phillips_perron_test` | function | Run the native Phillips-Perron Z_tau unit-root test. |
| `dfgls_test` | function | Elliott-Rothenberg-Stock DF-GLS unit-root test for a single series. |
| `zivot_andrews_test` | function | Zivot-Andrews unit-root test allowing one endogenous structural break. |
| `sample_coverage` | function | Return per-series sample start, end, observation count, and missingness. |
| `stationarity_tests` | function | Run ADF, Phillips-Perron, KPSS, or all three on one panel. |
| `summarize_data` | function | Run the standard single-panel summary suite. |
| `univariate_summary` | function | Return per-series descriptive statistics for numeric panel columns. |

## Data And Module Values

### `DEFAULT_DISTRIBUTION_METRICS`

Kind: `data`

```python
DEFAULT_DISTRIBUTION_METRICS = ("mean_change", "sd_change", "sd_ratio", "skew_change", "kurtosis_change", "ks_statistic")
```
### `DEFAULT_SUMMARY_METRICS`

Kind: `data`

```python
DEFAULT_SUMMARY_METRICS = ("mean", "sd", "min", "max", "n_obs", "n_missing")
```

## Callable And Class Reference

### DataAnalysisReport

Qualified name: `macroforecast.data_analysis.core.DataAnalysisReport`

#### Signature

```python
macroforecast.data_analysis.DataAnalysisReport(comparison: dict[str, Any], missing_shift: pd.DataFrame, distribution_shift: pd.DataFrame, correlation_shift: pd.DataFrame | None = None, cleaning_effect_summary: dict[str, Any] = <factory>, metadata: dict[str, Any] = <factory>) -> None
```

#### Description

Container returned by :func:`analyze_data`.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `comparison` | positional or keyword | `dict[str, Any]` | `required` |
| `missing_shift` | positional or keyword | `pd.DataFrame` | `required` |
| `distribution_shift` | positional or keyword | `pd.DataFrame` | `required` |
| `correlation_shift` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `cleaning_effect_summary` | positional or keyword | `dict[str, Any]` | `<factory>` |
| `metadata` | positional or keyword | `dict[str, Any]` | `<factory>` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.data_analysis.DataAnalysisReport(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `comparison` | `dict[str, Any]` | `required` |
| `missing_shift` | `pd.DataFrame` | `required` |
| `distribution_shift` | `pd.DataFrame` | `required` |
| `correlation_shift` | `pd.DataFrame \| None` | `None` |
| `cleaning_effect_summary` | `dict[str, Any]` | `default_factory` |
| `metadata` | `dict[str, Any]` | `default_factory` |

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `to_dict` | `to_dict(self) -> dict[str, Any]` | No public docstring is available. |
### DataSummaryReport

Qualified name: `macroforecast.data_analysis.summary.DataSummaryReport`

#### Signature

```python
macroforecast.data_analysis.DataSummaryReport(overview: dict[str, Any], coverage: pd.DataFrame, univariate: pd.DataFrame, missing: pd.DataFrame, correlation: pd.DataFrame | None = None, outliers: pd.DataFrame | None = None, stationarity: dict[str, Any] | None = None, metadata: dict[str, Any] = <factory>) -> None
```

#### Description

Container returned by :func:`summarize_data`.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `overview` | positional or keyword | `dict[str, Any]` | `required` |
| `coverage` | positional or keyword | `pd.DataFrame` | `required` |
| `univariate` | positional or keyword | `pd.DataFrame` | `required` |
| `missing` | positional or keyword | `pd.DataFrame` | `required` |
| `correlation` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `outliers` | positional or keyword | `pd.DataFrame \| None` | `None` |
| `stationarity` | positional or keyword | `dict[str, Any] \| None` | `None` |
| `metadata` | positional or keyword | `dict[str, Any]` | `<factory>` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.data_analysis.DataSummaryReport(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `overview` | `dict[str, Any]` | `required` |
| `coverage` | `pd.DataFrame` | `required` |
| `univariate` | `pd.DataFrame` | `required` |
| `missing` | `pd.DataFrame` | `required` |
| `correlation` | `pd.DataFrame \| None` | `None` |
| `outliers` | `pd.DataFrame \| None` | `None` |
| `stationarity` | `dict[str, Any] \| None` | `None` |
| `metadata` | `dict[str, Any]` | `default_factory` |

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `to_dict` | `to_dict(self) -> dict[str, Any]` | No public docstring is available. |
### changed_cell_count

Qualified name: `macroforecast.data_analysis.core.changed_cell_count`

#### Signature

```python
macroforecast.data_analysis.changed_cell_count(raw: pd.DataFrame, clean: pd.DataFrame, *, tolerance: float = 0.0) -> int
```

#### Description

Return the number of changed common-index/common-column cells.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `raw` | positional or keyword | `pd.DataFrame` | `required` |
| `clean` | positional or keyword | `pd.DataFrame` | `required` |
| `tolerance` | keyword only | `float` | `0.0` |

#### Returns

`int`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data_analysis.changed_cell_count(...)
```
### changed_cell_summary

Qualified name: `macroforecast.data_analysis.core.changed_cell_summary`

#### Signature

```python
macroforecast.data_analysis.changed_cell_summary(raw: pd.DataFrame, clean: pd.DataFrame, *, tolerance: float = 0.0) -> dict[str, Any]
```

#### Description

Return changed-cell count and rate for the common sample.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `raw` | positional or keyword | `pd.DataFrame` | `required` |
| `clean` | positional or keyword | `pd.DataFrame` | `required` |
| `tolerance` | keyword only | `float` | `0.0` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data_analysis.changed_cell_summary(...)
```
### changed_cells

Qualified name: `macroforecast.data_analysis.core.changed_cells`

#### Signature

```python
macroforecast.data_analysis.changed_cells(raw: pd.DataFrame, clean: pd.DataFrame, *, tolerance: float = 0.0) -> pd.DataFrame
```

#### Description

Return a boolean mask of changed common-index/common-column cells.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `raw` | positional or keyword | `pd.DataFrame` | `required` |
| `clean` | positional or keyword | `pd.DataFrame` | `required` |
| `tolerance` | keyword only | `float` | `0.0` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data_analysis.changed_cells(...)
```
### cleaning_effect_summary

Qualified name: `macroforecast.data_analysis.core.cleaning_effect_summary`

#### Signature

```python
macroforecast.data_analysis.cleaning_effect_summary(*, cleaning_metadata: Mapping[str, Any] | None = None, cleaning_log: Mapping[str, Any] | None = None, transform_map_applied: Mapping[str, int] | None = None, n_imputed_cells: int | None = None, n_outliers_flagged: int | None = None, n_truncated_obs: int | None = None, column_metadata: Mapping[str, Any] | None = None) -> dict[str, Any]
```

#### Description

Normalize preprocessing metadata into a compact data analysis summary.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `cleaning_metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `cleaning_log` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `transform_map_applied` | keyword only | `Mapping[str, int] \| None` | `None` |
| `n_imputed_cells` | keyword only | `int \| None` | `None` |
| `n_outliers_flagged` | keyword only | `int \| None` | `None` |
| `n_truncated_obs` | keyword only | `int \| None` | `None` |
| `column_metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data_analysis.cleaning_effect_summary(...)
```
### compare_panels

Qualified name: `macroforecast.data_analysis.core.compare_panels`

#### Signature

```python
macroforecast.data_analysis.compare_panels(raw: pd.DataFrame, clean: pd.DataFrame, *, tolerance: float = 0.0) -> dict[str, Any]
```

#### Description

Compare raw and cleaned panels at panel, column, index, and cell level.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `raw` | positional or keyword | `pd.DataFrame` | `required` |
| `clean` | positional or keyword | `pd.DataFrame` | `required` |
| `tolerance` | keyword only | `float` | `0.0` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data_analysis.compare_panels(...)
```
### correlation_matrix

Qualified name: `macroforecast.data_analysis.summary.correlation_matrix`

#### Signature

```python
macroforecast.data_analysis.correlation_matrix(data: Any, *, method: CorrelationMethod = "pearson", min_periods: int = 1) -> pd.DataFrame
```

#### Description

Return a numeric correlation matrix for one panel.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `Any` | `required` |
| `method` | keyword only | `CorrelationMethod` | `"pearson"` |
| `min_periods` | keyword only | `int` | `1` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data_analysis.correlation_matrix(...)
```
### correlation_shift

Qualified name: `macroforecast.data_analysis.core.correlation_shift`

#### Signature

```python
macroforecast.data_analysis.correlation_shift(raw: pd.DataFrame, clean: pd.DataFrame, *, method: CorrelationMethod = "pearson", fill_value: float | None = None, sample: AnalysisSample = "common_index") -> pd.DataFrame
```

#### Description

Return cleaned-minus-raw correlation matrix for common numeric columns.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `raw` | positional or keyword | `pd.DataFrame` | `required` |
| `clean` | positional or keyword | `pd.DataFrame` | `required` |
| `method` | keyword only | `CorrelationMethod` | `"pearson"` |
| `fill_value` | keyword only | `float \| None` | `None` |
| `sample` | keyword only | `AnalysisSample` | `"common_index"` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data_analysis.correlation_shift(...)
```
### mackinnon_pp_pvalue

Qualified name: `macroforecast.data_analysis.summary.mackinnon_pp_pvalue`

#### Signature

```python
macroforecast.data_analysis.mackinnon_pp_pvalue(z_tau: float, *, n: int, regression: str = "c") -> float
```

#### Description

Approximate MacKinnon p-value for the Phillips-Perron Z_tau statistic.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `z_tau` | positional or keyword | `float` | `required` |
| `n` | keyword only | `int` | `required` |
| `regression` | keyword only | `str` | `"c"` |

#### Returns

`float`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data_analysis.mackinnon_pp_pvalue(...)
```
### analyze_data

Qualified name: `macroforecast.data_analysis.core.analyze_data`

#### Signature

```python
macroforecast.data_analysis.analyze_data(raw: pd.DataFrame, clean: pd.DataFrame, *, distribution_metrics: Sequence[DistributionMetric] | None = None, include_correlation: bool = False, correlation_method: CorrelationMethod = "pearson", sample: AnalysisSample = "common_index", cleaning_metadata: Mapping[str, Any] | None = None, cleaning_log: Mapping[str, Any] | None = None, transform_map_applied: Mapping[str, int] | None = None, n_imputed_cells: int | None = None, n_outliers_flagged: int | None = None, n_truncated_obs: int | None = None, column_metadata: Mapping[str, Any] | None = None, tolerance: float = 0.0) -> DataAnalysisReport
```

#### Description

Run the standard data analysis suite on raw and cleaned panels.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `raw` | positional or keyword | `pd.DataFrame` | `required` |
| `clean` | positional or keyword | `pd.DataFrame` | `required` |
| `distribution_metrics` | keyword only | `Sequence[DistributionMetric] \| None` | `None` |
| `include_correlation` | keyword only | `bool` | `False` |
| `correlation_method` | keyword only | `CorrelationMethod` | `"pearson"` |
| `sample` | keyword only | `AnalysisSample` | `"common_index"` |
| `cleaning_metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `cleaning_log` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `transform_map_applied` | keyword only | `Mapping[str, int] \| None` | `None` |
| `n_imputed_cells` | keyword only | `int \| None` | `None` |
| `n_outliers_flagged` | keyword only | `int \| None` | `None` |
| `n_truncated_obs` | keyword only | `int \| None` | `None` |
| `column_metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `tolerance` | keyword only | `float` | `0.0` |

#### Returns

`DataAnalysisReport`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data_analysis.analyze_data(...)
```
### distribution_shift

Qualified name: `macroforecast.data_analysis.core.distribution_shift`

#### Signature

```python
macroforecast.data_analysis.distribution_shift(raw: pd.DataFrame, clean: pd.DataFrame, *, metrics: Sequence[DistributionMetric] | None = None, sample: AnalysisSample = "common_index") -> pd.DataFrame
```

#### Description

Return per-series distribution changes from raw to cleaned data.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `raw` | positional or keyword | `pd.DataFrame` | `required` |
| `clean` | positional or keyword | `pd.DataFrame` | `required` |
| `metrics` | keyword only | `Sequence[DistributionMetric] \| None` | `None` |
| `sample` | keyword only | `AnalysisSample` | `"common_index"` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data_analysis.distribution_shift(...)
```
### missing_rates

Qualified name: `macroforecast.data_analysis.summary.missing_rates`

#### Signature

```python
macroforecast.data_analysis.missing_rates(data: Any) -> pd.Series
```

#### Description

Return per-series missing rates.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `Any` | `required` |

#### Returns

`pd.Series`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data_analysis.missing_rates(...)
```
### missing_summary

Qualified name: `macroforecast.data_analysis.summary.missing_summary`

#### Signature

```python
macroforecast.data_analysis.missing_summary(data: Any) -> pd.DataFrame
```

#### Description

Return per-series missing-count, missing-rate, and longest-gap summary.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `Any` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data_analysis.missing_summary(...)
```
### missing_shift

Qualified name: `macroforecast.data_analysis.core.missing_shift`

#### Signature

```python
macroforecast.data_analysis.missing_shift(raw: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame
```

#### Description

Return per-column missing-count and missing-rate changes.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `raw` | positional or keyword | `pd.DataFrame` | `required` |
| `clean` | positional or keyword | `pd.DataFrame` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data_analysis.missing_shift(...)
```
### observation_counts

Qualified name: `macroforecast.data_analysis.summary.observation_counts`

#### Signature

```python
macroforecast.data_analysis.observation_counts(data: Any) -> pd.Series
```

#### Description

Return per-series non-missing observation counts.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `Any` | `required` |

#### Returns

`pd.Series`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data_analysis.observation_counts(...)
```
### outlier_summary

Qualified name: `macroforecast.data_analysis.summary.outlier_summary`

#### Signature

```python
macroforecast.data_analysis.outlier_summary(data: Any, *, method: OutlierMethod = "iqr", iqr_threshold: float = 10.0, zscore_threshold: float = 3.0) -> pd.DataFrame
```

#### Description

Return per-series outlier counts and rates for one panel.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `Any` | `required` |
| `method` | keyword only | `OutlierMethod` | `"iqr"` |
| `iqr_threshold` | keyword only | `float` | `10.0` |
| `zscore_threshold` | keyword only | `float` | `3.0` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data_analysis.outlier_summary(...)
```
### panel_overview

Qualified name: `macroforecast.data_analysis.summary.panel_overview`

#### Signature

```python
macroforecast.data_analysis.panel_overview(data: Any) -> dict[str, Any]
```

#### Description

Return panel-level shape, date range, frequency, and missingness.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `Any` | `required` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data_analysis.panel_overview(...)
```
### panel_snapshot

Qualified name: `macroforecast.data_analysis.summary.panel_snapshot`

#### Signature

```python
macroforecast.data_analysis.panel_snapshot(data: Any) -> dict[str, Any]
```

#### Description

Return a compact single-panel snapshot for reports and provenance.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `Any` | `required` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data_analysis.panel_snapshot(...)
```
### panel_snapshots

Qualified name: `macroforecast.data_analysis.core.panel_snapshots`

#### Signature

```python
macroforecast.data_analysis.panel_snapshots(raw: pd.DataFrame, clean: pd.DataFrame) -> dict[str, dict[str, Any]]
```

#### Description

Return compact before/after panel snapshots.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `raw` | positional or keyword | `pd.DataFrame` | `required` |
| `clean` | positional or keyword | `pd.DataFrame` | `required` |

#### Returns

`dict[str, dict[str, Any]]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data_analysis.panel_snapshots(...)
```
### adf_test

Qualified name: `macroforecast.data_analysis.summary.adf_test`

#### Signature

```python
macroforecast.data_analysis.adf_test(series: Any, *, regression: str = "c", autolag: str | None = "AIC", alpha: float = 0.05) -> dict[str, Any]
```

#### Description

Augmented Dickey-Fuller unit-root test for a single series.

``regression`` is the deterministic spec ('n','c','ct','ctt'); default 'c'
follows statsmodels (``tseries::adf.test`` defaults to 'ct'). Returns a flat
result dict (the multi-series entry point is ``stationarity_tests``).

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `series` | positional or keyword | `Any` | `required` |
| `regression` | keyword only | `str` | `"c"` |
| `autolag` | keyword only | `str \| None` | `"AIC"` |
| `alpha` | keyword only | `float` | `0.05` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data_analysis.adf_test(...)
```
### kpss_test

Qualified name: `macroforecast.data_analysis.summary.kpss_test`

#### Signature

```python
macroforecast.data_analysis.kpss_test(series: Any, *, regression: str = "c", nlags: Any = "auto", alpha: float = 0.05) -> dict[str, Any]
```

#### Description

KPSS stationarity test for a single series.

``regression='c'`` tests level stationarity (the ``tseries::kpss.test``
'Level' default); ``'ct'`` tests trend stationarity. Returns a flat result
dict (the multi-series entry point is ``stationarity_tests``).

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `series` | positional or keyword | `Any` | `required` |
| `regression` | keyword only | `str` | `"c"` |
| `nlags` | keyword only | `Any` | `"auto"` |
| `alpha` | keyword only | `float` | `0.05` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data_analysis.kpss_test(...)
```
### acf

Qualified name: `macroforecast.data_analysis.summary.acf`

#### Signature

```python
macroforecast.data_analysis.acf(series: Any, *, nlags: int = 20, alpha: float = 0.05, adjusted: bool = False) -> pd.DataFrame
```

#### Description

Sample autocorrelation function (stats::acf / forecast::Acf).

Returns a tidy table with the autocorrelation at lags 0..nlags and the
approximate ``1 - alpha`` confidence band (statsmodels acf). ``adjusted``
selects the n-k (unbiased) divisor instead of the biased 1/n estimator.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `series` | positional or keyword | `Any` | `required` |
| `nlags` | keyword only | `int` | `20` |
| `alpha` | keyword only | `float` | `0.05` |
| `adjusted` | keyword only | `bool` | `False` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data_analysis.acf(...)
```
### johansen_cointegration

Qualified name: `macroforecast.data_analysis.summary.johansen_cointegration`

#### Signature

```python
macroforecast.data_analysis.johansen_cointegration(panel: Any, *, det_order: int = 0, k_ar_diff: int = 1, significance: str = "95") -> dict[str, Any]
```

#### Description

Johansen cointegration test (R urca::ca.jo / statsmodels coint_johansen).

Tests the cointegration rank of a multivariate system via the trace and
maximum-eigenvalue statistics. ``det_order`` is the deterministic term
(-1 none, 0 constant, 1 linear trend); ``k_ar_diff`` the number of lagged
differences in the VECM. Returns, for each null rank r, the trace and
max-eigenvalue statistics with their 90/95/99% critical values, the selected
cointegration rank under each statistic (sequential test at ``significance``),
the eigenvalues, and the estimated cointegrating vectors.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `panel` | positional or keyword | `Any` | `required` |
| `det_order` | keyword only | `int` | `0` |
| `k_ar_diff` | keyword only | `int` | `1` |
| `significance` | keyword only | `str` | `"95"` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data_analysis.johansen_cointegration(...)
```
### engle_granger

Qualified name: `macroforecast.data_analysis.summary.engle_granger`

#### Signature

```python
macroforecast.data_analysis.engle_granger(y: Any, x: Any | None = None, *, trend: str = "c", max_lag: int | None = None, autolag: str | None = "aic", alpha: float = 0.05) -> dict[str, Any]
```

#### Description

Engle-Granger two-step residual-based cointegration test.

R analogue of the two-step Engle-Granger procedure (statsmodels
``tsa.stattools.coint``): regress ``y`` on ``x`` by OLS and apply an
augmented Dickey-Fuller test to the residuals. A rejection indicates the
residuals are stationary, i.e. ``y`` and ``x`` are cointegrated. ``trend`` is
the deterministic term in the cointegrating regression ('c', 'ct', or 'n');
``max_lag``/``autolag`` control the ADF lag length on the residuals.

Pass ``y`` and ``x`` separately, or a single panel whose first numeric column
is the dependent series and the remaining columns the regressors. Returns the
ADF statistic on the residuals, the MacKinnon ``p`` value, the 1/5/10%
critical values, the cointegrating-regression coefficients, and the
cointegration flag at ``alpha``.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `y` | positional or keyword | `Any` | `required` |
| `x` | positional or keyword | `Any \| None` | `None` |
| `trend` | keyword only | `str` | `"c"` |
| `max_lag` | keyword only | `int \| None` | `None` |
| `autolag` | keyword only | `str \| None` | `"aic"` |
| `alpha` | keyword only | `float` | `0.05` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data_analysis.engle_granger(...)
```
### phillips_ouliaris

Qualified name: `macroforecast.data_analysis.summary.phillips_ouliaris`

#### Signature

```python
macroforecast.data_analysis.phillips_ouliaris(y: Any, x: Any | None = None, *, trend: str = "c", test_type: str = "Zt", alpha: float = 0.05) -> dict[str, Any]
```

#### Description

Phillips-Ouliaris residual-based cointegration test.

R analogue of ``urca::ca.po`` / ``tseries::po.test`` (arch
``unitroot.cointegration.phillips_ouliaris``). Like Engle-Granger it tests
the stationarity of the cointegrating-regression residuals, but uses a
non-parametric (long-run-variance corrected) statistic that does not require
choosing an ADF lag length, so it is robust to residual autocorrelation.
``trend`` is the deterministic term ('n', 'c', 'ct', 'ctt'); ``test_type`` is
the statistic variant ('Zt' or 'Za' for the t-/rho-type, 'Pu'/'Pz' for the
variance-ratio forms). Returns the statistic, ``p`` value, 1/5/10% critical
values, the cointegrating vector, and the cointegration flag at ``alpha``.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `y` | positional or keyword | `Any` | `required` |
| `x` | positional or keyword | `Any \| None` | `None` |
| `trend` | keyword only | `str` | `"c"` |
| `test_type` | keyword only | `str` | `"Zt"` |
| `alpha` | keyword only | `float` | `0.05` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data_analysis.phillips_ouliaris(...)
```
### variance_ratio

Qualified name: `macroforecast.data_analysis.summary.variance_ratio`

#### Signature

```python
macroforecast.data_analysis.variance_ratio(series: Any, *, lags: int = 2, trend: str = "c", robust: bool = True, overlap: bool = True, alpha: float = 0.05) -> dict[str, Any]
```

#### Description

Lo-MacKinlay variance-ratio test of the random-walk null.

R analogue of ``vrtest``/``DescTools::VarianceRatioTest`` (arch
``unitroot.VarianceRatio``). Under a random walk the variance of ``lags``-period
returns grows linearly, so the variance ratio is one; values below one signal
mean reversion and above one signal momentum/positive autocorrelation.
``lags`` is the aggregation horizon ``q``; ``robust=True`` uses the
heteroskedasticity-robust standard error. Returns the variance ratio, the
standardised statistic, the ``p`` value, and the random-walk rejection flag at
``alpha`` (rejection means the series is NOT a random walk).

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `series` | positional or keyword | `Any` | `required` |
| `lags` | keyword only | `int` | `2` |
| `trend` | keyword only | `str` | `"c"` |
| `robust` | keyword only | `bool` | `True` |
| `overlap` | keyword only | `bool` | `True` |
| `alpha` | keyword only | `float` | `0.05` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data_analysis.variance_ratio(...)
```
### structural_stability

Qualified name: `macroforecast.data_analysis.summary.structural_stability`

#### Signature

```python
macroforecast.data_analysis.structural_stability(y: Any, x: Any | None = None, *, add_intercept: bool = True, alpha: float = 0.05) -> dict[str, Any]
```

#### Description

OLS-CUSUM test for parameter stability of a regression.

R analogue of ``strucchange::efp(type="OLS-CUSUM")`` / ``vars::stability``.
Fits ``y = X b + e`` by OLS and forms the empirical fluctuation process of the
cumulated standardised residuals, which converges to a Brownian bridge under
the null of constant coefficients. A large maximal fluctuation signals a
structural break. The supremum statistic has the Kolmogorov (sup Brownian
bridge) distribution.

Pass ``y`` and ``x`` separately, or a single panel whose first numeric column
is the dependent series. Returns the supremum statistic, the ``p`` value, the
10/5/1% critical values, the estimated break position (index of maximal
fluctuation), and the stability-rejection flag at ``alpha``.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `y` | positional or keyword | `Any` | `required` |
| `x` | positional or keyword | `Any \| None` | `None` |
| `add_intercept` | keyword only | `bool` | `True` |
| `alpha` | keyword only | `float` | `0.05` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data_analysis.structural_stability(...)
```
### newey_west

Qualified name: `macroforecast.data_analysis.summary.newey_west`

#### Signature

```python
macroforecast.data_analysis.newey_west(X: Any, y: Any | None = None, *, lags: int | str = "auto", add_intercept: bool = True, small_sample: bool = False) -> dict[str, Any]
```

#### Description

Newey-West HAC covariance for an OLS regression.

R analogue of ``sandwich::NeweyWest`` combined with ``lmtest::coeftest``:
fit ``y = X b + e`` by ordinary least squares, then form the
heteroskedasticity- and autocorrelation-consistent (HAC) covariance of the
coefficients using a Bartlett (Newey-West) kernel. ``lags`` is the bandwidth
``L``; ``"auto"`` uses the Newey-West fixed rule ``floor(4 (T/100)^(2/9))``.
With ``small_sample=True`` the meat is scaled by ``T / (T - k)`` (the
finite-sample adjustment used by ``lmtest::coeftest`` defaults).

Returns the coefficient estimates, HAC standard errors, ``t`` statistics,
two-sided ``p`` values (Student-``t`` with ``T - k`` degrees of freedom), the
HAC covariance matrix, the bandwidth, and the regressor names.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `X` | positional or keyword | `Any` | `required` |
| `y` | positional or keyword | `Any \| None` | `None` |
| `lags` | keyword only | `int \| str` | `"auto"` |
| `add_intercept` | keyword only | `bool` | `True` |
| `small_sample` | keyword only | `bool` | `False` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data_analysis.newey_west(...)
```
### vcov_hc

Qualified name: `macroforecast.data_analysis.summary.vcov_hc`

#### Signature

```python
macroforecast.data_analysis.vcov_hc(X: Any, y: Any | None = None, *, cov_type: str = "HC1", add_intercept: bool = True) -> dict[str, Any]
```

#### Description

Heteroskedasticity-consistent (White) covariance for an OLS regression.

R analogue of ``sandwich::vcovHC`` with ``lmtest::coeftest``. Fits
``y = X b + e`` by OLS and forms a robust covariance that is consistent
under heteroskedasticity but assumes no autocorrelation (for serial
correlation use :func:`newey_west`). ``cov_type`` selects the small-sample
weighting of the squared residuals:

- ``"HC0"`` -- White (1980), ``u_i^2``;
- ``"HC1"`` -- MacKinnon-White degrees-of-freedom scaling ``u_i^2 T/(T-k)``;
- ``"HC2"`` -- leverage-adjusted ``u_i^2/(1-h_i)``;
- ``"HC3"`` -- jackknife approximation ``u_i^2/(1-h_i)^2`` (default in R for
  small samples).

Returns the coefficient table (estimate, robust SE, ``t``, two-sided ``p``
with ``T - k`` degrees of freedom), the robust covariance matrix, and the
regressor names.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `X` | positional or keyword | `Any` | `required` |
| `y` | positional or keyword | `Any \| None` | `None` |
| `cov_type` | keyword only | `str` | `"HC1"` |
| `add_intercept` | keyword only | `bool` | `True` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data_analysis.vcov_hc(...)
```
### breusch_pagan_test

Qualified name: `macroforecast.data_analysis.summary.breusch_pagan_test`

#### Signature

```python
macroforecast.data_analysis.breusch_pagan_test(X: Any, y: Any | None = None, *, studentize: bool = True, add_intercept: bool = True) -> dict[str, Any]
```

#### Description

Breusch-Pagan test for heteroskedasticity in an OLS regression.

R analogue of ``lmtest::bptest``. Fits ``y = X b + e`` by OLS and tests the
null of homoskedasticity by an auxiliary regression of the squared residuals
on the regressors. With ``studentize=True`` (the ``lmtest`` default) the
Koenker robust version ``LM = n R^2`` is used, valid without normality; with
``studentize=False`` the classic Breusch-Pagan-Godfrey statistic
``0.5 * explained-SS`` of the scaled squared residuals is returned. Both are
chi-squared with ``p`` degrees of freedom (the number of regressors excluding
the intercept). Returns the statistic, degrees of freedom, ``p`` value, and
the auxiliary-regression R-squared.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `X` | positional or keyword | `Any` | `required` |
| `y` | positional or keyword | `Any \| None` | `None` |
| `studentize` | keyword only | `bool` | `True` |
| `add_intercept` | keyword only | `bool` | `True` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data_analysis.breusch_pagan_test(...)
```
### pacf

Qualified name: `macroforecast.data_analysis.summary.pacf`

#### Signature

```python
macroforecast.data_analysis.pacf(series: Any, *, nlags: int = 20, alpha: float = 0.05, method: str = "ywadjusted") -> pd.DataFrame
```

#### Description

Sample partial autocorrelation function (stats::pacf / forecast::Pacf).

Returns a tidy table with the partial autocorrelation at lags 0..nlags and
the approximate ``1 - alpha`` confidence band (statsmodels pacf). ``method``
is the statsmodels PACF estimator ('ywadjusted','ols','ld', ...).

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `series` | positional or keyword | `Any` | `required` |
| `nlags` | keyword only | `int` | `20` |
| `alpha` | keyword only | `float` | `0.05` |
| `method` | keyword only | `str` | `"ywadjusted"` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data_analysis.pacf(...)
```
### ndiffs

Qualified name: `macroforecast.data_analysis.summary.ndiffs`

#### Signature

```python
macroforecast.data_analysis.ndiffs(series: Any, *, test: str = "kpss", max_d: int = 2, alpha: float = 0.05) -> int
```

#### Description

Number of first differences to make a series stationary (forecast::ndiffs).

Repeatedly applies a unit-root / stationarity test until the differenced
series is judged stationary (KPSS: fail to reject; ADF/PP: reject the unit
root), up to ``max_d``.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `series` | positional or keyword | `Any` | `required` |
| `test` | keyword only | `str` | `"kpss"` |
| `max_d` | keyword only | `int` | `2` |
| `alpha` | keyword only | `float` | `0.05` |

#### Returns

`int`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data_analysis.ndiffs(...)
```
### nsdiffs

Qualified name: `macroforecast.data_analysis.summary.nsdiffs`

#### Signature

```python
macroforecast.data_analysis.nsdiffs(series: Any, *, m: int, max_D: int = 1, threshold: float = 0.64) -> int
```

#### Description

Number of seasonal differences via seasonal strength (forecast::nsdiffs).

Uses the Wang-Smyth-Hyndman seasonal strength F_s = max(0, 1 - Var(remainder)
/ Var(seasonal + remainder)) from an STL decomposition; a seasonal difference
is applied while F_s exceeds ``threshold`` (default 0.64), up to ``max_D``.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `series` | positional or keyword | `Any` | `required` |
| `m` | keyword only | `int` | `required` |
| `max_D` | keyword only | `int` | `1` |
| `threshold` | keyword only | `float` | `0.64` |

#### Returns

`int`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data_analysis.nsdiffs(...)
```
### phillips_perron_test

Qualified name: `macroforecast.data_analysis.summary.phillips_perron_test`

#### Signature

```python
macroforecast.data_analysis.phillips_perron_test(values: Sequence[float] | np.ndarray, *, alpha: float = 0.05) -> dict[str, Any]
```

#### Description

Run the native Phillips-Perron Z_tau unit-root test.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `values` | positional or keyword | `Sequence[float] \| np.ndarray` | `required` |
| `alpha` | keyword only | `float` | `0.05` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data_analysis.phillips_perron_test(...)
```
### dfgls_test

Qualified name: `macroforecast.data_analysis.summary.dfgls_test`

#### Signature

```python
macroforecast.data_analysis.dfgls_test(series: Any, *, trend: str = "c", method: str = "aic", alpha: float = 0.05) -> dict[str, Any]
```

#### Description

Elliott-Rothenberg-Stock DF-GLS unit-root test for a single series.

R analogue of ``urca::ur.ers`` (type ``"DF-GLS"``). The series is locally
GLS-detrended before an augmented Dickey-Fuller regression, giving higher
power than the standard ADF test against persistent stationary alternatives.
``trend`` is the deterministic specification, ``'c'`` (demeaned) or ``'ct'``
(de-trended); ``method`` selects the lag length ('aic', 'bic', or 't-stat').
Returns the statistic, the MacKinnon ``p`` value, the selected lag, the 1/5/10%
critical values, and the unit-root rejection flag at ``alpha``.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `series` | positional or keyword | `Any` | `required` |
| `trend` | keyword only | `str` | `"c"` |
| `method` | keyword only | `str` | `"aic"` |
| `alpha` | keyword only | `float` | `0.05` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data_analysis.dfgls_test(...)
```
### zivot_andrews_test

Qualified name: `macroforecast.data_analysis.summary.zivot_andrews_test`

#### Signature

```python
macroforecast.data_analysis.zivot_andrews_test(series: Any, *, regression: str = "c", trim: float = 0.15, maxlag: int | None = None, autolag: str | None = "AIC", alpha: float = 0.05) -> dict[str, Any]
```

#### Description

Zivot-Andrews unit-root test allowing one endogenous structural break.

R analogue of ``urca::ur.za``. Tests the unit-root null against a
trend-stationary alternative with a single break whose date is chosen
endogenously to be least favourable to the null. ``regression`` places the
break in the intercept ('c'), the trend ('t'), or both ('ct'); ``trim`` is
the fraction of the sample excluded at each end when searching for the break.
Returns the minimised statistic, the ``p`` value, the 1/5/10% critical
values, the selected lag, the estimated break position (index and label), and
the unit-root rejection flag at ``alpha``.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `series` | positional or keyword | `Any` | `required` |
| `regression` | keyword only | `str` | `"c"` |
| `trim` | keyword only | `float` | `0.15` |
| `maxlag` | keyword only | `int \| None` | `None` |
| `autolag` | keyword only | `str \| None` | `"AIC"` |
| `alpha` | keyword only | `float` | `0.05` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data_analysis.zivot_andrews_test(...)
```
### sample_coverage

Qualified name: `macroforecast.data_analysis.summary.sample_coverage`

#### Signature

```python
macroforecast.data_analysis.sample_coverage(data: Any) -> pd.DataFrame
```

#### Description

Return per-series sample start, end, observation count, and missingness.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `Any` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data_analysis.sample_coverage(...)
```
### stationarity_tests

Qualified name: `macroforecast.data_analysis.summary.stationarity_tests`

#### Signature

```python
macroforecast.data_analysis.stationarity_tests(data: Any, *, test: StationarityTest = "multi", scope: StationarityScope = "all", target: str | None = None, targets: Sequence[str] | None = None, alpha: float = 0.05, adf_regression: str = "c") -> dict[str, Any]
```

#### Description

Run ADF, Phillips-Perron, KPSS, or all three on one panel.

``adf_regression`` is the ADF deterministic specification passed to
statsmodels ``adfuller`` ('n', 'c', 'ct', 'ctt'); the default 'c' (constant
only) follows statsmodels. Note ``tseries::adf.test`` instead defaults to 'ct'
(constant + linear trend) with a fixed lag, so pass ``adf_regression='ct'`` to
reproduce that reference.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `Any` | `required` |
| `test` | keyword only | `StationarityTest` | `"multi"` |
| `scope` | keyword only | `StationarityScope` | `"all"` |
| `target` | keyword only | `str \| None` | `None` |
| `targets` | keyword only | `Sequence[str] \| None` | `None` |
| `alpha` | keyword only | `float` | `0.05` |
| `adf_regression` | keyword only | `str` | `"c"` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data_analysis.stationarity_tests(...)
```
### summarize_data

Qualified name: `macroforecast.data_analysis.summary.summarize_data`

#### Signature

```python
macroforecast.data_analysis.summarize_data(data: Any, *, metrics: Sequence[SummaryMetric] | None = None, include_correlation: bool = False, correlation_method: CorrelationMethod = "pearson", include_outliers: bool = False, outlier_method: OutlierMethod = "iqr", include_stationarity: bool = False, stationarity_test: StationarityTest = "multi", stationarity_scope: StationarityScope = "all") -> DataSummaryReport
```

#### Description

Run the standard single-panel summary suite.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `Any` | `required` |
| `metrics` | keyword only | `Sequence[SummaryMetric] \| None` | `None` |
| `include_correlation` | keyword only | `bool` | `False` |
| `correlation_method` | keyword only | `CorrelationMethod` | `"pearson"` |
| `include_outliers` | keyword only | `bool` | `False` |
| `outlier_method` | keyword only | `OutlierMethod` | `"iqr"` |
| `include_stationarity` | keyword only | `bool` | `False` |
| `stationarity_test` | keyword only | `StationarityTest` | `"multi"` |
| `stationarity_scope` | keyword only | `StationarityScope` | `"all"` |

#### Returns

`DataSummaryReport`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data_analysis.summarize_data(...)
```
### univariate_summary

Qualified name: `macroforecast.data_analysis.summary.univariate_summary`

#### Signature

```python
macroforecast.data_analysis.univariate_summary(data: Any, *, metrics: Sequence[SummaryMetric] | None = None) -> pd.DataFrame
```

#### Description

Return per-series descriptive statistics for numeric panel columns.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `Any` | `required` |
| `metrics` | keyword only | `Sequence[SummaryMetric] \| None` | `None` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data_analysis.univariate_summary(...)
```
