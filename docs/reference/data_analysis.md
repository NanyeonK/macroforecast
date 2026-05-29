# macroforecast.data_analysis

[Back to reference](index.md)

`macroforecast.data_analysis` compares two canonical pandas panels. The main
use case is raw data versus the panel returned by
`macroforecast.preprocessing.reprocess(...)`.

Use `macroforecast.data_summary` for a one-panel summary. Use
`macroforecast.preprocessing.report()` for the preprocessing execution log.

## Public Flow

```python
import macroforecast as mf

bundle = mf.data.load_fred_md()
spec = mf.data.spec(bundle, target="INDPRO", horizons=[1, 3, 6, 12])
processed = mf.preprocessing.reprocess(spec)

analysis = mf.data_analysis.analyze_data(
    spec.panel,
    processed.panel,
    include_correlation=True,
)
```

## analyze_data

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

| Name | Type | Default | Choices |
| --- | --- | --- | --- |
| `raw` | `pandas.DataFrame` | required | Canonical raw date-indexed panel. |
| `clean` | `pandas.DataFrame` | required | Canonical processed date-indexed panel. |
| `distribution_metrics` | sequence or `None` | all defaults | Any of `mean_change`, `sd_change`, `sd_ratio`, `skew_change`, `kurtosis_change`, `ks_statistic`. |
| `include_correlation` | `bool` | `False` | Whether to include cleaned-minus-raw correlations. |
| `correlation_method` | `str` | `"pearson"` | `"pearson"`, `"spearman"`, `"kendall"`. |
| `sample` | `str` | `"common_index"` | `"common_index"` compares distributions/correlations on dates present in both panels; `"full"` uses each full panel. |
| `cleaning_metadata` and related kwargs | mappings/counts or `None` | auto from `clean.attrs["macroforecast_metadata"]["preprocessing"]` when available | Optional preprocessing metadata to normalize into `cleaning_effect_summary`. |
| `tolerance` | `float` | `0.0` | Absolute tolerance for changed-cell counting. |

### Output

Returns `DataAnalysisReport`.

| Field | Type | Meaning |
| --- | --- | --- |
| `comparison` | `dict` | Shape, date range, common columns/index, missing totals, changed-cell count. |
| `missing_shift` | `pandas.DataFrame` | Per-column raw/clean missing counts and rate changes. |
| `distribution_shift` | `pandas.DataFrame` | Per-column distribution changes for common numeric columns. |
| `correlation_shift` | `pandas.DataFrame` or `None` | Cleaned-minus-raw correlation matrix when requested. |
| `cleaning_effect_summary` | `dict` | Normalized preprocessing counters, transform map, cleaning log, and column metadata. |
| `metadata` | `dict` | Input metadata plus a compact `data_analysis` stage with before/after snapshots. |

`DataAnalysisReport.to_dict()` converts DataFrame fields into nested
dictionaries for serialization.

### Metadata

`analyze_data(...)` stores raw and post-preprocessing snapshots under one short
stage key:

```python
analysis.metadata["data_analysis"]["before"]
analysis.metadata["data_analysis"]["after"]
```

The snapshots are intentionally compact. Full comparison tables remain in
`comparison`, `missing_shift`, `distribution_shift`, and `correlation_shift`.

| Key | Meaning |
| --- | --- |
| `before` | Raw panel snapshot: rows, columns, start, end, missing count. |
| `after` | Processed panel snapshot with the same fields. |
| `common` | Common row/column counts and changed-cell count. |
| `options` | Distribution metrics, correlation option, and changed-cell tolerance. |
| `effects` | Compact preprocessing counters: imputed cells, outliers, truncated observations, transform-code count, column-metadata count, cleaning-log presence. |
| `metadata_keys` | Metadata keys detected on raw and processed panels. |

### Comparison Sample

`distribution_shift` and `correlation_shift` default to `sample="common_index"`.
This avoids mixing distribution changes with rows that only exist before or
after preprocessing. Set `sample="full"` when you deliberately want each panel's
full available sample.

`ks_statistic` is the two-sample KS statistic only; it does not compute a p-value.

## Helper Functions

| Function | Input | Output | Purpose |
| --- | --- | --- | --- |
| `compare_panels(raw, clean, tolerance=...)` | two DataFrames | `dict` | Panel shape/date/column/index comparison and changed-cell count. |
| `missing_shift(raw, clean)` | two DataFrames | `DataFrame` | Missing-count and missing-rate changes with `column_status` (`common`, `raw_only`, `clean_only`). |
| `distribution_shift(raw, clean, metrics=..., sample=...)` | two DataFrames | `DataFrame` | Mean, variance, tail-shape, and KS-style distribution changes. |
| `correlation_shift(raw, clean, method=..., sample=...)` | two DataFrames | `DataFrame` | Cleaned-minus-raw correlation matrix. |
| `cleaning_effect_summary(...)` | metadata/counters | `dict` | Normalize preprocessing effects into one compact object. |

## Boundary

| Question | Use |
| --- | --- |
| Describe one raw or processed panel | `mf.data_summary.summarize_data(panel)` |
| Compare raw and processed panels | `mf.data_analysis.analyze_data(raw, processed)` |
| Report preprocessing choices and ordered steps | `mf.preprocessing.report(processed)` |
