# macroforecast.reporting

[Back to reference](index.md)

`macroforecast.reporting` is separate from `macroforecast.output`.

| Module | Owns | Does not own |
| --- | --- | --- |
| `macroforecast.output` | Generate output tables/JSON summaries and write artifacts. | Paper/table presentation style. |
| `macroforecast.reporting` | Format tables, render LaTeX/HTML/Markdown, create figure-ready data, and produce paper-facing metric/test tables. | Model fitting, evaluation, testing, or artifact writing. |

The reporting functions are callable and in-memory. They do not write files.
Call them through the namespace, for example `mf.reporting.report_table(...)`.
Reporting helpers are not exported as top-level shortcuts.

## report_table

```python
macroforecast.reporting.report_table(
    table,
    *,
    columns=None,
    rename=None,
    sort_by=None,
    ascending=True,
    index=False,
    precision=3,
    percent_columns=(),
    missing="",
    caption=None,
    label=None,
    notes=(),
    metadata=None,
) -> ReportTable
```

Creates a presentation-ready table from a pandas-like object.

| Argument | Default | Meaning |
| --- | --- | --- |
| `columns` | `None` | Optional selected column order. Missing columns raise `KeyError`. |
| `rename` | `None` | Mapping from source names to display names. |
| `sort_by` | `None` | Optional sort column or columns. |
| `ascending` | `True` | Sort direction. |
| `index` | `False` | Whether to include the original index as columns. |
| `precision` | `3` | Digits after the decimal for numeric values. |
| `percent_columns` | `()` | Display columns as percentages after renaming. |
| `missing` | `""` | Display value for missing or non-finite numeric values. |
| `caption`, `label`, `notes` | `None`, `None`, `()` | Presentation metadata for rendered output. |
| `metadata` | `None` | User metadata stored on the `ReportTable`. |

Output: `ReportTable(data, caption, label, notes, metadata)`.

`ReportTable.data` is a formatted `DataFrame` and carries
`attrs["macroforecast_metadata_schema"]["kind"] = "report_table"`.
Formatting metadata such as source shape, precision, and percentage columns is
stored separately in `attrs["macroforecast_metadata"]`.

## metric_report_table

```python
macroforecast.reporting.metric_report_table(
    results,
    *,
    table="scores",
    columns=None,
    rename=None,
    sort_by=None,
    ascending=True,
    precision=3,
    percent_columns=(),
    missing="",
    caption=None,
    label=None,
    notes=(),
    metadata=None,
) -> ReportTable
```

Creates a paper-facing metric table from an `EvaluationReport` or a raw
metric-like `DataFrame`. For an `EvaluationReport`, `table` selects which
component to display:

| `table` | Source |
| --- | --- |
| `"scores"` | `report.scores` |
| `"ranking"` | `report.ranking` |
| `"benchmark"` | `report.benchmark` |
| `"regime"` | `report.regime` |
| `"decomposition"` | `report.decomposition` |

Default display labels convert common columns such as `model`, `horizon`,
`rmse`, `relative_mse`, and `r2_oos` to `Model`, `H`, `RMSE`,
`Relative MSE`, and `R2 OOS`. Use `columns` to choose the paper table spine,
`rename` to override labels, and `percent_columns` when a column should be
displayed as a percentage.

## evaluation_report_tables

```python
macroforecast.reporting.evaluation_report_tables(
    report,
    *,
    include=("scores", "ranking", "benchmark", "regime", "decomposition"),
    include_aggregations=False,
    captions=None,
    labels=None,
    precision=3,
    percent_columns=(),
    missing="",
    metadata=None,
) -> ReportBundle
```

Creates a named `ReportBundle` from an `EvaluationReport`. Unavailable optional
tables are skipped. Set `include_aggregations=True` or include `"aggregations"`
in `include` to add all `report.aggregations` under names like
`aggregation_model_horizon_target`.

## test_report_table

```python
macroforecast.reporting.test_report_table(
    results,
    *,
    columns=None,
    include_reference=False,
    stars=True,
    star_levels=((0.01, "***"), (0.05, "**"), (0.1, "*")),
    precision=3,
    p_value_precision=3,
    missing="",
    caption=None,
    label=None,
    notes=(),
    metadata=None,
) -> ReportTable
```

Creates a paper-facing table from one or more forecast-test outputs. Input can
be a `TestResult`, a mapping or sequence of `TestResult` objects, a raw
`macroforecast.output.test_table(...)`, or a stacked table from functions such
as `macroforecast.tests.equal_predictive_tests(...)`.

Default output columns:

| Column | Meaning |
| --- | --- |
| `Test` | Requested key, such as `dm`, `gw`, `dmp`, or `hn`. |
| `Name` | Display name from component metadata. |
| `Ref.` | Statistic reference family, such as `t` or `z`. |
| `Statistic` | Formatted test statistic. |
| `p-value` | Formatted p-value, optionally with significance markers. |
| `Reject` | `Yes` or `No` based on the component decision flag. |
| `N` | Number of aligned observations used by the component test. |

Set `include_reference=True` to add a compact `Source` column. For detailed
R/source alignment, use `test_provenance_table(...)` instead of placing long
source text in the main paper table.

## test_provenance_table

```python
macroforecast.reporting.test_provenance_table(
    results,
    *,
    columns=None,
    missing="",
    caption=None,
    label=None,
    notes=(),
    metadata=None,
) -> ReportTable
```

Creates a source-alignment table for appendices, replication packages, or audit
logs. It keeps the null hypothesis, p-value reference distribution, source
reference, R reference, and alignment note separate from the compact paper
table.

Default output columns:

| Column | Meaning |
| --- | --- |
| `Test`, `Name` | Requested key and display name. |
| `Null` | Null hypothesis recorded by the component test. |
| `P-value ref.` | Reference distribution for the p-value. |
| `Status` | P-value availability flag. |
| `Source` | Package/source formula reference. |
| `R reference` | Exact R comparator when one is claimed; blank otherwise. |
| `Alignment` | Text description of exact, partial, or unavailable source alignment. |

## Renderers

```python
macroforecast.reporting.latex_table(table, *, booktabs=True) -> str
macroforecast.reporting.html_table(table) -> str
macroforecast.reporting.markdown_table(table) -> str
```

Each renderer accepts either a `ReportTable` or a raw table-like object. Raw
objects are passed through `report_table()` first.

`latex_table(..., booktabs=True)` uses `\toprule`, `\midrule`, and
`\bottomrule`. Set `booktabs=False` to use `\hline`.

## figure_data

```python
macroforecast.reporting.figure_data(
    data,
    *,
    x=None,
    y=None,
    group=None,
    columns=None,
    rename=None,
    dropna=True,
) -> pandas.DataFrame
```

Creates a tidy plotting/export frame. Use `columns` for an explicit selected
column set, or use `x`, `y`, and `group` to select plot roles.

Output: `DataFrame` with
`attrs["macroforecast_metadata_schema"]["kind"] = "figure_data"`.
Figure-role metadata such as `x`, `y`, `group`, `dropna`, and source shape is
stored in `attrs["macroforecast_metadata"]`.

## report_bundle

```python
macroforecast.reporting.report_bundle(
    *,
    tables=None,
    figures=None,
    metadata=None,
) -> ReportBundle
```

Collects named report tables and figure data without writing files.

| Field | Meaning |
| --- | --- |
| `tables` | Mapping from name to `ReportTable`. Raw table-like objects are converted with `report_table()`. |
| `figures` | Mapping from name to figure-ready `DataFrame`. Raw objects are converted with `figure_data()`. |
| `metadata` | Bundle-level metadata. |

## render_tables

```python
macroforecast.reporting.render_tables(bundle, *, format="latex") -> dict[str, str]
```

Renders every table in a `ReportBundle` or table mapping. Supported formats:
`"latex"`, `"html"`, and `"markdown"`.

## Example

```python
report = mf.reporting.report_table(
    scores,
    columns=("model", "horizon", "rmse", "r2_oos"),
    rename={"model": "Model", "rmse": "RMSE", "r2_oos": "R2 OOS"},
    sort_by="rmse",
    precision=3,
    percent_columns=("R2 OOS",),
    caption="Forecast accuracy",
    label="tab:forecast_accuracy",
    notes=("Lower RMSE is better.",),
)

latex = report.to_latex()
```

Evaluation reporting example:

```python
report = mf.evaluation.evaluate_report(
    forecast_result,
    metrics=("rmse", "mae", "relative_mse", "r2_oos"),
    benchmark_model="historical_mean",
    include_decomposition=True,
)

accuracy_table = mf.reporting.metric_report_table(
    report,
    columns=("model", "horizon", "rmse", "r2_oos"),
    percent_columns=("r2_oos",),
    caption="Forecast accuracy",
)

paper_tables = mf.reporting.evaluation_report_tables(
    report,
    include=("scores", "ranking", "benchmark", "decomposition"),
    percent_columns=("r2_oos",),
)
```

Forecast-test reporting example:

```python
tests = mf.tests.equal_predictive_tests(
    loss_a,
    loss_b,
    tests=("dm", "gw", "dmp", "hn"),
    error_a=error_a,
    error_b=error_b,
)

main_table = mf.reporting.test_report_table(
    tests,
    caption="Equal predictive ability tests",
    label="tab:equal_predictive_tests",
)
appendix_table = mf.reporting.test_provenance_table(tests)

latex = main_table.to_latex()
```
