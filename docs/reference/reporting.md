# macroforecast.reporting

[Back to reference](index.md)

`macroforecast.reporting` is separate from `macroforecast.output`.

| Module | Owns | Does not own |
| --- | --- | --- |
| `macroforecast.output` | Generate output tables/JSON summaries and write artifacts. | Paper/table presentation style. |
| `macroforecast.reporting` | Format tables, render LaTeX/HTML/Markdown, and create figure-ready data. | Model fitting, evaluation, testing, or artifact writing. |

The reporting functions are callable and in-memory. They do not write files.

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
