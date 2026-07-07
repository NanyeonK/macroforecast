# macroforecast.reporting

[Back to reference](index.md)

Markdown, HTML, and LaTeX report-table rendering.

## Public Symbols

| Symbol | Kind | Summary |
| --- | --- | --- |
| `ReportBundle` | class | Named in-memory report outputs. |
| `ReportTable` | class | Presentation-ready table that has not been written to disk. |
| `accuracy_table` | function | Return the default paper-facing forecast accuracy table. |
| `evaluation_report_tables` | function | Return named paper-facing tables from an evaluation report. |
| `figure_data` | function | Return a tidy frame intended for plotting or figure export. |
| `forecast_test_table` | function | Return the default paper-facing forecast-comparison test table. |
| `html_table` | function | Render a report table as HTML text. |
| `latex_table` | function | Render a report table as LaTeX text. |
| `markdown_table` | function | Render a report table as GitHub-flavored Markdown text. |
| `metric_report_table` | function | Return a paper-facing metric/evaluation table. |
| `model_comparison_table` | function | Return the default paper-facing model ranking/comparison table. |
| `paper_accuracy_table` | function | One line from a ``PipelineReport`` to a referee-ready horse-race table. |
| `render_tables` | function | Render all tables in a bundle or mapping. |
| `report_bundle` | function | Collect named reporting tables and figure data without writing files. |
| `report_table` | function | Return a presentation-ready table without writing files. |
| `test_provenance_table` | function | Return a source-alignment table for forecast-test outputs. |
| `test_report_table` | function | Return a paper-facing forecast-test result table. |

## Callable And Class Reference

### ReportBundle

Qualified name: `macroforecast.reporting.core.ReportBundle`

#### Signature

```python
macroforecast.reporting.ReportBundle(tables: dict[str, ReportTable] = <factory>, figures: dict[str, pd.DataFrame] = <factory>, metadata: dict[str, Any] = <factory>, metadata_schema: dict[str, Any] = <factory>) -> None
```

#### Description

Named in-memory report outputs.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `tables` | positional or keyword | `dict[str, ReportTable]` | `<factory>` |
| `figures` | positional or keyword | `dict[str, pd.DataFrame]` | `<factory>` |
| `metadata` | positional or keyword | `dict[str, Any]` | `<factory>` |
| `metadata_schema` | positional or keyword | `dict[str, Any]` | `<factory>` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.reporting.ReportBundle(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `tables` | `dict[str, ReportTable]` | `default_factory` |
| `figures` | `dict[str, pd.DataFrame]` | `default_factory` |
| `metadata` | `dict[str, Any]` | `default_factory` |
| `metadata_schema` | `dict[str, Any]` | `default_factory` |

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `render` | `render(self, *, format: RenderFormat = "latex") -> dict[str, str]` | No public docstring is available. |
| `to_dict` | `to_dict(self) -> dict[str, Any]` | No public docstring is available. |
### ReportTable

Qualified name: `macroforecast.reporting.core.ReportTable`

#### Signature

```python
macroforecast.reporting.ReportTable(data: pd.DataFrame, caption: str | None = None, label: str | None = None, notes: tuple[str, ...] = (), metadata: dict[str, Any] = <factory>, metadata_schema: dict[str, Any] = <factory>) -> None
```

#### Description

Presentation-ready table that has not been written to disk.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `pd.DataFrame` | `required` |
| `caption` | positional or keyword | `str \| None` | `None` |
| `label` | positional or keyword | `str \| None` | `None` |
| `notes` | positional or keyword | `tuple[str, ...]` | `()` |
| `metadata` | positional or keyword | `dict[str, Any]` | `<factory>` |
| `metadata_schema` | positional or keyword | `dict[str, Any]` | `<factory>` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.reporting.ReportTable(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `data` | `pd.DataFrame` | `required` |
| `caption` | `str \| None` | `None` |
| `label` | `str \| None` | `None` |
| `notes` | `tuple[str, ...]` | `()` |
| `metadata` | `dict[str, Any]` | `default_factory` |
| `metadata_schema` | `dict[str, Any]` | `default_factory` |

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `to_dict` | `to_dict(self) -> dict[str, Any]` | No public docstring is available. |
| `to_html` | `to_html(self) -> str` | No public docstring is available. |
| `to_latex` | `to_latex(self, *, booktabs: bool = True) -> str` | No public docstring is available. |
| `to_markdown` | `to_markdown(self) -> str` | No public docstring is available. |
### accuracy_table

Qualified name: `macroforecast.reporting.core.accuracy_table`

#### Signature

```python
macroforecast.reporting.accuracy_table(results: Any, *, columns: Sequence[str] | None = None, sort_by: str | Sequence[str] | None = ('horizon', 'rmse'), ascending: bool | Sequence[bool] = True, precision: int = 3, percent_columns: Sequence[str] = ('r2_oos', 'mse_reduction', 'success_ratio'), missing: str = "", caption: str | None = "Forecast accuracy", label: str | None = None, notes: Sequence[str] = ('Lower error metrics and higher R2 OOS are better.',), metadata: Mapping[str, Any] | None = None) -> ReportTable
```

#### Description

Return the default paper-facing forecast accuracy table.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `results` | positional or keyword | `Any` | `required` |
| `columns` | keyword only | `Sequence[str] \| None` | `None` |
| `sort_by` | keyword only | `str \| Sequence[str] \| None` | `("horizon", "rmse")` |
| `ascending` | keyword only | `bool \| Sequence[bool]` | `True` |
| `precision` | keyword only | `int` | `3` |
| `percent_columns` | keyword only | `Sequence[str]` | `("r2_oos", "mse_reduction", "success_ratio")` |
| `missing` | keyword only | `str` | `""` |
| `caption` | keyword only | `str \| None` | `"Forecast accuracy"` |
| `label` | keyword only | `str \| None` | `None` |
| `notes` | keyword only | `Sequence[str]` | `("Lower error metrics and higher R2 OOS are better.",)` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |

#### Returns

`ReportTable`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.reporting.accuracy_table(...)
```
### evaluation_report_tables

Qualified name: `macroforecast.reporting.core.evaluation_report_tables`

#### Signature

```python
macroforecast.reporting.evaluation_report_tables(report: Any, *, include: Sequence[str] = ('scores', 'ranking', 'benchmark', 'regime', 'decomposition'), include_aggregations: bool = False, captions: Mapping[str, str] | None = None, labels: Mapping[str, str] | None = None, precision: int = 3, percent_columns: Sequence[str] = (), missing: str = "", metadata: Mapping[str, Any] | None = None) -> ReportBundle
```

#### Description

Return named paper-facing tables from an evaluation report.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `report` | positional or keyword | `Any` | `required` |
| `include` | keyword only | `Sequence[str]` | `("scores", "ranking", "benchmark", "regime", "decomposition")` |
| `include_aggregations` | keyword only | `bool` | `False` |
| `captions` | keyword only | `Mapping[str, str] \| None` | `None` |
| `labels` | keyword only | `Mapping[str, str] \| None` | `None` |
| `precision` | keyword only | `int` | `3` |
| `percent_columns` | keyword only | `Sequence[str]` | `()` |
| `missing` | keyword only | `str` | `""` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |

#### Returns

`ReportBundle`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.reporting.evaluation_report_tables(...)
```
### figure_data

Qualified name: `macroforecast.reporting.core.figure_data`

#### Signature

```python
macroforecast.reporting.figure_data(data: Any, *, x: str | None = None, y: str | Sequence[str] | None = None, group: str | None = None, columns: Sequence[str] | None = None, rename: Mapping[str, str] | None = None, dropna: bool = True) -> pd.DataFrame
```

#### Description

Return a tidy frame intended for plotting or figure export.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `data` | positional or keyword | `Any` | `required` |
| `x` | keyword only | `str \| None` | `None` |
| `y` | keyword only | `str \| Sequence[str] \| None` | `None` |
| `group` | keyword only | `str \| None` | `None` |
| `columns` | keyword only | `Sequence[str] \| None` | `None` |
| `rename` | keyword only | `Mapping[str, str] \| None` | `None` |
| `dropna` | keyword only | `bool` | `True` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.reporting.figure_data(...)
```
### forecast_test_table

Qualified name: `macroforecast.reporting.core.forecast_test_table`

#### Signature

```python
macroforecast.reporting.forecast_test_table(results: Any, *, columns: Sequence[str] | None = None, include_reference: bool = False, stars: bool = True, star_levels: PValueStarLevels = ((0.01, '***'), (0.05, '**'), (0.1, '*')), precision: int = 3, p_value_precision: int = 3, missing: str = "", caption: str | None = "Forecast comparison tests", label: str | None = None, notes: Sequence[str] = (), metadata: Mapping[str, Any] | None = None) -> ReportTable
```

#### Description

Return the default paper-facing forecast-comparison test table.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `results` | positional or keyword | `Any` | `required` |
| `columns` | keyword only | `Sequence[str] \| None` | `None` |
| `include_reference` | keyword only | `bool` | `False` |
| `stars` | keyword only | `bool` | `True` |
| `star_levels` | keyword only | `PValueStarLevels` | `((0.01, "***"), (0.05, "**"), (0.1, "*"))` |
| `precision` | keyword only | `int` | `3` |
| `p_value_precision` | keyword only | `int` | `3` |
| `missing` | keyword only | `str` | `""` |
| `caption` | keyword only | `str \| None` | `"Forecast comparison tests"` |
| `label` | keyword only | `str \| None` | `None` |
| `notes` | keyword only | `Sequence[str]` | `()` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |

#### Returns

`ReportTable`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.reporting.forecast_test_table(...)
```
### html_table

Qualified name: `macroforecast.reporting.core.html_table`

#### Signature

```python
macroforecast.reporting.html_table(value: ReportTable | Any, **kwargs: Any) -> str
```

#### Description

Render a report table as HTML text.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `value` | positional or keyword | `ReportTable \| Any` | `required` |
| `kwargs` | var keyword | `Any` | `required` |

#### Returns

`str`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.reporting.html_table(...)
```
### latex_table

Qualified name: `macroforecast.reporting.core.latex_table`

#### Signature

```python
macroforecast.reporting.latex_table(value: ReportTable | Any, *, booktabs: bool = True, **kwargs: Any) -> str
```

#### Description

Render a report table as LaTeX text.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `value` | positional or keyword | `ReportTable \| Any` | `required` |
| `booktabs` | keyword only | `bool` | `True` |
| `kwargs` | var keyword | `Any` | `required` |

#### Returns

`str`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.reporting.latex_table(...)
```
### markdown_table

Qualified name: `macroforecast.reporting.core.markdown_table`

#### Signature

```python
macroforecast.reporting.markdown_table(value: ReportTable | Any, **kwargs: Any) -> str
```

#### Description

Render a report table as GitHub-flavored Markdown text.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `value` | positional or keyword | `ReportTable \| Any` | `required` |
| `kwargs` | var keyword | `Any` | `required` |

#### Returns

`str`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.reporting.markdown_table(...)
```
### metric_report_table

Qualified name: `macroforecast.reporting.core.metric_report_table`

#### Signature

```python
macroforecast.reporting.metric_report_table(results: Any, *, table: EvaluationTableName = "scores", columns: Sequence[str] | None = None, rename: Mapping[str, str] | None = None, sort_by: str | Sequence[str] | None = None, ascending: bool | Sequence[bool] = True, precision: int = 3, percent_columns: Sequence[str] = (), missing: str = "", caption: str | None = None, label: str | None = None, notes: Sequence[str] = (), metadata: Mapping[str, Any] | None = None) -> ReportTable
```

#### Description

Return a paper-facing metric/evaluation table.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `results` | positional or keyword | `Any` | `required` |
| `table` | keyword only | `EvaluationTableName` | `"scores"` |
| `columns` | keyword only | `Sequence[str] \| None` | `None` |
| `rename` | keyword only | `Mapping[str, str] \| None` | `None` |
| `sort_by` | keyword only | `str \| Sequence[str] \| None` | `None` |
| `ascending` | keyword only | `bool \| Sequence[bool]` | `True` |
| `precision` | keyword only | `int` | `3` |
| `percent_columns` | keyword only | `Sequence[str]` | `()` |
| `missing` | keyword only | `str` | `""` |
| `caption` | keyword only | `str \| None` | `None` |
| `label` | keyword only | `str \| None` | `None` |
| `notes` | keyword only | `Sequence[str]` | `()` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |

#### Returns

`ReportTable`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.reporting.metric_report_table(...)
```
### model_comparison_table

Qualified name: `macroforecast.reporting.core.model_comparison_table`

#### Signature

```python
macroforecast.reporting.model_comparison_table(results: Any, *, columns: Sequence[str] | None = None, sort_by: str | Sequence[str] | None = ('horizon', 'rank'), ascending: bool | Sequence[bool] = True, precision: int = 3, percent_columns: Sequence[str] = ('r2_oos',), missing: str = "", caption: str | None = "Model comparison", label: str | None = None, notes: Sequence[str] = ('Ranks are computed by the evaluation report rank metric.',), metadata: Mapping[str, Any] | None = None) -> ReportTable
```

#### Description

Return the default paper-facing model ranking/comparison table.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `results` | positional or keyword | `Any` | `required` |
| `columns` | keyword only | `Sequence[str] \| None` | `None` |
| `sort_by` | keyword only | `str \| Sequence[str] \| None` | `("horizon", "rank")` |
| `ascending` | keyword only | `bool \| Sequence[bool]` | `True` |
| `precision` | keyword only | `int` | `3` |
| `percent_columns` | keyword only | `Sequence[str]` | `("r2_oos",)` |
| `missing` | keyword only | `str` | `""` |
| `caption` | keyword only | `str \| None` | `"Model comparison"` |
| `label` | keyword only | `str \| None` | `None` |
| `notes` | keyword only | `Sequence[str]` | `("Ranks are computed by the evaluation report rank metric.",)` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |

#### Returns

`ReportTable`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.reporting.model_comparison_table(...)
```
### paper_accuracy_table

Qualified name: `macroforecast.reporting.core.paper_accuracy_table`

#### Signature

```python
macroforecast.reporting.paper_accuracy_table(report: Any, *, target: str | None = None, metric: str = "rel_rmse", star_levels: PValueStarLevels = ((0.01, '***'), (0.05, '**'), (0.1, '*')), mcs_mark: str = "†", benchmark_row: bool = True, precision: int = 3, missing: str = "--", caption: str | None = None, label: str | None = None, notes: Sequence[str] = (), metadata: Mapping[str, Any] | None = None) -> ReportTable | dict[str, ReportTable]
```

#### Description

One line from a ``PipelineReport`` to a referee-ready horse-race table.

``run_pipeline`` returns three separate long frames -- ``.accuracy`` (RMSE /
relative MSE / R2 OOS per target/horizon/contender), ``.significance`` (DM/CW
p-values per target/horizon/contender), and ``.mcs`` (Model Confidence Set
membership per target/horizon/contender). Nothing joined them into the wide
models-by-horizons table that is "Table 3" of almost every macro forecasting
paper -- rel-RMSE with significance stars and an MCS marker, one row per
model, one column per horizon. This does that join.

Accepts a ``PipelineReport`` (or any object/mapping exposing ``accuracy``,
and optionally ``significance``/``mcs``, with the same column contract as
``macroforecast.pipeline.evaluate``'s output).

``metric``: the accuracy column to display. ``"rel_rmse"`` (the default) is
computed here as ``sqrt(relative_mse)`` -- rel-RMSE is a common paper
convention that is not itself a column of the accuracy frame. Any other
column already on ``report.accuracy`` (``"relative_mse"``, ``"rmse"``,
``"r2_oos"``, ...) may be passed instead and is used as-is.

DM significance stars are joined from ``report.significance`` on
``(target, horizon, contender)`` and rendered with ``star_levels``
(``(threshold, marker)`` pairs, smallest threshold first); the benchmark row
itself is never starred (DM never compares a contender to itself). The MCS
marker (``mcs_mark``) is appended wherever ``report.mcs.in_mcs`` is True for
that ``(target, horizon, contender)``. Missing significance/MCS frames (or
missing rows within them) simply contribute no stars/marker rather than
raising, since not every ``PipelineReport`` carries them.

``benchmark_row=False`` drops the benchmark's own row (always rel-RMSE
1.000 by construction, since it is scored against itself).

Multi-target reports: with ``target=None`` and more than one distinct
target in ``report.accuracy``, this returns ``dict[target, ReportTable]``
rather than stacking targets into one frame. A ``ReportTable`` is one flat
2-D presentation grid (see ``ReportTable.to_latex()``); different targets
can have different available horizons and a different benchmark-present
status, so folding them into one MultiIndex-columned or MultiIndex-rowed
frame would either force a ragged union of horizons or break the plain
"one line to ``\begin{tabular}``" contract this function exists to provide.
A dict keeps each target's table independently well-formed and still lets a
caller do ``paper_accuracy_table(report)["INDPRO"].to_latex()``. Passing an
explicit ``target=...``, or a report that only has one target, returns that
target's ``ReportTable`` directly (no dict wrapping) so the common case is
truly one line to LaTeX.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `report` | positional or keyword | `Any` | `required` |
| `target` | keyword only | `str \| None` | `None` |
| `metric` | keyword only | `str` | `"rel_rmse"` |
| `star_levels` | keyword only | `PValueStarLevels` | `((0.01, "***"), (0.05, "**"), (0.1, "*"))` |
| `mcs_mark` | keyword only | `str` | `"†"` |
| `benchmark_row` | keyword only | `bool` | `True` |
| `precision` | keyword only | `int` | `3` |
| `missing` | keyword only | `str` | `"--"` |
| `caption` | keyword only | `str \| None` | `None` |
| `label` | keyword only | `str \| None` | `None` |
| `notes` | keyword only | `Sequence[str]` | `()` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |

#### Returns

`ReportTable | dict[str, ReportTable]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.reporting.paper_accuracy_table(...)
```
### render_tables

Qualified name: `macroforecast.reporting.core.render_tables`

#### Signature

```python
macroforecast.reporting.render_tables(value: ReportBundle | Mapping[str, ReportTable | Any], *, format: RenderFormat = "latex") -> dict[str, str]
```

#### Description

Render all tables in a bundle or mapping.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `value` | positional or keyword | `ReportBundle \| Mapping[str, ReportTable \| Any]` | `required` |
| `format` | keyword only | `RenderFormat` | `"latex"` |

#### Returns

`dict[str, str]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.reporting.render_tables(...)
```
### report_bundle

Qualified name: `macroforecast.reporting.core.report_bundle`

#### Signature

```python
macroforecast.reporting.report_bundle(*, tables: Mapping[str, ReportTable | Any] | None = None, figures: Mapping[str, Any] | None = None, metadata: Mapping[str, Any] | None = None) -> ReportBundle
```

#### Description

Collect named reporting tables and figure data without writing files.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `tables` | keyword only | `Mapping[str, ReportTable \| Any] \| None` | `None` |
| `figures` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |

#### Returns

`ReportBundle`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.reporting.report_bundle(...)
```
### report_table

Qualified name: `macroforecast.reporting.core.report_table`

#### Signature

```python
macroforecast.reporting.report_table(table: Any, *, columns: Sequence[str] | None = None, rename: Mapping[str, str] | None = None, sort_by: str | Sequence[str] | None = None, ascending: bool | Sequence[bool] = True, index: bool = False, precision: int = 3, percent_columns: Sequence[str] = (), missing: str = "", caption: str | None = None, label: str | None = None, notes: Sequence[str] = (), metadata: Mapping[str, Any] | None = None) -> ReportTable
```

#### Description

Return a presentation-ready table without writing files.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `table` | positional or keyword | `Any` | `required` |
| `columns` | keyword only | `Sequence[str] \| None` | `None` |
| `rename` | keyword only | `Mapping[str, str] \| None` | `None` |
| `sort_by` | keyword only | `str \| Sequence[str] \| None` | `None` |
| `ascending` | keyword only | `bool \| Sequence[bool]` | `True` |
| `index` | keyword only | `bool` | `False` |
| `precision` | keyword only | `int` | `3` |
| `percent_columns` | keyword only | `Sequence[str]` | `()` |
| `missing` | keyword only | `str` | `""` |
| `caption` | keyword only | `str \| None` | `None` |
| `label` | keyword only | `str \| None` | `None` |
| `notes` | keyword only | `Sequence[str]` | `()` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |

#### Returns

`ReportTable`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.reporting.report_table(...)
```
### test_provenance_table

Qualified name: `macroforecast.reporting.core.test_provenance_table`

#### Signature

```python
macroforecast.reporting.test_provenance_table(results: Any, *, columns: Sequence[str] | None = None, missing: str = "", caption: str | None = None, label: str | None = None, notes: Sequence[str] = (), metadata: Mapping[str, Any] | None = None) -> ReportTable
```

#### Description

Return a source-alignment table for forecast-test outputs.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `results` | positional or keyword | `Any` | `required` |
| `columns` | keyword only | `Sequence[str] \| None` | `None` |
| `missing` | keyword only | `str` | `""` |
| `caption` | keyword only | `str \| None` | `None` |
| `label` | keyword only | `str \| None` | `None` |
| `notes` | keyword only | `Sequence[str]` | `()` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |

#### Returns

`ReportTable`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.reporting.test_provenance_table(...)
```
### test_report_table

Qualified name: `macroforecast.reporting.core.test_report_table`

#### Signature

```python
macroforecast.reporting.test_report_table(results: Any, *, columns: Sequence[str] | None = None, include_reference: bool = False, stars: bool = True, star_levels: PValueStarLevels = ((0.01, '***'), (0.05, '**'), (0.1, '*')), precision: int = 3, p_value_precision: int = 3, missing: str = "", caption: str | None = None, label: str | None = None, notes: Sequence[str] = (), metadata: Mapping[str, Any] | None = None) -> ReportTable
```

#### Description

Return a paper-facing forecast-test result table.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `results` | positional or keyword | `Any` | `required` |
| `columns` | keyword only | `Sequence[str] \| None` | `None` |
| `include_reference` | keyword only | `bool` | `False` |
| `stars` | keyword only | `bool` | `True` |
| `star_levels` | keyword only | `PValueStarLevels` | `((0.01, "***"), (0.05, "**"), (0.1, "*"))` |
| `precision` | keyword only | `int` | `3` |
| `p_value_precision` | keyword only | `int` | `3` |
| `missing` | keyword only | `str` | `""` |
| `caption` | keyword only | `str \| None` | `None` |
| `label` | keyword only | `str \| None` | `None` |
| `notes` | keyword only | `Sequence[str]` | `()` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |

#### Returns

`ReportTable`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.reporting.test_report_table(...)
```
