from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from html import escape as _html_escape
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd


RenderFormat = Literal["latex", "html", "markdown"]
PValueStarLevels = Sequence[tuple[float, str]]
EvaluationTableName = Literal["scores", "ranking", "benchmark", "regime", "decomposition"]

_DEFAULT_METRIC_RENAME = {
    "model": "Model",
    "model_id": "Model",
    "model_spec": "Specification",
    "target": "Target",
    "state": "State",
    "regime": "Regime",
    "time_bucket": "Time",
    "horizon": "H",
    "n": "N",
    "rank": "Rank",
    "mse": "MSE",
    "rmse": "RMSE",
    "mae": "MAE",
    "bias": "Bias",
    "bias_squared": "Bias^2",
    "residual_variance": "Residual variance",
    "bias_share": "Bias share",
    "variance_share": "Variance share",
    "relative_mse": "Relative MSE",
    "relative_mae": "Relative MAE",
    "mse_reduction": "MSE reduction",
    "r2_oos": "R2 OOS",
    "theil_u2": "Theil U2",
    "success_ratio": "Success ratio",
    "gaussian_nll": "Gaussian NLL",
    "crps": "CRPS",
}


@dataclass(frozen=True)
class ReportTable:
    """Presentation-ready table that has not been written to disk."""

    data: pd.DataFrame
    caption: str | None = None
    label: str | None = None
    notes: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    metadata_schema: dict[str, Any] = field(
        default_factory=lambda: {"kind": "report_table", "version": 1}
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata_schema": dict(self.metadata_schema),
            "caption": self.caption,
            "label": self.label,
            "notes": list(self.notes),
            "metadata": _json_ready(self.metadata),
            "data": _json_ready(self.data.to_dict(orient="records")),
            "columns": [str(column) for column in self.data.columns],
        }

    def to_latex(self, *, booktabs: bool = True) -> str:
        return latex_table(self, booktabs=booktabs)

    def to_html(self) -> str:
        return html_table(self)

    def to_markdown(self) -> str:
        return markdown_table(self)


@dataclass(frozen=True)
class ReportBundle:
    """Named in-memory report outputs."""

    tables: dict[str, ReportTable] = field(default_factory=dict)
    figures: dict[str, pd.DataFrame] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    metadata_schema: dict[str, Any] = field(
        default_factory=lambda: {"kind": "report_bundle", "version": 1}
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata_schema": dict(self.metadata_schema),
            "metadata": _json_ready(self.metadata),
            "tables": {name: table.to_dict() for name, table in self.tables.items()},
            "figures": {
                name: _dataframe_summary(frame)
                for name, frame in self.figures.items()
            },
        }

    def render(self, *, format: RenderFormat = "latex") -> dict[str, str]:
        return render_tables(self, format=format)


def report_table(
    table: Any,
    *,
    columns: Sequence[str] | None = None,
    rename: Mapping[str, str] | None = None,
    sort_by: str | Sequence[str] | None = None,
    ascending: bool | Sequence[bool] = True,
    index: bool = False,
    precision: int = 3,
    percent_columns: Sequence[str] = (),
    missing: str = "",
    caption: str | None = None,
    label: str | None = None,
    notes: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> ReportTable:
    """Return a presentation-ready table without writing files."""

    frame = pd.DataFrame(table).copy()
    if columns is not None:
        missing_columns = [column for column in columns if column not in frame.columns]
        if missing_columns:
            raise KeyError(f"table missing columns: {missing_columns}")
        frame = frame.loc[:, list(columns)]
    if sort_by is not None:
        frame = frame.sort_values(sort_by, ascending=ascending)
    if rename:
        frame = frame.rename(columns=dict(rename))
    if index:
        frame = frame.reset_index()
    else:
        frame = frame.reset_index(drop=True)
    percent_set = {str(column) for column in percent_columns}
    formatted = frame.copy()
    for column in formatted.columns:
        formatted[column] = [
            _format_value(
                value,
                precision=precision,
                missing=missing,
                as_percent=str(column) in percent_set,
            )
            for value in formatted[column]
        ]
    report_metadata = {
        "source_shape": [int(frame.shape[0]), int(frame.shape[1])],
        "precision": int(precision),
        "percent_columns": sorted(percent_set),
        **dict(metadata or {}),
    }
    formatted.attrs["macroforecast_metadata_schema"] = {
        "kind": "report_table",
        "version": 1,
        "columns": [str(column) for column in formatted.columns],
        "n_rows": int(len(formatted)),
    }
    formatted.attrs["macroforecast_metadata"] = _json_ready(report_metadata)
    return ReportTable(
        data=formatted,
        caption=caption,
        label=label,
        notes=tuple(str(note) for note in notes),
        metadata=report_metadata,
    )


def test_report_table(
    results: Any,
    *,
    columns: Sequence[str] | None = None,
    include_reference: bool = False,
    stars: bool = True,
    star_levels: PValueStarLevels = ((0.01, "***"), (0.05, "**"), (0.1, "*")),
    precision: int = 3,
    p_value_precision: int = 3,
    missing: str = "",
    caption: str | None = None,
    label: str | None = None,
    notes: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> ReportTable:
    """Return a paper-facing forecast-test result table."""

    raw = _coerce_test_table(results)
    display = pd.DataFrame(
        [
            {
                "Test": _test_key(row, fallback=f"test_{pos}"),
                "Name": _test_name(row),
                "Ref.": _display_missing(row.get("statistic_type"), missing=missing),
                "Statistic": _format_value(
                    row.get("statistic"),
                    precision=precision,
                    missing=missing,
                    as_percent=False,
                ),
                "p-value": _format_p_value(
                    row.get("p_value"),
                    precision=p_value_precision,
                    missing=missing,
                    stars=stars,
                    star_levels=star_levels,
                ),
                "Reject": _format_decision(row.get("decision"), missing=missing),
                "N": _format_n_obs(row.get("n_obs"), missing=missing),
                "P-value status": _display_missing(row.get("p_value_status"), missing=missing),
                "Source": _source_label(row, missing=missing),
            }
            for pos, row in enumerate(raw.to_dict(orient="records"))
        ]
    )
    default_columns = ["Test", "Name", "Ref.", "Statistic", "p-value", "Reject", "N"]
    if include_reference:
        default_columns.append("Source")
    selected_columns = list(columns) if columns is not None else default_columns
    default_notes = (
        _star_note(star_levels) if stars else None,
        "Source alignment details are available from test_provenance_table(...).",
    )
    merged_notes = tuple(note for note in (*default_notes, *tuple(str(note) for note in notes)) if note)
    return report_table(
        display,
        columns=selected_columns,
        precision=precision,
        missing=missing,
        caption=caption,
        label=label,
        notes=merged_notes,
        metadata={
            "source_kind": "test_report_table",
            "input_rows": int(len(raw)),
            "stars": bool(stars),
            "star_levels": _json_ready(list(star_levels)),
            "p_value_precision": int(p_value_precision),
            **dict(metadata or {}),
        },
    )


def test_provenance_table(
    results: Any,
    *,
    columns: Sequence[str] | None = None,
    missing: str = "",
    caption: str | None = None,
    label: str | None = None,
    notes: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> ReportTable:
    """Return a source-alignment table for forecast-test outputs."""

    raw = _coerce_test_table(results)
    display = pd.DataFrame(
        [
            {
                "Test": _test_key(row, fallback=f"test_{pos}"),
                "Name": _test_name(row),
                "Null": _display_missing(row.get("null_hypothesis"), missing=missing),
                "P-value ref.": _display_missing(row.get("p_value_reference"), missing=missing),
                "Status": _display_missing(row.get("p_value_status"), missing=missing),
                "Source": _display_missing(row.get("source_reference"), missing=missing),
                "R reference": _display_missing(row.get("r_reference"), missing=missing),
                "Alignment": _display_missing(row.get("r_alignment"), missing=missing),
            }
            for pos, row in enumerate(raw.to_dict(orient="records"))
        ]
    )
    default_columns = [
        "Test",
        "Name",
        "Null",
        "P-value ref.",
        "Status",
        "Source",
        "R reference",
        "Alignment",
    ]
    return report_table(
        display,
        columns=list(columns) if columns is not None else default_columns,
        missing=missing,
        caption=caption,
        label=label,
        notes=tuple(str(note) for note in notes),
        metadata={
            "source_kind": "test_provenance_table",
            "input_rows": int(len(raw)),
            **dict(metadata or {}),
        },
    )


def metric_report_table(
    results: Any,
    *,
    table: EvaluationTableName = "scores",
    columns: Sequence[str] | None = None,
    rename: Mapping[str, str] | None = None,
    sort_by: str | Sequence[str] | None = None,
    ascending: bool | Sequence[bool] = True,
    precision: int = 3,
    percent_columns: Sequence[str] = (),
    missing: str = "",
    caption: str | None = None,
    label: str | None = None,
    notes: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> ReportTable:
    """Return a paper-facing metric/evaluation table."""

    raw = _coerce_evaluation_table(results, table=table)
    selected_columns = list(columns) if columns is not None else _default_metric_columns(raw)
    raw = _prepare_metric_display_frame(raw, selected_columns)
    rename_map = {**_DEFAULT_METRIC_RENAME, **dict(rename or {})}
    display_percent = _renamed_columns(percent_columns, rename_map)
    return report_table(
        raw,
        columns=selected_columns,
        rename=rename_map,
        sort_by=sort_by,
        ascending=ascending,
        precision=precision,
        percent_columns=display_percent,
        missing=missing,
        caption=caption,
        label=label,
        notes=tuple(str(note) for note in notes),
        metadata={
            "source_kind": "metric_report_table",
            "evaluation_table": str(table),
            "input_rows": int(len(raw)),
            "raw_columns": [str(column) for column in raw.columns],
            **dict(metadata or {}),
        },
    )


def evaluation_report_tables(
    report: Any,
    *,
    include: Sequence[str] = ("scores", "ranking", "benchmark", "regime", "decomposition"),
    include_aggregations: bool = False,
    captions: Mapping[str, str] | None = None,
    labels: Mapping[str, str] | None = None,
    precision: int = 3,
    percent_columns: Sequence[str] = (),
    missing: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> ReportBundle:
    """Return named paper-facing tables from an evaluation report."""

    tables: dict[str, ReportTable] = {}
    caption_map = dict(captions or {})
    label_map = dict(labels or {})
    for name in include:
        key = str(name)
        if key == "aggregations":
            include_aggregations = True
            continue
        if not _has_evaluation_table(report, key):
            continue
        tables[key] = metric_report_table(
            report,
            table=key,  # type: ignore[arg-type]
            precision=precision,
            percent_columns=percent_columns,
            missing=missing,
            caption=caption_map.get(key),
            label=label_map.get(key),
        )
    if include_aggregations:
        for name, frame in dict(getattr(report, "aggregations", {}) or {}).items():
            key = f"aggregation_{_safe_report_name(name)}"
            tables[key] = metric_report_table(
                frame,
                precision=precision,
                percent_columns=percent_columns,
                missing=missing,
                caption=caption_map.get(key),
                label=label_map.get(key),
                metadata={"aggregation": str(name)},
            )
    return ReportBundle(
        tables=tables,
        metadata={
            "source_kind": "evaluation_report_tables",
            "included": list(tables),
            "include_aggregations": bool(include_aggregations),
            **dict(metadata or {}),
        },
    )


def accuracy_table(
    results: Any,
    *,
    columns: Sequence[str] | None = None,
    sort_by: str | Sequence[str] | None = ("horizon", "rmse"),
    ascending: bool | Sequence[bool] = True,
    precision: int = 3,
    percent_columns: Sequence[str] = (
        "r2_oos",
        "mse_reduction",
        "success_ratio",
    ),
    missing: str = "",
    caption: str | None = "Forecast accuracy",
    label: str | None = None,
    notes: Sequence[str] = ("Lower error metrics and higher R2 OOS are better.",),
    metadata: Mapping[str, Any] | None = None,
) -> ReportTable:
    """Return the default paper-facing forecast accuracy table."""

    selected = (
        list(columns)
        if columns is not None
        else _available_columns(
            _coerce_evaluation_table(results, table="scores"),
            ("model", "horizon", "rmse", "mae", "r2_oos", "relative_mse"),
        )
    )
    return metric_report_table(
        results,
        table="scores",
        columns=selected,
        sort_by=_available_sort(sort_by, selected),
        ascending=ascending,
        precision=precision,
        percent_columns=percent_columns,
        missing=missing,
        caption=caption,
        label=label,
        notes=notes,
        metadata={"source_kind": "accuracy_table", **dict(metadata or {})},
    )


def model_comparison_table(
    results: Any,
    *,
    columns: Sequence[str] | None = None,
    sort_by: str | Sequence[str] | None = ("horizon", "rank"),
    ascending: bool | Sequence[bool] = True,
    precision: int = 3,
    percent_columns: Sequence[str] = ("r2_oos",),
    missing: str = "",
    caption: str | None = "Model comparison",
    label: str | None = None,
    notes: Sequence[str] = ("Ranks are computed by the evaluation report rank metric.",),
    metadata: Mapping[str, Any] | None = None,
) -> ReportTable:
    """Return the default paper-facing model ranking/comparison table."""

    selected = (
        list(columns)
        if columns is not None
        else _available_columns(
            _coerce_evaluation_table(results, table="ranking"),
            ("rank", "model", "horizon", "rmse", "mae", "r2_oos", "relative_mse"),
        )
    )
    return metric_report_table(
        results,
        table="ranking",
        columns=selected,
        sort_by=_available_sort(sort_by, selected),
        ascending=ascending,
        precision=precision,
        percent_columns=percent_columns,
        missing=missing,
        caption=caption,
        label=label,
        notes=notes,
        metadata={"source_kind": "model_comparison_table", **dict(metadata or {})},
    )


def forecast_test_table(
    results: Any,
    *,
    columns: Sequence[str] | None = None,
    include_reference: bool = False,
    stars: bool = True,
    star_levels: PValueStarLevels = ((0.01, "***"), (0.05, "**"), (0.1, "*")),
    precision: int = 3,
    p_value_precision: int = 3,
    missing: str = "",
    caption: str | None = "Forecast comparison tests",
    label: str | None = None,
    notes: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> ReportTable:
    """Return the default paper-facing forecast-comparison test table."""

    return test_report_table(
        results,
        columns=columns,
        include_reference=include_reference,
        stars=stars,
        star_levels=star_levels,
        precision=precision,
        p_value_precision=p_value_precision,
        missing=missing,
        caption=caption,
        label=label,
        notes=notes,
        metadata={"source_kind": "forecast_test_table", **dict(metadata or {})},
    )


def latex_table(value: ReportTable | Any, *, booktabs: bool = True, **kwargs: Any) -> str:
    """Render a report table as LaTeX text."""

    table = value if isinstance(value, ReportTable) else report_table(value, **kwargs)
    frame = table.data
    top = "\\toprule" if booktabs else "\\hline"
    mid = "\\midrule" if booktabs else "\\hline"
    bottom = "\\bottomrule" if booktabs else "\\hline"
    lines: list[str] = []
    if table.caption or table.label:
        lines.append("\\begin{table}[!htbp]")
        lines.append("\\centering")
    if table.caption:
        lines.append(f"\\caption{{{_latex_escape(table.caption)}}}")
    if table.label:
        lines.append(f"\\label{{{_latex_escape(table.label)}}}")
    lines.append("\\begin{tabular}{" + "l" * max(1, len(frame.columns)) + "}")
    lines.append(top)
    lines.append(" & ".join(_latex_escape(column) for column in frame.columns) + r" \\")
    lines.append(mid)
    for _, row in frame.iterrows():
        lines.append(" & ".join(_latex_escape(value) for value in row.to_list()) + r" \\")
    lines.append(bottom)
    lines.append("\\end{tabular}")
    if table.notes:
        for note in table.notes:
            lines.append(f"\\\\[-0.2em]{{\\footnotesize {_latex_escape(note)}}}")
    if table.caption or table.label:
        lines.append("\\end{table}")
    return "\n".join(lines)


def html_table(value: ReportTable | Any, **kwargs: Any) -> str:
    """Render a report table as HTML text."""

    table = value if isinstance(value, ReportTable) else report_table(value, **kwargs)
    html = table.data.to_html(index=False, escape=True, border=0)
    pieces = ["<figure class=\"macroforecast-report-table\">"]
    if table.caption:
        pieces.append(f"<figcaption>{_html_escape(table.caption)}</figcaption>")
    pieces.append(html)
    if table.notes:
        pieces.append("<ul class=\"notes\">")
        pieces.extend(f"<li>{_html_escape(note)}</li>" for note in table.notes)
        pieces.append("</ul>")
    pieces.append("</figure>")
    return "\n".join(pieces)


def markdown_table(value: ReportTable | Any, **kwargs: Any) -> str:
    """Render a report table as GitHub-flavored Markdown text."""

    table = value if isinstance(value, ReportTable) else report_table(value, **kwargs)
    frame = table.data
    rows: list[str] = []
    if table.caption:
        rows.append(f"**{_markdown_escape(table.caption)}**")
        rows.append("")
    headers = [_markdown_escape(column) for column in frame.columns]
    rows.append("| " + " | ".join(headers) + " |")
    rows.append("| " + " | ".join("---" for _ in headers) + " |")
    for _, row in frame.iterrows():
        rows.append("| " + " | ".join(_markdown_escape(value) for value in row.to_list()) + " |")
    if table.notes:
        rows.append("")
        rows.extend(f"Note: {_markdown_escape(note)}" for note in table.notes)
    return "\n".join(rows)


def figure_data(
    data: Any,
    *,
    x: str | None = None,
    y: str | Sequence[str] | None = None,
    group: str | None = None,
    columns: Sequence[str] | None = None,
    rename: Mapping[str, str] | None = None,
    dropna: bool = True,
) -> pd.DataFrame:
    """Return a tidy frame intended for plotting or figure export."""

    frame = pd.DataFrame(data).copy()
    source_shape = [int(frame.shape[0]), int(frame.shape[1])]
    selected = list(columns) if columns is not None else _figure_columns(x=x, y=y, group=group)
    if selected:
        missing = [column for column in selected if column not in frame.columns]
        if missing:
            raise KeyError(f"figure data missing columns: {missing}")
        frame = frame.loc[:, selected]
    if rename:
        frame = frame.rename(columns=dict(rename))
    if dropna:
        frame = frame.dropna()
    frame.attrs["macroforecast_metadata_schema"] = {
        "kind": "figure_data",
        "version": 1,
        "columns": [str(column) for column in frame.columns],
        "n_rows": int(len(frame)),
    }
    frame.attrs["macroforecast_metadata"] = {
        "source_shape": source_shape,
        "x": x,
        "y": [y] if isinstance(y, str) else list(y or ()),
        "group": group,
        "dropna": bool(dropna),
    }
    return frame


def report_bundle(
    *,
    tables: Mapping[str, ReportTable | Any] | None = None,
    figures: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ReportBundle:
    """Collect named reporting tables and figure data without writing files."""

    table_map = {
        str(name): value if isinstance(value, ReportTable) else report_table(value)
        for name, value in dict(tables or {}).items()
    }
    figure_map = {
        str(name): figure_data(value)
        for name, value in dict(figures or {}).items()
    }
    return ReportBundle(
        tables=table_map,
        figures=figure_map,
        metadata={
            "n_tables": int(len(table_map)),
            "n_figures": int(len(figure_map)),
            **dict(metadata or {}),
        },
    )


def render_tables(
    value: ReportBundle | Mapping[str, ReportTable | Any],
    *,
    format: RenderFormat = "latex",
) -> dict[str, str]:
    """Render all tables in a bundle or mapping."""

    if format not in {"latex", "html", "markdown"}:
        raise ValueError("format must be 'latex', 'html', or 'markdown'")
    tables = value.tables if isinstance(value, ReportBundle) else dict(value)
    rendered: dict[str, str] = {}
    for name, table_value in tables.items():
        table = table_value if isinstance(table_value, ReportTable) else report_table(table_value)
        if format == "latex":
            rendered[str(name)] = latex_table(table)
        elif format == "html":
            rendered[str(name)] = html_table(table)
        else:
            rendered[str(name)] = markdown_table(table)
    return rendered


def _available_columns(frame: pd.DataFrame, preferred: Sequence[str]) -> list[str]:
    selected = [column for column in preferred if column in frame.columns]
    return selected or _default_metric_columns(frame)


def _available_sort(
    sort_by: str | Sequence[str] | None,
    columns: Sequence[str],
) -> str | list[str] | None:
    if sort_by is None:
        return None
    if isinstance(sort_by, str):
        return sort_by if sort_by in columns else None
    selected = [column for column in sort_by if column in columns]
    return selected or None


def _figure_columns(
    *,
    x: str | None,
    y: str | Sequence[str] | None,
    group: str | None,
) -> list[str]:
    columns: list[str] = []
    if x is not None:
        columns.append(x)
    if y is not None:
        columns.extend([y] if isinstance(y, str) else list(y))
    if group is not None:
        columns.append(group)
    return list(dict.fromkeys(columns))


def _format_value(value: Any, *, precision: int, missing: str, as_percent: bool) -> str:
    if value is None or value is pd.NA or value is pd.NaT:
        return missing
    if isinstance(value, float) and not np.isfinite(value):
        return missing
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if as_percent:
            return f"{number * 100:.{precision}f}%"
        return f"{number:.{precision}f}"
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    return str(value)


def _coerce_test_table(results: Any) -> pd.DataFrame:
    from macroforecast.output import test_table

    return test_table(results).copy()


def _coerce_evaluation_table(results: Any, *, table: str) -> pd.DataFrame:
    if isinstance(results, pd.DataFrame):
        return results.copy()
    if table == "scores" and hasattr(results, "scores"):
        return getattr(results, "scores").copy()
    if table == "ranking" and hasattr(results, "ranking"):
        return getattr(results, "ranking").copy()
    if table == "benchmark" and getattr(results, "benchmark", None) is not None:
        return getattr(results, "benchmark").copy()
    if table == "regime" and getattr(results, "regime", None) is not None:
        return getattr(results, "regime").copy()
    if table == "decomposition" and getattr(results, "decomposition", None) is not None:
        return getattr(results, "decomposition").copy()
    raise ValueError(f"evaluation table {table!r} is not available")


def _has_evaluation_table(report: Any, table: str) -> bool:
    if table in {"scores", "ranking"}:
        return hasattr(report, table)
    if table in {"benchmark", "regime", "decomposition"}:
        return getattr(report, table, None) is not None
    return False


def _default_metric_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if str(column) != "metadata"]


def _prepare_metric_display_frame(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = frame.copy()
    identifier_columns = {
        "model",
        "model_id",
        "model_spec",
        "target",
        "state",
        "regime",
        "time_bucket",
        "forecast_policy",
        "horizon",
        "n",
        "rank",
    }
    for column in columns:
        if str(column) in identifier_columns and column in out.columns:
            out[column] = [_format_identifier(value) for value in out[column]]
    return out


def _format_identifier(value: Any) -> str:
    if _is_missing(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if number.is_integer():
            return str(int(number))
    return str(value)


def _renamed_columns(columns: Sequence[str], rename: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(str(rename.get(str(column), column)) for column in columns)


def _test_key(row: Mapping[str, Any], *, fallback: str) -> str:
    value = row.get("test")
    if _is_missing(value):
        value = row.get("name")
    if _is_missing(value):
        return fallback
    return str(value)


def _test_name(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping) and not _is_missing(metadata.get("name")):
        return str(metadata["name"])
    value = row.get("name")
    return "" if _is_missing(value) else str(value)


def _format_p_value(
    value: Any,
    *,
    precision: int,
    missing: str,
    stars: bool,
    star_levels: PValueStarLevels,
) -> str:
    if _is_missing(value):
        return missing
    p_value = float(value)
    threshold = 10.0 ** (-int(precision))
    if 0.0 < p_value < threshold:
        text = f"<{threshold:.{precision}f}"
    else:
        text = f"{p_value:.{precision}f}"
    return text + (_p_value_stars(p_value, star_levels) if stars else "")


def _p_value_stars(p_value: float, star_levels: PValueStarLevels) -> str:
    for threshold, marker in sorted(star_levels, key=lambda item: item[0]):
        if p_value <= float(threshold):
            return str(marker)
    return ""


def _star_note(star_levels: PValueStarLevels) -> str:
    pieces = [
        f"{marker} p<={float(threshold):.2g}"
        for threshold, marker in sorted(star_levels, key=lambda item: item[0])
    ]
    return "Significance markers: " + ", ".join(pieces) + "."


def _format_decision(value: Any, *, missing: str) -> str:
    if _is_missing(value):
        return missing
    return "Yes" if bool(value) else "No"


def _format_n_obs(value: Any, *, missing: str) -> str:
    if _is_missing(value):
        return missing
    return str(int(value))


def _source_label(row: Mapping[str, Any], *, missing: str) -> str:
    r_reference = row.get("r_reference")
    if not _is_missing(r_reference):
        return f"R: {r_reference}"
    source = row.get("source_reference")
    if not _is_missing(source):
        return str(source)
    return missing


def _display_missing(value: Any, *, missing: str) -> str:
    return missing if _is_missing(value) else str(value)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _safe_report_name(value: Any) -> str:
    text = str(value).strip().lower()
    out = "".join(char if char.isalnum() else "_" for char in text)
    out = "_".join(part for part in out.split("_") if part)
    return out or "table"


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _markdown_escape(value: Any) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def _dataframe_summary(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "shape": [int(frame.shape[0]), int(frame.shape[1])],
        "columns": [str(column) for column in frame.columns],
        "metadata_schema": _json_ready(frame.attrs.get("macroforecast_metadata_schema")),
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, pd.DataFrame):
        return {
            "columns": [str(column) for column in value.columns],
            "data": _json_ready(value.to_dict(orient="records")),
        }
    if isinstance(value, pd.Series):
        return _json_ready(value.to_list())
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


__all__ = [
    "ReportBundle",
    "ReportTable",
    "accuracy_table",
    "evaluation_report_tables",
    "figure_data",
    "forecast_test_table",
    "html_table",
    "latex_table",
    "markdown_table",
    "model_comparison_table",
    "render_tables",
    "report_bundle",
    "report_table",
    "metric_report_table",
    "test_provenance_table",
    "test_report_table",
]
