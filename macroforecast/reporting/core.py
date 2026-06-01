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
    formatted.attrs["macroforecast_metadata_schema"] = {
        "kind": "report_table",
        "version": 1,
        "columns": [str(column) for column in formatted.columns],
        "n_rows": int(len(formatted)),
    }
    return ReportTable(
        data=formatted,
        caption=caption,
        label=label,
        notes=tuple(str(note) for note in notes),
        metadata={
            "source_shape": [int(frame.shape[0]), int(frame.shape[1])],
            "precision": int(precision),
            "percent_columns": sorted(percent_set),
            **dict(metadata or {}),
        },
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
    "figure_data",
    "html_table",
    "latex_table",
    "markdown_table",
    "render_tables",
    "report_bundle",
    "report_table",
]
