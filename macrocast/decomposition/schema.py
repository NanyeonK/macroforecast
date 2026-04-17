"""decomposition_result.parquet schema v1."""
from __future__ import annotations

DECOMPOSITION_RESULT_SCHEMA_VERSION = "1.0"

COLUMNS: tuple[tuple[str, str], ...] = (
    ("component", "string"),
    ("axis_name", "string"),
    ("ss_between", "float64"),
    ("ss_total", "float64"),
    ("share", "float64"),
    ("n_variants", "int64"),
    ("n_groups", "int64"),
    ("significance_p", "float64"),
)


def expected_columns() -> tuple[str, ...]:
    return tuple(name for name, _ in COLUMNS)


__all__ = [
    "DECOMPOSITION_RESULT_SCHEMA_VERSION",
    "COLUMNS",
    "expected_columns",
]
