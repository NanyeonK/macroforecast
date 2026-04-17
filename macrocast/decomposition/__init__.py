"""Phase 7 decomposition engine — attribute sweep variance to components."""
from __future__ import annotations

from .attribution import AnovaResult, one_way_anova
from .components import COMPONENT_NAMES, COMPONENT_NAMES_SET, is_valid_component
from .engine import DecompositionPlan, DecompositionResult, run_decomposition
from .schema import (
    COLUMNS,
    DECOMPOSITION_RESULT_SCHEMA_VERSION,
    expected_columns,
)

__all__ = [
    "AnovaResult",
    "COLUMNS",
    "COMPONENT_NAMES",
    "COMPONENT_NAMES_SET",
    "DECOMPOSITION_RESULT_SCHEMA_VERSION",
    "DecompositionPlan",
    "DecompositionResult",
    "expected_columns",
    "is_valid_component",
    "one_way_anova",
    "run_decomposition",
]
