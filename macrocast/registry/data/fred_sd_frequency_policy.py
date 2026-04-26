from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name="fred_sd_frequency_policy",
    layer="1_data_task",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(
            id="report_only",
            description="record the selected FRED-SD native-frequency composition without blocking execution",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="allow_mixed_frequency",
            description="explicitly allow selected FRED-SD panels with mixed or unknown native frequencies",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="reject_mixed_known_frequency",
            description="reject selected FRED-SD panels containing more than one known native frequency",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="require_single_known_frequency",
            description="require exactly one known native frequency and no unknown-frequency FRED-SD series",
            status="operational",
            priority="A",
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
