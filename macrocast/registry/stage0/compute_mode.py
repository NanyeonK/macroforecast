from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


COMPUTE_MODE_ENTRIES: tuple[EnumRegistryEntry, ...] = (
    EnumRegistryEntry(id="serial", description="Default local execution: run one work unit at a time.", status="operational", priority="A"),
    EnumRegistryEntry(id="parallel", description="Parallel execution. Unit and worker count live in leaf_config.", status="operational", priority="A"),
)

AXIS_DEFINITION = AxisDefinition(
    axis_name="compute_mode",
    layer="0_meta",
    axis_type="enum",
    entries=COMPUTE_MODE_ENTRIES,
    compatible_with={},
    incompatible_with={},
    default_policy="fixed",
)
