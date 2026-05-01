from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


FAILURE_POLICY_ENTRIES: tuple[EnumRegistryEntry, ...] = (
    EnumRegistryEntry(id="fail_fast", description="Abort immediately on execution failure.", status="operational", priority="A"),
    EnumRegistryEntry(id="continue_on_failure", description="Continue large sweeps and record failed cells in the manifest.", status="operational", priority="A"),
)

AXIS_DEFINITION = AxisDefinition(
    axis_name="failure_policy",
    layer="0_meta",
    axis_type="enum",
    entries=FAILURE_POLICY_ENTRIES,
    compatible_with={},
    incompatible_with={},
    default_policy="fixed",
)
