from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


COMPUTE_MODE_ENTRIES: tuple[EnumRegistryEntry, ...] = (
    EnumRegistryEntry(id="serial", description="Serial local execution.", status="operational", priority="A"),
    EnumRegistryEntry(id="parallel_by_model", description="Parallel execution across models.", status="planned", priority="A"),
    EnumRegistryEntry(id="parallel_by_horizon", description="Parallel execution across horizons.", status="planned", priority="A"),
    EnumRegistryEntry(id="parallel_by_oos_date", description="Parallel execution across out-of-sample dates.", status="registry_only", priority="B"),
    EnumRegistryEntry(id="parallel_by_trial", description="Parallel execution across trials.", status="registry_only", priority="B"),
    EnumRegistryEntry(id="gpu_single", description="Single-GPU execution.", status="registry_only", priority="B"),
    EnumRegistryEntry(id="gpu_multi", description="Multi-GPU execution.", status="registry_only", priority="B"),
    EnumRegistryEntry(id="distributed_cluster", description="Distributed cluster execution.", status="registry_only", priority="B"),
)

AXIS_DEFINITION = AxisDefinition(
    axis_name="compute_mode",
    layer="0_meta",
    axis_type="enum",
    entries=COMPUTE_MODE_ENTRIES,
    compatible_with={},
    incompatible_with={},
    registry_type="enum_registry",
    default_policy="fixed",
)
