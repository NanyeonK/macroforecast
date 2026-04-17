from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


REGISTRY_TYPE_ENTRIES: tuple[EnumRegistryEntry, ...] = (
    EnumRegistryEntry(id="enum_registry", description="Finite enumerated registry.", status="operational", priority="A"),
    EnumRegistryEntry(id="numeric_registry", description="Numeric range/grid registry.", status="operational", priority="A"),
    EnumRegistryEntry(id="callable_registry", description="Callable-signature validated registry.", status="operational", priority="A"),
    EnumRegistryEntry(id="custom_plugin", description="Plugin-backed registry.", status="operational", priority="A"),
    EnumRegistryEntry(id="user_defined_yaml", description="User-supplied YAML schema registry.", status="registry_only", priority="B"),
    EnumRegistryEntry(id="external_adapter", description="Externally bridged adapter registry.", status="registry_only", priority="B"),
)

AXIS_DEFINITION = AxisDefinition(
    axis_name="registry_type",
    layer="0_meta",
    axis_type="enum",
    registry_type="enum_registry",
    default_policy="fixed",
    entries=REGISTRY_TYPE_ENTRIES,
    compatible_with={},
    incompatible_with={},
)
