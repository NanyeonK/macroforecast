from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .types import AxisRegistryEntry, SupportStatus


@dataclass(frozen=True)
class BaseRegistryEntry:
    id: str
    description: str
    status: SupportStatus
    priority: Literal["A", "B"]


@dataclass(frozen=True)
class EnumRegistryEntry(BaseRegistryEntry):
    pass


@dataclass(frozen=True)
class AxisDefinition:
    axis_name: str
    layer: str
    axis_type: Literal["enum", "numeric", "callable", "plugin"]
    default_policy: Literal["fixed", "sweep", "conditional"]
    entries: tuple[BaseRegistryEntry, ...]
    compatible_with: dict[str, tuple[str, ...]]
    incompatible_with: dict[str, tuple[str, ...]]


def axis_definition_to_legacy_entry(definition: AxisDefinition) -> AxisRegistryEntry:
    return AxisRegistryEntry(
        axis_name=definition.axis_name,
        layer=definition.layer,
        axis_type=definition.axis_type,
        allowed_values=tuple(entry.id for entry in definition.entries),
        current_status={entry.id: entry.status for entry in definition.entries},
        default_policy=definition.default_policy,
        compatible_with=dict(definition.compatible_with),
        incompatible_with=dict(definition.incompatible_with),
    )
