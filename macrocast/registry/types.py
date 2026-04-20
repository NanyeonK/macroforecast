from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SupportStatus = Literal[
    "operational",
    "registry_only",
    "planned",
    "future",
    "external_plugin",
    "not_supported_yet",
]

AxisType = Literal[
    "enum",
    "numeric",
    "callable",
    "plugin",
    "leaf_config",
]

AxisSelectionMode = Literal[
    "fixed",
    "sweep",
    "conditional",
    "nested_sweep",
    "derived",
]


@dataclass(frozen=True)
class AxisRegistryEntry:
    axis_name: str
    layer: str
    axis_type: AxisType | str
    allowed_values: tuple[str, ...]
    current_status: dict[str, SupportStatus | str]
    default_policy: AxisSelectionMode | str
    compatible_with: dict[str, tuple[str, ...]]
    incompatible_with: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class AxisSelection:
    axis_name: str
    layer: str
    selection_mode: AxisSelectionMode | str
    selected_values: tuple[str, ...]
    selected_status: dict[str, SupportStatus | str]
