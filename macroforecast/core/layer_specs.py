from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .dag import GatePredicate, LayerCategory, LayerId
from .ops.registry import Rule


@dataclass(frozen=True)
class Option:
    value: str
    label: str
    description: str
    status: str = "operational"
    leaf_config_required: tuple[str, ...] = ()
    leaf_config_optional: tuple[str, ...] = ()


@dataclass(frozen=True)
class AxisSpec:
    name: str
    options: tuple[Option, ...]
    default: Any
    status: str = "operational"
    sweepable: bool = True
    gate: GatePredicate | None = None
    dynamic_default_rule: Callable[..., Any] | None = None
    leaf_config_keys: tuple[str, ...] = ()
    # Optional parameter documentation for axis-level conditional leaf_config keys.
    # Set via macroforecast.scaffold.option_docs.types.ParameterDoc when the
    # axis acts as a categorical gate that unlocks further leaf_config fields.
    parameters: tuple["ParameterDoc", ...] = ()


@dataclass(frozen=True)
class SubLayerSpec:
    id: str
    name: str
    gate: GatePredicate | None = None
    axes: tuple[str, ...] = ()


@dataclass(frozen=True)
class LayerImplementationSpec:
    layer_id: LayerId
    name: str
    category: LayerCategory
    expected_inputs: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()
    ui_mode: str = "adaptive"
    layer_globals: tuple[str, ...] = ()
    sub_layers: tuple[SubLayerSpec, ...] = ()
    axes: dict[str, dict[str, AxisSpec]] = field(default_factory=dict)
    ops: tuple[str, ...] = ()
    layer_rules: tuple[Rule, ...] = ()
    sample_yaml: dict[str, str] = field(default_factory=dict)
    test_cases: tuple[str, ...] = ()


LAYER_SPEC_CHECKLIST: tuple[str, ...] = (
    "Layer ID, name, category",
    "Sub-layer structure",
    "Axis list per sub-layer",
    "Axis options, default, status",
    "Dynamic default rules",
    "Gate predicates",
    "Leaf config keys",
    "Layer globals",
    "Sink contract",
    "Layer-specific ops",
    "Hard/soft rules",
    "Default DAG",
    "Sample YAML",
    "Cross-layer references",
    "Test cases",
)

PHASE1_IMPLEMENTATION_ORDER: tuple[LayerId | str, ...] = (
    "l0",
    "l1",
    "l2",
    "l3",
    "l4",
    "l5",
    "l6",
    "l7",
    "l8",
    "diagnostics",
)
