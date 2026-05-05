from __future__ import annotations

from dataclasses import dataclass, field

from .dag import LayerId, SourceSelector
from .layers import LAYER_SINKS
from .ops.registry import TypeSpec


@dataclass(frozen=True)
class SourceContext:
    active_layers: frozenset[LayerId] = field(default_factory=frozenset)
    available_sinks: dict[LayerId, frozenset[str]] = field(default_factory=dict)

    @classmethod
    def all_declared_sinks_active(cls) -> "SourceContext":
        return cls(
            active_layers=frozenset(LAYER_SINKS),
            available_sinks={layer_id: frozenset(sinks) for layer_id, sinks in LAYER_SINKS.items()},
        )


def resolve_source_selector(selector: SourceSelector, context: SourceContext | None = None) -> TypeSpec:
    if selector.layer_ref == "external":
        raise ValueError("external source selectors require an explicit runtime adapter")
    active_context = context or SourceContext.all_declared_sinks_active()
    if selector.layer_ref not in active_context.active_layers:
        raise ValueError(f"{selector.layer_ref}.{selector.sink_name}: source layer is not active")
    available = active_context.available_sinks.get(selector.layer_ref, frozenset())
    if selector.sink_name not in available:
        raise ValueError(f"{selector.layer_ref}.{selector.sink_name}: sink is not available")
    try:
        return LAYER_SINKS[selector.layer_ref][selector.sink_name]
    except KeyError as exc:
        raise ValueError(f"{selector.layer_ref}.{selector.sink_name}: unknown sink") from exc
