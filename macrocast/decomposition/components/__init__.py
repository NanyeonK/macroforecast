"""Component enum for Phase 7 decomposition attribution.

Each component is a loss-variance attribution bucket. The Phase 7 engine
walks the registry's AxisDefinitions, groups axes by their .component tag,
and partitions the sweep's primary-metric sum-of-squares by each bucket.

The enum is a frozenset — callers checking membership get fast O(1)
lookup and the tuple below is what docs / reports enumerate when they
want a deterministic order.
"""
from __future__ import annotations

COMPONENT_NAMES: tuple[str, ...] = (
    "benchmark",
    "cv_scheme",
    "feature_representation",
    "importance",
    "loss",
    "nonlinearity",
    "preprocessing",
    "regularization",
)

COMPONENT_NAMES_SET = frozenset(COMPONENT_NAMES)
COMPONENT_ALIASES = {
    "feature_builder": "feature_representation",
}


def normalize_component(name: str | None) -> str | None:
    if name is None:
        return None
    return COMPONENT_ALIASES.get(name, name)


def is_valid_component(name: str | None) -> bool:
    return normalize_component(name) in COMPONENT_NAMES_SET if name is not None else True


__all__ = [
    "COMPONENT_ALIASES",
    "COMPONENT_NAMES",
    "COMPONENT_NAMES_SET",
    "is_valid_component",
    "normalize_component",
]
