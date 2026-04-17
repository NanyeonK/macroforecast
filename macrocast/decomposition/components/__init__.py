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
    "feature_builder",
    "importance",
    "loss",
    "nonlinearity",
    "preprocessing",
    "regularization",
)

COMPONENT_NAMES_SET = frozenset(COMPONENT_NAMES)


def is_valid_component(name: str | None) -> bool:
    return name is None or name in COMPONENT_NAMES_SET


__all__ = ["COMPONENT_NAMES", "COMPONENT_NAMES_SET", "is_valid_component"]
