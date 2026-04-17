"""8-axis statistical-test dispatch (Phase 2)."""

from .dispatch import (
    AXIS_NAMES,
    LEGACY_TO_NEW,
    dispatch_stat_tests,
)

__all__ = [
    "AXIS_NAMES",
    "LEGACY_TO_NEW",
    "dispatch_stat_tests",
]
