"""8-axis statistical-test dispatch (Phase 2)."""

from .dispatch import (
    AXIS_NAMES,
    DEFAULT_STAT_TEST_SPEC,
    META_AXIS_NAMES,
    STAT_TEST_AXIS_NAMES,
    active_stat_test_axes,
    canonicalize_stat_test_spec,
    dispatch_stat_tests,
)

__all__ = [
    "AXIS_NAMES",
    "DEFAULT_STAT_TEST_SPEC",
    "META_AXIS_NAMES",
    "STAT_TEST_AXIS_NAMES",
    "active_stat_test_axes",
    "canonicalize_stat_test_spec",
    "dispatch_stat_tests",
]
