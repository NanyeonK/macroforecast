"""Cycle 14 L1-4 -- selection_view=none does not false-error without feature_selection step.

Verifies that:
1. selection_view: none + no feature_selection step -> no error (the bug fix).
2. selection_view: multi + no feature_selection step -> error (guard still works).
"""
from __future__ import annotations

import pytest


def _issues_for_fixed(fixed: dict) -> list:
    """Call _validate_values with enabled=True and given fixed axes."""
    from macroforecast.core.layers.l3_5 import _validate_values, L3_5ResolvedAxes, DEFAULT_AXES
    values = {**DEFAULT_AXES, "enabled": True, **fixed}
    active = {k: True for k in values}
    resolved = L3_5ResolvedAxes(values, active)
    context = {"has_feature_selection_step": False}
    return _validate_values(resolved, fixed, context)


def test_selection_view_none_no_error():
    """selection_view=none without feature_selection step must produce no selection_view issues."""
    issues = _issues_for_fixed({"selection_view": "none"})
    selection_issues = [i for i in issues if "selection_view" in str(i)]
    assert not selection_issues, f"Unexpected issues for selection_view=none: {selection_issues}"


def test_selection_view_active_errors_without_selection_step():
    """selection_view=multi without feature_selection step must still raise an issue."""
    issues = _issues_for_fixed({"selection_view": "multi"})
    selection_issues = [i for i in issues if "selection_view" in str(i)]
    assert selection_issues, "Expected issue for selection_view=multi without feature_selection step"


def test_selection_view_none_does_not_propagate_error():
    """Explicitly setting none must not trigger the feature_selection requirement check."""
    issues = _issues_for_fixed({"selection_view": "none"})
    error_messages = [str(i) for i in issues if "feature_selection" in str(i) and "selection_view" in str(i)]
    assert not error_messages, f"Unexpected feature_selection error: {error_messages}"
