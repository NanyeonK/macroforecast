"""Cycle 50 — real_time_alfred vintage_policy is operational (not future-gated).

Inverted from Cycle 14 K-4 (test_l1_real_time_alfred_future_rejected.py) after
C50 promoted real_time_alfred from 'future' to 'operational'.

Tests:
1. real_time_alfred in fixed_axes is accepted by validate_layer (no hard error for policy).
2. real_time_alfred in leaf_config is accepted by validate_layer (no hard error for policy).
3. current_vintage (default) is still accepted without error (regression guard).
4. Option status for real_time_alfred is 'operational' in the AxisSpec registry.

Closes: Cycle 50 stale-test regression (CI F15/F17 equivalent).
"""
from __future__ import annotations

import pytest


_FRED_BASE_YAML = """
1_data:
  fixed_axes:
    dataset: fred_md
    {vintage_axis}
  leaf_config:
    target: INDPRO
    target_horizons: [1]
    {vintage_leaf}
"""


def test_real_time_alfred_accepted_in_fixed_axes():
    """validate_layer must NOT hard-reject real_time_alfred in fixed_axes (C50 operational)."""
    from macroforecast.core.layers.l1 import parse_layer_yaml, validate_layer

    yaml_text = _FRED_BASE_YAML.format(
        vintage_axis="vintage_policy: real_time_alfred",
        vintage_leaf="",
    )
    report = validate_layer(parse_layer_yaml(yaml_text))

    # No hard error must reference the vintage_policy value itself as future/not-implemented.
    future_rejection_msgs = [
        issue.message for issue in report.hard_errors
        if any(kw in issue.message.lower() for kw in ("not yet implemented", "future"))
        and "real_time_alfred" in issue.message.lower()
    ]
    assert not future_rejection_msgs, (
        f"real_time_alfred must NOT be hard-rejected as future in C50; "
        f"unexpected hard errors: {future_rejection_msgs}"
    )


def test_real_time_alfred_accepted_in_leaf_config():
    """validate_layer must NOT hard-reject real_time_alfred in leaf_config (C50 operational)."""
    from macroforecast.core.layers.l1 import parse_layer_yaml, validate_layer

    yaml_text = _FRED_BASE_YAML.format(
        vintage_axis="",
        vintage_leaf="vintage_policy: real_time_alfred",
    )
    report = validate_layer(parse_layer_yaml(yaml_text))

    future_rejection_msgs = [
        issue.message for issue in report.hard_errors
        if any(kw in issue.message.lower() for kw in ("not yet implemented", "future"))
        and "real_time_alfred" in issue.message.lower()
    ]
    assert not future_rejection_msgs, (
        f"real_time_alfred must NOT be hard-rejected as future in C50; "
        f"unexpected hard errors: {future_rejection_msgs}"
    )


def test_current_vintage_still_accepted():
    """Regression guard: current_vintage must still be accepted without any hard errors."""
    from macroforecast.core.layers.l1 import parse_layer_yaml, validate_layer

    yaml_text = _FRED_BASE_YAML.format(
        vintage_axis="vintage_policy: current_vintage",
        vintage_leaf="",
    )
    report = validate_layer(parse_layer_yaml(yaml_text))
    assert not report.has_hard_errors, (
        f"current_vintage must produce no hard errors, got: "
        f"{[i.message for i in report.hard_errors]}"
    )


def test_real_time_alfred_option_status_is_operational():
    """Option registry must carry status='operational' for real_time_alfred (C50 promotion)."""
    from macroforecast.core.layers.l1 import L1_LAYER_SPEC

    vintage_axis = L1_LAYER_SPEC.axes["l1_a"]["vintage_policy"]
    opt = next(
        (o for o in vintage_axis.options if o.value == "real_time_alfred"), None
    )
    assert opt is not None, (
        "real_time_alfred option not found in l1_a vintage_policy axis; "
        f"available options: {[o.value for o in vintage_axis.options]}"
    )
    assert opt.status == "operational", (
        f"Expected status='operational' for real_time_alfred after C50 promotion, "
        f"got status={opt.status!r}"
    )
