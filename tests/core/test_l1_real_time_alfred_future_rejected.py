"""Cycle 14 K-4 — real_time_alfred vintage_policy is hard-rejected at validation.

Tests:
1. real_time_alfred in fixed_axes raises before execution.
2. real_time_alfred in leaf_config raises before execution.
3. current_vintage (default) is accepted without error.
4. Rejection message mentions real_time_alfred or not-yet-implemented or future.

Closes: Cycle 14 F15/F17/F-H3 (P1-9)
"""
from __future__ import annotations

import pytest
import macroforecast as mf


_BASE_RECIPE = """
0_meta:
  fixed_axes:
    failure_policy: fail_fast
    reproducibility_mode: seeded_reproducible
  leaf_config:
    random_seed: 1
1_data:
  fixed_axes:
    custom_source_policy: custom_panel_only
    frequency: monthly
    horizon_set: custom_list
    {vintage_axis}
  leaf_config:
    target: y
    target_horizons: [1]
    {vintage_leaf}
    custom_panel_inline:
      date: [2020-01-01, 2020-02-01, 2020-03-01, 2020-04-01, 2020-05-01, 2020-06-01,
             2020-07-01, 2020-08-01, 2020-09-01, 2020-10-01, 2020-11-01, 2020-12-01]
      y: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
      x1: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
2_preprocessing:
  fixed_axes:
    transform_policy: no_transform
4_forecasting_model:
  nodes:
    - id: fit_ridge
      op: ridge
      params: {{alpha: 0.5}}
      inputs:
        - layer_ref: l3
          sink_name: l3_features_v1
"""

# Note: custom_panel_only sets vintage_policy=None internally, so to test
# real_time_alfred we use a FRED-MD-style recipe (dataset declared) where
# vintage_policy is read from fixed_axes.
_FRED_BASE_RECIPE = """
0_meta:
  fixed_axes:
    failure_policy: fail_fast
    reproducibility_mode: seeded_reproducible
  leaf_config:
    random_seed: 1
1_data:
  fixed_axes:
    dataset: fred_md
    {vintage_axis}
  leaf_config:
    target: INDPRO
    target_horizons: [1]
    {vintage_leaf}
4_forecasting_model:
  nodes:
    - id: fit_ridge
      op: ridge
      params: {{alpha: 0.5}}
      inputs:
        - layer_ref: l3
          sink_name: l3_features_v1
"""


def test_real_time_alfred_hard_rejected_in_fixed_axes():
    """vintage_policy=real_time_alfred in fixed_axes must raise before execution."""
    recipe = _FRED_BASE_RECIPE.format(
        vintage_axis="vintage_policy: real_time_alfred",
        vintage_leaf="",
    )
    with pytest.raises(Exception) as exc_info:
        mf.run(recipe)
    msg = str(exc_info.value).lower()
    assert any(kw in msg for kw in ("real_time_alfred", "not yet implemented", "future")), (
        f"Expected rejection message to mention real_time_alfred/future/not yet implemented, "
        f"got: {exc_info.value!r}"
    )


def test_real_time_alfred_hard_rejected_in_leaf_config():
    """vintage_policy=real_time_alfred in leaf_config must also be rejected (K-4 bug)."""
    recipe = _FRED_BASE_RECIPE.format(
        vintage_axis="",
        vintage_leaf="vintage_policy: real_time_alfred",
    )
    with pytest.raises(Exception) as exc_info:
        mf.run(recipe)
    msg = str(exc_info.value).lower()
    assert any(kw in msg for kw in ("real_time_alfred", "not yet implemented", "future")), (
        f"Expected rejection message to mention real_time_alfred/future/not yet implemented, "
        f"got: {exc_info.value!r}"
    )


def test_current_vintage_accepted():
    """vintage_policy=current_vintage (default) must not raise a validation error."""
    from macroforecast.core.layers.l1 import parse_layer_yaml, validate_layer

    yaml_text = """
1_data:
  fixed_axes:
    dataset: fred_md
    vintage_policy: current_vintage
  leaf_config:
    target: INDPRO
    target_horizons: [1]
"""
    report = validate_layer(parse_layer_yaml(yaml_text))
    alfred_errors = [
        issue for issue in report.hard_errors
        if "real_time_alfred" in issue.message or "not yet implemented" in issue.message
    ]
    assert not alfred_errors, (
        f"current_vintage should not produce alfred-related hard errors, got: {alfred_errors}"
    )


def test_real_time_alfred_validation_message_content():
    """Validation report for real_time_alfred must contain the expected rejection keywords."""
    from macroforecast.core.layers.l1 import parse_layer_yaml, validate_layer

    yaml_text = """
1_data:
  fixed_axes:
    dataset: fred_md
    vintage_policy: real_time_alfred
  leaf_config:
    target: INDPRO
    target_horizons: [1]
"""
    report = validate_layer(parse_layer_yaml(yaml_text))
    assert report.has_hard_errors, "real_time_alfred must produce hard validation errors"
    combined_msg = " ".join(issue.message for issue in report.hard_errors).lower()
    assert any(kw in combined_msg for kw in ("real_time_alfred", "not yet implemented", "future")), (
        f"Hard error messages do not mention real_time_alfred/future: {combined_msg!r}"
    )
