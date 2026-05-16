"""Tests for L1.B target_structure after Cycle 18 cleanup.

Verifies:
- multi_series_target is no longer a valid option (BREAKING change)
- single_target + target=<str> passes validation
- multi_target + targets=[...] passes validation
"""
from __future__ import annotations

import pytest

from macroforecast.core.layers.l1 import resolve_axes_from_raw, validate_layer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fixed(target_structure: str) -> dict:
    return {
        "dataset": "fred_md",
        "frequency": "monthly",
        "target_structure": target_structure,
    }


def _leaf_single(target: str = "CPIAUCSL") -> dict:
    return {"target": target}


def _leaf_multi(targets: list) -> dict:
    return {"targets": targets}


# ---------------------------------------------------------------------------
# C18-A: multi_series_target must be rejected
# ---------------------------------------------------------------------------

def test_multi_series_target_rejected_as_unknown():
    """multi_series_target is no longer in valid_options; validate_layer must hard-error."""
    from macroforecast.core.layers.l1 import validate_layer, parse_layer_yaml
    yaml_text = """
    1_data:
      fixed_axes:
        target_structure: multi_series_target
      leaf_config:
        targets: [CPIAUCSL, INDPRO]
    """
    report = validate_layer(parse_layer_yaml(yaml_text))
    assert report.has_hard_errors, "Expected hard error for unknown multi_series_target"
    assert any(
        "target_structure" in (issue.location or "")
        for issue in report.hard_errors
    )


def test_multi_series_target_not_in_target_structure_literal():
    """TargetStructure Literal only contains single_target and multi_target."""
    from macroforecast.core.layers.l1 import TargetStructure
    import typing
    args = typing.get_args(TargetStructure)
    assert "multi_series_target" not in args
    assert "single_target" in args
    assert "multi_target" in args


# ---------------------------------------------------------------------------
# C18-B: single_target + target: str passes
# ---------------------------------------------------------------------------

def test_single_target_with_target_passes():
    """single_target + leaf_config.target=str is valid."""
    fixed = _fixed("single_target")
    leaf = _leaf_single("UNRATE")
    resolved = resolve_axes_from_raw(fixed, leaf, tolerate_invalid=False)
    assert resolved["target_structure"] == "single_target"


# ---------------------------------------------------------------------------
# C18-B: multi_target + targets: list passes
# ---------------------------------------------------------------------------

def test_multi_target_with_targets_passes():
    """multi_target + leaf_config.targets=[...] is valid."""
    fixed = _fixed("multi_target")
    leaf = _leaf_multi(["CPIAUCSL", "UNRATE"])
    resolved = resolve_axes_from_raw(fixed, leaf, tolerate_invalid=False)
    assert resolved["target_structure"] == "multi_target"


def test_multi_target_single_element_targets_passes():
    """multi_target with a single-element list is still valid."""
    fixed = _fixed("multi_target")
    leaf = _leaf_multi(["GDPC1"])
    resolved = resolve_axes_from_raw(fixed, leaf, tolerate_invalid=False)
    assert resolved["target_structure"] == "multi_target"


# ---------------------------------------------------------------------------
# Regression: canonical function no longer aliases multi_series_target
# ---------------------------------------------------------------------------

def test_canonical_target_structure_is_identity():
    """_canonical_target_structure must be a pure passthrough after C18."""
    from macroforecast.core.layers.l1 import _canonical_target_structure
    assert _canonical_target_structure("single_target") == "single_target"
    assert _canonical_target_structure("multi_target") == "multi_target"
    # multi_series_target is no longer aliased -- it passes through as-is
    # (validation will then reject it as unknown, tested above)
    assert _canonical_target_structure("multi_series_target") == "multi_series_target"
