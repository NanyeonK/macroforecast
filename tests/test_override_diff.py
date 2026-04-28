"""Tests for macrocast.compiler.override_diff.apply_overrides."""
from __future__ import annotations

import copy

import pytest

from macrocast.compiler.override_diff import apply_overrides


def _baseline_recipe() -> dict:
    return {
        "recipe_id": "base",
        "path": {
            "0_meta": {"fixed_axes": {"experiment_unit": "single_target_single_generator"}},
            "2_preprocessing": {
                "fixed_axes": {
                    "scaling_policy": "standard",
                    "tcode_policy": "raw_only",
                }
            },
            "3_training": {
                "fixed_axes": {
                    "model_family": "ols",
                    "hyperparams": {"alpha": 0.001, "max_iter": 1000},
                }
            },
        },
    }


def test_noop_override_returns_deep_copy_and_empty_diff():
    base = _baseline_recipe()
    new, diffs = apply_overrides(base, {})
    assert new == base
    assert new is not base
    assert new["path"] is not base["path"]
    assert diffs == []


def test_single_override_applied_and_diff_reported():
    base = _baseline_recipe()
    new, diffs = apply_overrides(
        base,
        {"path.2_preprocessing.fixed_axes.scaling_policy": "robust"},
    )
    assert new["path"]["2_preprocessing"]["fixed_axes"]["scaling_policy"] == "robust"
    assert diffs == [
        {
            "path": "path.2_preprocessing.fixed_axes.scaling_policy",
            "old": "standard",
            "new": "robust",
        }
    ]


def test_multiple_overrides():
    base = _baseline_recipe()
    new, diffs = apply_overrides(
        base,
        {
            "path.2_preprocessing.fixed_axes.scaling_policy": "robust",
            "path.3_training.fixed_axes.model_family": "lasso",
        },
    )
    assert new["path"]["2_preprocessing"]["fixed_axes"]["scaling_policy"] == "robust"
    assert new["path"]["3_training"]["fixed_axes"]["model_family"] == "lasso"
    assert len(diffs) == 2


def test_nested_override_descends_into_subdict():
    base = _baseline_recipe()
    new, diffs = apply_overrides(
        base,
        {"path.3_training.fixed_axes.hyperparams.alpha": 0.1},
    )
    assert new["path"]["3_training"]["fixed_axes"]["hyperparams"]["alpha"] == 0.1
    assert new["path"]["3_training"]["fixed_axes"]["hyperparams"]["max_iter"] == 1000
    assert diffs[0] == {
        "path": "path.3_training.fixed_axes.hyperparams.alpha",
        "old": 0.001,
        "new": 0.1,
    }


def test_base_not_mutated():
    base = _baseline_recipe()
    before = copy.deepcopy(base)
    apply_overrides(
        base,
        {"path.3_training.fixed_axes.model_family": "lasso"},
    )
    assert base == before


def test_missing_leaf_raises_keyerror():
    base = _baseline_recipe()
    with pytest.raises(KeyError, match="leaf key"):
        apply_overrides(
            base,
            {"path.3_training.fixed_axes.nonexistent_axis": "x"},
        )


def test_missing_intermediate_raises_keyerror():
    base = _baseline_recipe()
    with pytest.raises(KeyError, match="intermediate key"):
        apply_overrides(
            base,
            {"path.nope.fixed_axes.scaling_policy": "robust"},
        )


def test_override_into_leaf_raises_valueerror():
    base = _baseline_recipe()
    with pytest.raises(ValueError, match="not a dict"):
        apply_overrides(
            base,
            {"path.3_training.fixed_axes.model_family.something": "x"},
        )


def test_empty_segment_rejected():
    base = _baseline_recipe()
    with pytest.raises(ValueError, match="empty segment"):
        apply_overrides(base, {"path..fixed_axes.x": "y"})


def test_empty_path_rejected():
    base = _baseline_recipe()
    with pytest.raises(ValueError, match="non-empty"):
        apply_overrides(base, {"": "y"})
