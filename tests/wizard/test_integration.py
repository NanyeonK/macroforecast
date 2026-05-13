"""Integration tests — Scenarios IT-01..IT-04."""
from __future__ import annotations

import copy
import pytest

from macroforecast.wizard.state import (
    RecipeState,
    current_recipe,
    validation_errors,
    yaml_text,
    yaml_edit_mode,
)


def _reset_state():
    current_recipe.set({})
    validation_errors.set([])
    yaml_text.set("")
    yaml_edit_mode.set(False)


# ---------------------------------------------------------------------------
# IT-01: L0 full form -> YAML export -> re-parse -> identical state
# ---------------------------------------------------------------------------

def test_it01_l0_full_roundtrip(tmp_path):
    """IT-01: Fill all L0 axes -> export YAML -> load fresh -> recipe identical."""
    _reset_state()

    # Step 2: simulate user filling all L0 axes
    RecipeState.set_axis("l0", "failure_policy", "continue_on_failure")
    RecipeState.set_axis("l0", "reproducibility_mode", "exploratory")
    RecipeState.set_axis("l0", "compute_mode", "parallel")
    RecipeState.set_leaf("l0", "random_seed", 42)

    # Step 3: capture before
    recipe_before = copy.deepcopy(current_recipe.value)

    # Step 4: write yaml to temp file
    yaml_file = tmp_path / "test_recipe.yaml"
    yaml_file.write_text(yaml_text.value, encoding="utf-8")

    # Step 5: reset state and load from file
    _reset_state()
    RecipeState.load_from_path(str(yaml_file))

    # Step 6: compare
    recipe_after = current_recipe.value
    assert recipe_before == recipe_after, (
        f"Recipe mismatch after round-trip.\n"
        f"Before: {recipe_before}\n"
        f"After:  {recipe_after}"
    )


# ---------------------------------------------------------------------------
# IT-02: YAML import -> all layer states restored
# ---------------------------------------------------------------------------

def test_it02_multi_layer_yaml_import(tmp_path):
    """IT-02: Multi-layer YAML import restores all axes correctly."""
    _reset_state()

    multi_layer_yaml = (
        "0_meta:\n"
        "  fixed_axes:\n"
        "    failure_policy: continue_on_failure\n"
        "1_data:\n"
        "  fixed_axes:\n"
        "    dataset: fred_md\n"
        "5_evaluation:\n"
        "  fixed_axes:\n"
        "    primary_metric: rmse\n"
    )
    yaml_file = tmp_path / "multi_layer.yaml"
    yaml_file.write_text(multi_layer_yaml, encoding="utf-8")

    RecipeState.load_from_path(str(yaml_file))

    assert RecipeState.get_axis("l0", "failure_policy") == "continue_on_failure", (
        f"l0/failure_policy mismatch: {RecipeState.get_axis('l0', 'failure_policy')}"
    )
    assert RecipeState.get_axis("l5", "primary_metric") == "rmse", (
        f"l5/primary_metric mismatch: {RecipeState.get_axis('l5', 'primary_metric')}"
    )


# ---------------------------------------------------------------------------
# IT-03: Validation errors surfaced for bad recipe
# ---------------------------------------------------------------------------

def test_it03_validation_errors_for_bad_recipe():
    """IT-03: run_validation on invalid dataset raises validation errors."""
    _reset_state()
    current_recipe.set({"1_data": {"fixed_axes": {"dataset": "nonexistent_dataset_xyz"}}})
    RecipeState.run_validation()

    # Tolerance per spec: if l1 validator rejects the dataset, errors non-empty.
    # If validator does not reject, verify it was called without crashing.
    # In practice: the validator DOES reject unknown dataset per our pre-test check.
    # So we assert non-empty.
    assert isinstance(validation_errors.value, list), (
        "validation_errors.value must be a list"
    )
    # Non-empty (the validator rejects nonexistent_dataset_xyz)
    assert len(validation_errors.value) > 0, (
        "Expected validation errors for dataset='nonexistent_dataset_xyz', got none"
    )


# ---------------------------------------------------------------------------
# IT-04: Validation passes for empty recipe
# ---------------------------------------------------------------------------

def test_it04_validation_passes_empty_recipe():
    """IT-04: run_validation on empty recipe returns []."""
    _reset_state()
    RecipeState.run_validation()
    assert validation_errors.value == [], (
        f"Expected no validation errors for empty recipe, got: {validation_errors.value}"
    )
