"""Tests for macroforecast.wizard.state — Scenarios S-01..S-11 + INV-01..INV-02."""
from __future__ import annotations

import copy
import tempfile
import os
import pytest

from macroforecast.wizard.state import (
    RecipeState,
    current_recipe,
    validation_errors,
    yaml_text,
    yaml_edit_mode,
)


def _reset_state():
    """Reset all reactive state to defaults before each test."""
    current_recipe.set({})
    validation_errors.set([])
    yaml_text.set("")
    yaml_edit_mode.set(False)


# ---------------------------------------------------------------------------
# S-01: Set axis and verify recipe update
# ---------------------------------------------------------------------------

def test_s01_set_axis_updates_recipe():
    """S-01: set_axis('l0', 'failure_policy', 'fail_fast') -> recipe dict set."""
    _reset_state()
    RecipeState.set_axis("l0", "failure_policy", "fail_fast")
    assert current_recipe.value == {
        "0_meta": {
            "fixed_axes": {"failure_policy": "fail_fast"},
            "leaf_config": {},
        }
    }


# ---------------------------------------------------------------------------
# S-02: Set axis triggers YAML sync
# ---------------------------------------------------------------------------

def test_s02_set_axis_triggers_yaml_sync():
    """S-02: set_axis -> yaml_text.value contains the new axis value."""
    _reset_state()
    RecipeState.set_axis("l0", "failure_policy", "continue_on_failure")
    assert "failure_policy: continue_on_failure" in yaml_text.value


# ---------------------------------------------------------------------------
# S-03: Get axis returns None for missing key
# ---------------------------------------------------------------------------

def test_s03_get_axis_missing_returns_none():
    """S-03: get_axis for nonexistent axis returns None."""
    _reset_state()
    result = RecipeState.get_axis("l0", "nonexistent_axis")
    assert result is None


# ---------------------------------------------------------------------------
# S-04: Get axis returns stored value
# ---------------------------------------------------------------------------

def test_s04_get_axis_returns_stored_value():
    """S-04: get_axis returns the value written into fixed_axes."""
    _reset_state()
    current_recipe.set({"0_meta": {"fixed_axes": {"compute_mode": "parallel"}}})
    result = RecipeState.get_axis("l0", "compute_mode")
    assert result == "parallel"


# ---------------------------------------------------------------------------
# S-05: Set leaf config
# ---------------------------------------------------------------------------

def test_s05_set_leaf_config():
    """S-05: set_leaf('l0', 'random_seed', 42) writes to leaf_config."""
    _reset_state()
    RecipeState.set_leaf("l0", "random_seed", 42)
    assert current_recipe.value == {
        "0_meta": {
            "fixed_axes": {},
            "leaf_config": {"random_seed": 42},
        }
    }


# ---------------------------------------------------------------------------
# S-06: YAML round-trip for L0 full block (idempotent)
# ---------------------------------------------------------------------------

def test_s06_yaml_roundtrip_idempotent():
    """S-06: sync_recipe_to_yaml() twice produces same yaml_text."""
    _reset_state()
    RecipeState.set_axis("l0", "failure_policy", "fail_fast")
    RecipeState.set_axis("l0", "reproducibility_mode", "seeded_reproducible")
    RecipeState.set_axis("l0", "compute_mode", "serial")
    RecipeState.set_leaf("l0", "random_seed", 0)
    yaml_captured = yaml_text.value
    # no-op sync (already synced)
    RecipeState.sync_yaml_to_recipe()
    # re-sync recipe -> yaml
    RecipeState.sync_recipe_to_yaml()
    assert yaml_text.value == yaml_captured


# ---------------------------------------------------------------------------
# S-07: YAML sync with full multi-layer recipe (deep equality)
# ---------------------------------------------------------------------------

def test_s07_yaml_sync_multilayer_roundtrip():
    """S-07: sync_recipe_to_yaml then sync_yaml_to_recipe preserves deep equality."""
    _reset_state()
    original = {
        "0_meta": {"fixed_axes": {"failure_policy": "fail_fast"}},
        "1_data": {"fixed_axes": {"dataset": "fred_md"}},
        "5_evaluation": {"fixed_axes": {"primary_metric": "mae"}},
    }
    current_recipe.set(copy.deepcopy(original))
    RecipeState.sync_recipe_to_yaml()
    # Now clear and re-parse from YAML
    current_recipe.set({})
    RecipeState.sync_yaml_to_recipe()
    assert current_recipe.value == original


# ---------------------------------------------------------------------------
# S-08: sync_yaml_to_recipe with malformed YAML
# ---------------------------------------------------------------------------

def test_s08_malformed_yaml_does_not_corrupt_recipe():
    """S-08: malformed YAML -> current_recipe unchanged + validation_errors set."""
    _reset_state()
    original = {"0_meta": {"fixed_axes": {"failure_policy": "fail_fast"}}}
    current_recipe.set(copy.deepcopy(original))
    # Inject malformed YAML
    yaml_text.set("this: is: : not: valid:: yaml{{{}")
    RecipeState.sync_yaml_to_recipe()
    # recipe must not be corrupted
    assert current_recipe.value == original
    # validation_errors must be non-empty and contain "YAML parse error"
    assert len(validation_errors.value) > 0
    assert any("YAML parse error" in e for e in validation_errors.value)


# ---------------------------------------------------------------------------
# S-09: run_validation with empty recipe -> no errors
# ---------------------------------------------------------------------------

def test_s09_run_validation_empty_recipe():
    """S-09: run_validation on empty recipe returns empty error list."""
    _reset_state()
    RecipeState.run_validation()
    assert validation_errors.value == []


# ---------------------------------------------------------------------------
# S-10: load_from_path loads valid YAML
# ---------------------------------------------------------------------------

def test_s10_load_from_path_valid_yaml(tmp_path):
    """S-10: load_from_path reads YAML into current_recipe and yaml_text."""
    _reset_state()
    yaml_content = "0_meta:\n  fixed_axes:\n    failure_policy: fail_fast\n"
    recipe_file = tmp_path / "recipe.yaml"
    recipe_file.write_text(yaml_content, encoding="utf-8")
    RecipeState.load_from_path(str(recipe_file))
    assert current_recipe.value == {"0_meta": {"fixed_axes": {"failure_policy": "fail_fast"}}}
    assert "failure_policy: fail_fast" in yaml_text.value


# ---------------------------------------------------------------------------
# S-11: load_from_path raises FileNotFoundError for missing file
# ---------------------------------------------------------------------------

def test_s11_load_from_path_missing_file_raises():
    """S-11: load_from_path on nonexistent path raises FileNotFoundError."""
    _reset_state()
    with pytest.raises(FileNotFoundError):
        RecipeState.load_from_path("/nonexistent/path/recipe.yaml")


# ---------------------------------------------------------------------------
# INV-01: sync_recipe_to_yaml is idempotent
# ---------------------------------------------------------------------------

def test_inv01_sync_recipe_to_yaml_idempotent():
    """INV-01: sync_recipe_to_yaml() twice -> same yaml_text (f(f(x)) == f(x))."""
    _reset_state()
    test_recipes = [
        {},
        {"0_meta": {"fixed_axes": {"failure_policy": "fail_fast"}}},
        {
            "0_meta": {"fixed_axes": {"failure_policy": "continue_on_failure"}},
            "1_data": {"fixed_axes": {"dataset": "fred_md"}},
        },
    ]
    for recipe in test_recipes:
        current_recipe.set(copy.deepcopy(recipe))
        RecipeState.sync_recipe_to_yaml()
        first_yaml = yaml_text.value
        RecipeState.sync_recipe_to_yaml()
        second_yaml = yaml_text.value
        assert first_yaml == second_yaml, (
            f"sync_recipe_to_yaml not idempotent for recipe={recipe!r}: "
            f"first={first_yaml!r}, second={second_yaml!r}"
        )


# ---------------------------------------------------------------------------
# INV-02: Round-trip identity (YAML -> recipe -> YAML)
# ---------------------------------------------------------------------------

def test_inv02_yaml_roundtrip_identity():
    """INV-02: For parseable YAML y: sync_recipe_to_yaml(sync_yaml_to_recipe(y)) == y."""
    import yaml as _yaml

    _reset_state()
    test_yamls = [
        "0_meta:\n  fixed_axes:\n    failure_policy: fail_fast\n",
        "0_meta:\n  fixed_axes:\n    failure_policy: fail_fast\n    compute_mode: serial\n",
    ]
    for y in test_yamls:
        # Parse y -> set current_recipe -> dump back -> compare
        parsed = _yaml.safe_load(y)
        current_recipe.set(parsed or {})
        RecipeState.sync_recipe_to_yaml()
        # The dumped YAML must parse to the same dict as y
        dumped = yaml_text.value
        assert _yaml.safe_load(dumped) == parsed, (
            f"Round-trip failed for y={y!r}: dumped={dumped!r}"
        )
