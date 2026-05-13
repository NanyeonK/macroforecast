"""Reactive state singletons and RecipeState helpers for the Wizard UI.

All reactive variables are module-level singletons so they can be shared
across Solara components without passing props down the tree.

Bidirectional sync invariant:
    set_axis / set_leaf  -> sync_recipe_to_yaml()
    sync_yaml_to_recipe  -> run_validation()

Do not call solara.reactive() inside a component body.
"""
from __future__ import annotations

from typing import Any

try:
    import solara
except ImportError as exc:
    raise ImportError(
        "macroforecast wizard requires the [wizard] extra. "
        "Install with: pip install 'macroforecast[wizard]'"
    ) from exc

import yaml

# ---------------------------------------------------------------------------
# Reactive singletons
# ---------------------------------------------------------------------------

current_recipe: solara.Reactive[dict] = solara.reactive({})
"""The current recipe dict (mutable). All components read/write via this."""

selected_layer: solara.Reactive[str] = solara.reactive("overview")
"""The currently selected layer ('l0'..'l8' or 'overview')."""

validation_errors: solara.Reactive[list] = solara.reactive([])
"""Validation errors: list of human-readable strings."""

yaml_text: solara.Reactive[str] = solara.reactive("")
"""The YAML text shown in the right pane. Kept in sync with current_recipe."""

yaml_edit_mode: solara.Reactive[bool] = solara.reactive(False)
"""Whether the user is editing the YAML pane directly."""


# ---------------------------------------------------------------------------
# Type coercion (copied from scaffold/wizard.py to avoid circular import)
# ---------------------------------------------------------------------------

def _coerce_value(raw: str, default: Any) -> Any:
    """Match the type of the default when possible.

    Keeps int defaults int after a numeric input, etc.
    Falls back to str on failure.
    """
    if isinstance(default, bool):
        return str(raw).lower() in {"true", "yes", "y", "1"}
    if isinstance(default, int) and not isinstance(default, bool):
        try:
            return int(raw)
        except (ValueError, TypeError):
            return raw
    if isinstance(default, float):
        try:
            return float(raw)
        except (ValueError, TypeError):
            return raw
    return raw


# ---------------------------------------------------------------------------
# RecipeState
# ---------------------------------------------------------------------------

class RecipeState:
    """Stateless helpers that operate on the module-level reactive variables.

    All methods are @staticmethod so they can be called without
    instantiating RecipeState.
    """

    LAYER_KEYS: dict[str, str] = {
        "l0":   "0_meta",
        "l1":   "1_data",
        "l1_5": "1_5_data_summary",
        "l2":   "2_preprocessing",
        "l2_5": "2_5_pre_post_preprocessing",
        "l3":   "3_feature_engineering",
        "l3_5": "3_5_feature_diagnostics",
        "l4":   "4_forecasting_model",
        "l4_5": "4_5_generator_diagnostics",
        "l5":   "5_evaluation",
        "l6":   "6_statistical_tests",
        "l7":   "7_interpretation",
        "l8":   "8_output",
    }

    @staticmethod
    def _ensure_layer_block(recipe: dict, layer_id: str) -> dict:
        """Ensure the layer block exists in the recipe and return it."""
        key = RecipeState.LAYER_KEYS[layer_id]
        if key not in recipe:
            recipe[key] = {}
        block = recipe[key]
        if "fixed_axes" not in block:
            block["fixed_axes"] = {}
        if "leaf_config" not in block:
            block["leaf_config"] = {}
        return block

    @staticmethod
    def set_axis(layer_id: str, axis_name: str, value: Any) -> None:
        """Write one fixed_axes value into current_recipe and sync yaml_text."""
        recipe = dict(current_recipe.value)
        block = RecipeState._ensure_layer_block(recipe, layer_id)
        block["fixed_axes"][axis_name] = value
        current_recipe.value = recipe
        RecipeState.sync_recipe_to_yaml()

    @staticmethod
    def get_axis(layer_id: str, axis_name: str) -> Any:
        """Read one fixed_axes value from current_recipe. Returns None if absent."""
        key = RecipeState.LAYER_KEYS.get(layer_id)
        if key is None:
            return None
        block = current_recipe.value.get(key, {})
        return block.get("fixed_axes", {}).get(axis_name)

    @staticmethod
    def set_leaf(layer_id: str, leaf_key: str, value: Any) -> None:
        """Write one leaf_config entry."""
        recipe = dict(current_recipe.value)
        block = RecipeState._ensure_layer_block(recipe, layer_id)
        block["leaf_config"][leaf_key] = value
        current_recipe.value = recipe
        RecipeState.sync_recipe_to_yaml()

    @staticmethod
    def get_leaf(layer_id: str, leaf_key: str) -> Any:
        """Read one leaf_config entry."""
        key = RecipeState.LAYER_KEYS.get(layer_id)
        if key is None:
            return None
        block = current_recipe.value.get(key, {})
        return block.get("leaf_config", {}).get(leaf_key)

    @staticmethod
    def sync_recipe_to_yaml() -> None:
        """Serialize current_recipe to YAML string -> yaml_text.value."""
        try:
            text = yaml.safe_dump(current_recipe.value, sort_keys=False, allow_unicode=True)
        except yaml.YAMLError:
            text = ""
        yaml_text.value = text

    @staticmethod
    def sync_yaml_to_recipe() -> None:
        """Parse yaml_text.value -> current_recipe.

        On parse error: set validation_errors with a parse error message;
        do not corrupt current_recipe.
        """
        try:
            parsed = yaml.safe_load(yaml_text.value)
        except yaml.YAMLError as e:
            validation_errors.value = [f"YAML parse error: {e}"]
            return
        if parsed is None:
            parsed = {}
        if not isinstance(parsed, dict):
            validation_errors.value = ["YAML parse error: top-level value must be a mapping"]
            return
        current_recipe.value = parsed
        RecipeState.run_validation()

    @staticmethod
    def run_validation() -> None:
        """Call RecipeBuilder.validate() on current_recipe -> validation_errors."""
        try:
            from macroforecast.scaffold.builder import RecipeBuilder
            builder = RecipeBuilder()
            # Inject the current recipe into the builder for validation
            builder._recipe.update(current_recipe.value)
            errors = builder.validate()
            validation_errors.value = list(errors) if errors else []
        except Exception as exc:  # noqa: BLE001
            # Validation is best-effort; do not crash the UI
            validation_errors.value = [f"Validation error: {exc}"]

    @staticmethod
    def load_from_path(path: str) -> None:
        """Read a YAML file -> current_recipe -> trigger sync_recipe_to_yaml()."""
        import os
        if not os.path.exists(path):
            raise FileNotFoundError(f"Recipe file not found: {path}")
        with open(path, encoding="utf-8") as fh:
            parsed = yaml.safe_load(fh)
        if parsed is None:
            parsed = {}
        current_recipe.value = parsed
        RecipeState.sync_recipe_to_yaml()

    @staticmethod
    def export_to_path(path: str) -> None:
        """Write yaml_text.value to disk at path."""
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(yaml_text.value)


__all__ = [
    "current_recipe",
    "selected_layer",
    "validation_errors",
    "yaml_text",
    "yaml_edit_mode",
    "RecipeState",
    "_coerce_value",
]
