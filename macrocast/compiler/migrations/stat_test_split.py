"""Phase 2 migration shim: legacy ``stat_test`` -> 8 axis split.

Rewrites a recipe dict's layer-6 fixed_axes so that
``{stat_test: <legacy_value>}`` becomes ``{<new_axis>: <legacy_value>}``,
emitting a DeprecationWarning exactly once per call. Idempotent; leaves
already-migrated recipe dicts untouched. Unknown legacy values raise
ValueError.
"""

from __future__ import annotations

import warnings
from typing import Any

from ...execution.stat_tests.dispatch import LEGACY_TO_NEW
from ...registry.naming import canonical_axis_value

_LAYER_STAT_TESTS = "6_stat_tests"


def _migrate_layer_block(layer_block: dict[str, Any]) -> dict[str, Any]:
    fixed = layer_block.get("fixed_axes") or {}
    if not isinstance(fixed, dict) or "stat_test" not in fixed:
        return layer_block
    legacy_value = canonical_axis_value("stat_test", str(fixed["stat_test"]))
    if legacy_value is None or legacy_value == "none":
        new_fixed = {k: v for k, v in fixed.items() if k != "stat_test"}
        new_block = dict(layer_block)
        new_block["fixed_axes"] = new_fixed
        return new_block
    if legacy_value not in LEGACY_TO_NEW:
        raise ValueError(
            f"unknown legacy stat_test value: {legacy_value!r}; "
            f"expected one of {sorted(LEGACY_TO_NEW)}"
        )
    new_axis, new_value = LEGACY_TO_NEW[legacy_value]
    warnings.warn(
        f"`stat_test: {legacy_value}` in layer_6 is deprecated; use "
        f"`{new_axis}: {new_value}`. The legacy field will be removed in "
        "v1.2 (ADR-006 breaking-change window).",
        DeprecationWarning,
        stacklevel=4,
    )
    new_fixed = dict(fixed)
    new_fixed.setdefault(new_axis, new_value)
    new_block = dict(layer_block)
    new_block["fixed_axes"] = new_fixed
    return new_block


def migrate_legacy_stat_test(recipe_dict: dict[str, Any]) -> dict[str, Any]:
    """Return a new recipe dict with legacy ``stat_test`` rewritten.

    Idempotent: recipes without a legacy field pass through unchanged.
    Mutates neither the input dict nor its nested blocks.
    """

    if not isinstance(recipe_dict, dict):
        return recipe_dict
    path = recipe_dict.get("path")
    if not isinstance(path, dict):
        return recipe_dict
    layer_block = path.get(_LAYER_STAT_TESTS)
    if not isinstance(layer_block, dict):
        return recipe_dict
    new_layer = _migrate_layer_block(layer_block)
    if new_layer is layer_block:
        return recipe_dict
    new_path = dict(path)
    new_path[_LAYER_STAT_TESTS] = new_layer
    new_recipe = dict(recipe_dict)
    new_recipe["path"] = new_path
    return new_recipe
