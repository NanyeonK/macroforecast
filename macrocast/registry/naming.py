from __future__ import annotations

from copy import deepcopy
from typing import Any

AXIS_NAME_ALIASES: dict[str, str] = {}

AXIS_VALUE_ALIASES: dict[tuple[str, str], str] = {}

def canonical_axis_name(axis_name: str) -> str:
    return AXIS_NAME_ALIASES.get(str(axis_name), str(axis_name))


def canonical_axis_value(axis_name: str, value: str) -> str:
    canonical_axis = canonical_axis_name(axis_name)
    return AXIS_VALUE_ALIASES.get((canonical_axis, str(value)), str(value))


def canonicalize_recipe_path(recipe_dict: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a recipe with canonical axis/value IDs in path blocks."""

    payload = deepcopy(recipe_dict)
    path = payload.get("path")
    if not isinstance(path, dict):
        return payload

    for layer_spec in path.values():
        if not isinstance(layer_spec, dict):
            continue
        for block_name in ("fixed_axes", "sweep_axes", "conditional_axes"):
            block = layer_spec.get(block_name)
            if not isinstance(block, dict):
                continue
            canonical_block: dict[str, Any] = {}
            for raw_axis, raw_value in block.items():
                axis = canonical_axis_name(raw_axis)
                if isinstance(raw_value, list):
                    value = [canonical_axis_value(axis, str(item)) for item in raw_value]
                else:
                    value = canonical_axis_value(axis, str(raw_value))
                canonical_block[axis] = value
            layer_spec[block_name] = canonical_block
        derived = layer_spec.get("derived_axes")
        if isinstance(derived, dict):
            layer_spec["derived_axes"] = {
                canonical_axis_name(raw_axis): rule_name for raw_axis, rule_name in derived.items()
            }
    return payload
