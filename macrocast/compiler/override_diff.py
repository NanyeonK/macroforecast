"""Dotted-path overrides over recipe dicts.

Callers who need a compiled spec afterwards run ``compile_recipe_dict`` on the
returned dict.

The dotted-path convention is **literal**: every segment is a dict key walked
from the recipe root. Concretely, to override the preprocessing scaling_policy
axis the override path is::

    path.2_preprocessing.fixed_axes.scaling_policy

not the abbreviated ``2_preprocessing.scaling_policy`` the plan's early drafts
sketched. The literal form avoids hidden layer↔fixed_axes↔leaf_config magic
and matches the recipe dict's actual shape one-to-one. The drafts at
``plans/phases/phase_06_ablation_replication.md`` 4.5 are updated to match.
"""
from __future__ import annotations

import copy
from typing import Any


def apply_overrides(
    base_recipe_dict: dict[str, Any],
    overrides: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply dotted-path overrides to a recipe dict.

    Parameters
    ----------
    base_recipe_dict : dict
        The YAML-form recipe dict (the input to ``compile_recipe_dict``).
    overrides : dict
        Mapping of dotted path → new value. Empty dict is a no-op and
        returns a deep copy + empty diff list.

    Returns
    -------
    new_recipe_dict : dict
        ``copy.deepcopy(base_recipe_dict)`` with each override applied.
        The input is never mutated.
    diff_entries : list of dict
        One entry per applied override:
        ``{"path": "path.2_preprocessing...", "old": <value>, "new": <value>}``.

    Raises
    ------
    ValueError
        An override path has an empty segment (``"..foo"`` or ``"a..b"``),
        or a non-final segment resolves to a non-dict intermediate.
    KeyError
        A path segment does not exist in the recipe dict. The dotted path
        is *strict*: creating new keys is not allowed in Phase 6 — the
        intent is to rewire existing fields, not to splice new ones.
    """

    new_recipe_dict = copy.deepcopy(base_recipe_dict)
    diff_entries: list[dict[str, Any]] = []

    for path, new_value in overrides.items():
        _validate_path_str(path)
        parts = path.split(".")
        cursor = new_recipe_dict
        for segment in parts[:-1]:
            if not isinstance(cursor, dict):
                raise ValueError(
                    f"override path {path!r}: intermediate before segment "
                    f"{segment!r} is not a dict"
                )
            if segment not in cursor:
                raise KeyError(
                    f"override path {path!r}: intermediate key {segment!r} not found"
                )
            cursor = cursor[segment]
        if not isinstance(cursor, dict):
            raise ValueError(
                f"override path {path!r}: final parent is not a dict"
            )
        leaf = parts[-1]
        if leaf not in cursor:
            raise KeyError(f"override path {path!r}: leaf key {leaf!r} not found")
        old_value = cursor[leaf]
        cursor[leaf] = new_value
        diff_entries.append({"path": path, "old": old_value, "new": new_value})

    return new_recipe_dict, diff_entries


def _validate_path_str(path: str) -> None:
    if not isinstance(path, str) or not path:
        raise ValueError(f"override path must be a non-empty string, got {path!r}")
    if "" in path.split("."):
        raise ValueError(f"override path {path!r} contains an empty segment")


__all__ = ["apply_overrides"]
