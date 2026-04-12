from __future__ import annotations

from typing import Any


def validate_recipe_schema(schema: dict[str, Any]) -> dict[str, Any]:
    root = schema.get('recipe_schema')
    if not isinstance(root, dict):
        raise ValueError('recipe_schema root must be a dict')
    for key in ['required_top_level', 'required_taxonomy_path_keys', 'allowed_kinds']:
        if key not in root or not isinstance(root[key], list) or not root[key]:
            raise ValueError(f'recipe schema missing non-empty list: {key}')
    return schema


def validate_recipe(recipe: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    root = schema['recipe_schema']
    for key in root['required_top_level']:
        if key not in recipe:
            raise ValueError(f'recipe missing key: {key}')
    if recipe['kind'] not in root['allowed_kinds']:
        raise ValueError(f'invalid recipe kind: {recipe["kind"]}')
    path = recipe['taxonomy_path']
    if not isinstance(path, dict):
        raise ValueError('taxonomy_path must be a dict')
    for key in root['required_taxonomy_path_keys']:
        if key not in path:
            raise ValueError(f'taxonomy_path missing key: {key}')
    if not isinstance(recipe['numeric_params'], dict):
        raise ValueError('numeric_params must be a dict')
    if not isinstance(recipe['outputs'], dict):
        raise ValueError('outputs must be a dict')
    return recipe
