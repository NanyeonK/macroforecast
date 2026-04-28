from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from ..registry import get_axis_registry
from .core import (
    NAVIGATOR_SCHEMA_VERSION,
    OPERATIONAL_NARROW_CONTRACTS,
    _LAYER_LABELS,
    _TREE_AXES,
    _VIRTUAL_AXES,
    _VIRTUAL_AXIS_STATUSES,
    build_navigation_view,
    load_recipe,
    navigator_state_engine_spec,
)
from .presentation import AXIS_PRESENTATION_SCHEMA_VERSION, axis_presentation_map
from .replications import REPLICATION_LIBRARY_VERSION, list_replication_entries

NAVIGATOR_UI_DATA_SCHEMA_VERSION = "navigator_ui_data_v1"

_DEFAULT_SAMPLE_PATHS = (
    "examples/recipes/model-benchmark.yaml",
    "examples/recipes/replications/synthetic-replication-roundtrip.yaml",
    "examples/recipes/replications/goulet-coulombe-2021-fred-md-ridge.yaml",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _stable_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _read_sample(path: str | Path, root: Path) -> dict[str, Any]:
    sample_path = Path(path)
    if sample_path.is_absolute():
        resolved = sample_path
    else:
        cwd_candidate = Path.cwd() / sample_path
        root_candidate = root / sample_path
        resolved = cwd_candidate if cwd_candidate.exists() else root_candidate
    recipe = load_recipe(resolved)
    return {
        "id": str(sample_path),
        "label": str(recipe.get("recipe_id", sample_path.stem)),
        "path": str(sample_path),
        "recipe": recipe,
        "recipe_yaml": yaml.safe_dump(recipe, sort_keys=False),
        "view": build_navigation_view(recipe),
    }


def axis_catalog() -> dict[str, Any]:
    registry = get_axis_registry()
    catalog: dict[str, Any] = {}
    for axis_name, entry in sorted(registry.items()):
        catalog[axis_name] = {
            "axis_name": entry.axis_name,
            "layer": entry.layer,
            "axis_type": entry.axis_type,
            "allowed_values": list(entry.allowed_values),
            "current_status": dict(entry.current_status),
            "default_policy": entry.default_policy,
            "compatible_with": {key: list(value) for key, value in entry.compatible_with.items()},
            "incompatible_with": {key: list(value) for key, value in entry.incompatible_with.items()},
        }
    for axis_name, values in sorted(_VIRTUAL_AXES.items()):
        catalog[axis_name] = {
            "axis_name": axis_name,
            "layer": "3_training",
            "axis_type": "virtual",
            "allowed_values": list(values),
            "current_status": {
                value: _VIRTUAL_AXIS_STATUSES.get(axis_name, {}).get(value, "unknown") for value in values
            },
            "default_policy": "fixed",
            "compatible_with": {},
            "incompatible_with": {},
        }
    return catalog


def axis_presentation() -> dict[str, Any]:
    return axis_presentation_map()


def navigator_ui_data(sample_paths: tuple[str | Path, ...] | None = None) -> dict[str, Any]:
    root = _repo_root()
    paths = sample_paths or _DEFAULT_SAMPLE_PATHS
    return {
        "schema_version": NAVIGATOR_UI_DATA_SCHEMA_VERSION,
        "navigator_schema_version": NAVIGATOR_SCHEMA_VERSION,
        "axis_presentation_schema_version": AXIS_PRESENTATION_SCHEMA_VERSION,
        "replication_library_version": REPLICATION_LIBRARY_VERSION,
        "layer_labels": dict(_LAYER_LABELS),
        "tree_axes": {layer: list(axes) for layer, axes in _TREE_AXES.items()},
        "axis_catalog": axis_catalog(),
        "axis_presentation": axis_presentation(),
        "operational_narrow_contracts": [dict(item) for item in OPERATIONAL_NARROW_CONTRACTS],
        "state_engine": navigator_state_engine_spec(),
        "samples": [_read_sample(path, root) for path in paths],
        "replications": list_replication_entries(),
    }


def write_navigator_ui_data(
    output_path: str | Path,
    *,
    sample_paths: tuple[str | Path, ...] | None = None,
    check: bool = False,
) -> Path:
    output = Path(output_path)
    payload = _stable_json(navigator_ui_data(sample_paths))
    if check:
        current = output.read_text(encoding="utf-8") if output.exists() else ""
        if current != payload:
            raise ValueError(f"navigator UI data is stale: regenerate {output}")
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
    return output
