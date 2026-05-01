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
    _LAYER_AXIS_GROUPS,
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


def layer_axis_groups() -> dict[str, list[dict[str, Any]]]:
    return {
        layer: [
            {
                **{key: value for key, value in group.items() if key != "axes"},
                "axes": list(group.get("axes", ())),
            }
            for group in groups
        ]
        for layer, groups in _LAYER_AXIS_GROUPS.items()
    }


_RUNTIME_SUPPORT = {
    "schema_version": "navigator_runtime_support_v1",
    "status_map": {
        "operational": "runtime_supported",
        "operational_narrow": "supported_with_contract",
        "registry_only": "schema_only",
        "gated_named": "schema_only",
        "not_supported_yet": "schema_only",
        "external_plugin": "plugin_required",
        "future": "future",
    },
    "legend": {
        "runtime_supported": {
            "label": "Runtime supported",
            "summary": "Implemented in the current local execution path for ordinary recipes.",
        },
        "supported_with_contract": {
            "label": "Supported with contract",
            "summary": "Implemented for the documented narrow contract; companion choices still matter.",
        },
        "schema_only": {
            "label": "Schema only",
            "summary": "Valid grammar or planning surface, but no full runtime path is open yet.",
        },
        "plugin_required": {
            "label": "Plugin required",
            "summary": "Needs a registered external callable or integration supplied by the user.",
        },
        "future": {
            "label": "Future",
            "summary": "Design placeholder rejected by validation or unavailable at runtime.",
        },
    },
    "layer_notes": {
        "0_meta": {
            "label": "Runtime planner",
            "summary": "Study scope, reproducibility, and serial/local execution controls are active runtime inputs.",
        },
        "1_data_task": {
            "label": "Core runtime",
            "summary": "FRED-MD/FRED-QD fixtures, custom panels, target definitions, and availability checks execute locally.",
        },
        "2_preprocessing": {
            "label": "Core runtime",
            "summary": "Transform codes, missing/outlier policies, scaling, lags, factor blocks, and selection have local execution coverage for supported options.",
        },
        "3_training": {
            "label": "Core runtime plus stubs",
            "summary": "Linear, benchmark, AR, and lightweight sklearn paths run; advanced families may remain schema-only or plugin-backed.",
        },
        "4_evaluation": {
            "label": "Core runtime",
            "summary": "Point metrics, benchmark-relative metrics, aggregation, slicing, ranking, and decomposition materialize runtime artifacts.",
        },
        "5_output_provenance": {
            "label": "Core runtime",
            "summary": "JSON/CSV exports, manifests, selected objects, diagnostics, tests, and importance summaries write to disk.",
        },
        "6_stat_tests": {
            "label": "Lightweight runtime",
            "summary": "Enabled tests produce deterministic lightweight results for point-forecast workflows; density and heavy bootstrap methods remain limited.",
        },
        "7_importance": {
            "label": "Lightweight runtime",
            "summary": "Linear coefficients, permutation-style importance, group/lineage aggregation, and transformation attribution run; full SHAP and deep methods remain schema-oriented.",
        },
    },
}


def runtime_support() -> dict[str, Any]:
    return {
        "schema_version": _RUNTIME_SUPPORT["schema_version"],
        "status_map": dict(_RUNTIME_SUPPORT["status_map"]),
        "legend": {key: dict(value) for key, value in _RUNTIME_SUPPORT["legend"].items()},
        "layer_notes": {key: dict(value) for key, value in _RUNTIME_SUPPORT["layer_notes"].items()},
    }

def navigator_ui_data(sample_paths: tuple[str | Path, ...] | None = None) -> dict[str, Any]:
    root = _repo_root()
    paths = sample_paths or _DEFAULT_SAMPLE_PATHS
    return {
        "schema_version": NAVIGATOR_UI_DATA_SCHEMA_VERSION,
        "navigator_schema_version": NAVIGATOR_SCHEMA_VERSION,
        "axis_presentation_schema_version": AXIS_PRESENTATION_SCHEMA_VERSION,
        "replication_library_version": REPLICATION_LIBRARY_VERSION,
        "layer_labels": dict(_LAYER_LABELS),
        "layer_axis_groups": layer_axis_groups(),
        "tree_axes": {layer: list(axes) for layer, axes in _TREE_AXES.items()},
        "axis_catalog": axis_catalog(),
        "axis_presentation": axis_presentation(),
        "operational_narrow_contracts": [dict(item) for item in OPERATIONAL_NARROW_CONTRACTS],
        "runtime_support": runtime_support(),
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
