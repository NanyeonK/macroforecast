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
from .replications import REPLICATION_LIBRARY_VERSION, list_replication_entries

NAVIGATOR_UI_DATA_SCHEMA_VERSION = "navigator_ui_data_v1"

_DEFAULT_SAMPLE_PATHS = (
    "examples/recipes/model-benchmark.yaml",
    "examples/recipes/replications/synthetic-replication-roundtrip.yaml",
    "examples/recipes/replications/goulet-coulombe-2021-fred-md-ridge.yaml",
)

_L0_AXIS_PRESENTATION: dict[str, dict[str, Any]] = {
    "research_design": {
        "order": 1,
        "label": "Study route",
        "question": "What kind of forecasting study is this?",
        "summary": "Selects the top-level runner route: single run, controlled comparison, wrapper bundle, or replication.",
        "docs_url": "../user_guide/design.html#research-design",
        "contract": "User-facing runner route. This affects executor family, sweep interpretation, and artifact shape.",
        "selection_kind": "user_choice",
        "values": {
            "single_path_benchmark": {
                "label": "Single recipe run",
                "summary": "One chosen path produces forecasts, metrics, and artifacts.",
            },
            "controlled_variation": {
                "label": "Controlled comparison",
                "summary": "One or more axes vary while the rest of the path is held fixed.",
            },
            "orchestrated_bundle": {
                "label": "Wrapper bundle",
                "summary": "Routes through a higher-level bundle runner when that runner has a concrete contract.",
            },
            "replication_override": {
                "label": "Replication route",
                "summary": "Reruns a known recipe route and records replication-oriented provenance.",
            },
        },
    },
    "experiment_unit": {
        "order": 2,
        "label": "Runner unit",
        "question": "Which execution unit owns this recipe?",
        "summary": "Usually derived by the compiler from study route, target structure, and sweep shape.",
        "docs_url": "../user_guide/design.html#experiment-unit",
        "contract": "Derived runner contract. Explicit values must match target and sweep constraints.",
        "selection_kind": "usually_derived",
        "values": {
            "single_target_single_model": {
                "label": "Single target, single model",
                "summary": "Default unit for one target with no model sweep.",
            },
            "single_target_model_grid": {
                "label": "Single target model grid",
                "summary": "One target with model-family variants.",
            },
            "single_target_full_sweep": {
                "label": "Single target full sweep",
                "summary": "Reserved grammar for wider sweep orchestration.",
            },
            "multi_target_separate_runs": {
                "label": "Multi-target separate runs",
                "summary": "Each target runs independently and writes separate run outputs.",
            },
            "multi_target_shared_design": {
                "label": "Multi-target shared design",
                "summary": "Targets share the same data and representation design.",
            },
            "hierarchical_forecasting_run": {
                "label": "Hierarchical forecasting run",
                "summary": "Reserved route for hierarchy-aware forecast execution.",
            },
            "panel_forecasting_run": {
                "label": "Panel forecasting run",
                "summary": "Reserved route for panel-oriented forecast execution.",
            },
            "state_space_run": {
                "label": "State-space run",
                "summary": "Reserved route for state-space forecast execution.",
            },
            "replication_recipe": {
                "label": "Replication recipe",
                "summary": "Execution unit derived for replication routes.",
            },
            "benchmark_suite": {
                "label": "Benchmark suite",
                "summary": "Reserved route for a bundled benchmark suite.",
            },
            "ablation_study": {
                "label": "Ablation study",
                "summary": "Reserved route for ablation-oriented execution.",
            },
        },
    },
    "failure_policy": {
        "order": 3,
        "label": "Failure handling",
        "question": "What should happen when a variant or cell fails?",
        "summary": "Controls whether the run stops immediately, skips failed cells, or preserves partial results.",
        "docs_url": "../user_guide/design.html#failure-policy",
        "contract": "Runtime failure contract. Sweep-compatible modes decide whether invalid variants stop or are skipped.",
        "selection_kind": "user_choice",
        "values": {
            "fail_fast": {
                "label": "Stop on first failure",
                "summary": "Abort immediately so the first error can be investigated.",
            },
            "skip_failed_cell": {
                "label": "Skip failed cell",
                "summary": "Continue a sweep while recording the failed variant status.",
            },
            "skip_failed_model": {
                "label": "Skip failed model",
                "summary": "Continue after failures scoped to a model-family branch.",
            },
            "retry_then_skip": {
                "label": "Retry, then skip",
                "summary": "Reserved policy for retryable cells before skipping.",
            },
            "fallback_to_default_hp": {
                "label": "Fallback hyperparameters",
                "summary": "Reserved policy for using default hyperparameters after tuning failure.",
            },
            "save_partial_results": {
                "label": "Save partial results",
                "summary": "Persist completed artifacts before aborting or reporting failure.",
            },
            "warn_only": {
                "label": "Warn only",
                "summary": "Emit warnings and continue when failures are recoverable.",
            },
        },
    },
    "reproducibility_mode": {
        "order": 4,
        "label": "Reproducibility",
        "question": "How strictly should stochastic components be pinned?",
        "summary": "Controls seeds and deterministic-library settings before model execution.",
        "docs_url": "../user_guide/design.html#reproducibility-mode",
        "contract": "Seed and determinism contract applied before stochastic model code runs.",
        "selection_kind": "user_choice",
        "values": {
            "strict_reproducible": {
                "label": "Strict reproducibility",
                "summary": "Use deterministic-library settings for replication-grade reruns.",
            },
            "seeded_reproducible": {
                "label": "Seeded reproducibility",
                "summary": "Seed Python, numpy, and optional torch without strict backend flags.",
            },
            "best_effort": {
                "label": "Best effort",
                "summary": "Apply available seeds and mark the run as non-strict.",
            },
            "exploratory": {
                "label": "Exploratory",
                "summary": "Do not force deterministic behavior.",
            },
        },
    },
    "compute_mode": {
        "order": 5,
        "label": "Compute layout",
        "question": "Where should parallel work be attempted?",
        "summary": "Requests serial or parallel execution across models, horizons, targets, OOS dates, or trials.",
        "docs_url": "../user_guide/design.html#compute-mode",
        "contract": "Execution parallelism contract. Unsupported modes or singleton work units degrade to serial/no-op behavior.",
        "selection_kind": "user_choice",
        "values": {
            "serial": {
                "label": "Serial execution",
                "summary": "Run one unit of work at a time.",
            },
            "parallel_by_model": {
                "label": "Parallel by model",
                "summary": "Parallelize across model-family variants when a model sweep exists.",
            },
            "parallel_by_horizon": {
                "label": "Parallel by horizon",
                "summary": "Parallelize across forecast horizons when multiple horizons exist.",
            },
            "parallel_by_target": {
                "label": "Parallel by target",
                "summary": "Parallelize across targets in a multi-target run.",
            },
            "parallel_by_oos_date": {
                "label": "Parallel by OOS date",
                "summary": "Parallelize across origin dates in a long pseudo-OOS window.",
            },
            "parallel_by_trial": {
                "label": "Parallel by trial",
                "summary": "Reserved route for trial-level parallelism.",
            },
            "distributed_cluster": {
                "label": "Distributed cluster",
                "summary": "Reserved route for cluster-managed execution.",
            },
        },
    },
}


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
    return {
        axis_name: {
            **{key: value for key, value in spec.items() if key != "values"},
            "values": {value_name: dict(value_spec) for value_name, value_spec in spec.get("values", {}).items()},
        }
        for axis_name, spec in _L0_AXIS_PRESENTATION.items()
    }


def navigator_ui_data(sample_paths: tuple[str | Path, ...] | None = None) -> dict[str, Any]:
    root = _repo_root()
    paths = sample_paths or _DEFAULT_SAMPLE_PATHS
    return {
        "schema_version": NAVIGATOR_UI_DATA_SCHEMA_VERSION,
        "navigator_schema_version": NAVIGATOR_SCHEMA_VERSION,
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
