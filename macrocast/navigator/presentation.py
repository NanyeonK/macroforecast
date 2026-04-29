from __future__ import annotations

from typing import Any

AXIS_PRESENTATION_SCHEMA_VERSION = "axis_presentation_v1"

# User-facing labels live here. Registry IDs remain the canonical API/YAML
# values; labels are the docs/Navigator surface shown to researchers.
AXIS_PRESENTATION_MAP: dict[str, dict[str, Any]] = {
    "study_scope": {
        "order": 1,
        "label": "Study Scope",
        "short_label": "Scope",
        "question": "How many targets and methods should this study compare?",
        "summary": "Sets target cardinality and whether the method path is fixed or compared across alternatives.",
        "docs_url": "../detail/layer0/study_scope.html",
        "contract": "Primary Layer 0 study-shape contract. It derives the Layer 1 target_structure and determines whether downstream method axes are fixed or sweep-aware.",
        "selection_kind": "user_choice",
        "values": {
            "one_target_one_method": {
                "label": "One Target, One Method",
                "short_label": "1 Target / 1 Method",
                "summary": "Use one target and one fixed forecasting method path.",
            },
            "one_target_compare_methods": {
                "label": "One Target, Compare Methods",
                "short_label": "1 Target / Compare",
                "summary": "Use one target and compare multiple model, representation, preprocessing, horizon, or tuning choices.",
            },
            "multiple_targets_one_method": {
                "label": "Multiple Targets, One Method",
                "short_label": "Multi Target / 1 Method",
                "summary": "Use multiple targets and one fixed forecasting method path shared across targets.",
            },
            "multiple_targets_compare_methods": {
                "label": "Multiple Targets, Compare Methods",
                "short_label": "Multi Target / Compare",
                "summary": "Use multiple targets and compare one or more downstream method axes across the same study.",
            },
        },
    },
    "failure_policy": {
        "order": 3,
        "label": "Failure Handling",
        "short_label": "Failures",
        "question": "What should happen when a run, variant, or cell fails?",
        "summary": "Controls whether the run stops, skips failed units, warns, or preserves partial results.",
        "docs_url": "../detail/layer0/failure_policy.html",
        "contract": "Runtime failure contract. Sweep-compatible modes decide whether invalid variants stop or are skipped.",
        "selection_kind": "user_choice",
        "values": {
            "fail_fast": {
                "label": "Stop on First Failure",
                "short_label": "Stop",
                "summary": "Abort immediately so the first error can be investigated.",
            },
            "skip_failed_cell": {
                "label": "Skip Failed Sweep Cells",
                "short_label": "Skip Cells",
                "summary": "Continue a sweep while recording failed variant status.",
            },
            "skip_failed_model": {
                "label": "Skip Failed Model Branches",
                "short_label": "Skip Models",
                "summary": "Continue after failures scoped to one model branch.",
            },
            "retry_then_skip": {
                "label": "Retry Then Skip",
                "short_label": "Retry/Skip",
                "summary": "Reserved policy for retryable cells before skipping.",
            },
            "fallback_to_default_hp": {
                "label": "Use Default Hyperparameters",
                "short_label": "Default HP",
                "summary": "Reserved policy for using default hyperparameters after tuning failure.",
            },
            "save_partial_results": {
                "label": "Save Partial Results",
                "short_label": "Save Partial",
                "summary": "Persist completed artifacts before aborting or reporting failure.",
            },
            "warn_only": {
                "label": "Warn Only",
                "short_label": "Warn",
                "summary": "Emit warnings and continue when failures are recoverable.",
            },
        },
    },
    "reproducibility_mode": {
        "order": 4,
        "label": "Reproducibility",
        "short_label": "Reproducibility",
        "question": "How strictly should stochastic components be pinned?",
        "summary": "Controls seeds and deterministic-library settings before model execution.",
        "docs_url": "../detail/layer0/reproducibility_mode.html",
        "contract": "Seed and determinism contract applied before stochastic model code runs.",
        "selection_kind": "user_choice",
        "values": {
            "strict_reproducible": {
                "label": "Strict Reproducible Run",
                "short_label": "Strict",
                "summary": "Use deterministic-library settings for replication-grade reruns.",
            },
            "seeded_reproducible": {
                "label": "Seeded Run",
                "short_label": "Seeded",
                "summary": "Seed Python, NumPy, and optional torch without strict backend flags.",
            },
            "best_effort": {
                "label": "Best-Effort Seeded Run",
                "short_label": "Best Effort",
                "summary": "Apply available seeds and mark the run as non-strict.",
            },
            "exploratory": {
                "label": "Exploratory Run",
                "short_label": "Exploratory",
                "summary": "Do not force deterministic behavior.",
            },
        },
    },
    "compute_mode": {
        "order": 5,
        "label": "Compute Layout",
        "short_label": "Compute",
        "question": "Where should parallel work be attempted?",
        "summary": "Requests serial or parallel execution across models, horizons, targets, OOS dates, or trials.",
        "docs_url": "../detail/layer0/compute_mode.html",
        "contract": "Execution parallelism contract. Unsupported modes or singleton work units degrade to serial/no-op behavior.",
        "selection_kind": "user_choice",
        "values": {
            "serial": {
                "label": "Serial",
                "short_label": "Serial",
                "summary": "Run one unit of work at a time.",
            },
            "parallel_by_model": {
                "label": "Parallelize Models",
                "short_label": "By Model",
                "summary": "Parallelize across model-family variants when a model sweep exists.",
            },
            "parallel_by_horizon": {
                "label": "Parallelize Horizons",
                "short_label": "By Horizon",
                "summary": "Parallelize across forecast horizons when multiple horizons exist.",
            },
            "parallel_by_target": {
                "label": "Parallelize Targets",
                "short_label": "By Target",
                "summary": "Parallelize across targets in a multi-target run.",
            },
            "parallel_by_oos_date": {
                "label": "Parallelize OOS Dates",
                "short_label": "By OOS Date",
                "summary": "Parallelize across origin dates in a long pseudo-OOS window.",
            },
            "parallel_by_trial": {
                "label": "Parallelize Trials",
                "short_label": "By Trial",
                "summary": "Reserved route for trial-level parallelism.",
            },
            "distributed_cluster": {
                "label": "Distributed Cluster",
                "short_label": "Cluster",
                "summary": "Reserved route for cluster-managed execution.",
            },
        },
    },
}


def axis_presentation_map() -> dict[str, Any]:
    return {
        axis_name: {
            **{key: value for key, value in spec.items() if key != "values"},
            "values": {value_name: dict(value_spec) for value_name, value_spec in spec.get("values", {}).items()},
        }
        for axis_name, spec in AXIS_PRESENTATION_MAP.items()
    }
