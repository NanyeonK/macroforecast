"""Multi-target separate-runs wrapper.

`execute_separate_runs` fans out an N-target recipe into N independent
single-target executions, each under its own artifact directory, and
writes a top-level aggregate manifest linking them.

Distinct from the `multiple_targets_one_method` path inside
`execute_recipe`, which loops over targets internally but aggregates
predictions into one table.

Typical use:

    from macrocast.studies.multi_target import execute_separate_runs

    result = execute_separate_runs(
        source_recipe_dict=recipe_dict,    # target_structure=multi_target
        output_root="out/",
        local_raw_source="tests/fixtures/fred_md_ar_sample.csv",
    )
    # out/
    #   separate_runs_manifest.json
    #   targets/
    #     INDPRO/...
    #     RPI/...
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SEPARATE_RUNS_MANIFEST_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class SeparateRunsResult:
    """Aggregate of one fan-out call.

    Attributes:
        source_recipe_id: The multi-target source recipe's ``recipe_id``.
        output_root: The top-level output directory.
        manifest_path: Path to ``separate_runs_manifest.json``.
        per_target_results: Each entry carries ``target``, ``artifact_dir``,
            and ``recipe_id`` of the single-target run.
        successful_targets: Targets whose runs completed without error.
        failed_targets: Targets whose runs raised an exception.
    """

    source_recipe_id: str
    output_root: str
    manifest_path: str
    per_target_results: tuple[dict[str, Any], ...]
    successful_targets: tuple[str, ...]
    failed_targets: tuple[str, ...]


def _build_single_target_recipe_dict(
    source_recipe_dict: dict[str, Any],
    target_name: str,
) -> dict[str, Any]:
    """Clone the source recipe into a single-target variant for ``target_name``.

    Changes applied to the clone:
    - ``1_data_task.fixed_axes.target_structure`` → ``single_target``
    - ``1_data_task.leaf_config.target`` → ``target_name``; ``targets`` removed
    - ``0_meta.fixed_axes.study_scope`` cleared if it was set to
      ``multiple_targets_compare_methods`` (the child runs derive their own unit)
    - ``recipe_id`` suffixed with ``__target__<name>`` for traceability
    """
    variant = copy.deepcopy(source_recipe_dict)
    parent_recipe_id = variant.get("recipe_id", "recipe")
    variant["recipe_id"] = f"{parent_recipe_id}__target__{target_name}"

    path = variant.setdefault("path", {})
    data_task = path.setdefault("1_data_task", {})
    data_task_fixed = data_task.setdefault("fixed_axes", {})
    data_task_fixed["target_structure"] = "single_target"
    data_task_leaf = data_task.setdefault("leaf_config", {})
    data_task_leaf["target"] = target_name
    data_task_leaf.pop("targets", None)

    meta_fixed = path.setdefault("0_meta", {}).setdefault("fixed_axes", {})
    if meta_fixed.get("study_scope") == "multiple_targets_compare_methods":
        meta_fixed.pop("study_scope")

    return variant


def _extract_targets(source_recipe_dict: dict[str, Any]) -> list[str]:
    data_task = source_recipe_dict.get("path", {}).get("1_data_task", {})
    leaf = data_task.get("leaf_config", {}) or {}
    targets = leaf.get("targets")
    if not isinstance(targets, (list, tuple)) or len(targets) < 2:
        raise ValueError(
            "execute_separate_runs requires source_recipe_dict with "
            "leaf_config.targets listing at least two target names"
        )
    return [str(t) for t in targets]


def execute_separate_runs(
    *,
    source_recipe_dict: dict[str, Any],
    output_root: str | Path,
    local_raw_source: str | Path | None = None,
    provenance_payload: dict[str, Any] | None = None,
    fail_fast: bool = True,
) -> SeparateRunsResult:
    """Fan out a multi-target recipe into per-target single-target runs.

    Args:
        source_recipe_dict: The multi-target source recipe (must set
            ``target_structure=multi_target`` and list two or more
            targets in ``leaf_config.targets``).
        output_root: Directory that will hold
            ``targets/<target_name>/`` per sub-run plus the top-level
            ``separate_runs_manifest.json``.
        local_raw_source: Passed through to each ``execute_recipe`` call
            so tests can use fixture CSVs.
        provenance_payload: Optional base provenance dict merged into
            each sub-run's ``provenance_payload``; the per-target name is
            added automatically.
        fail_fast: When True (default), the first target failure raises.
            When False, failures are recorded in the aggregate manifest
            and remaining targets still execute.

    Returns:
        A :class:`SeparateRunsResult` with per-target artifact paths.
    """

    from ..compiler.build import compile_recipe_dict
    from ..execution.build import execute_recipe

    targets = _extract_targets(source_recipe_dict)
    source_recipe_id = source_recipe_dict.get("recipe_id", "recipe")

    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)
    targets_dir = output_root_path / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)

    per_target_results: list[dict[str, Any]] = []
    successful: list[str] = []
    failed: list[str] = []

    for target_name in targets:
        sub_dict = _build_single_target_recipe_dict(source_recipe_dict, target_name)
        sub_payload: dict[str, Any] = dict(provenance_payload or {})
        sub_payload["separate_runs_target"] = target_name
        sub_payload["separate_runs_source_recipe_id"] = source_recipe_id
        try:
            compile_result = compile_recipe_dict(sub_dict)
            compiled = compile_result.compiled
            if compiled.execution_status != "executable":
                raise RuntimeError(
                    f"single-target variant for {target_name!r} is not executable: "
                    f"status={compiled.execution_status!r} "
                    f"warnings={list(compiled.warnings)} "
                    f"blocked={list(compiled.blocked_reasons)}"
                )
            execution = execute_recipe(
                recipe=compiled.recipe_spec,
                preprocess=compiled.preprocess_contract,
                output_root=targets_dir / target_name,
                local_raw_source=local_raw_source,
                provenance_payload=sub_payload,
            )
            per_target_results.append({
                "target": target_name,
                "recipe_id": sub_dict["recipe_id"],
                "status": "success",
                "artifact_dir": str(execution.artifact_dir),
                "artifact_subdir": execution.run.artifact_subdir,
            })
            successful.append(target_name)
        except Exception as exc:
            if fail_fast:
                raise
            per_target_results.append({
                "target": target_name,
                "recipe_id": sub_dict.get("recipe_id"),
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            })
            failed.append(target_name)

    manifest_path = output_root_path / "separate_runs_manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": SEPARATE_RUNS_MANIFEST_SCHEMA_VERSION,
        "study_scope": "multiple_targets_compare_methods",
        "source_recipe_id": source_recipe_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "targets": targets,
        "runs": per_target_results,
        "summary": {
            "total": len(targets),
            "successful": len(successful),
            "failed": len(failed),
        },
    }, indent=2))

    return SeparateRunsResult(
        source_recipe_id=source_recipe_id,
        output_root=str(output_root_path),
        manifest_path=str(manifest_path),
        per_target_results=tuple(per_target_results),
        successful_targets=tuple(successful),
        failed_targets=tuple(failed),
    )


__all__ = [
    "SEPARATE_RUNS_MANIFEST_SCHEMA_VERSION",
    "SeparateRunsResult",
    "execute_separate_runs",
]
