"""Horse-race sweep runner (Phase 1 sub-task 01.2).

Iterates the variants of a compiled :class:`SweepPlan`, calls
:func:`execute_recipe` once per variant into an isolated sub-directory
under ``output_root``, shares a single FRED cache across variants, and
writes one Schema-v1 ``study_manifest.json`` at the study root.

See plans/phases/phase_01_sweep_executor.md section 4.2.
"""

from __future__ import annotations
import warnings

import contextvars
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..compiler.sweep_plan import SweepPlan
from ..studies.manifest import (
    VariantManifestEntry,
    build_study_manifest,
    validate_study_manifest,
)
from .build import execute_recipe
from .errors import ExecutionError


DEFAULT_RESEARCH_DESIGN = "controlled_variation"


@dataclass(frozen=True)
class VariantResult:
    """Outcome of one variant within a sweep."""

    variant_id: str
    axis_values: dict[str, str]
    status: str
    artifact_dir: str | None
    runtime_seconds: float
    error: str | None = None
    metrics_summary: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SweepResult:
    """Aggregate result of an ``execute_sweep`` call."""

    study_id: str
    output_root: str
    manifest_path: str
    per_variant_results: tuple[VariantResult, ...]
    successful_count: int
    failed_count: int

    @property
    def size(self) -> int:
        return len(self.per_variant_results)


def _try_package_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("macrocast")
    except Exception:
        pass
    try:
        from .. import __version__

        return str(__version__)
    except Exception:
        return None


def _load_metrics_summary(artifact_dir: str | Path) -> dict[str, Any]:
    metrics_path = Path(artifact_dir) / "metrics.json"
    if not metrics_path.exists():
        return {}
    try:
        data = json.loads(metrics_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    if isinstance(data, dict):
        return data
    return {"raw": data}


def _relative_to(child: Path, parent: Path) -> str:
    try:
        return str(child.relative_to(parent))
    except ValueError:
        return str(child)


def _extract_parent_compute_mode(plan: SweepPlan) -> str:
    """Read compute_mode from the plan's parent recipe (defaults to "serial")."""
    parent = plan.parent_recipe_dict.get("path", {}) if isinstance(plan.parent_recipe_dict, dict) else {}
    meta = parent.get("0_meta", {}) if isinstance(parent, dict) else {}
    if not isinstance(meta, dict):
        return "serial"
    for section in ("fixed_axes", "sweep_axes"):
        block = meta.get(section, {}) or {}
        if not isinstance(block, dict):
            continue
        if "compute_mode" in block:
            val = block["compute_mode"]
            if isinstance(val, list) and val:
                return str(val[0])
            return str(val)
    return "serial"


def _extract_parent_failure_policy(plan: SweepPlan) -> str:
    """Read failure_policy from the plan's parent recipe (defaults to "fail_fast")."""
    parent = plan.parent_recipe_dict.get("path", {}) if isinstance(plan.parent_recipe_dict, dict) else {}
    meta = parent.get("0_meta", {}) if isinstance(parent, dict) else {}
    if not isinstance(meta, dict):
        return "fail_fast"
    for section in ("fixed_axes", "sweep_axes"):
        block = meta.get(section, {}) or {}
        if not isinstance(block, dict):
            continue
        if "failure_policy" in block:
            val = block["failure_policy"]
            if isinstance(val, list) and val:
                return str(val[0])
            return str(val)
    return "fail_fast"


# Policies that signal "continue past variant failure" at the sweep level.
_CONTINUE_ON_VARIANT_FAILURE = frozenset({
    "skip_failed_cell",
    "skip_failed_model",
    "save_partial_results",
    "warn_only",
})


def execute_sweep(
    *,
    plan: SweepPlan,
    output_root: str | Path,
    local_raw_source: str | Path | None = None,
    research_design: str = DEFAULT_RESEARCH_DESIGN,
    extra_provenance: dict[str, Any] | None = None,
) -> SweepResult:
    """Execute every variant of ``plan`` under ``output_root``.

    Failure handling is driven by the parent recipe's ``failure_policy``
    meta axis (Layer 0, section 0.4). Supported policies and sweep-level
    behaviour:

    - ``fail_fast`` - the first variant failure re-raises (abort).
    - ``skip_failed_cell`` / ``skip_failed_model`` / ``save_partial_results``
      - record the failure in the study manifest and continue.
    - ``warn_only`` - same as the continue set above, plus emit a
      ``RuntimeWarning`` per failed variant.

    Recipes that do not specify ``failure_policy`` default to
    ``fail_fast``.

    Args:
        plan: A :class:`SweepPlan` produced by
            :func:`macrocast.compile_sweep_plan`.
        output_root: Directory that will contain ``variants/<vid>/`` per
            variant, the shared FRED cache at ``.raw_cache_shared``, and
            the aggregate ``study_manifest.json``.
        local_raw_source: Passed through to :func:`execute_recipe` so
            tests can use fixture CSVs.
        research_design: Recorded on the study manifest. Defaults to
            ``controlled_variation``.
        extra_provenance: Optional dict merged into each variant's
            ``provenance_payload`` on top of ``variant_id`` / ``study_id``.

    Returns:
        A :class:`SweepResult` summarising per-variant outcomes.
    """

    failure_policy = _extract_parent_failure_policy(plan)
    continue_on_failure = failure_policy in _CONTINUE_ON_VARIANT_FAILURE

    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)
    variants_dir = output_root_path / "variants"
    cache_dir = output_root_path / ".raw_cache_shared"
    variants_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    entries: list[VariantManifestEntry] = []
    results: list[VariantResult] = []

    from ..compiler.build import compile_recipe_dict, compiled_spec_to_dict

    def _run_variant(variant) -> tuple[VariantManifestEntry, VariantResult]:
        variant_output = variants_dir / variant.variant_id
        t0 = time.monotonic()
        try:
            compile_result = compile_recipe_dict(variant.variant_recipe_dict)
            compiled = compile_result.compiled
            if compiled.execution_status != "executable":
                raise ExecutionError(
                    f"variant {variant.variant_id} is not executable: "
                    f"status={compiled.execution_status!r} "
                    f"warnings={list(compiled.warnings)} "
                    f"blocked={list(compiled.blocked_reasons)}"
                )

            default_payload: dict[str, Any] = {
                "compiler": compiled_spec_to_dict(compiled),
                "tree_context": dict(compiled.tree_context),
            }
            default_payload["variant_id"] = variant.variant_id
            default_payload["study_id"] = plan.study_id
            default_payload["axis_values"] = dict(variant.axis_values)
            if extra_provenance:
                default_payload.update(extra_provenance)

            execution = execute_recipe(
                recipe=compiled.recipe_spec,
                preprocess=compiled.preprocess_contract,
                output_root=variant_output,
                local_raw_source=local_raw_source,
                provenance_payload=default_payload,
                cache_root=cache_dir,
            )

            runtime = time.monotonic() - t0
            metrics = _load_metrics_summary(execution.artifact_dir)
            rel_artifact = _relative_to(Path(execution.artifact_dir), output_root_path)
            return (
                VariantManifestEntry(
                    variant_id=variant.variant_id,
                    axis_values=dict(variant.axis_values),
                    status="success",
                    artifact_dir=rel_artifact,
                    metrics_summary=metrics,
                    runtime_seconds=runtime,
                ),
                VariantResult(
                    variant_id=variant.variant_id,
                    axis_values=dict(variant.axis_values),
                    status="success",
                    artifact_dir=execution.artifact_dir,
                    runtime_seconds=runtime,
                    metrics_summary=metrics,
                ),
            )
        except Exception as exc:
            runtime = time.monotonic() - t0
            if not continue_on_failure:
                raise
            if failure_policy == "warn_only":
                warnings.warn(
                    f"variant {variant.variant_id} failed: {type(exc).__name__}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            rel_artifact = _relative_to(variant_output, output_root_path)
            return (
                VariantManifestEntry(
                    variant_id=variant.variant_id,
                    axis_values=dict(variant.axis_values),
                    status="failed",
                    artifact_dir=rel_artifact,
                    runtime_seconds=runtime,
                    error=f"{type(exc).__name__}: {exc}",
                ),
                VariantResult(
                    variant_id=variant.variant_id,
                    axis_values=dict(variant.axis_values),
                    status="failed",
                    artifact_dir=None,
                    runtime_seconds=runtime,
                    error=f"{type(exc).__name__}: {exc}",
                ),
            )

    parent_compute_mode = _extract_parent_compute_mode(plan)
    has_model_family_sweep = any(
        axis.endswith(".model_family") for axis in plan.axes_swept
    )
    should_parallelize = (
        parent_compute_mode == "parallel_by_model"
        and has_model_family_sweep
        and len(plan.variants) > 1
    )

    if should_parallelize:
        # variant-level threading: each variant runs its own execute_recipe;
        # workers capped at 4 (matching target/horizon parallel ceilings).
        with ThreadPoolExecutor(max_workers=min(len(plan.variants), 4)) as ex:
            ordered = list(plan.variants)
            futures = [
                ex.submit(contextvars.copy_context().run, _run_variant, variant)
                for variant in ordered
            ]
            for future in futures:
                entry, result = future.result()
                entries.append(entry)
                results.append(result)
    else:
        for variant in plan.variants:
            entry, result = _run_variant(variant)
            entries.append(entry)
            results.append(result)

    manifest = build_study_manifest(
        study_id=plan.study_id,
        research_design=research_design,
        parent_recipe_id=plan.parent_recipe_id,
        parent_recipe_dict=plan.parent_recipe_dict,
        axes_swept=plan.axes_swept,
        variants=entries,
        package_version=_try_package_version(),
    )
    validate_study_manifest(manifest)
    manifest_path = output_root_path / "study_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str)
    )

    return SweepResult(
        study_id=plan.study_id,
        output_root=str(output_root_path),
        manifest_path=str(manifest_path),
        per_variant_results=tuple(results),
        successful_count=sum(1 for r in results if r.status == "success"),
        failed_count=sum(1 for r in results if r.status == "failed"),
    )


__all__ = [
    "DEFAULT_RESEARCH_DESIGN",
    "SweepResult",
    "VariantResult",
    "execute_sweep",
]
