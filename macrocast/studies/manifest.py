"""Study-level manifest (Schema v1) for horse-race sweeps.

One manifest per study. Summarises the sweep plan, per-variant outcomes,
and provenance. Schema v1 is frozen for v0.3; future extensions bump the
``schema_version`` field.

See plans/infra/study_manifest_schema.md and phase_01_sweep_executor.md
section 4.4 for the authoritative spec.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any, Iterable

STUDY_MANIFEST_SCHEMA_VERSION = "1.0"

_REQUIRED_TOP_LEVEL_KEYS = (
    "schema_version",
    "study_id",
    "research_design",
    "created_at_utc",
    "parent_recipe_id",
    "sweep_plan",
)

_REQUIRED_SWEEP_PLAN_KEYS = ("axes_swept", "variants")

_REQUIRED_VARIANT_KEYS = (
    "variant_id",
    "axis_values",
    "status",
    "artifact_dir",
)

_ALLOWED_VARIANT_STATUS = ("success", "failed", "skipped")


class StudyManifestSchemaError(ValueError):
    """Raised when a manifest dict fails Schema v1 validation."""


@dataclass(frozen=True)
class VariantManifestEntry:
    """One variant's entry in the study manifest (Schema v1)."""

    variant_id: str
    axis_values: dict[str, Any]
    status: str
    artifact_dir: str
    metrics_summary: dict[str, Any] = field(default_factory=dict)
    seed_used: int | None = None
    runtime_seconds: float | None = None
    error: str | None = None
    compiler_status: str | None = None
    compiler_warnings: list[str] = field(default_factory=list)
    compiler_blocked_reasons: list[str] = field(default_factory=list)
    layer3_capability_cell: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "variant_id": self.variant_id,
            "axis_values": dict(self.axis_values),
            "status": self.status,
            "artifact_dir": self.artifact_dir,
            "metrics_summary": dict(self.metrics_summary),
        }
        if self.seed_used is not None:
            payload["seed_used"] = int(self.seed_used)
        if self.runtime_seconds is not None:
            payload["runtime_seconds"] = float(self.runtime_seconds)
        if self.error is not None:
            payload["error"] = str(self.error)
        if self.compiler_status is not None:
            payload["compiler_status"] = str(self.compiler_status)
        if self.compiler_warnings:
            payload["compiler_warnings"] = list(self.compiler_warnings)
        if self.compiler_blocked_reasons:
            payload["compiler_blocked_reasons"] = list(self.compiler_blocked_reasons)
        if self.layer3_capability_cell:
            payload["layer3_capability_cell"] = dict(self.layer3_capability_cell)
        return payload


def _utc_now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_study_manifest(
    *,
    study_id: str,
    research_design: str,
    parent_recipe_id: str,
    parent_recipe_dict: dict[str, Any],
    axes_swept: Iterable[str],
    variants: Iterable[VariantManifestEntry],
    package_version: str | None = None,
    git_commit: str | None = None,
    tree_context: dict[str, Any] | None = None,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Produce a Schema-v1 study manifest dict ready for JSON write."""

    variant_list = [v.to_dict() for v in variants]
    successful = sum(1 for v in variant_list if v["status"] == "success")
    failed = sum(1 for v in variant_list if v["status"] == "failed")
    skipped = sum(1 for v in variant_list if v["status"] == "skipped")
    invalid_cells = sum(
        1
        for v in variant_list
        if v["status"] == "skipped"
        and v.get("compiler_status") in {"blocked_by_incompatibility", "not_supported"}
    )
    manifest: dict[str, Any] = {
        "schema_version": STUDY_MANIFEST_SCHEMA_VERSION,
        "study_id": study_id,
        "research_design": research_design,
        "created_at_utc": created_at_utc or _utc_now_iso(),
        "parent_recipe_id": parent_recipe_id,
        "parent_recipe": parent_recipe_dict,
        "sweep_plan": {
            "axes_swept": list(axes_swept),
            "variants": variant_list,
        },
        "summary": {
            "total_variants": len(variant_list),
            "successful": successful,
            "failed": failed,
            "skipped": skipped,
            "invalid_cells": invalid_cells,
            "runnable_variants": successful + failed,
            "variants_by_status": {
                "success": successful,
                "failed": failed,
                "skipped": skipped,
            },
        },
    }
    if package_version is not None:
        manifest["package_version"] = str(package_version)
    if git_commit is not None:
        manifest["git_commit"] = str(git_commit)
    if tree_context is not None:
        manifest["tree_context"] = dict(tree_context)
    return manifest


def validate_study_manifest(manifest: dict[str, Any]) -> None:
    """Raise StudyManifestSchemaError if ``manifest`` does not match v1."""

    if not isinstance(manifest, dict):
        raise StudyManifestSchemaError("manifest must be a dict")

    for key in _REQUIRED_TOP_LEVEL_KEYS:
        if key not in manifest:
            raise StudyManifestSchemaError(f"missing top-level key: {key}")

    if manifest["schema_version"] != STUDY_MANIFEST_SCHEMA_VERSION:
        raise StudyManifestSchemaError(
            f"unsupported schema_version: {manifest['schema_version']!r} "
            f"(expected {STUDY_MANIFEST_SCHEMA_VERSION!r})"
        )

    sweep_plan = manifest.get("sweep_plan")
    if not isinstance(sweep_plan, dict):
        raise StudyManifestSchemaError("sweep_plan must be a dict")
    for key in _REQUIRED_SWEEP_PLAN_KEYS:
        if key not in sweep_plan:
            raise StudyManifestSchemaError(f"sweep_plan missing key: {key}")

    if not isinstance(sweep_plan["axes_swept"], list):
        raise StudyManifestSchemaError("sweep_plan.axes_swept must be a list")

    variants = sweep_plan.get("variants")
    if not isinstance(variants, list):
        raise StudyManifestSchemaError("sweep_plan.variants must be a list")

    seen_variant_ids: set[str] = set()
    for idx, variant in enumerate(variants):
        if not isinstance(variant, dict):
            raise StudyManifestSchemaError(
                f"sweep_plan.variants[{idx}] must be a dict"
            )
        for key in _REQUIRED_VARIANT_KEYS:
            if key not in variant:
                raise StudyManifestSchemaError(
                    f"sweep_plan.variants[{idx}] missing key: {key}"
                )
        vid = variant["variant_id"]
        if vid in seen_variant_ids:
            raise StudyManifestSchemaError(
                f"duplicate variant_id in sweep_plan.variants: {vid!r}"
            )
        seen_variant_ids.add(vid)
        if variant["status"] not in _ALLOWED_VARIANT_STATUS:
            raise StudyManifestSchemaError(
                f"sweep_plan.variants[{idx}].status must be one of "
                f"{_ALLOWED_VARIANT_STATUS}, got {variant['status']!r}"
            )
        if not isinstance(variant["axis_values"], dict):
            raise StudyManifestSchemaError(
                f"sweep_plan.variants[{idx}].axis_values must be a dict"
            )


__all__ = [
    "STUDY_MANIFEST_SCHEMA_VERSION",
    "StudyManifestSchemaError",
    "VariantManifestEntry",
    "build_study_manifest",
    "validate_study_manifest",
]
