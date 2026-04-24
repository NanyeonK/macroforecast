"""Tests for study_manifest Schema v1 validation (Phase 1 sub-task 01.3 / 01.7)."""

from __future__ import annotations

import pytest

from macrocast.studies.manifest import (
    STUDY_MANIFEST_SCHEMA_VERSION,
    StudyManifestSchemaError,
    VariantManifestEntry,
    build_study_manifest,
    validate_study_manifest,
)


def _minimal_manifest() -> dict:
    entry = VariantManifestEntry(
        variant_id="v-deadbeef",
        axis_values={"3_training.model_family": "ridge"},
        status="success",
        artifact_dir="variants/v-deadbeef",
        metrics_summary={"msfe": 0.01},
        runtime_seconds=1.2,
    )
    return build_study_manifest(
        study_id="sha256-abcd1234abcd1234",
        research_design="controlled_variation",
        parent_recipe_id="rt",
        parent_recipe_dict={"recipe_id": "rt", "path": {}},
        axes_swept=("3_training.model_family",),
        variants=[entry],
        package_version="0.3.0",
    )


def test_build_manifest_is_schema_valid() -> None:
    manifest = _minimal_manifest()
    assert manifest["schema_version"] == STUDY_MANIFEST_SCHEMA_VERSION
    validate_study_manifest(manifest)


def test_missing_top_level_key_raises() -> None:
    manifest = _minimal_manifest()
    del manifest["study_id"]
    with pytest.raises(StudyManifestSchemaError, match="study_id"):
        validate_study_manifest(manifest)


def test_bad_schema_version_raises() -> None:
    manifest = _minimal_manifest()
    manifest["schema_version"] = "0.9"
    with pytest.raises(StudyManifestSchemaError, match="schema_version"):
        validate_study_manifest(manifest)


def test_missing_sweep_plan_variants_raises() -> None:
    manifest = _minimal_manifest()
    del manifest["sweep_plan"]["variants"]
    with pytest.raises(StudyManifestSchemaError, match="variants"):
        validate_study_manifest(manifest)


def test_variant_missing_required_field_raises() -> None:
    manifest = _minimal_manifest()
    del manifest["sweep_plan"]["variants"][0]["status"]
    with pytest.raises(StudyManifestSchemaError, match="status"):
        validate_study_manifest(manifest)


def test_invalid_variant_status_raises() -> None:
    manifest = _minimal_manifest()
    manifest["sweep_plan"]["variants"][0]["status"] = "maybe"
    with pytest.raises(StudyManifestSchemaError, match="status"):
        validate_study_manifest(manifest)


def test_duplicate_variant_id_raises() -> None:
    manifest = _minimal_manifest()
    extra = dict(manifest["sweep_plan"]["variants"][0])
    manifest["sweep_plan"]["variants"].append(extra)
    with pytest.raises(StudyManifestSchemaError, match="duplicate"):
        validate_study_manifest(manifest)


def test_summary_counts_match_variants() -> None:
    entries = [
        VariantManifestEntry(
            variant_id="v-1",
            axis_values={"a": "x"},
            status="success",
            artifact_dir="variants/v-1",
        ),
        VariantManifestEntry(
            variant_id="v-2",
            axis_values={"a": "y"},
            status="failed",
            artifact_dir="variants/v-2",
            error="boom",
        ),
        VariantManifestEntry(
            variant_id="v-3",
            axis_values={"a": "z"},
            status="skipped",
            artifact_dir="variants/v-3",
            compiler_status="blocked_by_incompatibility",
            compiler_blocked_reasons=["blocked"],
            layer3_capability_cell={"runtime_status": "blocked_by_incompatibility"},
        ),
    ]
    manifest = build_study_manifest(
        study_id="sha256-test",
        research_design="controlled_variation",
        parent_recipe_id="rt",
        parent_recipe_dict={"recipe_id": "rt", "path": {}},
        axes_swept=("a",),
        variants=entries,
    )
    assert manifest["summary"]["successful"] == 1
    assert manifest["summary"]["failed"] == 1
    assert manifest["summary"]["skipped"] == 1
    assert manifest["summary"]["invalid_cells"] == 1
    assert manifest["summary"]["runnable_variants"] == 2
    assert manifest["summary"]["total_variants"] == 3
    skipped = manifest["sweep_plan"]["variants"][2]
    assert skipped["compiler_status"] == "blocked_by_incompatibility"
    assert skipped["compiler_blocked_reasons"] == ["blocked"]
