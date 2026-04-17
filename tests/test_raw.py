from __future__ import annotations

import json
from pathlib import Path

from macrocast.raw import (
    RawArtifactRecord,
    RawDatasetMetadata,
    RawLoadResult,
    RawVersionRequest,
    append_raw_manifest_entry,
    build_raw_artifact_record,
    get_manifest_path,
    get_raw_cache_root,
    get_raw_file_path,
    list_vintages,
    normalize_version_request,
    read_raw_manifest,
)


def test_normalize_version_request_current_and_vintage() -> None:
    current = normalize_version_request("fred_md")
    vintage = normalize_version_request("fred_qd", vintage="2020-01")

    assert current.mode == "current"
    assert current.vintage is None
    assert vintage.mode == "vintage"
    assert vintage.vintage == "2020-01"


def test_list_vintages_respects_dataset_defaults_and_bounds() -> None:
    vintages = list_vintages("fred_qd", start="2005-01", end="2005-03")

    assert vintages == ["2005-01", "2005-02", "2005-03"]


def test_cache_paths_are_deterministic(tmp_path: Path) -> None:
    root = get_raw_cache_root(tmp_path)
    current_req = RawVersionRequest(dataset="fred_md", mode="current", vintage=None)
    vintage_req = RawVersionRequest(dataset="fred_qd", mode="vintage", vintage="2020-01")

    current_path = get_raw_file_path(current_req, root, suffix="csv")
    vintage_path = get_raw_file_path(vintage_req, root, suffix="csv")
    manifest_path = get_manifest_path(root)

    assert current_path == root / "fred_md" / "current" / "raw.csv"
    assert vintage_path == root / "fred_qd" / "vintages" / "2020-01.csv"
    assert manifest_path == root / "manifest" / "raw_artifacts.jsonl"


def test_manifest_round_trip(tmp_path: Path) -> None:
    root = get_raw_cache_root(tmp_path)
    req = RawVersionRequest(dataset="fred_md", mode="vintage", vintage="2020-01")
    raw_path = get_raw_file_path(req, root, suffix="csv")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text("date,INDPRO\n2020-01-01,1.0\n")

    artifact = build_raw_artifact_record(
        request=req,
        source_url="https://example.com/2020-01.csv",
        local_path=raw_path,
        file_format="csv",
        cache_hit=False,
    )
    metadata = RawDatasetMetadata(
        dataset="fred_md",
        source_family="fred-md",
        frequency="monthly",
        version_mode="vintage",
        vintage="2020-01",
        data_through="2020-01",
        support_tier="stable",
    )
    result = RawLoadResult(data=None, dataset_metadata=metadata, artifact=artifact)

    append_raw_manifest_entry(result, cache_root=root)
    entries = read_raw_manifest(cache_root=root)

    assert len(entries) == 1
    assert entries[0]["dataset"] == "fred_md"
    assert entries[0]["vintage"] == "2020-01"
    assert entries[0]["file_format"] == "csv"
    assert entries[0]["support_tier"] == "stable"
