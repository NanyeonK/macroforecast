from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from macrocast.execution.build import (
    _combine_raw_results,
    _raw_artifact_payload,
    _raw_dataset_metadata_payload,
    _source_availability_contract,
)
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


def _raw_component(
    *,
    dataset: str,
    source_family: str,
    frequency: str,
    source_url: str,
    local_path: str,
    file_sha256: str,
    data_through: str,
    cache_hit: bool,
) -> RawLoadResult:
    frame = pd.DataFrame(
        {f"{dataset}_VALUE": [1.0, 2.0]},
        index=pd.date_range("2000-01-01", periods=2, freq="MS"),
    )
    metadata = RawDatasetMetadata(
        dataset=dataset,
        source_family=source_family,
        frequency=frequency,
        version_mode="vintage",
        vintage="2020-01",
        data_through=data_through,
        support_tier="stable",
    )
    artifact = RawArtifactRecord(
        dataset=dataset,
        version_mode="vintage",
        vintage="2020-01",
        source_url=source_url,
        local_path=local_path,
        file_format="csv",
        downloaded_at="2026-04-26T00:00:00+00:00",
        file_sha256=file_sha256,
        file_size_bytes=123,
        cache_hit=cache_hit,
        manifest_version="v1",
    )
    return RawLoadResult(data=frame, dataset_metadata=metadata, artifact=artifact, transform_codes={f"{dataset}_VALUE": 1})


def test_composite_source_contracts_flow_into_source_availability_contract(tmp_path: Path) -> None:
    md_path = tmp_path / "fred_md.csv"
    sd_path = tmp_path / "fred_sd.csv"
    md_path.write_text("date,fred_md_VALUE\n2000-01-01,1.0\n")
    sd_path.write_text("date,fred_sd_VALUE\n2000-01-01,1.0\n")
    md = _raw_component(
        dataset="fred_md",
        source_family="fred-md",
        frequency="monthly",
        source_url=str(md_path),
        local_path=str(md_path),
        file_sha256="a" * 64,
        data_through="2000-02",
        cache_hit=False,
    )
    sd = _raw_component(
        dataset="fred_sd",
        source_family="fred-sd",
        frequency="state_monthly",
        source_url=str(sd_path),
        local_path=str(sd_path),
        file_sha256="b" * 64,
        data_through="2000-01",
        cache_hit=True,
    )

    combined = _combine_raw_results("fred_md+fred_sd", "monthly", [("fred_md", md), ("fred_sd", sd)])
    metadata = _raw_dataset_metadata_payload(combined)
    artifact = _raw_artifact_payload(combined)
    source_contract = _source_availability_contract(
        combined,
        SimpleNamespace(raw_dataset="fred_md+fred_sd", data_vintage="2020-01", data_task_spec={}),
        index_start="2000-01-01",
        index_end="2000-02-01",
        metadata=metadata,
        artifact_payload=artifact,
    )

    assert metadata["dataset"] == "fred_md+fred_sd"
    assert metadata["data_through"] == "2000-01"
    assert artifact["file_format"] == "mixed"
    assert source_contract["contract_version"] == "source_availability_contract_v1"
    assert source_contract["component_count"] == 2
    assert source_contract["data_vintage_requested"] == "2020-01"
    assert source_contract["observed_data_window"] == {
        "index_start": "2000-01-01",
        "index_end": "2000-02-01",
        "data_through": "2000-01",
    }
    components = source_contract["component_source_contracts"]
    assert [component["component"] for component in components] == ["fred_md", "fred_sd"]
    assert [component["dataset"] for component in components] == ["fred_md", "fred_sd"]
    assert [component["source_url_kind"] for component in components] == ["local_path", "local_path"]
    assert [component["artifact_file_sha256"] for component in components] == ["a" * 64, "b" * 64]
    assert [component["artifact_cache_hit"] for component in components] == [False, True]
