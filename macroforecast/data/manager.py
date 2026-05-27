from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import cast

from .errors import RawVersionFormatError
from .types import ArtifactFormat, DatasetId, RawArtifactRecord, RawVersionRequest

_FIRST_VINTAGE: dict[DatasetId, str] = {
    "fred_md": "1999-01",
    "fred_qd": "2005-01",
    "fred_sd": "2005-06",
}
_ARTIFACT_FORMATS: set[ArtifactFormat] = {"csv", "xlsx", "mixed"}


def _dataset_id(dataset: str) -> DatasetId:
    if dataset not in _FIRST_VINTAGE:
        raise RawVersionFormatError(f"unknown dataset={dataset!r}")
    return cast(DatasetId, dataset)


def normalize_version_request(dataset: str, vintage: str | None = None) -> RawVersionRequest:
    dataset_id = _dataset_id(dataset)
    if vintage is None:
        return RawVersionRequest(dataset=dataset_id, mode="current", vintage=None)
    if not re.fullmatch(r"\d{4}-\d{2}", vintage):
        raise RawVersionFormatError(f"invalid vintage format: {vintage!r}")
    return RawVersionRequest(dataset=dataset_id, mode="vintage", vintage=vintage)


def list_vintages(dataset: str, start: str | None = None, end: str | None = None) -> list[str]:
    dataset_id = _dataset_id(dataset)
    request = normalize_version_request(dataset_id, vintage=start or _FIRST_VINTAGE[dataset_id])
    if request.vintage is None:
        raise RawVersionFormatError("start vintage must be supplied")
    start_year, start_month = map(int, request.vintage.split("-"))
    if end is None:
        raise RawVersionFormatError("end must be supplied explicitly in the v1 skeleton")
    end_request = normalize_version_request(dataset_id, vintage=end)
    if end_request.vintage is None:
        raise RawVersionFormatError("end vintage must be supplied")
    end_year, end_month = map(int, end_request.vintage.split("-"))
    vintages: list[str] = []
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        vintages.append(f"{year:04d}-{month:02d}")
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1
    return vintages


def build_raw_artifact_record(
    *,
    request: RawVersionRequest,
    source_url: str,
    local_path: str | Path,
    file_format: str,
    cache_hit: bool,
) -> RawArtifactRecord:
    if file_format not in _ARTIFACT_FORMATS:
        raise RawVersionFormatError(f"unknown artifact file_format={file_format!r}")
    path = Path(local_path)
    content = path.read_bytes()
    return RawArtifactRecord(
        dataset=request.dataset,
        version_mode=request.mode,
        vintage=request.vintage,
        source_url=source_url,
        local_path=str(path),
        file_format=cast(ArtifactFormat, file_format),
        downloaded_at=datetime.now(timezone.utc).isoformat(),
        file_sha256=hashlib.sha256(content).hexdigest(),
        file_size_bytes=len(content),
        cache_hit=cache_hit,
        manifest_version="v1",
    )
