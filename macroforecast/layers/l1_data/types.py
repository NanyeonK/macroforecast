from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

DatasetId = Literal["fred_md", "fred_qd", "fred_sd", "fred_md+fred_sd", "fred_qd+fred_sd"]
VersionMode = Literal["current", "vintage"]
ArtifactFormat = Literal["csv", "xlsx", "mixed"]
SupportTier = Literal["stable", "provisional"]


@dataclass(frozen=True)
class RawVersionRequest:
    dataset: DatasetId
    mode: VersionMode
    vintage: str | None


@dataclass(frozen=True)
class RawDatasetMetadata:
    dataset: str
    source_family: str
    frequency: str
    version_mode: VersionMode
    vintage: str | None
    data_through: str | None
    support_tier: SupportTier
    parse_notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class RawArtifactRecord:
    dataset: str
    version_mode: VersionMode
    vintage: str | None
    source_url: str
    local_path: str
    file_format: ArtifactFormat
    downloaded_at: str
    file_sha256: str
    file_size_bytes: int
    cache_hit: bool
    manifest_version: str


@dataclass(frozen=True)
class RawLoadResult:
    data: Any
    dataset_metadata: RawDatasetMetadata
    artifact: RawArtifactRecord
    transform_codes: dict[str, int] = field(default_factory=dict)
