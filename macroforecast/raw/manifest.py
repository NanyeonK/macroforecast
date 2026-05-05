from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .cache import get_manifest_path
from .errors import RawManifestError
from .types import RawLoadResult


def append_raw_manifest_entry(result: RawLoadResult, *, cache_root: str | Path | None = None) -> None:
    path = get_manifest_path(cache_root)
    entry = {
        **asdict(result.artifact),
        "data_through": result.dataset_metadata.data_through,
        "support_tier": result.dataset_metadata.support_tier,
        "parse_notes": list(result.dataset_metadata.parse_notes),
    }
    try:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as exc:
        raise RawManifestError(f"failed to append manifest entry to {path}") from exc


def read_raw_manifest(*, cache_root: str | Path | None = None) -> list[dict]:
    path = get_manifest_path(cache_root)
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
    except OSError as exc:
        raise RawManifestError(f"failed to read manifest {path}") from exc
