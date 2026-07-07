"""Maintenance helpers for pipeline model-fit stores."""
from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from macroforecast.pipeline.result_store import _parse_datetime, _unlink_quietly


def purge_model_store(
    store: str | Path,
    *,
    before: str | datetime | None = None,
    aliases: Sequence[str] | None = None,
) -> int:
    """Delete saved model fits matching the supplied filters and return a count.

    Pipeline model stores contain one alias directory per arm/model and one JSON
    sidecar per fitted origin/horizon. Each matching sidecar is deleted together
    with its recorded pickle path when that pickle lives under ``store``.
    ``before`` filters by the sidecar file modification time because legacy model
    sidecars do not record a creation timestamp.
    """

    root = Path(store)
    alias_filter = {str(alias) for alias in aliases} if aliases is not None else None
    before_dt = _parse_datetime(before) if before is not None else None
    deleted = 0

    for metadata_path in _model_metadata_paths(root, alias_filter):
        if before_dt is not None and not _path_mtime_before(metadata_path, before_dt):
            continue
        manifest = _read_manifest(metadata_path)
        model_path = _model_path_from_manifest(root, metadata_path, manifest)
        if model_path is not None:
            _unlink_quietly(model_path)
        _unlink_quietly(metadata_path)
        deleted += 1
        _remove_empty_parent(metadata_path.parent, root)
    return deleted


def _model_metadata_paths(root: Path, aliases: set[str] | None) -> list[Path]:
    if aliases is not None:
        return [
            path
            for alias in sorted(aliases)
            for path in sorted((root / alias).glob("*.json"))
        ]
    return sorted(root.glob("*/*.json"))


def _path_mtime_before(path: Path, before: datetime) -> bool:
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return False
    return mtime < before


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _model_path_from_manifest(
    root: Path,
    metadata_path: Path,
    manifest: dict[str, Any],
) -> Path | None:
    raw = manifest.get("model_path")
    if raw is None:
        candidate = metadata_path.with_suffix(".pkl")
    else:
        candidate = Path(str(raw))
    try:
        candidate.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return None
    return candidate


def _remove_empty_parent(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError:
        return
    try:
        path.rmdir()
    except OSError:
        return


__all__ = ["purge_model_store"]
