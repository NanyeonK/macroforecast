from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from .types import RawVersionRequest


def get_raw_cache_root(cache_root: str | Path | None = None) -> Path:
    root = Path(cache_root).expanduser() if cache_root is not None else Path("~/.macroforecast/raw").expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_manifest_path(cache_root: str | Path | None = None) -> Path:
    root = get_raw_cache_root(cache_root)
    path = root / "manifest" / "raw_artifacts.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_raw_file_path(
    request: RawVersionRequest,
    cache_root: str | Path | None = None,
    *,
    suffix: str,
) -> Path:
    root = get_raw_cache_root(cache_root)
    if request.mode == "current":
        path = root / request.dataset / "current" / f"raw.{suffix}"
    else:
        if request.vintage is None:
            raise ValueError("vintage mode requires vintage string")
        path = root / request.dataset / "vintages" / f"{request.vintage}.{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def atomic_copy_to_cache(source: str | Path, target: str | Path) -> None:
    """Copy ``source`` to ``target`` without exposing partial cache files."""
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target_path.name}.",
        suffix=".tmp",
        dir=target_path.parent,
    )
    tmp_path = Path(tmp_name)
    os.close(fd)
    try:
        shutil.copyfile(source, tmp_path)
        os.replace(tmp_path, target_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def atomic_write_bytes_to_cache(content: bytes, target: str | Path) -> None:
    """Write bytes to ``target`` without exposing partial cache files."""
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target_path.name}.",
        suffix=".tmp",
        dir=target_path.parent,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(content)
        os.replace(tmp_path, target_path)
    finally:
        tmp_path.unlink(missing_ok=True)
