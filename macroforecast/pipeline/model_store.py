"""Maintenance helpers for pipeline model-fit stores."""
from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from macroforecast.pipeline.result_store import (
    _resolved_within,
    _unlink_quietly,
    _validated_purge_cutoff,
    _validated_store_component,
)


def purge_model_store(
    store: str | Path,
    *,
    before: str | datetime | None = None,
    aliases: Sequence[str] | None = None,
) -> int:
    """Delete saved model fits matching the supplied filters and return a count.

    Pipeline model stores contain one directory per resolved model alias and one JSON
    sidecar per fitted identity. Multiple arms may share a model alias and thus a
    directory, but current filenames carry a versioned digest of the full effective fit
    identity so those arms keep separate sidecars. Legacy origin/horizon filenames and
    current digested filenames are both discovered. Each matching sidecar is deleted
    together with the pickle it owns. ``before`` filters by the sidecar file modification
    time because legacy model sidecars do not record a creation timestamp.

    Every filter is validated BEFORE anything is enumerated or removed, so a refused
    call deletes nothing: an unparseable ``before`` raises rather than acting as no
    cutoff, and an ``aliases`` entry that is not a plain alias directory name -- empty,
    ``.``, ``..``, an absolute path, anything containing a separator -- raises rather
    than reaching outside ``store``. An alias directory that resolves outside the store
    (a symlink) is never followed, whether it was named or reached by enumerating them
    all; it is skipped, so such a call reports 0 rather than deleting anything.

    ``aliases=`` names on-disk alias directory components, not arbitrary raw model
    aliases. Lowercase ASCII aliases normally match their directory names; aliases that
    require collision-resistant encoding can be discovered from the saved sidecar's raw
    ``alias`` field and its parent directory. Filters are validated but never normalized
    or encoded; a raw unsafe alias generally matches no encoded directory and returns 0.

    Only files inside the resolved store are removed. A sidecar whose recorded
    ``model_path`` points outside is deleted, but the file it names is left alone --
    see :func:`_model_path_from_manifest` for how a generated pickle is identified
    without depending on the working directory the store was written from.

    Deletion is best effort and idempotent. The returned count is the number of stored
    FITS for which at least one file (the pickle or its sidecar) was actually removed
    -- not the number of files, and not the number of sidecars considered.
    """

    root = Path(store)
    before_dt = _validated_purge_cutoff(before, label="before")
    alias_filter = (
        {_validated_store_component(alias, label="aliases") for alias in aliases}
        if aliases is not None
        else None
    )
    deleted = 0

    for metadata_path in _model_metadata_paths(root, alias_filter):
        if before_dt is not None and not _path_mtime_before(metadata_path, before_dt):
            continue
        manifest = _read_manifest(metadata_path)
        model_path = _model_path_from_manifest(root, metadata_path, manifest)
        # Both are attempted, and the fit counts when either actually went away.
        model_removed = False if model_path is None else _unlink_quietly(model_path)
        metadata_removed = _unlink_quietly(metadata_path)
        if model_removed or metadata_removed:
            deleted += 1
        _remove_empty_parent(metadata_path.parent, root)
    return deleted


def _model_metadata_paths(root: Path, aliases: set[str] | None) -> list[Path]:
    """Sidecars under each in-store alias directory, in a deterministic order.

    Directories are walked one at a time rather than through ``root.glob("*/*.json")``
    so each can be containment-checked first: a glob follows a symlinked child, which
    would let a link inside the store enumerate -- and therefore delete -- sidecars
    that live somewhere else entirely.
    """
    if aliases is not None:
        directories = [root / alias for alias in sorted(aliases)]
    else:
        try:
            directories = sorted(path for path in root.glob("*") if path.is_dir())
        except OSError:
            return []
    paths: list[Path] = []
    for directory in directories:
        if _resolved_within(directory, root) is None:
            continue
        paths.extend(sorted(directory.glob("*.json")))
    return paths


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
    """The in-store pickle this sidecar owns, or ``None`` when there is none to remove.

    The writer settles this. ``forecasting/policies/base.py::_store_model_fit`` builds
    ``<store>/<alias>/<stem>.pkl`` and ``<store>/<alias>/<stem>.json`` and hands both to
    :func:`macroforecast.models.save_fit`, so a generated pickle is ALWAYS the sidecar's
    same-stem sibling. That sibling is also the only spelling that does not depend on
    where the store was written from: ``save_fit`` records ``model_path`` verbatim, so a
    store created with a relative ``model_store=`` records a relative path, and resolving
    it in a later process would silently address that process's working directory
    instead -- deleting nothing, or something else, and leaving the pickle orphaned.

    So a relative record is not resolved. It is only used to confirm it still names the
    sibling; a hand-edited one that names anything else is refused rather than quietly
    redirected onto the sibling. An absolute record is honoured when it resolves inside
    the store, and refused when it does not -- a store may legitimately be purged
    through a different absolute prefix, but nothing outside it is ever this function's
    to delete.
    """
    sibling = metadata_path.with_suffix(".pkl")
    raw = manifest.get("model_path")
    if raw is None:
        # ``save_fit`` records None when the fit could not be pickled (the sidecar is
        # written either way), and legacy sidecars may carry no field at all. The
        # sibling is the only candidate; it simply may not exist, which is a no-op.
        return _resolved_within(sibling, root)
    candidate = Path(str(raw))
    if not candidate.is_absolute():
        if candidate.name != sibling.name:
            return None
        return _resolved_within(sibling, root)
    return _resolved_within(candidate, root)


def _remove_empty_parent(path: Path, root: Path) -> None:
    """Drop an alias directory once its last sidecar is gone.

    Restricted to a DIRECT child of the resolved store. The store root itself has to
    survive a purge -- an empty alias made ``root / ""`` the store root and reached
    ``rmdir(root)`` -- and a deeper directory is not something this store created.
    ``rmdir`` refuses a non-empty directory, so an alias that still holds files stays.
    """
    resolved = _resolved_within(path, root)
    if resolved is None:
        return
    try:
        resolved_root = root.resolve()
    except OSError:
        return
    if resolved.parent != resolved_root:
        return
    try:
        path.rmdir()
    except OSError:
        return


__all__ = ["purge_model_store"]
