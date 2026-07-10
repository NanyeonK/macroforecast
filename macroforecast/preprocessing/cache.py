"""Content-addressed on-disk store for fitted preprocessing objects.

``PreprocessorStore`` is a directory-backed cache that maps a
``(PreprocessSpec, target, origin_pos)`` triple -- optionally salted by a
caller-supplied ``namespace`` -- to a pickled ``FittedPreprocessor`` (or any
picklable object). It is designed for safe concurrent multi-process access:

- Writes use a temp-file + ``os.replace`` (atomic on POSIX), so a concurrent
  reader never observes a partially-written file.
- Reads are lock-free: a corrupt or missing file is treated as a cache miss.
- Duplicate concurrent writes of the same key are harmless (idempotent
  content; the last ``os.replace`` wins, which is fine because the content is
  identical).

Canonical key construction
--------------------------
The cache key is a SHA-256 hex digest over the deterministic JSON
serialisation of ``(spec_dict, target, origin_pos)`` where ``spec_dict`` is
``spec.to_dict()``.  ``PreprocessSpec.to_dict()`` calls ``_json_ready`` on the
options dict, converting callables to their qualified name strings and tuples to
lists, so the output is always JSON-serialisable with ``json.dumps(sort_keys=True)``.
When a preprocessing spec configures ``standardize_scope`` (for example
``origin_available_predictors``), that option is part of ``spec_dict`` and
therefore separates fitted/prepared cache entries from other scopes.

When the store is constructed with a ``namespace`` (any JSON-serialisable
value -- e.g. the effective ``StagePolicy.to_dict()`` the caller fitted under),
the namespace is folded into the same digest. Two callers sharing one
``root_dir`` but constructing the store with DIFFERENT namespaces therefore
never collide: their keys differ even for an identical ``(spec, target,
origin_pos)`` triple, so a fit produced under one namespace is never served to
a reader in another. ``namespace=None`` (the default) reproduces the original,
pre-namespace digest exactly, so existing stores built before this parameter
existed keep working unchanged for callers that do not pass one.

Frame payloads
--------------
The same store also persists prepared-base DataFrame payloads used by the
forecasting runner. They use the same content-addressed digest/namespace
mechanics, with parquet for the frame and a JSON metadata sidecar for index,
columns, dtypes, and lightweight attrs. This is an opt-in tier: callers reach it
only by passing a store, normally through ``preprocessing_cache_dir``.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd


class UndigestiblePreprocessorSpec(Exception):
    """Raised when a preprocessing spec is unsafe for disk-cache identity."""


class PreprocessorStore:
    """Directory-backed content-addressed store for picklable objects.

    Parameters
    ----------
    root_dir:
        Directory where cached files are stored.  Created on first ``put`` if
        it does not already exist.
    namespace:
        Optional JSON-serialisable value folded into every key digest (see
        :meth:`key`). Use this to safely share one ``root_dir`` across callers
        whose fitted objects would otherwise collide for the same
        ``(spec, target, origin_pos)`` triple but differ in some other
        run-level knob (e.g. the ``preprocessing_policy`` scope) that is not
        already part of ``spec``. ``None`` (the default) reproduces the
        original pre-namespace digest.
    """

    def __init__(self, root_dir: str | Path, *, namespace: Any | None = None) -> None:
        self._root = Path(root_dir)
        self._namespace = namespace

    # ------------------------------------------------------------------
    # Key construction
    # ------------------------------------------------------------------

    def key(
        self,
        prep_spec: Any,
        *,
        target: str,
        origin_pos: int,
    ) -> str:
        """Return a SHA-256 hex digest for the (spec, target, origin_pos) triple.

        The digest is computed over the deterministic JSON serialisation of::

            {
                "spec": prep_spec.to_dict(),
                "target": target,
                "origin_pos": origin_pos,
            }

        The disk identity path refuses user callables unless they carry an
        explicit ``__mf_digest__``. Qualified names alone are not content
        identity: editing a named function in place must not replay stale fitted
        preprocessors or prepared-base panels from a previous run.

        Second caveat: the key does NOT independently encode the ``fit_policy``
        (``origin_available`` vs ``fit_window``) used when calling
        ``PreprocessSpec.fit`` -- UNLESS the store was constructed with a
        ``namespace`` that carries it (see :meth:`__init__`). The preprocessing
        spec's own ``standardize_scope`` option is encoded separately through
        the spec identity. Within a single run/pipeline the policy scope is
        constant, so reuse is safe without a namespace. Never share one store
        directory across runs that differ in
        ``preprocessing_policy.scope`` for the same spec WITHOUT namespacing by
        that scope -- a fit_window fit would be served where an
        origin_available fit is expected (or vice versa), silently producing
        wrong transforms. ``macroforecast.pipeline`` always constructs its
        shared store with a namespace derived from the effective
        ``StagePolicy`` so pipeline-driven runs are safe by construction; a
        caller wiring ``PreprocessorStore`` directly (bypassing the pipeline)
        is responsible for passing a namespace when it shares one directory
        across differing scopes.

        Parameters
        ----------
        prep_spec:
            A ``PreprocessSpec`` instance (must expose ``to_dict()``).
        target:
            Name of the target series, e.g. ``"INDPRO"``.
        origin_pos:
            Integer index of the forecast origin in the rolling window.

        Returns
        -------
        str
            64-character lowercase hex digest.
        """
        payload: dict[str, Any] = {
            "spec": _disk_spec_identity(prep_spec),
            "target": str(target),
            "origin_pos": int(origin_pos),
        }
        if self._namespace is not None:
            payload["namespace"] = self._namespace
        canonical = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def frame_key(
        self,
        prep_spec: Any,
        *,
        target: str,
        cache_key: Any,
        kind: str,
    ) -> str:
        """Return a SHA-256 digest for a stored frame payload.

        ``cache_key`` is the same horizon-independent identity used by the
        runner's in-memory cache (for prepared-base panels, this is
        ``_preprocessing_cache_key(item, vintage_id=...)``). The store namespace
        is folded in exactly as :meth:`key` does, so one shared directory remains
        isolated across preprocessing-policy scopes and vintage tags.
        """

        payload: dict[str, Any] = {
            "kind": str(kind),
            "spec": _disk_spec_identity(prep_spec),
            "target": str(target),
            "cache_key": _json_ready(cache_key),
        }
        if self._namespace is not None:
            payload["namespace"] = self._namespace
        canonical = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, key: str) -> Any | None:
        """Return the unpickled object for *key*, or ``None`` on a miss or error.

        A corrupt, truncated, or partially-written file is silently treated as
        a cache miss — it is never re-raised to the caller.

        Parameters
        ----------
        key:
            A hex digest previously returned by :meth:`key`.

        Returns
        -------
        object or None
            The cached object, or ``None`` if the key is not present or the
            stored data is unreadable.
        """
        path = self._path(key)
        if not path.exists():
            return None
        try:
            data = path.read_bytes()
            return pickle.loads(data)  # noqa: S301 — trusted local filesystem
        except Exception:  # noqa: BLE001 — any unpickling / IO error → miss
            return None

    def get_frame(self, key: str) -> pd.DataFrame | None:
        """Return a stored DataFrame payload for *key*, or ``None`` on a miss.

        The JSON metadata must be readable and match the parquet payload shape;
        corrupt/truncated files are treated as cache misses, mirroring
        :meth:`get`.
        """

        parquet_path, meta_path = self._frame_paths(key)
        if not parquet_path.exists() or not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text())
            frame = pd.read_parquet(parquet_path)
            columns = list(meta.get("columns", []))
            if columns and list(frame.columns) != columns:
                frame = frame.loc[:, columns]
            for column, dtype in dict(meta.get("dtypes", {})).items():
                if column in frame.columns:
                    frame[column] = frame[column].astype(dtype)
            index_freq = meta.get("index_freq")
            if index_freq and isinstance(frame.index, pd.DatetimeIndex):
                try:
                    frame.index.freq = index_freq
                except ValueError:
                    pass
            frame.attrs = dict(meta.get("attrs", {}))
            return frame
        except Exception:  # noqa: BLE001 - corrupt frame/meta -> cache miss
            return None

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def put(self, key: str, value: Any) -> None:
        """Pickle *value* and store it under *key* via an atomic temp-file replace.

        The write sequence is:

        1. Serialize *value* to bytes with ``pickle.dumps``.
        2. Write the bytes to a temporary file in ``root_dir`` (same filesystem
           as the final destination, so the subsequent rename is always atomic).
        3. Call ``os.replace(tmp, final)`` — atomic on POSIX; on Windows it is
           best-effort atomic.

        A concurrent writer racing to store the same key is harmless: both
        writers produce identical content, and ``os.replace`` is atomic, so the
        final file is always complete.

        Parameters
        ----------
        key:
            A hex digest previously returned by :meth:`key`.
        value:
            Any picklable object.
        """
        self._root.mkdir(parents=True, exist_ok=True)
        final = self._path(key)
        data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)

        # Write to a temp file in the same directory so os.replace is atomic
        # (same filesystem, no cross-device move).
        fd, tmp_path = tempfile.mkstemp(
            suffix=".tmp",
            prefix=f"_{key}_",
            dir=self._root,
        )
        try:
            os.write(fd, data)
        except Exception:
            os.close(fd)
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        else:
            os.close(fd)

        # Atomic rename — on POSIX this is guaranteed to be atomic.
        os.replace(tmp_path, str(final))

    def put_frame(self, key: str, frame: pd.DataFrame) -> None:
        """Store a DataFrame payload under *key* via atomic parquet/json writes."""

        self._root.mkdir(parents=True, exist_ok=True)
        parquet_path, meta_path = self._frame_paths(key)

        fd, tmp_parquet = tempfile.mkstemp(
            suffix=".parquet",
            prefix=f"_{key}_",
            dir=self._root,
        )
        os.close(fd)
        try:
            frame.to_parquet(tmp_parquet, index=True)
            os.replace(tmp_parquet, str(parquet_path))
        except Exception:
            try:
                os.unlink(tmp_parquet)
            except OSError:
                pass
            raise

        meta = {
            "columns": [str(column) for column in frame.columns],
            "dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
            "index_names": list(frame.index.names),
            "index_dtype": str(frame.index.dtype),
            "index_freq": getattr(frame.index, "freqstr", None),
            "attrs": _json_ready(dict(frame.attrs)),
        }
        fd, tmp_meta = tempfile.mkstemp(
            suffix=".json",
            prefix=f"_{key}_",
            dir=self._root,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(meta, fh, sort_keys=True)
                fh.write("\n")
            os.replace(tmp_meta, str(meta_path))
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                os.unlink(tmp_meta)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _path(self, key: str) -> Path:
        """Return the filesystem path for a given cache key."""
        return self._root / f"{key}.pkl"

    def _frame_paths(self, key: str) -> tuple[Path, Path]:
        """Return ``(parquet_path, meta_path)`` for a DataFrame payload."""

        return self._root / f"{key}.parquet", self._root / f"{key}.json"


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


def _disk_spec_identity(prep_spec: Any) -> dict[str, Any]:
    options = getattr(prep_spec, "options", None)
    if isinstance(options, Mapping):
        return {"options": _json_ready_for_disk(dict(options), path="preprocessing.options")}
    if hasattr(prep_spec, "to_dict"):
        return _json_ready_for_disk(prep_spec.to_dict(), path="preprocessing")
    return _json_ready_for_disk(prep_spec, path="preprocessing")


def _json_ready_for_disk(value: Any, *, path: str) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_ready_for_disk(item, path=f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [_json_ready_for_disk(item, path=f"{path}[]") for item in value]
    if isinstance(value, list):
        return [_json_ready_for_disk(item, path=f"{path}[]") for item in value]
    if callable(value):
        return {
            "callable": _callable_name(value),
            "mf_digest": _callable_digest(value, path=path),
        }
    return _json_ready(value)


def _callable_digest(func: Callable[..., Any], *, path: str) -> str:
    marker = getattr(func, "__mf_digest__", None)
    if marker is not None:
        return str(marker)
    raise UndigestiblePreprocessorSpec(
        f"{path} is a custom preprocessing callable without __mf_digest__; "
        "skipping disk preprocessing cache tiers for this spec"
    )


def _callable_name(func: Callable[..., Any]) -> str:
    module = getattr(func, "__module__", "")
    qualname = getattr(func, "__qualname__", getattr(func, "__name__", repr(func)))
    return f"{module}.{qualname}" if module else str(qualname)


__all__ = ["PreprocessorStore"]
