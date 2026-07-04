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

When the store is constructed with a ``namespace`` (any JSON-serialisable
value -- e.g. the effective ``StagePolicy.to_dict()`` the caller fitted under),
the namespace is folded into the same digest. Two callers sharing one
``root_dir`` but constructing the store with DIFFERENT namespaces therefore
never collide: their keys differ even for an identical ``(spec, target,
origin_pos)`` triple, so a fit produced under one namespace is never served to
a reader in another. ``namespace=None`` (the default) reproduces the original,
pre-namespace digest exactly, so existing stores built before this parameter
existed keep working unchanged for callers that do not pass one.

Dependencies
------------
stdlib only: ``hashlib``, ``json``, ``os``, ``pathlib``, ``pickle``,
``tempfile``.  No third-party imports.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any


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

        ``prep_spec.to_dict()`` already converts the options dict to a
        JSON-ready form (callables become qualified name strings, tuples
        become lists), so ``json.dumps(..., sort_keys=True)`` is stable across
        Python runs.

        Caveat: a callable carried in the spec (e.g. a ``custom_steps`` entry)
        is identified only by its qualified name, so two DIFFERENT anonymous
        ``lambda`` callables both serialise to ``<module>.<lambda>`` and would
        share a key. Use named module-level functions in ``custom_steps`` to
        guarantee distinct keys. The standard pipeline specs (impute /
        standardize / transform / outliers options) carry no anonymous
        callables and are unaffected.

        Second caveat: the key does NOT independently encode the ``fit_policy``
        (``origin_available`` vs ``fit_window``) used when calling
        ``PreprocessSpec.fit`` -- UNLESS the store was constructed with a
        ``namespace`` that carries it (see :meth:`__init__`). Within a single
        run/pipeline the scope is constant, so reuse is safe without a
        namespace. Never share one store directory across runs that differ in
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
            "spec": prep_spec.to_dict(),
            "target": str(target),
            "origin_pos": int(origin_pos),
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _path(self, key: str) -> Path:
        """Return the filesystem path for a given cache key."""
        return self._root / f"{key}.pkl"


__all__ = ["PreprocessorStore"]
