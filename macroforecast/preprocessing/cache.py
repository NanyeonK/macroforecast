"""Content-addressed on-disk store for fitted preprocessing objects.

``PreprocessorStore`` is a directory-backed cache that maps a
``(PreprocessSpec, target, origin_pos)`` triple to a pickled
``FittedPreprocessor`` (or any picklable object).  It is designed for safe
concurrent multi-process access:

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
    """

    def __init__(self, root_dir: str | Path) -> None:
        self._root = Path(root_dir)

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

        Second caveat: the key does NOT encode the ``fit_policy``
        (``origin_available`` vs ``fit_window``) used when calling
        ``PreprocessSpec.fit``. Within a single run/pipeline the scope is
        constant, so reuse is safe. Never share one store directory across runs
        that differ in ``preprocessing_policy.scope`` for the same spec -- a
        fit_window fit would be served where an origin_available fit is expected
        (or vice versa), silently producing wrong transforms.

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
