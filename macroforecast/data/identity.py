"""Content identity for a data panel.

A fingerprint is a property of the data, not of a study, so it belongs here rather
than in ``pipeline``. ``pipeline.run``, ``data.vintage``, the result store and the
artifact manifest all need the same one -- and until 2026-08-09 ``data.vintage``
reached UP into ``pipeline.run`` to get it, one of two known layering exceptions.

Moved verbatim: same algorithm, same subsampling cap, same returned dict. A digest
that changed here would invalidate every cached cell and every recorded manifest,
so this move is required to be a no-op on the bytes.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

#: Above this many cells the digest comes from a deterministic strided subsample
#: rather than full content, and the returned dict records that.
_FINGERPRINT_FULL_CELL_CAP = 20_000_000


def panel_fingerprint(frame: pd.DataFrame) -> dict[str, Any]:
    """A stable sha256 fingerprint over the panel's index, columns, and values.

    Full content by default (index as int64 ns timestamps, column names in
    order, values as explicit little-endian float64 bytes -- so the digest is
    stable across platforms/byte orders, not just across runs on one machine).
    Above :data:`_FINGERPRINT_FULL_CELL_CAP` cells the digest is computed from
    a deterministic strided subsample instead (same row/col stride every call
    for the same shape), and ``method``/``row_stride``/``col_stride`` record
    this so a referee never mistakes it for a full-content digest.
    """
    n_rows, n_cols = frame.shape
    total_cells = n_rows * n_cols
    row_stride = col_stride = 1
    method = "full_content"
    sampled = frame
    if total_cells > _FINGERPRINT_FULL_CELL_CAP and n_rows > 0 and n_cols > 0:
        reduction = total_cells / _FINGERPRINT_FULL_CELL_CAP
        row_stride = max(1, round(reduction ** 0.5))
        col_stride = max(1, round(reduction / row_stride))
        sampled = frame.iloc[::row_stride, ::col_stride]
        method = "strided_subsample"

    digest = hashlib.sha256()
    try:
        digest.update(np.ascontiguousarray(sampled.index.asi8).tobytes())
    except AttributeError:
        # Non-datetime index (should not happen for a canonical panel, but the
        # fingerprint must never raise): fall back to a stable string form.
        digest.update("\x1f".join(str(v) for v in sampled.index).encode())
    digest.update("\x1f".join(str(c) for c in sampled.columns).encode())
    values = np.ascontiguousarray(sampled.to_numpy(dtype="float64"))
    digest.update(values.astype("<f8", copy=False).tobytes())

    return {
        "algorithm": "sha256",
        "method": method,
        "value": digest.hexdigest(),
        "row_stride": row_stride,
        "col_stride": col_stride,
        "sampled_shape": [int(sampled.shape[0]), int(sampled.shape[1])],
    }


#: Old private name, kept for one release so an in-flight import still resolves.
_panel_fingerprint = panel_fingerprint
