"""Content identity for a data panel.

A fingerprint is a property of the data, not of a study, so it belongs here rather
than in ``pipeline``. ``pipeline.run``, ``data.vintage``, the result store and the
artifact manifest all need the same one -- and until 2026-08-09 ``data.vintage``
reached UP into ``pipeline.run`` to get it, one of two known layering exceptions.

The digest covers the whole panel. It used to fall back to a strided subsample above
a cell cap, which meant a cell the stride skipped could change without the digest
moving -- and this digest is part of result-cell identity, so a stale forecast could
be served for changed data (F-005). Size is now handled by streaming the values in
row chunks instead of by looking at fewer of them.

The returned mapping is unchanged, deliberately. ``result_cell_identity`` serialises
the whole dict into the canonical payload and ``ResultStore.load`` compares the whole
manifest mapping, so adding or dropping a key misses every existing cache even when
the digest is identical. Every panel old code hashed in full therefore keeps both its
value and its mapping; only the oversized panels, whose identity was unsound, move.

The digest is also a property of the data rather than of the pandas build that read it.
A ``DatetimeIndex`` remembers the resolution it was constructed at -- pandas 2 defaults
to nanoseconds, pandas 3 to microseconds -- and ``.asi8`` reports the RAW stored
integers, so feeding it directly made the same instants hash to two different values on
the two versions, and let two indexes holding DIFFERENT instants at different
resolutions hash to the same one whenever their stored integers happened to coincide.
Datetime index values are therefore canonicalised to nanoseconds before they are read as
integers, which keeps the long-standing pandas-2 digest and makes every resolution agree
with it.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import OutOfBoundsDatetime

#: Values are fed to the digest in row chunks of at most this many cells, so a large
#: panel never needs a second full-size copy. This is a memory bound and nothing
#: else: the byte stream, and therefore the digest, is identical for every chunk
#: size, because a row-major buffer split by rows concatenates back to itself.
_FINGERPRINT_CHUNK_CELLS = 4_000_000

#: Datetime index values are converted to this resolution before being read as int64.
#: It is nanoseconds because that is what every digest recorded before pandas began
#: defaulting a new index to microseconds, so canonicalising here repairs portability
#: without moving a single fingerprint that a pandas-2 run had already stored.
_FINGERPRINT_DATETIME_UNIT = "ns"

#: Written before the semantic index form below, so that path cannot produce the same
#: bytes as the int64 DATETIME path. It claims nothing against the non-datetime string
#: fallback, which is written untagged, so a contrived non-datetime index could in
#: principle spell these bytes itself. Only panels dated outside the nanosecond range
#: reach the tagged path, and those never had a resolution-independent digest to begin
#: with, so no fingerprint that was ever portable moves because of it.
_OUT_OF_NS_RANGE_DATETIME_TAG = b"\x00mf.datetime.iso\x00"


def _chunk_bounds(n_rows: int, n_cols: int) -> list[tuple[int, int]]:
    """Row slices whose cell count stays under :data:`_FINGERPRINT_CHUNK_CELLS`."""

    if n_rows <= 0:
        return []
    rows_per_chunk = max(1, _FINGERPRINT_CHUNK_CELLS // max(1, n_cols))
    return [
        (start, min(start + rows_per_chunk, n_rows))
        for start in range(0, n_rows, rows_per_chunk)
    ]


def _update_separated_strings(digest: Any, values: Any) -> None:
    """Feed ``values`` as ``\x1f``-separated text, without building the joined string.

    Byte-for-byte what ``"\x1f".join(...)`` produced -- the separator goes between
    elements and not after the last one -- so this is an allocation change and not a
    digest change. It matters for an extremely wide panel, where joining every column
    name first would materialise the whole header as one string.
    """
    for position, value in enumerate(values):
        if position:
            digest.update(b"\x1f")
        digest.update(str(value).encode())


def _canonical_datetime_asi8(index: pd.DatetimeIndex) -> np.ndarray | None:
    """*index* re-expressed at :data:`_FINGERPRINT_DATETIME_UNIT`, or ``None``.

    ``as_unit`` returns a NEW index and leaves the caller's alone, which matters
    because the panel being fingerprinted belongs to the caller and identity must not
    be something that edits its input.

    ``None`` means at least one instant lies outside the nanosecond range (roughly
    1677 to 2262), so there is no int64 form of it to compare across pandas versions.
    Returning the raw ``.asi8`` instead would be the very defect this repairs, so the
    caller writes a semantic form for those panels rather than a resolution-dependent
    integer one.
    """
    try:
        return index.as_unit(_FINGERPRINT_DATETIME_UNIT).asi8
    except (OutOfBoundsDatetime, OverflowError):
        return None


def _utc_view_of_aware_datetimes(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """*index* re-expressed in UTC when it is aware, and unchanged when it is naive.

    ``tz_convert`` returns a NEW index and renames the same instants at the same stored
    resolution, so this neither edits the caller's panel nor moves a moment inside it.

    A naive index is returned as it came rather than localised, because localising
    would assert an offset the panel never stated. What that costs is recorded in
    :func:`_update_semantic_datetimes`.
    """
    if index.tz is None:
        return index
    return index.tz_convert("UTC")


def _update_semantic_datetimes(digest: Any, index: pd.DatetimeIndex) -> None:
    """Feed out-of-nanosecond-range instants as ISO text, tagged and unambiguous.

    An aware index is read in UTC first. ``isoformat()`` spells the local wall clock
    and its offset, so one instant filed under two zones would otherwise leave two
    digests, while inside the nanosecond range ``.asi8`` is already a count from the
    epoch and the zone an index is displayed in never reaches the hash at all. Reading
    in UTC is what lets this path keep that promise: one instant, one fingerprint,
    whatever zone names it, at whatever resolution it was built.

    A naive index stays naive, so it does not agree with an aware one here even when
    their wall clocks read alike. That is stricter than the int64 path, where a naive
    index and a UTC one do collide because both spell the same integers. Stricter is
    the safe direction, since it can only miss a reuse and never merge two panels into
    one cell, and only dates outside the nanosecond range reach this code, where no
    digest was portable to begin with, so nothing already stored moves.
    """
    digest.update(_OUT_OF_NS_RANGE_DATETIME_TAG)
    _update_separated_strings(
        digest,
        (
            value.isoformat() if hasattr(value, "isoformat") else str(value)
            for value in _utc_view_of_aware_datetimes(index)
        ),
    )


def _update_index(digest: Any, frame: pd.DataFrame, bounds: list[tuple[int, int]]) -> None:
    """Feed every index value to ``digest`` in row order.

    Int64 nanoseconds for the usual case, but not for every one. A datetime index
    whose instants do not all fit the nanosecond range takes the tagged semantic
    ISO path below instead, and an index with no ``.asi8`` at all falls back to a
    string form.
    """

    index = frame.index
    if isinstance(index, pd.DatetimeIndex):
        # Canonicalised rather than read raw, because ``.asi8`` reports the resolution
        # the index happens to be stored at. An index already at nanoseconds converts
        # to itself, so this repairs portability without moving its digest.
        asi8 = _canonical_datetime_asi8(index)
        if asi8 is None:
            _update_semantic_datetimes(digest, index)
            return
    else:
        try:
            # Period and Timedelta indexes keep this path byte for byte, because only
            # a DatetimeIndex is canonicalised above. That is stability HERE, not
            # portability in general. ``validate_panel`` requires a canonical panel's
            # index to be a DatetimeIndex, so a canonical panel never reaches this
            # branch, and an arbitrary TimedeltaIndex still reports ``.asi8`` at
            # whatever resolution it happens to be stored at. Nothing here makes that
            # resolution-independent.
            asi8 = index.asi8
        except AttributeError:
            # Non-datetime index (should not happen for a canonical panel, but the
            # fingerprint must never raise): fall back to a stable string form. The
            # separator is written between elements rather than by joining the whole
            # index into one string first, which is the same bytes without the
            # whole-index allocation.
            _update_separated_strings(digest, index)
            return
    for start, stop in bounds:
        digest.update(np.ascontiguousarray(asi8[start:stop]).tobytes())


def _update_values(digest: Any, frame: pd.DataFrame, bounds: list[tuple[int, int]]) -> None:
    """Feed the panel's values to ``digest`` in row order.

    Explicit little-endian so the digest is stable across byte orders, not just across
    runs on one machine.

    A canonical panel cannot be complex -- ``validate_panel`` rejects it (F-007) -- but
    this helper is also called on feature panels that did not come through that gate.
    Those are hashed as interleaved ``<c16`` rather than by writing each chunk's real
    part followed by its imaginary part: the interleaved layout is what makes chunk
    concatenation give the same bytes as one pass, so the chunk size stays a memory
    choice and never enters content identity.
    """
    has_complex = any(
        pd.api.types.is_complex_dtype(dtype) for dtype in frame.dtypes
    )
    dtype = "complex128" if has_complex else "float64"
    itemtype = "<c16" if has_complex else "<f8"
    for start, stop in bounds:
        values = np.ascontiguousarray(frame.iloc[start:stop].to_numpy(dtype=dtype))
        digest.update(values.astype(itemtype, copy=False).tobytes())


def panel_fingerprint(frame: pd.DataFrame) -> dict[str, Any]:
    """A stable sha256 fingerprint over the panel's index, columns, and values.

    Every index value, every column name, and every cell influences the digest,
    whatever the panel's size. Large panels are streamed in row chunks rather than
    sampled, so the bound is on memory and not on how much of the data is looked at.

    ``method``/``row_stride``/``col_stride``/``sampled_shape`` are kept and now report
    what is always true: the full content, at stride 1, over the panel's own shape.
    They are not vestigial -- the whole mapping is part of stored cache identity, so
    dropping them would invalidate every existing manifest for no gain. What they no
    longer do is describe a subsample, because there isn't one.

    Determinism holds across copies, repeated calls, and chunk sizes: the same content
    gives the same value, and changing any single cell changes it.
    """
    n_rows, n_cols = frame.shape
    bounds = _chunk_bounds(int(n_rows), int(n_cols))

    digest = hashlib.sha256()
    _update_index(digest, frame, bounds)
    _update_separated_strings(digest, frame.columns)
    _update_values(digest, frame, bounds)

    return {
        "algorithm": "sha256",
        "method": "full_content",
        "value": digest.hexdigest(),
        "row_stride": 1,
        "col_stride": 1,
        "sampled_shape": [int(n_rows), int(n_cols)],
    }


#: Retired with the subsample path and read by nothing: there is no cap left to
#: govern. The name stays bound because ``pipeline.run`` re-exports it and that module
#: is outside this change's scope; ``None`` rather than the old 20_000_000 so a reader
#: cannot mistake it for a live threshold.
_FINGERPRINT_FULL_CELL_CAP = None

#: Old private name, kept for one release so an in-flight import still resolves.
_panel_fingerprint = panel_fingerprint
