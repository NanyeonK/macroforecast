from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from macroforecast.model_selection.types import ValidationSplitterSpec, WithinFoldMode
from macroforecast.window import Split, make_splitter, normalize_window_name


ValidationSplitterLike = ValidationSplitterSpec | Callable[[pd.Index], Any] | str


def validation_splitter(method: str, **params: Any) -> ValidationSplitterSpec:
    """Build a named validation-splitter override for a ``SearchSpec``."""

    return ValidationSplitterSpec(method=str(method), params=dict(params))


def explicit_folds(
    boundaries: Sequence[Any],
    *,
    within_fold: str = "fixed",
) -> ValidationSplitterSpec:
    """Build fixed-boundary validation folds for a ``SearchSpec``."""

    return ValidationSplitterSpec(
        method="explicit_folds",
        explicit_folds=tuple(boundaries),
        within_fold=_coerce_within_fold(within_fold),
    )


def recursive_threefold() -> ValidationSplitterSpec:
    """Build recursive three-fold validation with expanding train blocks."""

    return ValidationSplitterSpec(method="recursive_threefold")


def resolve_validation_splitter(
    index: pd.Index,
    splitter: ValidationSplitterLike,
) -> tuple[list[Split], str, dict[str, Any]]:
    """Resolve a validation-splitter override against an index."""

    splits, name, metadata, _ = _resolve_validation_splitter_with_fold_ids(
        index,
        splitter,
    )
    return splits, name, metadata


def _resolve_validation_splitter_with_fold_ids(
    index: pd.Index,
    splitter: ValidationSplitterLike,
) -> tuple[list[Split], str, dict[str, Any], list[int]]:
    """Resolve a validation splitter and logical fold IDs."""

    labels = pd.Index(index)
    if callable(splitter) and not isinstance(splitter, ValidationSplitterSpec):
        splits = _normalize_callable_splits(splitter(labels), labels)
        name = "callable_splitter"
        metadata = {
            "split_source": "validation_splitter",
            "validation_splitter": _callable_metadata(splitter),
            "window": None,
            "temporal_order": _splits_are_temporal(splits),
        }
        return splits, name, metadata, list(range(len(splits)))

    spec = (
        ValidationSplitterSpec(method=splitter)
        if isinstance(splitter, str)
        else splitter
    )
    if spec.method == "explicit_folds":
        splits, fold_ids = _explicit_fold_splits_with_fold_ids(
            labels,
            spec.explicit_folds,
            spec.within_fold,
        )
        name = "explicit_folds"
        temporal = True
    elif spec.method == "recursive_threefold":
        splits = _recursive_threefold_splits(labels)
        fold_ids = list(range(len(splits)))
        name = "recursive_threefold"
        temporal = True
    else:
        method = normalize_window_name(spec.method)
        splits = make_splitter(method, len(labels), **dict(spec.params))
        fold_ids = list(range(len(splits)))
        name = method
        temporal = method != "random_kfold"
    metadata = {
        "split_source": "validation_splitter",
        "validation_splitter": spec.to_dict(),
        "window": None,
        "temporal_order": temporal,
    }
    return splits, name, metadata, fold_ids


def _coerce_within_fold(value: str) -> WithinFoldMode:
    key = str(value).lower().replace("-", "_")
    if key not in {"fixed", "expanding"}:
        raise ValueError("within_fold must be 'fixed' or 'expanding'")
    return "expanding" if key == "expanding" else "fixed"


def _explicit_fold_splits(
    index: pd.Index,
    boundaries: Sequence[Any],
    within_fold: str,
) -> list[Split]:
    splits, _ = _explicit_fold_splits_with_fold_ids(index, boundaries, within_fold)
    return splits


def _explicit_fold_splits_with_fold_ids(
    index: pd.Index,
    boundaries: Sequence[Any],
    within_fold: str,
) -> tuple[list[Split], list[int]]:
    positions = _boundary_positions(index, boundaries)
    splits: list[Split] = []
    fold_ids: list[int] = []
    for fold_id, (start, end) in enumerate(zip(positions[:-1], positions[1:], strict=True)):
        if within_fold == "fixed":
            splits.append((
                np.arange(start, dtype=int),
                np.arange(start, end, dtype=int),
            ))
            fold_ids.append(fold_id)
        else:
            for val_pos in range(start, end):
                splits.append((
                    np.arange(val_pos, dtype=int),
                    np.asarray([val_pos], dtype=int),
                ))
                fold_ids.append(fold_id)
    if not splits:
        raise ValueError("explicit_folds produced no validation splits")
    return splits, fold_ids


def _recursive_threefold_splits(index: pd.Index) -> list[Split]:
    boundaries = np.linspace(0, len(index), 5, dtype=int)[1:]
    return _explicit_fold_splits(index, boundaries.tolist(), "fixed")


def _boundary_positions(index: pd.Index, boundaries: Sequence[Any]) -> list[int]:
    if len(boundaries) < 2:
        raise ValueError("explicit_folds requires at least two boundaries")
    positions = [_boundary_position(index, boundary) for boundary in boundaries]
    n_obs = len(index)
    previous = 0
    for pos in positions:
        if pos <= previous:
            raise ValueError("explicit_folds boundaries must be strictly increasing")
        if pos > n_obs:
            raise ValueError("explicit_folds boundaries cannot exceed index length")
        previous = pos
    if positions[0] <= 0:
        raise ValueError("the first explicit_folds boundary must leave training rows")
    return positions


def _boundary_position(index: pd.Index, boundary: Any) -> int:
    if isinstance(boundary, (int, np.integer)) and not isinstance(boundary, bool):
        return int(boundary)
    loc = index.get_loc(boundary)
    if isinstance(loc, slice):
        if loc.stop is None:
            raise ValueError(f"boundary label {boundary!r} is not uniquely located")
        return int(loc.stop)
    if isinstance(loc, np.ndarray):
        matches = np.flatnonzero(loc) if loc.dtype == bool else loc
        if len(matches) != 1:
            raise ValueError(f"boundary label {boundary!r} is not uniquely located")
        return int(matches[0]) + 1
    return int(loc) + 1


def _normalize_callable_splits(value: Any, index: pd.Index) -> list[Split]:
    if not isinstance(value, Sequence):
        raise TypeError("callable validation splitter must return a sequence of splits")
    splits: list[Split] = []
    for split_id, pair in enumerate(value):
        if not isinstance(pair, Sequence) or len(pair) != 2:
            raise ValueError("each callable split must contain train and validation indices")
        train_idx = _normalize_split_side(pair[0], index, split_id=split_id, side="train")
        val_idx = _normalize_split_side(pair[1], index, split_id=split_id, side="validation")
        if np.intersect1d(train_idx, val_idx).size:
            raise ValueError(f"split {split_id} train and validation positions overlap")
        splits.append((train_idx, val_idx))
    if not splits:
        raise ValueError("callable validation splitter produced no splits")
    return splits


def _normalize_split_side(
    values: Any,
    index: pd.Index,
    *,
    split_id: int,
    side: str,
) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"split {split_id} {side} indices must be one-dimensional")
    if arr.dtype == bool:
        if len(arr) != len(index):
            raise ValueError(
                f"split {split_id} {side} boolean mask length must match index length"
            )
        arr = np.flatnonzero(arr)
    elif np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(int, copy=False)
    else:
        loc = index.get_indexer(pd.Index(arr))
        if (loc < 0).any():
            missing = pd.Index(arr)[loc < 0]
            raise ValueError(
                f"split {split_id} {side} labels are not in the selection index: "
                f"{list(missing[:3])}"
            )
        arr = loc.astype(int, copy=False)
    if len(arr) == 0:
        raise ValueError(f"split {split_id} {side} indices must not be empty")
    if len(np.unique(arr)) != len(arr):
        raise ValueError(f"split {split_id} {side} indices must not contain duplicates")
    if int(arr.min()) < 0 or int(arr.max()) >= len(index):
        raise ValueError(f"split {split_id} {side} indices are outside index bounds")
    return arr.astype(int, copy=False)


def _splits_are_temporal(splits: list[Split]) -> bool:
    return all(int(train_idx.max()) < int(val_idx.min()) for train_idx, val_idx in splits)


def _callable_metadata(func: Callable[..., Any]) -> dict[str, Any]:
    module = getattr(func, "__module__", "")
    qualname = getattr(func, "__qualname__", getattr(func, "__name__", repr(func)))
    name = f"{module}.{qualname}" if module else str(qualname)
    return {"callable": name, "mf_digest": getattr(func, "__mf_digest__", None)}


__all__ = [
    "ValidationSplitterLike",
    "explicit_folds",
    "recursive_threefold",
    "resolve_validation_splitter",
    "validation_splitter",
]
