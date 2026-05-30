from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

Split = tuple[np.ndarray, np.ndarray]

_VALIDATION_ALIASES = {
    "last": "last_block",
    "last_block": "last_block",
    "holdout": "last_block",
    "poos": "poos",
    "pseudo_out_of_sample": "poos",
    "expanding": "expanding",
    "expanding_walk_forward": "expanding",
    "time_series_split": "expanding",
    "rolling": "rolling_blocks",
    "rolling_blocks": "rolling_blocks",
    "rolling_walk_forward": "rolling_blocks",
    "blocked_kfold": "blocked_kfold",
    "block_cv": "blocked_kfold",
    "kfold": "blocked_kfold",
}


def _check_n_samples(n_samples: int) -> int:
    n = int(n_samples)
    if n < 2:
        raise ValueError("n_samples must be at least 2")
    return n


def _check_nonnegative_int(name: str, value: int) -> int:
    out = int(value)
    if out < 0:
        raise ValueError(f"{name} must be non-negative")
    return out


@dataclass(frozen=True)
class WindowSpec:
    """Reusable temporal window specification."""

    method: str = "expanding"
    validation_size: int | None = None
    validation_ratio: float = 0.2
    min_train_size: int | None = None
    n_splits: int = 5
    step: int = 1
    horizon: int = 1
    embargo: int = 0
    metadata: dict[str, Any] | None = None

    def split(self, n_samples: int) -> list[Split]:
        """Return train/validation index splits for ``n_samples``."""

        return make_splitter(
            self.method,
            n_samples,
            validation_size=self.validation_size,
            validation_ratio=self.validation_ratio,
            min_train_size=self.min_train_size,
            n_splits=self.n_splits,
            step=self.step,
            horizon=self.horizon,
            embargo=self.embargo,
        )

    def to_table(self, n_samples: int, *, index: pd.Index | None = None) -> pd.DataFrame:
        """Return this window as an inspectable split table."""

        return split_table(
            self.method,
            n_samples,
            index=index,
            validation_size=self.validation_size,
            validation_ratio=self.validation_ratio,
            min_train_size=self.min_train_size,
            n_splits=self.n_splits,
            step=self.step,
            horizon=self.horizon,
            embargo=self.embargo,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a metadata representation of the window."""

        return {
            "method": normalize_window_name(self.method),
            "validation_size": self.validation_size,
            "validation_ratio": self.validation_ratio,
            "min_train_size": self.min_train_size,
            "n_splits": self.n_splits,
            "step": self.step,
            "horizon": self.horizon,
            "embargo": self.embargo,
            "metadata": dict(self.metadata or {}),
        }


def normalize_window_name(window: str) -> str:
    """Return the canonical window method name for a method or alias."""

    key = str(window).lower().replace("-", "_")
    try:
        return _VALIDATION_ALIASES[key]
    except KeyError as exc:
        allowed = "last_block, poos, expanding, rolling_blocks, blocked_kfold"
        raise ValueError(
            f"Unknown window method {window!r}. Available methods: {allowed}."
        ) from exc


def resolve_window(window: WindowSpec | str | None = None) -> WindowSpec:
    """Return a ``WindowSpec`` from a spec, method name, or default."""

    if window is None:
        return expanding()
    if isinstance(window, WindowSpec):
        return window
    return WindowSpec(method=normalize_window_name(window))


def last_block(
    *,
    validation_size: int | None = None,
    validation_ratio: float = 0.2,
    embargo: int = 0,
) -> WindowSpec:
    """Configure one final validation block."""

    return WindowSpec(
        method="last_block",
        validation_size=validation_size,
        validation_ratio=validation_ratio,
        embargo=embargo,
    )


def poos(
    *,
    validation_size: int | None = None,
    validation_ratio: float = 0.25,
    embargo: int = 0,
) -> WindowSpec:
    """Configure pseudo-out-of-sample one-step tail splits."""

    return WindowSpec(
        method="poos",
        validation_size=validation_size,
        validation_ratio=validation_ratio,
        embargo=embargo,
    )


def expanding(
    *,
    min_train_size: int | None = None,
    step: int = 1,
    horizon: int = 1,
    embargo: int = 0,
) -> WindowSpec:
    """Configure expanding-window validation splits."""

    return WindowSpec(
        method="expanding",
        min_train_size=min_train_size,
        step=step,
        horizon=horizon,
        embargo=embargo,
    )


def rolling_blocks(
    *,
    n_blocks: int = 3,
    block_size: int | None = None,
    embargo: int = 0,
) -> WindowSpec:
    """Configure consecutive validation blocks over the sample tail."""

    return WindowSpec(
        method="rolling_blocks",
        validation_size=block_size,
        n_splits=n_blocks,
        embargo=embargo,
    )


def blocked_kfold(*, n_splits: int = 5, embargo: int = 0) -> WindowSpec:
    """Configure chronological blocked k-fold validation."""

    return WindowSpec(method="blocked_kfold", n_splits=n_splits, embargo=embargo)


def _resolve_validation_size(
    n_samples: int,
    *,
    validation_size: int | None,
    validation_ratio: float,
) -> int:
    if validation_size is not None:
        size = int(validation_size)
    else:
        if not 0 < validation_ratio < 1:
            raise ValueError("validation_ratio must be between 0 and 1")
        size = int(np.ceil(n_samples * validation_ratio))
    if size < 1:
        raise ValueError("validation_size must be at least 1")
    if size >= n_samples:
        raise ValueError("validation_size must be smaller than n_samples")
    return size


def _train_val(train_end: int, val_start: int, val_end: int) -> Split:
    if train_end <= 0:
        raise ValueError("split has no training observations")
    if val_start >= val_end:
        raise ValueError("split has no validation observations")
    return np.arange(train_end, dtype=int), np.arange(val_start, val_end, dtype=int)


def last_block_split(
    n_samples: int,
    *,
    validation_size: int | None = None,
    validation_ratio: float = 0.2,
    embargo: int = 0,
) -> Iterator[Split]:
    """Yield one split with the last block held out for validation."""

    n = _check_n_samples(n_samples)
    gap = _check_nonnegative_int("embargo", embargo)
    holdout = _resolve_validation_size(
        n, validation_size=validation_size, validation_ratio=validation_ratio
    )
    val_start = n - holdout
    train_end = val_start - gap
    yield _train_val(train_end, val_start, n)


def poos_split(
    n_samples: int,
    *,
    validation_size: int | None = None,
    validation_ratio: float = 0.25,
    embargo: int = 0,
) -> Iterator[Split]:
    """Yield pseudo-out-of-sample one-step validation splits over the tail block."""

    n = _check_n_samples(n_samples)
    gap = _check_nonnegative_int("embargo", embargo)
    holdout = _resolve_validation_size(
        n, validation_size=validation_size, validation_ratio=validation_ratio
    )
    start = n - holdout
    for val_start in range(start, n):
        train_end = val_start - gap
        yield _train_val(train_end, val_start, val_start + 1)


def expanding_split(
    n_samples: int,
    *,
    min_train_size: int | None = None,
    step: int = 1,
    horizon: int = 1,
    embargo: int = 0,
) -> Iterator[Split]:
    """Yield expanding-window validation splits."""

    n = _check_n_samples(n_samples)
    h = int(horizon)
    if h < 1:
        raise ValueError("horizon must be at least 1")
    if step < 1:
        raise ValueError("step must be at least 1")
    gap = _check_nonnegative_int("embargo", embargo)
    minimum = int(min_train_size) if min_train_size is not None else max(1, n // 2)
    if minimum >= n:
        raise ValueError("min_train_size must be smaller than n_samples")
    for train_end in range(minimum, n - gap - h + 1, int(step)):
        val_start = train_end + gap
        yield _train_val(train_end, val_start, val_start + h)


def rolling_blocks_split(
    n_samples: int,
    *,
    n_blocks: int = 3,
    block_size: int | None = None,
    embargo: int = 0,
) -> Iterator[Split]:
    """Yield consecutive validation blocks with all prior observations as training data."""

    n = _check_n_samples(n_samples)
    blocks = int(n_blocks)
    if blocks < 1:
        raise ValueError("n_blocks must be at least 1")
    size = int(block_size) if block_size is not None else max(1, n // (blocks + 2))
    if size < 1:
        raise ValueError("block_size must be at least 1")
    if blocks * size >= n:
        raise ValueError("n_blocks * block_size must be smaller than n_samples")
    gap = _check_nonnegative_int("embargo", embargo)
    start = n - blocks * size
    for block in range(blocks):
        val_start = start + block * size
        val_end = min(val_start + size, n)
        train_end = val_start - gap
        yield _train_val(train_end, val_start, val_end)


def blocked_kfold_split(
    n_samples: int,
    *,
    n_splits: int = 5,
    embargo: int = 0,
) -> Iterator[Split]:
    """Yield chronological blocked-fold splits using only past data for training."""

    n = _check_n_samples(n_samples)
    splits = int(n_splits)
    if splits < 2:
        raise ValueError("n_splits must be at least 2")
    if splits > n:
        raise ValueError("n_splits must be smaller than or equal to n_samples")
    boundaries = np.linspace(0, n, splits + 1, dtype=int)
    gap = _check_nonnegative_int("embargo", embargo)
    emitted = 0
    for fold in range(splits):
        val_start = int(boundaries[fold])
        val_end = int(boundaries[fold + 1])
        train_end = val_start - gap
        if train_end <= 0 or val_start >= val_end:
            continue
        emitted += 1
        yield _train_val(train_end, val_start, val_end)
    if emitted == 0:
        raise ValueError("blocked_kfold_split produced no valid chronological folds")


def make_splitter(
    validation: str,
    n_samples: int,
    *,
    validation_size: int | None = None,
    validation_ratio: float = 0.2,
    min_train_size: int | None = None,
    n_splits: int = 5,
    step: int = 1,
    horizon: int = 1,
    embargo: int = 0,
) -> list[Split]:
    """Build validation splits from a validation method name."""

    key = normalize_window_name(validation)
    if key == "last_block":
        splits = list(
            last_block_split(
                n_samples,
                validation_size=validation_size,
                validation_ratio=validation_ratio,
                embargo=embargo,
            )
        )
    elif key == "poos":
        splits = list(
            poos_split(
                n_samples,
                validation_size=validation_size,
                validation_ratio=validation_ratio,
                embargo=embargo,
            )
        )
    elif key == "expanding":
        splits = list(
            expanding_split(
                n_samples,
                min_train_size=min_train_size,
                step=step,
                horizon=horizon,
                embargo=embargo,
            )
        )
    elif key == "rolling_blocks":
        splits = list(
            rolling_blocks_split(
                n_samples,
                n_blocks=n_splits,
                block_size=validation_size,
                embargo=embargo,
            )
        )
    elif key == "blocked_kfold":
        splits = list(blocked_kfold_split(n_samples, n_splits=n_splits, embargo=embargo))
    if not splits:
        raise ValueError(f"Validation method {key!r} produced no splits")
    return splits


def split_table(
    validation: str,
    n_samples: int,
    *,
    index: pd.Index | None = None,
    validation_size: int | None = None,
    validation_ratio: float = 0.2,
    min_train_size: int | None = None,
    n_splits: int = 5,
    step: int = 1,
    horizon: int = 1,
    embargo: int = 0,
) -> pd.DataFrame:
    """Return validation splits as an inspectable table."""

    splits = make_splitter(
        validation,
        n_samples,
        validation_size=validation_size,
        validation_ratio=validation_ratio,
        min_train_size=min_train_size,
        n_splits=n_splits,
        step=step,
        horizon=horizon,
        embargo=embargo,
    )
    labels = index if index is not None else pd.RangeIndex(int(n_samples))
    if len(labels) != int(n_samples):
        raise ValueError("index length must equal n_samples")
    rows = []
    for i, (train_idx, val_idx) in enumerate(splits):
        rows.append({
            "split": i,
            "n_train": int(len(train_idx)),
            "n_validation": int(len(val_idx)),
            "train_start": labels[int(train_idx[0])],
            "train_end": labels[int(train_idx[-1])],
            "validation_start": labels[int(val_idx[0])],
            "validation_end": labels[int(val_idx[-1])],
            "train_start_pos": int(train_idx[0]),
            "train_end_pos": int(train_idx[-1]),
            "validation_start_pos": int(val_idx[0]),
            "validation_end_pos": int(val_idx[-1]),
        })
    return pd.DataFrame(rows)


__all__ = [
    "Split",
    "WindowSpec",
    "blocked_kfold",
    "blocked_kfold_split",
    "expanding",
    "expanding_split",
    "last_block",
    "last_block_split",
    "make_splitter",
    "normalize_window_name",
    "poos",
    "poos_split",
    "resolve_window",
    "rolling_blocks",
    "rolling_blocks_split",
    "split_table",
]
