from __future__ import annotations

from typing import Iterator
import numpy as np


class TemporalCVSplitter:
    def split(self, n_samples: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError


class LastBlockSplitter(TemporalCVSplitter):
    def __init__(self, validation_size: int, embargo_gap: int = 0):
        self.validation_size = validation_size
        self.embargo_gap = embargo_gap

    def split(self, n_samples: int):
        val_start = max(1, n_samples - self.validation_size)
        train_end = max(1, val_start - self.embargo_gap)
        yield np.arange(0, train_end), np.arange(val_start, n_samples)


class RollingBlocksSplitter(TemporalCVSplitter):
    def __init__(self, n_blocks: int = 3, block_size: int | None = None, embargo_gap: int = 0):
        self.n_blocks = n_blocks
        self.block_size = block_size
        self.embargo_gap = embargo_gap

    def split(self, n_samples: int):
        block = self.block_size or max(1, n_samples // (self.n_blocks + 1))
        starts = range(max(1, n_samples - self.n_blocks * block), n_samples, block)
        for val_start in starts:
            val_end = min(n_samples, val_start + block)
            train_end = max(1, val_start - self.embargo_gap)
            train_idx = np.arange(0, train_end)
            val_idx = np.arange(val_start, val_end)
            if len(train_idx) and len(val_idx):
                yield train_idx, val_idx


class ExpandingValidationSplitter(TemporalCVSplitter):
    def __init__(self, min_train_size: int, step_size: int = 1, embargo_gap: int = 0):
        self.min_train_size = min_train_size
        self.step_size = step_size
        self.embargo_gap = embargo_gap

    def split(self, n_samples: int):
        for t in range(self.min_train_size, n_samples - self.embargo_gap, self.step_size):
            train_idx = np.arange(0, t)
            val_idx = np.arange(t + self.embargo_gap, min(n_samples, t + self.embargo_gap + 1))
            if len(train_idx) and len(val_idx):
                yield train_idx, val_idx


class BlockedKFoldSplitter(TemporalCVSplitter):
    def __init__(self, n_splits: int = 5, embargo_gap: int = 0):
        self.n_splits = n_splits
        self.embargo_gap = embargo_gap

    def split(self, n_samples: int):
        fold_size = max(1, n_samples // self.n_splits)
        for k in range(self.n_splits):
            val_start = k * fold_size
            val_end = min(n_samples, val_start + fold_size)
            val_idx = np.arange(val_start, val_end)
            train_end = max(0, val_start - self.embargo_gap)
            train_idx = np.arange(0, train_end)
            if len(train_idx) and len(val_idx):
                yield train_idx, val_idx
