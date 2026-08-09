"""Typed context for one forecast origin."""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class OriginContext(Mapping):
    """One origin's absolute-position slices and cadence flags.

    Behaves as the mapping ``iter_origins`` has always yielded -- ``ctx["fit_idx"]``,
    ``"row" in ctx``, ``dict(ctx)`` all work -- while also exposing typed attributes.

    The distinction the attributes exist to protect is ``fit_idx`` vs
    ``estimation_idx``. They are the same rows only when ``retrain_every == 1``; with a
    retrain cadence the fit window is frozen at the last retrain point while the
    estimation window keeps growing. Reading the wrong one is a silent, plausible
    mistake, and a ``Mapping[str, Any]`` cannot catch it.
    """

    row: Mapping[str, Any]
    estimation_idx: np.ndarray
    fit_idx: np.ndarray
    test_idx: np.ndarray
    val_splits: list = field(default_factory=list)

    # -- typed accessors over the plan row -------------------------------------
    @property
    def origin(self) -> Any:
        return self.row["origin"]

    @property
    def origin_pos(self) -> int:
        return int(self.row["origin_pos"])

    @property
    def horizon(self) -> int:
        return int(self.row["horizon"])

    @property
    def retrain(self) -> bool:
        return bool(self.row["retrain"])

    @property
    def retrain_group(self) -> int:
        return int(self.row["retrain_group"])

    @property
    def retune(self) -> bool:
        return bool(self.row["retune"])

    @property
    def retune_group(self) -> int:
        return int(self.row["retune_group"])

    # -- Mapping compatibility --------------------------------------------------
    #: The keys ``iter_origins`` has always yielded. Kept exactly, so existing
    #: subscript access is unchanged and migration is per-consumer rather than a
    #: single breaking commit.
    _KEYS = ("row", "estimation_idx", "fit_idx", "test_idx", "val_splits")

    def __getitem__(self, key: str) -> Any:
        if key not in self._KEYS:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._KEYS)

    def __len__(self) -> int:
        return len(self._KEYS)


@dataclass(frozen=True)
class VintageContext:
    """Which vintage each part of an origin was resolved against.

    An origin under a retrain cadence has TWO: the vintage its fit sample came from
    (frozen at the last retrain point) and the vintage its features come from (the
    current origin). Conflating them is the design question #450 turns on, and
    ``vintage_boundary_audit`` currently reports one -- which is why the guard at
    ``runner.py`` refuses ``retrain_every > 1`` for vintage-aware runs.
    """

    origin_vintage: Any = None
    fit_vintage: Any = None
    actuals_vintage: Any = None


__all__ = ["OriginContext", "VintageContext"]
