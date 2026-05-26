"""Base protocol for standalone function-op result objects.

All fit-result classes in ``mf.functions`` must structurally conform to
:class:`FitResultBase`.  Conformance is structural (duck-typing via
``typing.Protocol``) -- result dataclasses do NOT inherit from this class.

Protocol requirements:
- ``.summary() -> str``: returns a human-readable text summary.
- ``.predict(X) -> np.ndarray``: returns predictions for new input X.

For scalar result objects (L5 metrics returning ``float``) and test-result
objects (L6 returning a dataclass with ``.statistic`` / ``.pvalue``), a
separate protocol hierarchy will be defined in Cycle 28.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class FitResultBase(Protocol):
    """Structural protocol for fit-result objects returned by L4 standalone callables.

    Objects conforming to this protocol must implement ``.summary()`` and
    ``.predict()``.  Use ``isinstance(result, FitResultBase)`` to check
    conformance at runtime (requires ``@runtime_checkable``).

    Note: This is a *protocol*, not a base class.  Result dataclasses
    do NOT inherit from ``FitResultBase``; they conform structurally.
    Frozen dataclasses are incompatible with ABC inheritance.
    """

    def summary(self) -> str:
        """Return a human-readable text summary of the fit result."""
        ...

    def predict(self, X: "np.ndarray | object") -> np.ndarray:
        """Return predictions for new input X."""
        ...


__all__ = ["FitResultBase"]
