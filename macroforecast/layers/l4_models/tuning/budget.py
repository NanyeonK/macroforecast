"""Compatibility shim for macroforecast.models.tuning.budget."""

from __future__ import annotations

from macroforecast.models.tuning.budget import *  # noqa: F401,F403
try:
    from macroforecast.models.tuning.budget import __all__ as __all__  # type: ignore[attr-defined]
except ImportError:
    __all__ = [name for name in globals() if not name.startswith("_")]
