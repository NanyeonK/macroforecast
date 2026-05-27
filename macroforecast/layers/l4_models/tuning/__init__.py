"""Compatibility shim for macroforecast.models.tuning."""

from __future__ import annotations

from macroforecast.models.tuning import *  # noqa: F401,F403
try:
    from macroforecast.models.tuning import __all__ as __all__  # type: ignore[attr-defined]
except ImportError:
    __all__ = [name for name in globals() if not name.startswith("_")]
