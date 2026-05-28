"""Compatibility shim for macroforecast.diagnostics.features.schema."""

from __future__ import annotations

from macroforecast.diagnostics.features.schema import *  # noqa: F401,F403
try:
    from macroforecast.diagnostics.features.schema import __all__ as __all__  # type: ignore[attr-defined]
except ImportError:
    __all__ = [name for name in globals() if not name.startswith("_")]
