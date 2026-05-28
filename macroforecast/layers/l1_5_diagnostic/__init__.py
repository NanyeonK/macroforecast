"""Compatibility shim for macroforecast.diagnostics.data_summary."""

from __future__ import annotations

from macroforecast.diagnostics.data_summary import *  # noqa: F401,F403
try:
    from macroforecast.diagnostics.data_summary import __all__ as __all__  # type: ignore[attr-defined]
except ImportError:
    __all__ = [name for name in globals() if not name.startswith("_")]
