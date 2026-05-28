"""Compatibility shim for macroforecast.data.sources.fred_qd."""

from __future__ import annotations

from macroforecast.data.sources.fred_qd import *  # noqa: F401,F403
try:
    from macroforecast.data.sources.fred_qd import __all__ as __all__  # type: ignore[attr-defined]
except ImportError:
    __all__ = [name for name in globals() if not name.startswith("_")]
