"""Compatibility shim for macroforecast.data.option_docs."""

from __future__ import annotations

from macroforecast.data.option_docs import *  # noqa: F401,F403
try:
    from macroforecast.data.option_docs import __all__ as __all__  # type: ignore[attr-defined]
except ImportError:
    __all__ = [name for name in globals() if not name.startswith("_")]
