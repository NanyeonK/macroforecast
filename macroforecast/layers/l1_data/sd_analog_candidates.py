"""Compatibility shim for macroforecast.data.sd_analog_candidates."""

from __future__ import annotations

from macroforecast.data.sd_analog_candidates import *  # noqa: F401,F403
try:
    from macroforecast.data.sd_analog_candidates import __all__ as __all__  # type: ignore[attr-defined]
except ImportError:
    __all__ = [name for name in globals() if not name.startswith("_")]
