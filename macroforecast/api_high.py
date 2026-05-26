"""Backward-compatibility shim for macroforecast.api_high.

This module was moved to macroforecast.api.quick in v0.10 Phase 4
restructure. This shim re-exports everything so existing imports continue
to work.
"""
from macroforecast.api.quick import *  # noqa: F401, F403
from macroforecast.api.quick import (  # noqa: F401
    Experiment,
    ForecastResult,
    forecast,
    _set_at,
    _build_default_recipe,
)
