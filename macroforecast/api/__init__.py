"""macroforecast.api -- consolidated API surface (v0.10 restructure, Phase 4).

Submodules
----------
- ``macroforecast.api.recipe``   -- ``run`` / ``replicate`` / ``run_file``
- ``macroforecast.api.quick``    -- ``forecast`` / ``Experiment`` / ``ForecastResult``
- ``macroforecast.api.custom``   -- user-defined model / preprocessor / feature registration
- ``macroforecast.api.defaults`` -- default profile dict template
- ``macroforecast.api.functions`` -- standalone function-op namespace
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY: tuple[str, ...] = ("recipe", "quick", "custom", "defaults", "functions")


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
