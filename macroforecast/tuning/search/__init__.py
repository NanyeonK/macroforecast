from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "grid_search": ".grid",
    "random_search": ".random",
    "bayesian_optimization": ".bayesian",
    "genetic_algorithm": ".genetic",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
