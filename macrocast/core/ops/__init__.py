from __future__ import annotations

from .registry import OpSpec, Rule, clear_op_registry, get_op, list_ops, register_op

# Import universal ops for side-effect registration.
from . import universal as universal

__all__ = [
    "OpSpec",
    "Rule",
    "clear_op_registry",
    "get_op",
    "list_ops",
    "register_op",
]
