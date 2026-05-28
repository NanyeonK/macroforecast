from __future__ import annotations

from .registry import OpSpec, Rule, clear_op_registry, get_op, list_ops, register_op

# Import universal ops for side-effect registration.
from . import universal as universal
from . import diagnostic_ops as diagnostic_ops
from macroforecast.features import ops as l3_ops
from macroforecast.models import ops as l4_ops
from macroforecast.evaluation import ops as l5_ops
from macroforecast.stat_tests import ops as l6_ops
from macroforecast.interpretation import ops as l7_ops
from macroforecast.output import ops as l8_ops

_LAYER_OP_MODULES = (l3_ops, l4_ops, l5_ops, l6_ops, l7_ops, l8_ops)

__all__ = [
    "OpSpec",
    "Rule",
    "clear_op_registry",
    "get_op",
    "list_ops",
    "register_op",
]
