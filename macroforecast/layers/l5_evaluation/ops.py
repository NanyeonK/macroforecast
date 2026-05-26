"""L5 evaluation ops (collocated).

This module re-exports the op registrations from ``macroforecast.core.ops.l5_ops``
so callers can import from the collocated path. The side-effect registrations
happen exactly once via the original module.
"""
from __future__ import annotations

# Trigger op registration side-effects from the original core module.
import macroforecast.core.ops.l5_ops  # noqa: F401

from macroforecast.core.ops.l5_ops import (  # noqa: F401
    l5_collect_inputs,
    metric_compute,
    benchmark_relative,
    aggregate,
    slice_and_decompose,
    rank_and_report,
    blocked_oob_reality_check,
)
