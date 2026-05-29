"""Compatibility namespace for pre-semantic layer imports.

New code should import from macroforecast.meta, data, preprocessing, features,
models, evaluation, stat_tests, interpretation, output, or diagnostics.
"""

from __future__ import annotations

__all__ = [
    "l1_data", "l3_features",
    "l4_models", "l5_evaluation", "l6_tests", "l7_interpretation",
    "l8_output", "l3_5_diagnostic",
    "l4_5_diagnostic",
]
