"""Cycle 14 L1-1 -- error messages include layer name and recipe key path.

Verifies that:
1. L1 error for missing target columns includes "[L1/1_data..." prefix.
2. L4 error for missing fit_model node includes "[L4/4_forecasting_model..." prefix.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# L1: target column missing from custom panel
# ---------------------------------------------------------------------------

def test_l1_target_missing_includes_layer_prefix():
    """L1 'target columns missing' error must include [L1/1_data...] prefix."""
    from macroforecast.core.runtime import _validate_targets_present
    import pandas as pd

    frame = pd.DataFrame({"x1": [1.0, 2.0]})
    leaf_config = {"target": "NONEXISTENT", "targets": ["NONEXISTENT"]}
    resolved = {}
    with pytest.raises(ValueError, match=r"\[L1/1_data"):
        _validate_targets_present(frame, leaf_config, resolved)


# ---------------------------------------------------------------------------
# L4: missing fit_model node - verify error message text directly
# ---------------------------------------------------------------------------

def test_l4_missing_fit_model_error_text_includes_layer():
    """The L4 fit_model node error message must include [L4/4_forecasting_model...]."""
    import macroforecast.core.runtime as rt
    import inspect

    src = inspect.getsource(rt.materialize_l4_minimal)
    assert "[L4/4_forecasting_model" in src, (
        "L4 fit_model error message missing [L4/4_forecasting_model...] prefix in source"
    )


def test_l4_error_string_contains_prefix():
    """The actual L4 ValueError for fit_model must have layer prefix."""
    import macroforecast.core.runtime as rt

    # Call with empty nodes list that passes layer validation (no sinks needed at this check)
    # We need to get past validate_layer — provide an empty recipe with minimal structure
    from unittest.mock import MagicMock, patch
    import pandas as pd

    l3 = MagicMock()
    l3.horizon_set = [1]
    l3.X_final.data = pd.DataFrame({"x1": range(5)})
    l3.y_final.metadata.values = {"data": pd.Series(range(5), name="y")}
    l3.y_final.name = "y"

    # Patch validate_layer to return no hard errors
    mock_report = MagicMock()
    mock_report.has_hard_errors = False

    with patch("macroforecast.core.layers.l4.validate_layer", return_value=mock_report):
        recipe = {"4_forecasting_model": {"nodes": []}}
        with pytest.raises(ValueError, match=r"\[L4/4_forecasting_model"):
            rt.materialize_l4_minimal(recipe, l3)
