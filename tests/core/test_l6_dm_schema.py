"""Cycle 14 L1-5 -- DM/CW result dicts expose decision, alternative, correction_method.

Verifies that every L6 result dict has the three new fields:
  - decision (bool)
  - alternative (str: one_sided / two_sided)
  - correction_method (str: hln_nw / nw)
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Build minimal errors DataFrame for testing
# ---------------------------------------------------------------------------

def _make_errors(n: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    origins = pd.date_range("2010-01-01", periods=n, freq="MS")
    return pd.DataFrame({
        "model_id": ["model_a"] * n + ["model_b"] * n,
        "target": ["y"] * (2 * n),
        "horizon": [1] * (2 * n),
        "origin": list(origins) * 2,
        "squared": rng.uniform(0.01, 1.0, 2 * n),
        "absolute": rng.uniform(0.01, 1.0, 2 * n),
        "forecast": rng.normal(0, 1, 2 * n),
        "actual": rng.normal(0, 1, 2 * n),
    })


def test_dm_result_has_decision_field():
    """DM per-horizon result must have 'decision' key."""
    from macroforecast.core.runtime import _l6_equal_predictive_results
    from unittest.mock import MagicMock

    errors = _make_errors()
    sub = {
        "equal_predictive_test": "dm_diebold_mariano",
        "equal_pair_strategy": "all_vs_benchmark",
        "loss_function": "squared",
        "hln_correction": True,
    }
    leaf = {"dependence_correction": "newey_west"}
    l4_models = MagicMock()
    l4_models.benchmark_flags = {"model_a": True, "model_b": False}

    results = _l6_equal_predictive_results(errors, sub, leaf, l4_models)
    assert results, "No DM results returned"
    for key, val in results.items():
        assert "decision" in val, f"Missing 'decision' in DM result at {key}: {list(val)}"
        assert "alternative" in val, f"Missing 'alternative' in DM result at {key}: {list(val)}"
        assert "correction_method" in val, f"Missing 'correction_method' in DM result at {key}: {list(val)}"


def test_cw_result_has_decision_field():
    """CW (nested) result must have 'decision', 'alternative', 'correction_method'."""
    from macroforecast.core.runtime import _l6_nested_results
    from unittest.mock import MagicMock

    errors = _make_errors()
    sub = {
        "nested_test": "clark_west",
        "nested_pair_strategy": "all_pairs",
        "loss_function": "squared",
        "cw_adjustment": True,
    }
    leaf = {"dependence_correction": "newey_west"}
    l4_models = MagicMock()
    l4_models.benchmark_flags = {}

    results = _l6_nested_results(errors, sub, leaf, l4_models)
    assert results, "No CW results returned"
    for key, val in results.items():
        assert "decision" in val, f"Missing 'decision' in CW result at {key}: {list(val)}"
        assert "alternative" in val, f"Missing 'alternative' in CW result at {key}: {list(val)}"
        assert "correction_method" in val, f"Missing 'correction_method' in CW result at {key}: {list(val)}"
        assert val["alternative"] == "one_sided", f"CW should be one_sided, got {val['alternative']}"


def test_dm_backward_compat_decision_at_5pct():
    """Backward compat: decision_at_5pct must still exist alongside decision."""
    from macroforecast.core.runtime import _l6_equal_predictive_results
    from unittest.mock import MagicMock

    errors = _make_errors()
    sub = {
        "equal_predictive_test": "dm_diebold_mariano",
        "equal_pair_strategy": "all_vs_benchmark",
        "loss_function": "squared",
        "hln_correction": False,
    }
    leaf = {"dependence_correction": "newey_west"}
    l4_models = MagicMock()
    l4_models.benchmark_flags = {"model_a": True, "model_b": False}

    results = _l6_equal_predictive_results(errors, sub, leaf, l4_models)
    for key, val in results.items():
        assert "decision_at_5pct" in val, f"decision_at_5pct backward compat missing at {key}"
        # decision_at_5pct now emits DeprecationWarning on __getitem__ — suppress here
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            legacy_val = val["decision_at_5pct"]
        # decision and decision_at_5pct should agree
        assert val["decision"] == legacy_val, \
            f"decision != decision_at_5pct at {key}: {val['decision']} vs {legacy_val}"
