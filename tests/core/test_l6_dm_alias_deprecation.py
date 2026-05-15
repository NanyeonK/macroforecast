"""Cycle 15 M-3 -- L6 DM/CW result dict: decision_at_5pct alias deprecation.

Verifies that:
  - decision_at_5pct is still accessible via __getitem__ (backward compat)
  - accessing decision_at_5pct emits DeprecationWarning
  - decision_at_5pct value equals decision value
  - decision_at_5pct is NOT in keys() / __iter__ / len()
  - `in` operator (via __contains__) still finds the alias
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Shared fixture: build a _L6ResultWithDeprecatedAlias dict directly
# ---------------------------------------------------------------------------

def _make_alias_dict():
    """Return a fresh _L6ResultWithDeprecatedAlias as returned by DM functions."""
    from macroforecast.core.runtime import _L6ResultWithDeprecatedAlias

    return _L6ResultWithDeprecatedAlias({
        "statistic": 1.23,
        "p_value": 0.03,
        "decision": True,
        "decision_at_5pct": True,  # alias — same value
        "alternative": "two_sided",
        "correction_method": "hln_nw",
        "n_obs": 30,
        "mean_loss_difference": 0.01,
        "hln_correction": True,
    })


# ---------------------------------------------------------------------------
# Unit tests exercising the class directly (fast, no mf.run needed)
# ---------------------------------------------------------------------------

def test_keys_excludes_deprecated_alias():
    """keys() must NOT contain decision_at_5pct."""
    d = _make_alias_dict()
    keys = list(d.keys())
    assert "decision_at_5pct" not in keys, (
        f"deprecated alias leaked into keys(): {keys}"
    )
    assert "decision" in keys, "decision must be present in keys()"


def test_iter_excludes_deprecated_alias():
    """__iter__ must NOT yield decision_at_5pct."""
    d = _make_alias_dict()
    iterated = list(d)
    assert "decision_at_5pct" not in iterated, (
        f"deprecated alias leaked into __iter__: {iterated}"
    )


def test_len_excludes_deprecated_alias():
    """len() must NOT count decision_at_5pct."""
    d = _make_alias_dict()
    # 8 real keys: statistic, p_value, decision, alternative, correction_method,
    # n_obs, mean_loss_difference, hln_correction
    assert len(d) == 8, f"expected 8 keys (excl. alias), got {len(d)}"


def test_getitem_emits_deprecation_warning():
    """d['decision_at_5pct'] must emit DeprecationWarning."""
    d = _make_alias_dict()
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        _ = d["decision_at_5pct"]
    deprec = [w for w in ws if issubclass(w.category, DeprecationWarning)]
    assert deprec, "expected DeprecationWarning when accessing decision_at_5pct"
    assert "decision" in str(deprec[0].message).lower(), (
        "warning message should mention 'decision'"
    )


def test_legacy_value_still_correct():
    """decision_at_5pct must still return the same value as decision."""
    d = _make_alias_dict()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert d["decision_at_5pct"] == d["decision"], (
            "decision_at_5pct value does not match decision"
        )


def test_contains_still_finds_alias():
    """'decision_at_5pct' in d must still return True (backward compat for if-in guard)."""
    d = _make_alias_dict()
    assert "decision_at_5pct" in d, (
        "__contains__ should still find deprecated alias"
    )


def test_items_excludes_deprecated_alias():
    """items() must NOT include decision_at_5pct."""
    d = _make_alias_dict()
    item_keys = [k for k, v in d.items()]
    assert "decision_at_5pct" not in item_keys, (
        f"deprecated alias leaked into items(): {item_keys}"
    )


def test_values_count_excludes_alias():
    """values() must have same count as keys() (no extra entry for alias)."""
    d = _make_alias_dict()
    assert len(list(d.values())) == len(list(d.keys())), (
        "values() and keys() counts differ"
    )


def test_get_emits_deprecation_warning():
    """d.get('decision_at_5pct') must also emit DeprecationWarning."""
    d = _make_alias_dict()
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        _ = d.get("decision_at_5pct")
    deprec = [w for w in ws if issubclass(w.category, DeprecationWarning)]
    assert deprec, "expected DeprecationWarning from .get('decision_at_5pct')"


# ---------------------------------------------------------------------------
# Integration test via _l6_equal_predictive_results (no mf.run needed)
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


def test_dm_result_keys_exclude_alias_via_function():
    """_l6_equal_predictive_results results must not include decision_at_5pct in keys()."""
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
        keys = list(val.keys())
        assert "decision_at_5pct" not in keys, (
            f"deprecated alias in keys() at {key}: {keys}"
        )
        assert "decision" in keys, f"'decision' missing from keys() at {key}"


def test_cw_result_keys_exclude_alias_via_function():
    """_l6_nested_results results must not include decision_at_5pct in keys()."""
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
        keys = list(val.keys())
        assert "decision_at_5pct" not in keys, (
            f"deprecated alias in keys() at {key}: {keys}"
        )
        assert "decision" in keys, f"'decision' missing from keys() at {key}"
