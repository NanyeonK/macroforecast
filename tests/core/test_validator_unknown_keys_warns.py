"""Cycle 14 L1-3 -- unknown top-level and leaf_config keys emit UserWarning.

Verifies that:
1. Unknown top-level recipe key (e.g., 99_fake_layer) emits UserWarning.
2. Unknown leaf_config key at a known layer emits UserWarning.
3. Known keys do NOT produce spurious warnings.
"""
from __future__ import annotations

import warnings
import pytest


def test_unknown_top_level_key_warns():
    """An unrecognized top-level recipe key must emit UserWarning."""
    from macroforecast.core.execution import _warn_unknown_recipe_keys
    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter("always")
        _warn_unknown_recipe_keys({"99_fake_layer": {}})
    messages = [str(x.message) for x in w_list if issubclass(x.category, UserWarning)]
    assert any("99_fake_layer" in m for m in messages), f"Expected UserWarning for 99_fake_layer, got: {messages}"


def test_unknown_leaf_config_key_warns():
    """An unrecognized leaf_config key at a known layer must emit UserWarning."""
    from macroforecast.core.execution import _warn_unknown_recipe_keys
    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter("always")
        _warn_unknown_recipe_keys({"1_data": {"leaf_config": {"typo_key": "value"}}})
    messages = [str(x.message) for x in w_list if issubclass(x.category, UserWarning)]
    assert any("typo_key" in m for m in messages), f"Expected UserWarning for typo_key, got: {messages}"


def test_known_keys_no_spurious_warning():
    """Known top-level keys must not produce UserWarning."""
    from macroforecast.core.execution import _warn_unknown_recipe_keys
    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter("always")
        _warn_unknown_recipe_keys({
            "0_meta": {},
            "1_data": {"leaf_config": {"target": "y"}},
            "4_forecasting_model": {},
        })
    uw = [x for x in w_list if issubclass(x.category, UserWarning)]
    assert not uw, f"Unexpected UserWarning for known keys: {[str(x.message) for x in uw]}"
