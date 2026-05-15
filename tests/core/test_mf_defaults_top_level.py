"""Regression test for Cycle 14 K-1: mf.defaults accessible at top level.

Verifies that ``import macroforecast as mf; mf.defaults`` does not raise
AttributeError, and that the submodule exposes documented constant values.

Closes: Cycle 14 F7 (P1-7)
"""
from __future__ import annotations

import pytest


def test_mf_defaults_accessible_top_level():
    """mf.defaults must be accessible via the top-level module attribute."""
    import macroforecast as mf

    assert hasattr(mf, "defaults"), (
        "mf.defaults not accessible — expected lazy module export"
    )


def test_mf_defaults_is_module():
    """mf.defaults must be a module, not a plain value."""
    import types
    import macroforecast as mf

    assert isinstance(mf.defaults, types.ModuleType), (
        f"mf.defaults should be a module, got {type(mf.defaults)}"
    )


def test_mf_defaults_has_random_seed():
    """mf.defaults must expose DEFAULT_RANDOM_SEED as documented."""
    import macroforecast as mf

    assert hasattr(mf.defaults, "DEFAULT_RANDOM_SEED"), (
        "mf.defaults.DEFAULT_RANDOM_SEED not found"
    )
    assert isinstance(mf.defaults.DEFAULT_RANDOM_SEED, int), (
        f"DEFAULT_RANDOM_SEED must be int, got {type(mf.defaults.DEFAULT_RANDOM_SEED)}"
    )


def test_mf_defaults_has_default_profile():
    """mf.defaults must expose DEFAULT_PROFILE as documented."""
    import macroforecast as mf

    assert hasattr(mf.defaults, "DEFAULT_PROFILE"), (
        "mf.defaults.DEFAULT_PROFILE not found"
    )
    assert isinstance(mf.defaults.DEFAULT_PROFILE, dict), (
        f"DEFAULT_PROFILE must be dict, got {type(mf.defaults.DEFAULT_PROFILE)}"
    )


def test_mf_defaults_accessible_twice():
    """Repeated access must return the same module object (no re-import side effects)."""
    import macroforecast as mf

    first = mf.defaults
    second = mf.defaults
    assert first is second, "mf.defaults should return the same module object on repeated access"
