"""Tests for macroforecast.api._deprecations shims.

Verifies that deprecated parameters emit DeprecationWarning and that
the canonical parameter equivalents work without warnings.
"""
from __future__ import annotations

import warnings

import pytest


# ---------------------------------------------------------------------------
# resolve_model: model_family -> model
# ---------------------------------------------------------------------------

def test_model_family_emits_deprecation_warning():
    from macroforecast.api._deprecations import resolve_model
    with pytest.warns(DeprecationWarning, match="model_family="):
        result = resolve_model(model=None, model_family="ridge", default="ar_p")
    assert result == "ridge"


def test_model_canonical_no_warning():
    from macroforecast.api._deprecations import resolve_model
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        result = resolve_model(model="lasso", model_family=None, default="ar_p")
    assert result == "lasso"


def test_model_default_no_warning():
    from macroforecast.api._deprecations import resolve_model
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        result = resolve_model(model=None, model_family=None, default="ar_p")
    assert result == "ar_p"


def test_model_conflict_raises_value_error():
    from macroforecast.api._deprecations import resolve_model
    with pytest.raises(ValueError, match="not both"):
        resolve_model(model="ridge", model_family="lasso", default="ar_p")


def test_model_family_removal_version_in_warning():
    from macroforecast.api._deprecations import resolve_model
    with pytest.warns(DeprecationWarning, match="v0.10.0"):
        resolve_model(model=None, model_family="ridge", default="ar_p")


# ---------------------------------------------------------------------------
# resolve_models: model_families -> models
# ---------------------------------------------------------------------------

def test_model_families_emits_deprecation_warning():
    from macroforecast.api._deprecations import resolve_models
    with pytest.warns(DeprecationWarning, match="model_families="):
        result = resolve_models(models=None, model_families=["ridge", "lasso"])
    assert result == ["ridge", "lasso"]


def test_models_canonical_no_warning():
    from macroforecast.api._deprecations import resolve_models
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        result = resolve_models(models=["ridge"], model_families=None)
    assert result == ["ridge"]


def test_models_conflict_raises_value_error():
    from macroforecast.api._deprecations import resolve_models
    with pytest.raises(ValueError, match="not both"):
        resolve_models(models=["ridge"], model_families=["lasso"])


# ---------------------------------------------------------------------------
# resolve_benchmark_model: benchmark_family -> benchmark_model
# ---------------------------------------------------------------------------

def test_benchmark_family_emits_deprecation_warning():
    from macroforecast.api._deprecations import resolve_benchmark_model
    with pytest.warns(DeprecationWarning, match="benchmark_family="):
        result = resolve_benchmark_model(
            benchmark_model=None, benchmark_family="ar_p", default="ar_p"
        )
    assert result == "ar_p"


def test_benchmark_model_canonical_no_warning():
    from macroforecast.api._deprecations import resolve_benchmark_model
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        result = resolve_benchmark_model(
            benchmark_model="ridge", benchmark_family=None, default="ar_p"
        )
    assert result == "ridge"


def test_benchmark_family_conflict_raises_value_error():
    from macroforecast.api._deprecations import resolve_benchmark_model
    with pytest.raises(ValueError, match="not both"):
        resolve_benchmark_model(
            benchmark_model="ridge", benchmark_family="lasso", default="ar_p"
        )


# ---------------------------------------------------------------------------
# OPERATIONAL_MODEL_FAMILIES / FUTURE_MODEL_FAMILIES constant shims
# ---------------------------------------------------------------------------

def test_operational_model_families_emits_deprecation_warning():
    """OPERATIONAL_MODEL_FAMILIES access via __getattr__ must emit DeprecationWarning."""
    import macroforecast.layers.l4_models.ops as ops_mod
    with pytest.warns(DeprecationWarning, match="OPERATIONAL_MODEL_FAMILIES"):
        result = ops_mod.OPERATIONAL_MODEL_FAMILIES
    # Value must equal the canonical OPERATIONAL_MODELS tuple.
    from macroforecast.layers.l4_models.ops import OPERATIONAL_MODELS
    assert result == OPERATIONAL_MODELS


def test_future_model_families_emits_deprecation_warning():
    """FUTURE_MODEL_FAMILIES access via __getattr__ must emit DeprecationWarning."""
    import macroforecast.layers.l4_models.ops as ops_mod
    with pytest.warns(DeprecationWarning, match="FUTURE_MODEL_FAMILIES"):
        result = ops_mod.FUTURE_MODEL_FAMILIES
    from macroforecast.layers.l4_models.ops import FUTURE_MODELS
    assert result == FUTURE_MODELS


def test_operational_model_families_removal_version_in_warning():
    import macroforecast.layers.l4_models.ops as ops_mod
    with pytest.warns(DeprecationWarning, match="v0.10.0"):
        _ = ops_mod.OPERATIONAL_MODEL_FAMILIES
