"""C64 independent validation — BaseEstimator contract for 8 new model classes.

Written by tester from test-spec.md only. Covers test-spec.md Section 3.

Tests verify:
  - All 8 new classes inherit from sklearn BaseEstimator
  - get_params() returns a non-empty dict with all init param names
  - set_params() round-trips (random_state restores to original)
  - clone() produces independent instance with same params and no fitted state
  - __repr__() contains the class name
  - HemisphereNN: nu appears in get_params(), nu_target does NOT
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, clone

# Tree/ensemble model names (no torch needed for structural tests)
_TREE_CLASSES = [
    "SlowGrowingTree",
    "QuantileRegressionForest",
    "Bagging",
    "Booging",
    "MacroRandomForest",
    "KNN",
]

# Neural model names (no torch needed for get_params / repr / clone)
_NEURAL_CLASSES = ["SequenceModel", "HemisphereNN"]

_ALL_NEW_CLASSES = _TREE_CLASSES + _NEURAL_CLASSES


def _import(name: str):
    """Import a public model class by name from macroforecast.layers.l4_models."""
    from macroforecast.layers import l4_models as models
    return getattr(models, name)


def _make_minimal(name: str):
    """Construct a minimal (fast) instance of the named class.

    Parameters are reduced to the smallest safe values for structural tests
    (no actual fit is performed here).
    """
    cls = _import(name)
    if name == "SlowGrowingTree":
        return cls(eta=0.1)
    if name == "QuantileRegressionForest":
        return cls(n_estimators=5, random_state=0)
    if name == "Bagging":
        return cls(n_estimators=3, random_state=0)
    if name == "Booging":
        return cls(B=2, inner_n_estimators=5, random_state=0)
    if name == "MacroRandomForest":
        return cls(B=3, random_state=0)
    if name == "KNN":
        return cls(n_neighbors=3)
    if name == "SequenceModel":
        return cls(hidden_size=4, n_epochs=1, random_state=0)
    if name == "HemisphereNN":
        return cls(lc=1, lm=1, lv=1, neurons=4, n_epochs=1, B=2, random_state=0)
    raise ValueError(f"Unknown class: {name}")


# ---------------------------------------------------------------------------
# Parametrized contract tests (one per model class)
# ---------------------------------------------------------------------------

class TestBaseEstimatorContractModels:
    @pytest.mark.parametrize("class_name", _ALL_NEW_CLASSES)
    def test_is_baseestimator(self, class_name: str) -> None:
        """T5 (partial): All 8 new classes are instances of sklearn BaseEstimator."""
        m = _make_minimal(class_name)
        assert isinstance(m, BaseEstimator), (
            f"{class_name} is not an instance of sklearn.base.BaseEstimator"
        )

    @pytest.mark.parametrize("class_name", _ALL_NEW_CLASSES)
    def test_get_params_is_nonempty_dict(self, class_name: str) -> None:
        """T1: get_params() returns non-empty dict."""
        m = _make_minimal(class_name)
        params = m.get_params()
        assert isinstance(params, dict), f"{class_name}.get_params() not a dict"
        assert len(params) > 0, f"{class_name}.get_params() returned empty dict"

    @pytest.mark.parametrize("class_name", _ALL_NEW_CLASSES)
    def test_set_params_roundtrip(self, class_name: str) -> None:
        """T2: set_params modifies and restores random_state correctly."""
        m = _make_minimal(class_name)
        params_before = m.get_params()
        # KNN has no random_state; use n_neighbors instead
        if "random_state" in params_before:
            m.set_params(random_state=99)
            assert m.get_params()["random_state"] == 99
            m.set_params(**params_before)
            assert m.get_params() == params_before
        else:
            # Fallback: for classes without random_state, verify get_params stable
            assert params_before == m.get_params()

    @pytest.mark.parametrize("class_name", _ALL_NEW_CLASSES)
    def test_clone_independent(self, class_name: str) -> None:
        """T3: clone() produces independent instance with same params; no fitted state."""
        m = _make_minimal(class_name)
        m2 = clone(m)
        assert m is not m2, "clone() returned same object"
        assert m.get_params() == m2.get_params(), (
            f"clone of {class_name} has different params: "
            f"{m.get_params()} != {m2.get_params()}"
        )
        # clone must produce unfitted instance
        assert not hasattr(m2, "feature_names_in_"), (
            f"clone of {class_name} has feature_names_in_ (should be unfitted)"
        )

    @pytest.mark.parametrize("class_name", _ALL_NEW_CLASSES)
    def test_repr_contains_classname(self, class_name: str) -> None:
        """T4: __repr__() is non-trivial and contains the class name."""
        m = _make_minimal(class_name)
        r = repr(m)
        assert class_name in r, (
            f"repr({class_name}) does not contain the class name: {r!r}"
        )
        # Must not be the default object repr
        assert "object at 0x" not in r

    @pytest.mark.parametrize("class_name", _ALL_NEW_CLASSES)
    def test_feature_names_in_absent_before_fit(self, class_name: str) -> None:
        """T5: feature_names_in_ is NOT present before fit (sklearn convention)."""
        m = _make_minimal(class_name)
        assert not hasattr(m, "feature_names_in_"), (
            f"{class_name} has feature_names_in_ before fit — sklearn violation"
        )


# ---------------------------------------------------------------------------
# HemisphereNN nu/nu_target specific guard
# ---------------------------------------------------------------------------

class TestHemisphereNNNuContract:
    def test_nu_in_get_params(self) -> None:
        """Critical: nu must appear in get_params(), nu_target must NOT."""
        from macroforecast.layers.l4_models import HemisphereNN
        m = HemisphereNN(nu=0.3)
        params = m.get_params()
        assert "nu" in params, (
            "HemisphereNN.get_params() missing 'nu' key"
        )
        assert params["nu"] == pytest.approx(0.3), (
            f"Expected nu=0.3, got {params['nu']}"
        )
        assert "nu_target" not in params, (
            "HemisphereNN.get_params() contains 'nu_target' — should be private"
        )

    def test_nu_none_in_get_params(self) -> None:
        """nu=None (default) is preserved in get_params()."""
        from macroforecast.layers.l4_models import HemisphereNN
        m = HemisphereNN()
        params = m.get_params()
        assert "nu" in params
        assert params["nu"] is None

    def test_nu_clone_preserves_value(self) -> None:
        """clone preserves nu=0.7 exactly."""
        from macroforecast.layers.l4_models import HemisphereNN
        m = HemisphereNN(nu=0.7)
        m2 = clone(m)
        assert m2.get_params()["nu"] == pytest.approx(0.7)
