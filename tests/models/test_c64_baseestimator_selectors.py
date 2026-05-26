"""C64 tests -- feature_selection BaseEstimator refactor (5 selectors).

Tests verify:
- All 5 selector classes inherit from BaseEstimator + TransformerMixin
- get_params() works correctly (no manual implementation needed)
- feature_names_in_ and n_features_in_ are set during fit()
- Existing behavior preserved: transform(), fit_transform(), selected_features_
- NotFittedError still raised before fit()
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.exceptions import NotFittedError

from macroforecast.layers.l3_features.selection import (
    Boruta,
    RFE,
    LassoPathSelector,
    StabilitySelection,
    GeneticSelection,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def signal_df() -> tuple[pd.DataFrame, pd.Series]:
    """30-sample, 10-feature DataFrame with 2 strong signal features."""
    rng = np.random.RandomState(0)
    X = pd.DataFrame(
        rng.randn(30, 10),
        columns=[f"x{i}" for i in range(10)],
    )
    # y depends only on x0 and x1 with small noise
    y = pd.Series(2 * X["x0"] + 3 * X["x1"] + 0.1 * rng.randn(30), name="y")
    return X, y


# All 5 selector classes parameterized together
ALL_SELECTORS = [Boruta, RFE, LassoPathSelector, StabilitySelection, GeneticSelection]

# Fast-to-fit selectors only (for tests that actually call fit)
FAST_SELECTORS = [RFE, LassoPathSelector, StabilitySelection]


# ---------------------------------------------------------------------------
# B1: Inheritance checks
# ---------------------------------------------------------------------------

class TestB1Inheritance:
    """All 5 selectors inherit from BaseEstimator + TransformerMixin."""

    @pytest.mark.parametrize("cls", ALL_SELECTORS)
    def test_is_baseestimator_subclass(self, cls) -> None:
        assert issubclass(cls, BaseEstimator), f"{cls.__name__} not BaseEstimator"

    @pytest.mark.parametrize("cls", ALL_SELECTORS)
    def test_is_transformermixin_subclass(self, cls) -> None:
        assert issubclass(cls, TransformerMixin), f"{cls.__name__} not TransformerMixin"

    @pytest.mark.parametrize("cls", ALL_SELECTORS)
    def test_instance_is_baseestimator(self, cls) -> None:
        obj = cls()
        assert isinstance(obj, BaseEstimator)

    @pytest.mark.parametrize("cls", ALL_SELECTORS)
    def test_instance_is_transformermixin(self, cls) -> None:
        obj = cls()
        assert isinstance(obj, TransformerMixin)


# ---------------------------------------------------------------------------
# B2: get_params (no manual implementation; BaseEstimator provides it)
# ---------------------------------------------------------------------------

class TestB2GetParams:
    """get_params() works correctly for all 5 selectors."""

    def test_boruta_get_params(self) -> None:
        b = Boruta(n_estimators_rf=50, max_iter=10, alpha=0.01, random_state=7)
        params = b.get_params()
        assert params["n_estimators_rf"] == 50
        assert params["max_iter"] == 10
        assert params["alpha"] == 0.01
        assert params["random_state"] == 7

    def test_rfe_get_params(self) -> None:
        r = RFE(n_features_to_select=3, step=2, estimator="ridge")
        params = r.get_params()
        assert params["n_features_to_select"] == 3
        assert params["step"] == 2
        assert params["estimator"] == "ridge"

    def test_lasso_path_get_params(self) -> None:
        lp = LassoPathSelector(n_features_to_select=4, normalize_features=False)
        params = lp.get_params()
        assert params["n_features_to_select"] == 4
        assert params["normalize_features"] is False

    def test_stability_get_params(self) -> None:
        ss = StabilitySelection(n_subsamples=50, pi_thr=0.7)
        params = ss.get_params()
        assert params["n_subsamples"] == 50
        assert params["pi_thr"] == 0.7

    def test_genetic_get_params(self) -> None:
        gs = GeneticSelection(population_size=20, n_generations=30)
        params = gs.get_params()
        assert params["population_size"] == 20
        assert params["n_generations"] == 30

    @pytest.mark.parametrize("cls", ALL_SELECTORS)
    def test_no_manual_get_params_override(self, cls) -> None:
        """get_params is NOT overridden in the class body (comes from BaseEstimator)."""
        # If the class had a manual get_params, it would be in cls.__dict__
        assert "get_params" not in cls.__dict__, (
            f"{cls.__name__} should not manually define get_params"
        )

    @pytest.mark.parametrize("cls", ALL_SELECTORS)
    def test_no_manual_set_params_override(self, cls) -> None:
        """set_params is NOT overridden (comes from BaseEstimator)."""
        assert "set_params" not in cls.__dict__, (
            f"{cls.__name__} should not manually define set_params"
        )


# ---------------------------------------------------------------------------
# B3: Feature tracking attributes after fit
# ---------------------------------------------------------------------------

class TestB3FeatureTracking:
    """feature_names_in_ and n_features_in_ are set during fit()."""

    @pytest.mark.parametrize("cls", FAST_SELECTORS)
    def test_feature_names_in_set(self, cls, signal_df) -> None:
        X, y = signal_df
        obj = cls()
        obj.fit(X, y)
        assert hasattr(obj, "feature_names_in_"), f"{cls.__name__} missing feature_names_in_"
        assert list(obj.feature_names_in_) == list(X.columns)

    @pytest.mark.parametrize("cls", FAST_SELECTORS)
    def test_n_features_in_set(self, cls, signal_df) -> None:
        X, y = signal_df
        obj = cls()
        obj.fit(X, y)
        assert hasattr(obj, "n_features_in_"), f"{cls.__name__} missing n_features_in_"
        assert obj.n_features_in_ == 10

    def test_boruta_feature_tracking(self, signal_df) -> None:
        """Boruta sets feature_names_in_ and n_features_in_ during fit."""
        X, y = signal_df
        b = Boruta(n_estimators_rf=10, max_iter=5, random_state=0)
        b.fit(X, y)
        assert b.n_features_in_ == 10
        assert list(b.feature_names_in_) == list(X.columns)


# ---------------------------------------------------------------------------
# B4: Existing behavior preserved
# ---------------------------------------------------------------------------

class TestB4ExistingBehavior:
    """Existing transform, fit_transform, selected_features_ behavior is unchanged."""

    def test_rfe_transform_returns_subset(self, signal_df) -> None:
        X, y = signal_df
        r = RFE(n_features_to_select=3)
        r.fit(X, y)
        X_sel = r.transform(X)
        assert isinstance(X_sel, pd.DataFrame)
        assert X_sel.shape[1] == len(r.selected_features_)

    def test_rfe_fit_transform(self, signal_df) -> None:
        X, y = signal_df
        r = RFE(n_features_to_select=3)
        X_sel = r.fit_transform(X, y)
        assert isinstance(X_sel, pd.DataFrame)
        assert X_sel.shape[1] <= 10

    def test_lasso_path_transform(self, signal_df) -> None:
        X, y = signal_df
        lp = LassoPathSelector(n_features_to_select=4)
        lp.fit(X, y)
        X_sel = lp.transform(X)
        assert X_sel.shape[1] == len(lp.selected_features_)

    def test_stability_fit_transform(self, signal_df) -> None:
        X, y = signal_df
        ss = StabilitySelection(n_subsamples=20, random_state=0)
        X_sel = ss.fit_transform(X, y)
        assert isinstance(X_sel, pd.DataFrame)


# ---------------------------------------------------------------------------
# B5: NotFittedError still raised before fit
# ---------------------------------------------------------------------------

class TestB5NotFittedError:
    """Existing NotFittedError guard on transform() is preserved."""

    @pytest.mark.parametrize("cls", ALL_SELECTORS)
    def test_transform_before_fit_raises(self, cls, signal_df) -> None:
        X, _ = signal_df
        obj = cls()
        with pytest.raises(NotFittedError):
            obj.transform(X)


# ---------------------------------------------------------------------------
# B6: clone() works on selectors
# ---------------------------------------------------------------------------

class TestB6Clone:
    """clone() works on unfitted selectors."""

    @pytest.mark.parametrize("cls", ALL_SELECTORS)
    def test_clone_unfitted(self, cls) -> None:
        obj = cls()
        cloned = clone(obj)
        assert type(cloned) is cls
        assert cloned.get_params() == obj.get_params()
