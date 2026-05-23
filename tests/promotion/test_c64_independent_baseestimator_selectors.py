"""C64 independent validation — BaseEstimator contract for 5 feature selectors.

Written by tester from test-spec.md only. Covers test-spec.md Section 4.

Tests verify (7 per selector = 35 total + 4 cross-selector parametrized = 39):
  B-T1: get_params() returns dict with all __init__ param names
  B-T2: set_params() round-trips
  B-T3: feature_names_in_ populated after fit
  B-T4: n_features_in_ populated after fit
  B-T5: __repr__ contains class name
  B-T6: clone() produces independent instance with same params, no fitted state
  B-T7: existing fit/transform/fit_transform behavior preserved

Cross-selector parametrized (4 tests × 5 selectors = 20):
  test_selector_is_baseestimator
  test_selector_get_params_nonempty
  test_selector_feature_tracking_after_fit
  test_selector_clone_clears_fitted_state
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin, clone


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _panel_signal(n: int = 100, p: int = 10, seed: int = 42):
    """Panel with 2 truly informative features (feat_0, feat_2)."""
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.randn(n, p), columns=[f"feat_{i}" for i in range(p)])
    y = pd.Series(
        2.0 * X["feat_0"] - 1.5 * X["feat_2"] + 0.05 * rng.randn(n), name="y"
    )
    return X, y


_SELECTOR_NAMES = [
    "Boruta",
    "RFE",
    "LassoPathSelector",
    "StabilitySelection",
    "GeneticSelection",
]


def _make_fast_selector(name: str):
    """Construct a fast-running instance of each selector for parametrized tests."""
    import macroforecast.feature_selection as fs
    cls = getattr(fs, name)
    if name == "Boruta":
        return cls(n_estimators_rf=10, max_iter=3, random_state=0)
    if name == "RFE":
        return cls(n_features_to_select=3, random_state=0)
    if name == "LassoPathSelector":
        return cls(n_features_to_select=3, random_state=0)
    if name == "StabilitySelection":
        return cls(n_subsamples=10, random_state=0)
    if name == "GeneticSelection":
        return cls(population_size=5, n_generations=3, random_state=0)
    raise ValueError(f"Unknown selector: {name}")


# ---------------------------------------------------------------------------
# Boruta (7 tests + isinstance check)
# ---------------------------------------------------------------------------

class TestBorutaBaseEstimator:
    def test_BT1_get_params(self):
        """B-T1: get_params() returns dict with all Boruta init param names."""
        from macroforecast.feature_selection import Boruta
        sel = Boruta(n_estimators_rf=50, max_iter=20, alpha=0.1)
        params = sel.get_params()
        assert isinstance(params, dict)
        assert "n_estimators_rf" in params
        assert params["n_estimators_rf"] == 50
        assert "max_iter" in params
        assert params["max_iter"] == 20
        assert "alpha" in params
        assert params["alpha"] == pytest.approx(0.1)
        # Additional params that must be present
        assert "include_tentative" in params
        assert "random_state" in params
        assert "n_shadow_copies" in params

    def test_BT2_set_params_roundtrip(self):
        """B-T2: set_params changes are reflected in get_params."""
        from macroforecast.feature_selection import Boruta
        sel = Boruta(alpha=0.05)
        sel.set_params(alpha=0.10, max_iter=50)
        assert sel.get_params()["alpha"] == pytest.approx(0.10)
        assert sel.get_params()["max_iter"] == 50

    def test_BT3_feature_names_in_after_fit(self):
        """B-T3: feature_names_in_ populated after fit with correct column names."""
        from macroforecast.feature_selection import Boruta
        X, y = _panel_signal()
        sel = Boruta(n_estimators_rf=20, max_iter=5, random_state=0)
        sel.fit(X, y)
        assert hasattr(sel, "feature_names_in_")
        assert list(sel.feature_names_in_) == list(X.columns)

    def test_BT4_n_features_in_after_fit(self):
        """B-T4: n_features_in_ populated after fit."""
        from macroforecast.feature_selection import Boruta
        X, y = _panel_signal()
        sel = Boruta(n_estimators_rf=20, max_iter=5, random_state=0)
        sel.fit(X, y)
        assert hasattr(sel, "n_features_in_")
        assert sel.n_features_in_ == X.shape[1]

    def test_BT5_repr_contains_classname(self):
        """B-T5: __repr__ contains 'Boruta' and is not the default object repr."""
        from macroforecast.feature_selection import Boruta
        sel = Boruta()
        r = repr(sel)
        assert "Boruta" in r
        assert "object at 0x" not in r

    def test_BT6_clone_independent(self):
        """B-T6: clone produces independent instance with same params; clears fitted state."""
        from macroforecast.feature_selection import Boruta
        X, y = _panel_signal()
        sel = Boruta(n_estimators_rf=20, max_iter=5, random_state=0)
        sel.fit(X, y)
        sel2 = clone(sel)
        # Same params
        assert sel2.get_params() == sel.get_params()
        # clone clears fitted state
        assert not hasattr(sel2, "selected_features_")
        assert not hasattr(sel2, "feature_names_in_")

    def test_BT7_fit_transform_behavior_preserved(self):
        """B-T7: fit+transform gives same result as fit_transform."""
        from macroforecast.feature_selection import Boruta
        X, y = _panel_signal()
        sel = Boruta(n_estimators_rf=20, max_iter=5, random_state=0)
        sel.fit(X, y)
        X_t1 = sel.transform(X)
        X_ft = sel.fit_transform(X, y)
        assert isinstance(X_t1, pd.DataFrame)
        assert isinstance(X_ft, pd.DataFrame)
        assert set(X_t1.columns).issubset(set(X.columns))
        # Both paths must give identical result
        pd.testing.assert_frame_equal(X_t1, X_ft)

    def test_BT_is_baseestimator(self):
        """Boruta inherits from both BaseEstimator and TransformerMixin."""
        from macroforecast.feature_selection import Boruta
        sel = Boruta()
        assert isinstance(sel, BaseEstimator)
        assert isinstance(sel, TransformerMixin)


# ---------------------------------------------------------------------------
# RFE (7 tests + isinstance check)
# ---------------------------------------------------------------------------

class TestRFEBaseEstimator:
    def test_BT1_get_params(self):
        """B-T1: get_params() returns dict with all RFE init param names."""
        from macroforecast.feature_selection import RFE
        sel = RFE(n_features_to_select=3, step=2)
        params = sel.get_params()
        assert isinstance(params, dict)
        assert "n_features_to_select" in params
        assert params["n_features_to_select"] == 3
        assert "step" in params
        assert params["step"] == 2
        assert "estimator" in params
        assert "random_state" in params

    def test_BT2_set_params_roundtrip(self):
        """B-T2: set_params changes are reflected in get_params."""
        from macroforecast.feature_selection import RFE
        sel = RFE()
        sel.set_params(n_features_to_select=4, estimator="lasso")
        assert sel.get_params()["n_features_to_select"] == 4
        assert sel.get_params()["estimator"] == "lasso"

    def test_BT3_feature_names_in_after_fit(self):
        """B-T3: feature_names_in_ populated after fit."""
        from macroforecast.feature_selection import RFE
        X, y = _panel_signal()
        sel = RFE(n_features_to_select=3, random_state=0)
        sel.fit(X, y)
        assert hasattr(sel, "feature_names_in_")
        assert list(sel.feature_names_in_) == list(X.columns)

    def test_BT4_n_features_in_after_fit(self):
        """B-T4: n_features_in_ populated after fit."""
        from macroforecast.feature_selection import RFE
        X, y = _panel_signal()
        sel = RFE(random_state=0)
        sel.fit(X, y)
        assert sel.n_features_in_ == X.shape[1]

    def test_BT5_repr_contains_classname(self):
        """B-T5: __repr__ contains 'RFE'."""
        from macroforecast.feature_selection import RFE
        assert "RFE" in repr(RFE())

    def test_BT6_clone_independent(self):
        """B-T6: clone produces independent instance; clears fitted state."""
        from macroforecast.feature_selection import RFE
        X, y = _panel_signal()
        sel = RFE(n_features_to_select=3, random_state=0)
        sel.fit(X, y)
        sel2 = clone(sel)
        assert sel2.get_params() == sel.get_params()
        assert not hasattr(sel2, "selected_features_")
        assert not hasattr(sel2, "feature_names_in_")

    def test_BT7_fit_transform_preserved(self):
        """B-T7: fit_transform returns DataFrame with subset of original columns."""
        from macroforecast.feature_selection import RFE
        X, y = _panel_signal()
        sel = RFE(n_features_to_select=3, random_state=0)
        X_ft = sel.fit_transform(X, y)
        assert isinstance(X_ft, pd.DataFrame)
        assert X_ft.shape[1] <= X.shape[1]
        assert set(X_ft.columns).issubset(set(X.columns))

    def test_BT_is_baseestimator(self):
        """RFE inherits from both BaseEstimator and TransformerMixin."""
        from macroforecast.feature_selection import RFE
        sel = RFE()
        assert isinstance(sel, BaseEstimator)
        assert isinstance(sel, TransformerMixin)


# ---------------------------------------------------------------------------
# LassoPathSelector (7 tests + isinstance check)
# ---------------------------------------------------------------------------

class TestLassoPathSelectorBaseEstimator:
    def test_BT1_get_params(self):
        """B-T1: get_params() returns dict with LassoPathSelector init param names."""
        from macroforecast.feature_selection import LassoPathSelector
        sel = LassoPathSelector(n_features_to_select=4, normalize_features=False)
        params = sel.get_params()
        assert isinstance(params, dict)
        assert "n_features_to_select" in params
        assert params["n_features_to_select"] == 4
        assert "normalize_features" in params
        assert params["normalize_features"] is False
        assert "random_state" in params

    def test_BT2_set_params_roundtrip(self):
        """B-T2: set_params changes are reflected in get_params."""
        from macroforecast.feature_selection import LassoPathSelector
        sel = LassoPathSelector()
        sel.set_params(n_features_to_select=5, normalize_features=False)
        assert sel.get_params()["n_features_to_select"] == 5
        assert sel.get_params()["normalize_features"] is False

    def test_BT3_feature_names_in_after_fit(self):
        """B-T3: feature_names_in_ populated after fit."""
        from macroforecast.feature_selection import LassoPathSelector
        X, y = _panel_signal()
        sel = LassoPathSelector(n_features_to_select=3, random_state=0)
        sel.fit(X, y)
        assert hasattr(sel, "feature_names_in_")
        assert list(sel.feature_names_in_) == list(X.columns)

    def test_BT4_n_features_in_after_fit(self):
        """B-T4: n_features_in_ populated after fit."""
        from macroforecast.feature_selection import LassoPathSelector
        X, y = _panel_signal()
        sel = LassoPathSelector(n_features_to_select=3, random_state=0)
        sel.fit(X, y)
        assert sel.n_features_in_ == X.shape[1]

    def test_BT5_repr_contains_classname(self):
        """B-T5: __repr__ contains 'LassoPathSelector'."""
        from macroforecast.feature_selection import LassoPathSelector
        assert "LassoPathSelector" in repr(LassoPathSelector())

    def test_BT6_clone_independent(self):
        """B-T6: clone produces independent instance; clears fitted state."""
        from macroforecast.feature_selection import LassoPathSelector
        X, y = _panel_signal()
        sel = LassoPathSelector(n_features_to_select=3, random_state=0)
        sel.fit(X, y)
        sel2 = clone(sel)
        assert sel2.get_params() == sel.get_params()
        assert not hasattr(sel2, "selected_features_")
        assert not hasattr(sel2, "feature_names_in_")

    def test_BT7_fit_transform_preserved(self):
        """B-T7: fit_transform returns DataFrame with subset of original columns."""
        from macroforecast.feature_selection import LassoPathSelector
        X, y = _panel_signal()
        sel = LassoPathSelector(n_features_to_select=3, random_state=0)
        X_ft = sel.fit_transform(X, y)
        assert isinstance(X_ft, pd.DataFrame)
        assert X_ft.shape[1] <= X.shape[1]

    def test_BT_is_baseestimator(self):
        """LassoPathSelector inherits from both BaseEstimator and TransformerMixin."""
        from macroforecast.feature_selection import LassoPathSelector
        sel = LassoPathSelector()
        assert isinstance(sel, BaseEstimator)
        assert isinstance(sel, TransformerMixin)


# ---------------------------------------------------------------------------
# StabilitySelection (7 tests + isinstance check)
# ---------------------------------------------------------------------------

class TestStabilitySelectionBaseEstimator:
    def test_BT1_get_params(self):
        """B-T1: get_params() returns dict with StabilitySelection init param names."""
        from macroforecast.feature_selection import StabilitySelection
        sel = StabilitySelection(n_subsamples=50, subsample_fraction=0.6, pi_thr=0.7)
        params = sel.get_params()
        assert isinstance(params, dict)
        assert "n_subsamples" in params
        assert params["n_subsamples"] == 50
        assert "subsample_fraction" in params
        assert params["subsample_fraction"] == pytest.approx(0.6)
        assert "pi_thr" in params
        assert params["pi_thr"] == pytest.approx(0.7)
        assert "base_estimator" in params
        assert "alpha" in params
        assert "random_state" in params

    def test_BT2_set_params_roundtrip(self):
        """B-T2: set_params changes are reflected in get_params."""
        from macroforecast.feature_selection import StabilitySelection
        sel = StabilitySelection()
        sel.set_params(n_subsamples=20, pi_thr=0.8)
        assert sel.get_params()["n_subsamples"] == 20
        assert sel.get_params()["pi_thr"] == pytest.approx(0.8)

    def test_BT3_feature_names_in_after_fit(self):
        """B-T3: feature_names_in_ populated after fit."""
        from macroforecast.feature_selection import StabilitySelection
        X, y = _panel_signal(n=60, p=6)
        sel = StabilitySelection(n_subsamples=10, random_state=0)
        sel.fit(X, y)
        assert hasattr(sel, "feature_names_in_")
        assert list(sel.feature_names_in_) == list(X.columns)

    def test_BT4_n_features_in_after_fit(self):
        """B-T4: n_features_in_ populated after fit."""
        from macroforecast.feature_selection import StabilitySelection
        X, y = _panel_signal(n=60, p=6)
        sel = StabilitySelection(n_subsamples=10, random_state=0)
        sel.fit(X, y)
        assert sel.n_features_in_ == X.shape[1]

    def test_BT5_repr_contains_classname(self):
        """B-T5: __repr__ contains 'StabilitySelection'."""
        from macroforecast.feature_selection import StabilitySelection
        assert "StabilitySelection" in repr(StabilitySelection())

    def test_BT6_clone_independent(self):
        """B-T6: clone produces independent instance; clears fitted state."""
        from macroforecast.feature_selection import StabilitySelection
        X, y = _panel_signal(n=60, p=6)
        sel = StabilitySelection(n_subsamples=10, random_state=0)
        sel.fit(X, y)
        sel2 = clone(sel)
        assert sel2.get_params() == sel.get_params()
        assert not hasattr(sel2, "selected_features_")
        assert not hasattr(sel2, "feature_names_in_")

    def test_BT7_fit_transform_preserved(self):
        """B-T7: fit_transform returns DataFrame with subset of original columns."""
        from macroforecast.feature_selection import StabilitySelection
        X, y = _panel_signal(n=60, p=6)
        sel = StabilitySelection(n_subsamples=10, random_state=0)
        X_ft = sel.fit_transform(X, y)
        assert isinstance(X_ft, pd.DataFrame)
        assert X_ft.shape[1] <= X.shape[1]

    def test_BT_is_baseestimator(self):
        """StabilitySelection inherits from both BaseEstimator and TransformerMixin."""
        from macroforecast.feature_selection import StabilitySelection
        sel = StabilitySelection()
        assert isinstance(sel, BaseEstimator)
        assert isinstance(sel, TransformerMixin)


# ---------------------------------------------------------------------------
# GeneticSelection (7 tests + isinstance check; BT7 marked @pytest.mark.slow)
# ---------------------------------------------------------------------------

class TestGeneticSelectionBaseEstimator:
    def test_BT1_get_params(self):
        """B-T1: get_params() returns dict with GeneticSelection init param names."""
        from macroforecast.feature_selection import GeneticSelection
        sel = GeneticSelection(population_size=20, n_generations=10, crossover_prob=0.9)
        params = sel.get_params()
        assert isinstance(params, dict)
        assert "population_size" in params
        assert params["population_size"] == 20
        assert "n_generations" in params
        assert params["n_generations"] == 10
        assert "crossover_prob" in params
        assert params["crossover_prob"] == pytest.approx(0.9)
        assert "fitness_estimator" in params
        assert "cv_folds" in params
        assert "random_state" in params

    def test_BT2_set_params_roundtrip(self):
        """B-T2: set_params changes are reflected in get_params."""
        from macroforecast.feature_selection import GeneticSelection
        sel = GeneticSelection()
        sel.set_params(population_size=15, fitness_estimator="lasso")
        assert sel.get_params()["population_size"] == 15
        assert sel.get_params()["fitness_estimator"] == "lasso"

    def test_BT3_feature_names_in_after_fit(self):
        """B-T3: feature_names_in_ populated after fit."""
        from macroforecast.feature_selection import GeneticSelection
        X, y = _panel_signal(n=60, p=6)
        sel = GeneticSelection(population_size=5, n_generations=3, random_state=0)
        sel.fit(X, y)
        assert hasattr(sel, "feature_names_in_")
        assert list(sel.feature_names_in_) == list(X.columns)

    def test_BT4_n_features_in_after_fit(self):
        """B-T4: n_features_in_ populated after fit."""
        from macroforecast.feature_selection import GeneticSelection
        X, y = _panel_signal(n=60, p=6)
        sel = GeneticSelection(population_size=5, n_generations=3, random_state=0)
        sel.fit(X, y)
        assert sel.n_features_in_ == X.shape[1]

    def test_BT5_repr_contains_classname(self):
        """B-T5: __repr__ contains 'GeneticSelection'."""
        from macroforecast.feature_selection import GeneticSelection
        assert "GeneticSelection" in repr(GeneticSelection())

    def test_BT6_clone_independent(self):
        """B-T6: clone produces independent instance; clears fitted state."""
        from macroforecast.feature_selection import GeneticSelection
        X, y = _panel_signal(n=60, p=6)
        sel = GeneticSelection(population_size=5, n_generations=3, random_state=0)
        sel.fit(X, y)
        sel2 = clone(sel)
        assert sel2.get_params() == sel.get_params()
        assert not hasattr(sel2, "selected_features_")
        assert not hasattr(sel2, "feature_names_in_")

    @pytest.mark.slow
    def test_BT7_fit_transform_preserved(self):
        """B-T7: fit_transform returns DataFrame (marked slow — GA runs iterative loop)."""
        from macroforecast.feature_selection import GeneticSelection
        X, y = _panel_signal(n=60, p=6)
        sel = GeneticSelection(population_size=5, n_generations=3, random_state=0)
        X_ft = sel.fit_transform(X, y)
        assert isinstance(X_ft, pd.DataFrame)
        assert X_ft.shape[1] <= X.shape[1]

    def test_BT_is_baseestimator(self):
        """GeneticSelection inherits from both BaseEstimator and TransformerMixin."""
        from macroforecast.feature_selection import GeneticSelection
        sel = GeneticSelection()
        assert isinstance(sel, BaseEstimator)
        assert isinstance(sel, TransformerMixin)


# ---------------------------------------------------------------------------
# Cross-selector parametrized tests (4 tests × 5 selectors = 20 tests)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("selector_name", _SELECTOR_NAMES)
def test_selector_is_baseestimator(selector_name: str) -> None:
    """All 5 selectors must be instances of BaseEstimator + TransformerMixin."""
    sel = _make_fast_selector(selector_name)
    assert isinstance(sel, BaseEstimator), (
        f"{selector_name} is not an instance of BaseEstimator"
    )
    assert isinstance(sel, TransformerMixin), (
        f"{selector_name} is not an instance of TransformerMixin"
    )


@pytest.mark.parametrize("selector_name", _SELECTOR_NAMES)
def test_selector_get_params_nonempty(selector_name: str) -> None:
    """All 5 selectors must return non-empty dict from get_params() with random_state."""
    sel = _make_fast_selector(selector_name)
    params = sel.get_params()
    assert isinstance(params, dict)
    assert len(params) > 0
    assert "random_state" in params, (
        f"{selector_name}.get_params() missing 'random_state'"
    )


@pytest.mark.parametrize("selector_name", _SELECTOR_NAMES)
def test_selector_feature_tracking_after_fit(selector_name: str) -> None:
    """All 5 selectors must set feature_names_in_ and n_features_in_ after fit."""
    X, y = _panel_signal(n=60, p=6)
    sel = _make_fast_selector(selector_name)
    sel.fit(X, y)
    assert hasattr(sel, "feature_names_in_"), (
        f"{selector_name} missing feature_names_in_ after fit"
    )
    assert hasattr(sel, "n_features_in_"), (
        f"{selector_name} missing n_features_in_ after fit"
    )
    assert sel.n_features_in_ == X.shape[1], (
        f"{selector_name}: n_features_in_ {sel.n_features_in_} != {X.shape[1]}"
    )
    assert list(sel.feature_names_in_) == list(X.columns), (
        f"{selector_name}: feature_names_in_ mismatch"
    )


@pytest.mark.parametrize("selector_name", _SELECTOR_NAMES)
def test_selector_clone_clears_fitted_state(selector_name: str) -> None:
    """clone() must clear both feature_names_in_ and selected_features_ for all selectors."""
    X, y = _panel_signal(n=60, p=6)
    sel = _make_fast_selector(selector_name)
    sel.fit(X, y)
    sel2 = clone(sel)
    assert not hasattr(sel2, "feature_names_in_"), (
        f"clone of {selector_name} still has feature_names_in_"
    )
    assert not hasattr(sel2, "selected_features_"), (
        f"clone of {selector_name} still has selected_features_"
    )
