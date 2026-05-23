"""C64 independent validation — tree (6) + neural (2) promotions.

Written by tester from test-spec.md only. Covers:
  Section A: Per new model class (8 × 5 = 40 tests)
  Section C: HemisphereNN specific (3 extra)
  Regression: C63 classes still importable, __all__ count, etc.
  Gap callables: Section 6 of test-spec.md

Tests marked @pytest.mark.deep require torch; skipped in fast CI.
Tests marked @pytest.mark.slow are skipped in fast CI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _panel(n: int = 80, p: int = 6, seed: int = 42):
    """Synthetic signal panel: y = 2*x0 - x2 + 0.1*noise."""
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.randn(n, p), columns=[f"feat_{i}" for i in range(p)])
    y = pd.Series(2.0 * X["feat_0"] - 1.0 * X["feat_2"] + 0.1 * rng.randn(n), name="y")
    return X, y


def _panel_ts(n: int = 120, p: int = 4, seed: int = 7):
    """Longer time-series panel for walk-forward models."""
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.randn(n, p), columns=[f"v{i}" for i in range(p)])
    y = pd.Series(rng.randn(n), name="target")
    return X, y


# ---------------------------------------------------------------------------
# SlowGrowingTree — 5 tests
# ---------------------------------------------------------------------------

class TestSlowGrowingTree:
    def test_ST1_importable(self):
        """T1: Importable from flat and grouped namespaces."""
        from macroforecast.models import SlowGrowingTree
        from macroforecast.models.tree import SlowGrowingTree as SGT2
        assert SlowGrowingTree is SGT2

    def test_ST2_isinstance(self):
        """T2: isinstance check against private runtime class."""
        from macroforecast.models import SlowGrowingTree
        from macroforecast.core.runtime import _SlowGrowingTree
        m = SlowGrowingTree()
        assert isinstance(m, _SlowGrowingTree)

    def test_ST3_smoke_fit_predict(self):
        """T3: fit returns self; predict returns correct shape."""
        from macroforecast.models import SlowGrowingTree
        X, y = _panel()
        m = SlowGrowingTree(eta=0.1)
        result = m.fit(X, y)
        assert result is m
        preds = m.predict(X)
        assert preds.shape == (len(X),)

    def test_ST4_get_params(self):
        """T4: get_params() returns dict containing init param names."""
        from macroforecast.models import SlowGrowingTree
        m = SlowGrowingTree(eta=0.2, random_state=1)
        params = m.get_params()
        assert isinstance(params, dict)
        assert "eta" in params
        assert params["eta"] == 0.2
        assert "random_state" in params
        assert params["random_state"] == 1

    def test_ST5_feature_tracking(self):
        """T5: feature_names_in_ and n_features_in_ set after fit."""
        from macroforecast.models import SlowGrowingTree
        X, y = _panel()
        m = SlowGrowingTree()
        assert not hasattr(m, "feature_names_in_")
        m.fit(X, y)
        assert hasattr(m, "feature_names_in_")
        assert hasattr(m, "n_features_in_")
        assert m.n_features_in_ == X.shape[1]
        assert list(m.feature_names_in_) == list(X.columns)


# ---------------------------------------------------------------------------
# QuantileRegressionForest — 5 tests
# ---------------------------------------------------------------------------

class TestQuantileRegressionForest:
    def test_QRF1_importable(self):
        """T1: Importable from flat and grouped namespaces."""
        from macroforecast.models import QuantileRegressionForest
        from macroforecast.models.tree import QuantileRegressionForest as QRF2
        assert QuantileRegressionForest is QRF2

    def test_QRF2_isinstance(self):
        """T2: isinstance check against private runtime class."""
        from macroforecast.models import QuantileRegressionForest
        from macroforecast.core.runtime import _QuantileRegressionForest
        m = QuantileRegressionForest()
        assert isinstance(m, _QuantileRegressionForest)

    def test_QRF3_smoke_fit_predict(self):
        """T3: fit/predict smoke + predict_quantiles extended interface."""
        from macroforecast.models import QuantileRegressionForest
        X, y = _panel()
        m = QuantileRegressionForest(n_estimators=10, random_state=0)
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(X),)
        # Extended interface: predict_quantiles must return a dict with 0.05 key
        q_dict = m.predict_quantiles(X)
        assert 0.05 in q_dict
        assert q_dict[0.05].shape == (len(X),)

    def test_QRF4_get_params(self):
        """T4: get_params() returns dict with n_estimators key."""
        from macroforecast.models import QuantileRegressionForest
        m = QuantileRegressionForest(n_estimators=50)
        params = m.get_params()
        assert isinstance(params, dict)
        assert "n_estimators" in params
        assert params["n_estimators"] == 50

    def test_QRF5_feature_tracking(self):
        """T5: feature_names_in_ and n_features_in_ set after fit."""
        from macroforecast.models import QuantileRegressionForest
        X, y = _panel()
        m = QuantileRegressionForest(n_estimators=5, random_state=0)
        assert not hasattr(m, "feature_names_in_")
        m.fit(X, y)
        assert m.n_features_in_ == X.shape[1]
        assert list(m.feature_names_in_) == list(X.columns)


# ---------------------------------------------------------------------------
# Bagging — 5 tests
# ---------------------------------------------------------------------------

class TestBagging:
    def test_BA1_importable(self):
        """T1: Importable from flat and grouped namespaces."""
        from macroforecast.models import Bagging
        from macroforecast.models.tree import Bagging as BA2
        assert Bagging is BA2

    def test_BA2_isinstance(self):
        """T2: isinstance check against private runtime class."""
        from macroforecast.models import Bagging
        from macroforecast.core.runtime import _BaggingWrapper
        m = Bagging()
        assert isinstance(m, _BaggingWrapper)

    def test_BA3_smoke_fit_predict(self):
        """T3: fit/predict smoke on synthetic data."""
        from macroforecast.models import Bagging
        X, y = _panel()
        m = Bagging(n_estimators=5, random_state=0)
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(X),)

    def test_BA4_get_params(self):
        """T4: get_params() returns dict with init param names."""
        from macroforecast.models import Bagging
        m = Bagging(base_family="ridge", n_estimators=20)
        params = m.get_params()
        assert isinstance(params, dict)
        assert params["base_family"] == "ridge"
        assert params["n_estimators"] == 20

    def test_BA5_feature_tracking(self):
        """T5: feature_names_in_ and n_features_in_ set after fit."""
        from macroforecast.models import Bagging
        X, y = _panel()
        m = Bagging(n_estimators=3, random_state=0)
        assert not hasattr(m, "feature_names_in_")
        m.fit(X, y)
        assert m.n_features_in_ == X.shape[1]
        assert hasattr(m, "feature_names_in_")


# ---------------------------------------------------------------------------
# Booging — 5 tests (BO3 is @pytest.mark.slow)
# ---------------------------------------------------------------------------

class TestBooging:
    def test_BO1_importable(self):
        """T1: Importable from flat and grouped namespaces."""
        from macroforecast.models import Booging
        from macroforecast.models.tree import Booging as BO2
        assert Booging is BO2

    def test_BO2_isinstance(self):
        """T2: isinstance check against private runtime class."""
        from macroforecast.models import Booging
        from macroforecast.core.runtime import _BoogingWrapper
        m = Booging()
        assert isinstance(m, _BoogingWrapper)

    @pytest.mark.slow
    def test_BO3_smoke_fit_predict(self):
        """T3: fit/predict smoke (marked slow — runs minimal B/inner_n_estimators)."""
        from macroforecast.models import Booging
        X, y = _panel(n=60)
        m = Booging(B=3, inner_n_estimators=10, random_state=0)
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(X),)

    def test_BO4_get_params(self):
        """T4: get_params() returns dict with B and sample_frac."""
        from macroforecast.models import Booging
        m = Booging(B=5, sample_frac=0.6)
        params = m.get_params()
        assert isinstance(params, dict)
        assert params["B"] == 5
        assert params["sample_frac"] == pytest.approx(0.6)

    def test_BO5_feature_tracking(self):
        """T5: feature_names_in_ and n_features_in_ set after fit."""
        from macroforecast.models import Booging
        X, y = _panel(n=60)
        m = Booging(B=2, inner_n_estimators=5, random_state=0)
        assert not hasattr(m, "feature_names_in_")
        m.fit(X, y)
        assert m.n_features_in_ == X.shape[1]


# ---------------------------------------------------------------------------
# MacroRandomForest — 5 tests (MRF3 is @pytest.mark.slow + importorskip)
# ---------------------------------------------------------------------------

class TestMacroRandomForest:
    def test_MRF1_importable(self):
        """T1: Importable from flat and grouped namespaces."""
        from macroforecast.models import MacroRandomForest
        from macroforecast.models.tree import MacroRandomForest as MRF2
        assert MacroRandomForest is MRF2

    def test_MRF2_isinstance(self):
        """T2: isinstance check against private runtime class."""
        from macroforecast.models import MacroRandomForest
        from macroforecast.core.runtime import _MRFExternalWrapper
        m = MacroRandomForest()
        assert isinstance(m, _MRFExternalWrapper)

    @pytest.mark.slow
    def test_MRF3_smoke_fit_predict(self):
        """T3: fit/predict smoke (guarded by importorskip)."""
        pytest.importorskip("macroforecast._vendor.macro_random_forest")
        from macroforecast.models import MacroRandomForest
        X, y = _panel_ts(n=60, p=4)
        m = MacroRandomForest(B=5, random_state=0)
        m.fit(X, y)
        preds = m.predict(X.tail(5))
        assert preds.shape == (5,)

    def test_MRF4_get_params(self):
        """T4: get_params() returns dict with B and ridge_lambda."""
        from macroforecast.models import MacroRandomForest
        m = MacroRandomForest(B=10, ridge_lambda=0.2)
        params = m.get_params()
        assert isinstance(params, dict)
        assert params["B"] == 10
        assert params["ridge_lambda"] == pytest.approx(0.2)

    def test_MRF5_feature_tracking(self):
        """T5: feature_names_in_ NOT set before fit (sklearn convention)."""
        from macroforecast.models import MacroRandomForest
        m = MacroRandomForest()
        assert not hasattr(m, "feature_names_in_")


# ---------------------------------------------------------------------------
# KNN — 5 tests
# ---------------------------------------------------------------------------

class TestKNN:
    def test_KNN1_importable(self):
        """T1: Importable from flat and grouped namespaces."""
        from macroforecast.models import KNN
        from macroforecast.models.tree import KNN as KNN2
        assert KNN is KNN2

    def test_KNN2_isinstance(self):
        """T2: isinstance check against private runtime class."""
        from macroforecast.models import KNN
        from macroforecast.core.runtime import _AutoClipKNN
        m = KNN()
        assert isinstance(m, _AutoClipKNN)

    def test_KNN3_smoke_fit_predict(self):
        """T3: fit/predict smoke on synthetic data."""
        from macroforecast.models import KNN
        X, y = _panel()
        m = KNN(n_neighbors=3)
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(X),)

    def test_KNN4_get_params(self):
        """T4: get_params() returns dict with n_neighbors and weights."""
        from macroforecast.models import KNN
        m = KNN(n_neighbors=7, weights="distance")
        params = m.get_params()
        assert isinstance(params, dict)
        assert params["n_neighbors"] == 7
        assert params["weights"] == "distance"

    def test_KNN5_feature_tracking(self):
        """T5: feature_names_in_ and n_features_in_ set after fit."""
        from macroforecast.models import KNN
        X, y = _panel()
        m = KNN()
        assert not hasattr(m, "feature_names_in_")
        m.fit(X, y)
        assert m.n_features_in_ == X.shape[1]
        assert list(m.feature_names_in_) == list(X.columns)


# ---------------------------------------------------------------------------
# SequenceModel — 5 tests (SM3 and SM5 are @pytest.mark.deep)
# ---------------------------------------------------------------------------

class TestSequenceModel:
    def test_SM1_importable(self):
        """T1: Importable from flat and grouped namespaces."""
        from macroforecast.models import SequenceModel
        from macroforecast.models.neural import SequenceModel as SM2
        assert SequenceModel is SM2

    def test_SM2_isinstance(self):
        """T2: isinstance check against private runtime class."""
        from macroforecast.models import SequenceModel
        from macroforecast.core.runtime import _TorchSequenceModel
        m = SequenceModel()
        assert isinstance(m, _TorchSequenceModel)

    @pytest.mark.deep
    def test_SM3_smoke_fit_predict(self):
        """T3: fit/predict smoke (requires torch)."""
        pytest.importorskip("torch")
        from macroforecast.models import SequenceModel
        X, y = _panel()
        m = SequenceModel(kind="lstm", hidden_size=8, n_epochs=2, random_state=0)
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(X),)

    def test_SM4_get_params(self):
        """T4: get_params() returns dict with kind and hidden_size (no torch needed)."""
        from macroforecast.models import SequenceModel
        m = SequenceModel(kind="gru", hidden_size=16)
        params = m.get_params()
        assert isinstance(params, dict)
        assert params["kind"] == "gru"
        assert params["hidden_size"] == 16

    @pytest.mark.deep
    def test_SM5_feature_tracking(self):
        """T5: feature_names_in_ and n_features_in_ set after fit (requires torch)."""
        pytest.importorskip("torch")
        from macroforecast.models import SequenceModel
        X, y = _panel()
        m = SequenceModel(kind="lstm", hidden_size=4, n_epochs=1, random_state=0)
        m.fit(X, y)
        assert m.n_features_in_ == X.shape[1]
        assert list(m.feature_names_in_) == list(X.columns)


# ---------------------------------------------------------------------------
# HemisphereNN — 5 tests (HNN3 is @pytest.mark.deep)
# ---------------------------------------------------------------------------

class TestHemisphereNN:
    def test_HNN1_importable(self):
        """T1: Importable from flat and grouped namespaces."""
        from macroforecast.models import HemisphereNN
        from macroforecast.models.neural import HemisphereNN as HNN2
        assert HemisphereNN is HNN2

    def test_HNN2_isinstance(self):
        """T2: isinstance check against private runtime class."""
        from macroforecast.models import HemisphereNN
        from macroforecast.core.runtime import _HemisphereNN
        m = HemisphereNN()
        assert isinstance(m, _HemisphereNN)

    @pytest.mark.deep
    def test_HNN3_smoke_fit_predict(self):
        """T3: fit/predict smoke (requires torch)."""
        pytest.importorskip("torch")
        from macroforecast.models import HemisphereNN
        X, y = _panel()
        m = HemisphereNN(lc=1, lm=1, lv=1, neurons=8, n_epochs=2, B=2, random_state=0)
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(X),)

    def test_HNN4_get_params_nu_roundtrip(self):
        """T4: Critical — nu must appear in get_params(), nu_target must NOT."""
        from macroforecast.models import HemisphereNN
        m = HemisphereNN(nu=0.4, B=5)
        params = m.get_params()
        assert isinstance(params, dict)
        assert "nu" in params, "nu must be a get_params key (not nu_target)"
        assert params["nu"] == pytest.approx(0.4)
        assert "B" in params
        assert "nu_target" not in params  # nu_target is a private implementation detail

    def test_HNN5_set_params_roundtrip(self):
        """T5: set_params changes are reflected in get_params (no torch needed)."""
        from macroforecast.models import HemisphereNN
        m = HemisphereNN()
        m.set_params(nu=0.6, lc=3)
        assert m.get_params()["nu"] == pytest.approx(0.6)
        assert m.get_params()["lc"] == 3


# ---------------------------------------------------------------------------
# Section C: HemisphereNN specific (3 extra tests)
# ---------------------------------------------------------------------------

class TestHemisphereNNSpecific:
    def test_H1_nu_get_params(self):
        """H1: HemisphereNN(nu=0.5).get_params()['nu'] == 0.5."""
        from macroforecast.models import HemisphereNN
        m = HemisphereNN(nu=0.5)
        assert m.get_params()["nu"] == pytest.approx(0.5)

    def test_H2_nu_and_nu_target_attrs(self):
        """H2: Both .nu and .nu_target attributes present with correct value."""
        from macroforecast.models import HemisphereNN
        m = HemisphereNN(nu=0.5)
        assert hasattr(m, "nu")
        assert m.nu == pytest.approx(0.5)
        # nu_target is set by _HemisphereNN.__init__; must also equal 0.5
        assert hasattr(m, "nu_target")
        assert m.nu_target == pytest.approx(0.5)

    def test_H3_clone_preserves_nu(self):
        """H3: clone(HemisphereNN(nu=0.5)) produces unfitted clone with nu=0.5."""
        from macroforecast.models import HemisphereNN
        m = HemisphereNN(nu=0.5)
        m2 = clone(m)
        assert m2 is not m
        assert m2.get_params()["nu"] == pytest.approx(0.5)
        # Clone must be unfitted
        assert not hasattr(m2, "feature_names_in_")


# ---------------------------------------------------------------------------
# Regression: C63 classes still importable + count check
# ---------------------------------------------------------------------------

class TestC63Regression:
    def test_all_22_c63_classes_still_importable(self):
        """Regression: all 22 C63 promotions remain importable after C64 changes."""
        from macroforecast.models import (
            MidasAlmon, MidasBeta, MidasStep, UnrestrictedMidas,
            LinearAR, FactorAugmentedAR,
            NonNegRidge, TwoStageRandomWalkRidge,
            ShrinkToTargetRidge, FusedDifferenceRidge,
            PrincipalComponentRegression, FactorAugmentedVAR,
            VAR, GLMBoost,
            BVAR, BVARMinnesota, DFMMixedFrequency,
            GARCH, RealizedGARCH,
            ETS, Theta, HoltWinters,
        )
        # Spot check a few
        assert GARCH is not None
        assert BVARMinnesota is not None
        assert MidasAlmon is not None

    def test_total_models_count_is_30(self):
        """After C64, mf.models.__all__ must have exactly 30 entries."""
        from macroforecast import models
        assert len(models.__all__) == 30, f"Expected 30, got {len(models.__all__)}"

    def test_feature_selection_fit_transform_still_works(self):
        """Regression: C63 feature_selection behavior unchanged by BaseEstimator refactor."""
        from macroforecast.feature_selection import Boruta

        def _panel_signal(n=80, p=6, seed=42):
            rng = np.random.RandomState(seed)
            X = pd.DataFrame(rng.randn(n, p), columns=[f"feat_{i}" for i in range(p)])
            y = pd.Series(2.0 * X["feat_0"] - 1.5 * X["feat_2"] + 0.05 * rng.randn(n), name="y")
            return X, y

        X, y = _panel_signal()
        sel = Boruta(n_estimators_rf=10, max_iter=3, random_state=0)
        X_fit_trans = sel.fit_transform(X, y)
        X_transform = sel.transform(X)
        assert isinstance(X_fit_trans, pd.DataFrame)
        pd.testing.assert_frame_equal(X_fit_trans, X_transform)

    def test_not_fitted_error_still_raised(self):
        """Regression: transform() before fit() still raises NotFittedError."""
        from macroforecast.feature_selection import RFE
        from sklearn.exceptions import NotFittedError

        def _panel_signal(n=100, p=10, seed=42):
            rng = np.random.RandomState(seed)
            X = pd.DataFrame(rng.randn(n, p), columns=[f"feat_{i}" for i in range(p)])
            y = pd.Series(2.0 * X["feat_0"] - 1.5 * X["feat_2"] + 0.05 * rng.randn(n), name="y")
            return X, y

        X, _ = _panel_signal()
        sel = RFE()
        with pytest.raises(NotFittedError):
            sel.transform(X)


# ---------------------------------------------------------------------------
# Gap Callables — Section 6 of test-spec.md
# ---------------------------------------------------------------------------

class TestGapCallables:
    def test_slow_growing_tree_fit_importable(self):
        """slow_growing_tree_fit is importable from mf.functions."""
        from macroforecast.functions import slow_growing_tree_fit
        assert callable(slow_growing_tree_fit)

    def test_slow_growing_tree_fit_smoke(self):
        """slow_growing_tree_fit returns result with .predict() of correct shape."""
        from macroforecast.functions import slow_growing_tree_fit
        X, y = _panel()
        result = slow_growing_tree_fit(X, y, eta=0.1, random_state=0)
        preds = result.predict(X)
        assert preds.shape == (len(X),)

    def test_quantile_regression_forest_fit_importable(self):
        """quantile_regression_forest_fit is importable from mf.functions."""
        from macroforecast.functions import quantile_regression_forest_fit
        assert callable(quantile_regression_forest_fit)

    def test_bagging_fit_importable(self):
        """bagging_fit is importable from mf.functions."""
        from macroforecast.functions import bagging_fit
        assert callable(bagging_fit)

    def test_booging_fit_importable(self):
        """booging_fit is importable from mf.functions."""
        from macroforecast.functions import booging_fit
        assert callable(booging_fit)

    def test_macro_random_forest_fit_importable(self):
        """macro_random_forest_fit is importable from mf.functions."""
        from macroforecast.functions import macro_random_forest_fit
        assert callable(macro_random_forest_fit)

    @pytest.mark.deep
    def test_hemisphere_nn_fit_importable(self):
        """hemisphere_nn_fit is importable from mf.functions (deep mark: no torch required for import)."""
        from macroforecast.functions import hemisphere_nn_fit
        assert callable(hemisphere_nn_fit)
