"""C64 promotion tests -- tree (6) + neural (2) public classes.

Tests focus on:
T1. Public import surface: all 8 classes importable from mf.models
T2. isinstance in both directions (public/private)
T3. BaseEstimator contract: get_params, set_params, clone, __repr__
T4. feature_names_in_ and n_features_in_ set after fit
T5. Gap callable smoke: FitResult.predict returns correct shape

Tests marked @pytest.mark.slow are skipped by default CI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin, clone

from macroforecast.core.runtime import (
    _SlowGrowingTree,
    _QuantileRegressionForest,
    _BaggingWrapper,
    _BoogingWrapper,
    _MRFExternalWrapper,
    _AutoClipKNN,
    _TorchSequenceModel,
    _HemisphereNN,
)
from macroforecast.models import (
    SlowGrowingTree,
    QuantileRegressionForest,
    Bagging,
    Booging,
    MacroRandomForest,
    KNN,
    SequenceModel,
    HemisphereNN,
    # Existing C63 classes should still be importable
    MidasAlmon,
    LinearAR,
    BVAR,
    GARCH,
    ETS,
)
from macroforecast.functions import (
    slow_growing_tree_fit,
    SlowGrowingTreeFitResult,
    quantile_regression_forest_fit,
    QuantileRegressionForestFitResult,
    bagging_fit,
    BaggingFitResult,
    hemisphere_nn_fit,
    HemisphereNNFitResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_df() -> tuple[pd.DataFrame, pd.Series]:
    """Small (30, 5) DataFrame and target Series with deterministic seed."""
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.randn(30, 5), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(rng.randn(30), name="y")
    return X, y


# ---------------------------------------------------------------------------
# T1: Public import surface
# ---------------------------------------------------------------------------

class TestT1ImportSurface:
    """All 8 new classes are importable from mf.models."""

    def test_tree_classes_importable(self) -> None:
        """Tree-family public classes importable from macroforecast.models."""
        assert SlowGrowingTree is not None
        assert QuantileRegressionForest is not None
        assert Bagging is not None
        assert Booging is not None
        assert MacroRandomForest is not None
        assert KNN is not None

    def test_neural_classes_importable(self) -> None:
        """Neural public classes importable from macroforecast.models."""
        assert SequenceModel is not None
        assert HemisphereNN is not None

    def test_c63_regression_still_importable(self) -> None:
        """C63 classes are not broken by C64 additions."""
        assert MidasAlmon is not None
        assert LinearAR is not None
        assert BVAR is not None
        assert GARCH is not None
        assert ETS is not None

    def test_all_8_in_models_all(self) -> None:
        """All 8 new classes appear in mf.models.__all__."""
        import macroforecast.models as mods
        for name in [
            "SlowGrowingTree", "QuantileRegressionForest", "Bagging", "Booging",
            "MacroRandomForest", "KNN", "SequenceModel", "HemisphereNN",
        ]:
            assert name in mods.__all__, f"{name} missing from mf.models.__all__"


# ---------------------------------------------------------------------------
# T2: isinstance in both directions
# ---------------------------------------------------------------------------

class TestT2IsInstance:
    """isinstance checks in both directions (public/private)."""

    @pytest.mark.parametrize("pub_cls, priv_cls", [
        (SlowGrowingTree, _SlowGrowingTree),
        (QuantileRegressionForest, _QuantileRegressionForest),
        (Bagging, _BaggingWrapper),
        (Booging, _BoogingWrapper),
        (MacroRandomForest, _MRFExternalWrapper),
        (KNN, _AutoClipKNN),
        (SequenceModel, _TorchSequenceModel),
        (HemisphereNN, _HemisphereNN),
    ])
    def test_isinstance_private(self, pub_cls, priv_cls) -> None:
        """Public instance is also an instance of the private parent class."""
        obj = pub_cls()
        assert isinstance(obj, priv_cls)

    @pytest.mark.parametrize("pub_cls", [
        SlowGrowingTree, QuantileRegressionForest, Bagging, Booging,
        MacroRandomForest, KNN, SequenceModel, HemisphereNN,
    ])
    def test_isinstance_baseestimator(self, pub_cls) -> None:
        """Public instance is an instance of sklearn.base.BaseEstimator."""
        obj = pub_cls()
        assert isinstance(obj, BaseEstimator)

    @pytest.mark.parametrize("pub_cls", [
        SlowGrowingTree, QuantileRegressionForest, Bagging, Booging,
        MacroRandomForest, KNN, SequenceModel, HemisphereNN,
    ])
    def test_isinstance_regressormixin(self, pub_cls) -> None:
        """Public instance is an instance of sklearn.base.RegressorMixin."""
        obj = pub_cls()
        assert isinstance(obj, RegressorMixin)


# ---------------------------------------------------------------------------
# T3: BaseEstimator contract (get_params, set_params, clone, __repr__)
# ---------------------------------------------------------------------------

class TestT3BaseEstimatorContract:
    """BaseEstimator contract tests for 8 new public model classes."""

    @pytest.mark.parametrize("pub_cls, param_name, param_val", [
        (SlowGrowingTree, "eta", 0.2),
        (KNN, "n_neighbors", 7),
        (QuantileRegressionForest, "n_estimators", 50),
        (Bagging, "n_estimators", 20),
        (Booging, "B", 20),
        (MacroRandomForest, "B", 20),
        (SequenceModel, "hidden_size", 16),
        (HemisphereNN, "neurons", 32),
    ])
    def test_get_params_roundtrip(self, pub_cls, param_name, param_val) -> None:
        """get_params() returns the exact values passed to __init__."""
        obj = pub_cls(**{param_name: param_val})
        params = obj.get_params()
        assert param_name in params
        assert params[param_name] == param_val

    @pytest.mark.parametrize("pub_cls, param_name, new_val", [
        (SlowGrowingTree, "eta", 0.3),
        (KNN, "n_neighbors", 9),
        (QuantileRegressionForest, "n_estimators", 30),
        (Bagging, "n_estimators", 30),
        (Booging, "B", 30),
        (MacroRandomForest, "B", 30),
        (SequenceModel, "hidden_size", 64),
        (HemisphereNN, "neurons", 128),
    ])
    def test_set_params(self, pub_cls, param_name, new_val) -> None:
        """set_params() updates the parameter value correctly."""
        obj = pub_cls()
        obj.set_params(**{param_name: new_val})
        assert getattr(obj, param_name) == new_val

    @pytest.mark.parametrize("pub_cls", [
        SlowGrowingTree, KNN, QuantileRegressionForest, Bagging,
        Booging, MacroRandomForest, SequenceModel, HemisphereNN,
    ])
    def test_clone(self, pub_cls) -> None:
        """clone() produces a new unfitted instance with identical params."""
        obj = pub_cls()
        cloned = clone(obj)
        assert type(cloned) is pub_cls
        assert cloned.get_params() == obj.get_params()
        # Cloned instance is unfitted (no feature_names_in_)
        assert not hasattr(cloned, "feature_names_in_")

    @pytest.mark.parametrize("pub_cls", [
        SlowGrowingTree, KNN, QuantileRegressionForest,
        Bagging, Booging, MacroRandomForest, SequenceModel, HemisphereNN,
    ])
    def test_repr(self, pub_cls) -> None:
        """__repr__() returns a string containing the class name."""
        obj = pub_cls()
        r = repr(obj)
        assert pub_cls.__name__ in r

    def test_hemisphere_nn_nu_in_get_params(self) -> None:
        """HemisphereNN.get_params() contains 'nu', not 'nu_target'."""
        hnn = HemisphereNN(nu=0.5)
        params = hnn.get_params()
        assert "nu" in params, "nu must appear in get_params()"
        assert "nu_target" not in params, "nu_target must NOT appear in get_params()"
        assert params["nu"] == 0.5

    def test_hemisphere_nn_nu_target_coexists(self) -> None:
        """HemisphereNN.nu_target is also set (used internally by private class)."""
        hnn = HemisphereNN(nu=0.5)
        assert hasattr(hnn, "nu_target")
        assert hnn.nu_target == 0.5


# ---------------------------------------------------------------------------
# T4: feature_names_in_ and n_features_in_ after fit
# ---------------------------------------------------------------------------

class TestT4FeatureTracking:
    """feature_names_in_ and n_features_in_ are set after fit."""

    def test_slow_growing_tree_feature_tracking(self, tiny_df) -> None:
        X, y = tiny_df
        m = SlowGrowingTree(eta=0.1, min_leaf_size=2)
        m.fit(X, y)
        assert hasattr(m, "feature_names_in_")
        assert hasattr(m, "n_features_in_")
        assert m.n_features_in_ == 5
        assert list(m.feature_names_in_) == list(X.columns)

    def test_knn_feature_tracking(self, tiny_df) -> None:
        X, y = tiny_df
        m = KNN(n_neighbors=3)
        m.fit(X, y)
        assert m.n_features_in_ == 5
        assert list(m.feature_names_in_) == list(X.columns)

    def test_qrf_feature_tracking(self, tiny_df) -> None:
        X, y = tiny_df
        m = QuantileRegressionForest(n_estimators=10)
        m.fit(X, y)
        assert m.n_features_in_ == 5

    def test_predict_output_shape(self, tiny_df) -> None:
        """predict() returns 1-D array of length n_samples."""
        X, y = tiny_df
        for cls in [SlowGrowingTree, KNN]:
            obj = cls()
            obj.fit(X, y)
            preds = obj.predict(X)
            assert preds.shape == (30,), f"{cls.__name__} predict shape wrong"


# ---------------------------------------------------------------------------
# T5: Gap callable smoke tests
# ---------------------------------------------------------------------------

class TestT5GapCallables:
    """Gap callables return FitResult with correct predict() shape."""

    def test_slow_growing_tree_fit(self, tiny_df) -> None:
        X, y = tiny_df
        result = slow_growing_tree_fit(X, y, eta=0.1, min_leaf_size=2)
        assert isinstance(result, SlowGrowingTreeFitResult)
        preds = result.predict(X)
        assert preds.shape == (30,)
        s = result.summary()
        assert isinstance(s, str)
        assert "SlowGrowingTree" in s

    def test_slow_growing_tree_fit_numpy_input(self) -> None:
        """Gap callable accepts numpy arrays as X and y."""
        rng = np.random.RandomState(0)
        X_np = rng.randn(20, 3)
        y_np = rng.randn(20)
        result = slow_growing_tree_fit(X_np, y_np, eta=0.1, min_leaf_size=2)
        preds = result.predict(X_np)
        assert preds.shape == (20,)

    def test_quantile_regression_forest_fit(self, tiny_df) -> None:
        X, y = tiny_df
        result = quantile_regression_forest_fit(X, y, n_estimators=10)
        assert isinstance(result, QuantileRegressionForestFitResult)
        preds = result.predict(X)
        assert preds.shape == (30,)
        # predict_quantiles should return a dict
        qdict = result.predict_quantiles(X)
        assert isinstance(qdict, dict)

    def test_bagging_fit(self, tiny_df) -> None:
        X, y = tiny_df
        result = bagging_fit(X, y, base_family="ridge", n_estimators=5)
        assert isinstance(result, BaggingFitResult)
        preds = result.predict(X)
        assert preds.shape == (30,)

    def test_gap_callable_all_importable(self) -> None:
        """All 6 C64 gap callables are importable from mf.functions.__all__."""
        import macroforecast.functions as funcs
        for name in [
            "slow_growing_tree_fit", "SlowGrowingTreeFitResult",
            "quantile_regression_forest_fit", "QuantileRegressionForestFitResult",
            "bagging_fit", "BaggingFitResult",
            "booging_fit", "BoogingFitResult",
            "macro_random_forest_fit", "MacroRandomForestFitResult",
            "hemisphere_nn_fit", "HemisphereNNFitResult",
        ]:
            assert name in funcs.__all__, f"{name} missing from mf.functions.__all__"
