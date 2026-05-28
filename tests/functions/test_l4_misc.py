"""Tests for Cycle 37 L4 misc family standalone callables.

6 test classes cover the 6 new callables in ``mf.functions``:
``svr_linear_fit``, ``svr_rbf_fit``, ``svr_poly_fit``, ``knn_fit``,
``kernel_ridge_fit``, ``mars_fit``.

Bit-exact assertions compare against ``_build_l4_model`` recipe path
where deterministic (SVR/KNN/KernelRidge are deterministic).
mars_fit requires optional ``pyearth`` dep: skipif guard.

Protocol conformance: ``isinstance(r, FitResultBase)``.

Uses small panels (50x3) for CI speed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.functions import FitResultBase, SVRFitResult
from macroforecast.core.runtime import _build_l4_model


# ---------------------------------------------------------------------------
# Optional dep availability checks
# ---------------------------------------------------------------------------

def _pyearth_available() -> bool:
    try:
        from pyearth import Earth  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Shared deterministic fixtures
# ---------------------------------------------------------------------------

def _make_xy_rng42(n: int = 50, p: int = 3):
    """RNG-42 small panel: X ~ N(0,1), y = X @ [1,2,3] + 0.5*noise."""
    rng = np.random.RandomState(42)
    X = rng.randn(n, p)
    beta = np.arange(1, p + 1, dtype=float)
    y = X @ beta + 0.5 * rng.randn(n)
    return X, y


@pytest.fixture(scope="module")
def xy_rng42():
    return _make_xy_rng42()


def _recipe_predict(family: str, params: dict, X_arr: np.ndarray, y_arr: np.ndarray) -> np.ndarray:
    X = pd.DataFrame(X_arr, columns=[f"x{i}" for i in range(X_arr.shape[1])])
    y = pd.Series(y_arr.ravel(), name="y")
    model = _build_l4_model(family, params)
    if family in ("svr_linear", "svr_rbf", "svr_poly", "kernel_ridge"):
        model.fit(X.values, y.values)
    else:
        model.fit(X, y)
    return np.asarray(model.predict(X.values), dtype=float).ravel()


# ===========================================================================
# TestSVRLinearFit
# ===========================================================================

class TestSVRLinearFit:
    """svr_linear_fit: correctness, predict, summary, protocol, validation."""

    def test_returns_result(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.svr_linear_fit(X, y, C=1.0)
        assert isinstance(r, SVRFitResult)
        assert r.kernel == "linear"
        assert r.C == 1.0
        assert r.n_support_vectors >= 0

    def test_bit_exact_with_recipe(self, xy_rng42):
        X_arr, y_arr = xy_rng42
        r = mf.functions.svr_linear_fit(X_arr, y_arr, C=1.0)
        ref = _recipe_predict("svr_linear", {"C": 1.0}, X_arr, y_arr)
        np.testing.assert_allclose(r.predict(X_arr), ref, rtol=1e-10)

    def test_predict_shape(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.svr_linear_fit(X, y)
        preds = r.predict(X)
        assert preds.shape == (50,)
        assert preds.dtype == float

    def test_predict_accepts_dataframe(self, xy_rng42):
        X_arr, y_arr = xy_rng42
        r = mf.functions.svr_linear_fit(X_arr, y_arr)
        assert r.predict(pd.DataFrame(X_arr)).shape == (50,)

    def test_summary_contains_required_fields(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.svr_linear_fit(X, y)
        s = r.summary()
        assert "SVR" in s
        assert "LINEAR" in s.upper()
        assert "n_support_vectors" in s

    def test_protocol_conformance(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.svr_linear_fit(X, y)
        assert isinstance(r, FitResultBase)

    def test_validation_C(self, xy_rng42):
        X, y = xy_rng42
        with pytest.raises(ValueError, match="C"):
            mf.functions.svr_linear_fit(X, y, C=0.0)

    def test_namespace_wiring(self):
        assert "svr_linear_fit" in mf.functions.__all__
        assert "SVRLinearFitResult" in mf.functions.__all__
        assert "SVRFitResult" in mf.functions.__all__


# ===========================================================================
# TestSVRRBFFit
# ===========================================================================

class TestSVRRBFFit:
    """svr_rbf_fit: correctness, predict, summary, protocol, validation."""

    def test_returns_result(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.svr_rbf_fit(X, y, C=1.0, gamma="scale")
        assert isinstance(r, SVRFitResult)
        assert r.kernel == "rbf"
        assert r.C == 1.0
        assert r.gamma == "scale"

    def test_bit_exact_with_recipe(self, xy_rng42):
        X_arr, y_arr = xy_rng42
        r = mf.functions.svr_rbf_fit(X_arr, y_arr, C=1.0)
        ref = _recipe_predict("svr_rbf", {"C": 1.0}, X_arr, y_arr)
        np.testing.assert_allclose(r.predict(X_arr), ref, rtol=1e-10)

    def test_predict_shape(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.svr_rbf_fit(X, y)
        assert r.predict(X).shape == (50,)

    def test_summary_contains_required_fields(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.svr_rbf_fit(X, y)
        s = r.summary()
        assert "SVR" in s
        assert "RBF" in s
        assert "gamma" in s

    def test_protocol_conformance(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.svr_rbf_fit(X, y)
        assert isinstance(r, FitResultBase)

    def test_validation_C(self, xy_rng42):
        X, y = xy_rng42
        with pytest.raises(ValueError, match="C"):
            mf.functions.svr_rbf_fit(X, y, C=-1.0)

    def test_namespace_wiring(self):
        assert "svr_rbf_fit" in mf.functions.__all__
        assert "SVRRBFFitResult" in mf.functions.__all__


# ===========================================================================
# TestSVRPolyFit
# ===========================================================================

class TestSVRPolyFit:
    """svr_poly_fit: correctness, predict, summary, protocol, validation."""

    def test_returns_result(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.svr_poly_fit(X, y, C=1.0, degree=2)
        assert isinstance(r, SVRFitResult)
        assert r.kernel == "poly"
        assert r.C == 1.0
        assert r.degree == 2

    def test_bit_exact_with_recipe(self, xy_rng42):
        X_arr, y_arr = xy_rng42
        r = mf.functions.svr_poly_fit(X_arr, y_arr, C=1.0, degree=2)
        # Recipe uses default degree=3; set degree explicitly to 2 via set_params post-hoc.
        # Both paths produce the same model since we call set_params in both.
        ref = _recipe_predict("svr_poly", {"C": 1.0}, X_arr, y_arr)
        # With degree=2 (non-default) preds will differ from recipe degree=3 default;
        # confirm same-degree consistency instead.
        r_default = mf.functions.svr_poly_fit(X_arr, y_arr, C=1.0, degree=3)
        np.testing.assert_allclose(r_default.predict(X_arr), ref, rtol=1e-10)

    def test_predict_shape(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.svr_poly_fit(X, y)
        assert r.predict(X).shape == (50,)

    def test_summary_contains_required_fields(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.svr_poly_fit(X, y)
        s = r.summary()
        assert "SVR" in s
        assert "POLY" in s.upper()
        assert "degree" in s

    def test_protocol_conformance(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.svr_poly_fit(X, y)
        assert isinstance(r, FitResultBase)

    def test_validation_C(self, xy_rng42):
        X, y = xy_rng42
        with pytest.raises(ValueError, match="C"):
            mf.functions.svr_poly_fit(X, y, C=0.0)

    def test_validation_degree(self, xy_rng42):
        X, y = xy_rng42
        with pytest.raises(ValueError, match="degree"):
            mf.functions.svr_poly_fit(X, y, degree=0)

    def test_namespace_wiring(self):
        assert "svr_poly_fit" in mf.functions.__all__
        assert "SVRPolyFitResult" in mf.functions.__all__


# ===========================================================================
# TestKNNFit
# ===========================================================================

class TestKNNFit:
    """knn_fit: correctness, predict, summary, protocol, validation."""

    def test_returns_result(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.knn_fit(X, y, n_neighbors=5)
        assert r.n_neighbors == 5
        assert r.n_neighbors_used >= 1
        assert r.n_features_in_ == 3
        assert r.weights in ("uniform", "distance")

    def test_bit_exact_with_recipe(self, xy_rng42):
        X_arr, y_arr = xy_rng42
        r = mf.functions.knn_fit(X_arr, y_arr, n_neighbors=5)
        # Build recipe model (AutoClipKNN)
        X = pd.DataFrame(X_arr, columns=[f"x{i}" for i in range(X_arr.shape[1])])
        y = pd.Series(y_arr.ravel(), name="y")
        model = _build_l4_model("knn", {"n_neighbors": 5})
        model.fit(X, y)
        ref = np.asarray(model.predict(X), dtype=float)
        np.testing.assert_allclose(r.predict(X_arr), ref, rtol=1e-12)

    def test_predict_shape(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.knn_fit(X, y)
        preds = r.predict(X)
        assert preds.shape == (50,)
        assert preds.dtype == float

    def test_predict_accepts_dataframe(self, xy_rng42):
        X_arr, y_arr = xy_rng42
        r = mf.functions.knn_fit(X_arr, y_arr)
        assert r.predict(pd.DataFrame(X_arr)).shape == (50,)

    def test_n_neighbors_clipped(self):
        """Small training set: n_neighbors_used <= len(y)."""
        rng = np.random.RandomState(0)
        X = rng.randn(3, 2)
        y = rng.randn(3)
        r = mf.functions.knn_fit(X, y, n_neighbors=10)
        assert r.n_neighbors_used <= 3

    def test_summary_contains_required_fields(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.knn_fit(X, y)
        s = r.summary()
        assert "KNN" in s
        assert "n_neighbors" in s

    def test_protocol_conformance(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.knn_fit(X, y)
        assert isinstance(r, FitResultBase)

    def test_validation_n_neighbors(self, xy_rng42):
        X, y = xy_rng42
        with pytest.raises(ValueError, match="n_neighbors"):
            mf.functions.knn_fit(X, y, n_neighbors=0)

    def test_namespace_wiring(self):
        assert "knn_fit" in mf.functions.__all__
        assert "KNNFitResult" in mf.functions.__all__


# ===========================================================================
# TestKernelRidgeFit
# ===========================================================================

class TestKernelRidgeFit:
    """kernel_ridge_fit: correctness, predict, summary, protocol, validation."""

    def test_returns_result(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.kernel_ridge_fit(X, y, alpha=1.0, kernel="rbf")
        assert r.alpha == 1.0
        assert r.kernel == "rbf"
        assert r.n_features_in_ == 3

    def test_bit_exact_with_recipe(self, xy_rng42):
        X_arr, y_arr = xy_rng42
        r = mf.functions.kernel_ridge_fit(X_arr, y_arr, alpha=1.0, kernel="rbf")
        ref = _recipe_predict("kernel_ridge", {"alpha": 1.0, "kernel": "rbf"}, X_arr, y_arr)
        np.testing.assert_allclose(r.predict(X_arr), ref, rtol=1e-10)

    def test_predict_shape(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.kernel_ridge_fit(X, y)
        preds = r.predict(X)
        assert preds.shape == (50,)
        assert preds.dtype == float

    def test_predict_accepts_dataframe(self, xy_rng42):
        X_arr, y_arr = xy_rng42
        r = mf.functions.kernel_ridge_fit(X_arr, y_arr)
        assert r.predict(pd.DataFrame(X_arr)).shape == (50,)

    def test_linear_kernel(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.kernel_ridge_fit(X, y, kernel="linear")
        assert r.kernel == "linear"
        assert r.predict(X).shape == (50,)

    def test_summary_contains_required_fields(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.kernel_ridge_fit(X, y)
        s = r.summary()
        assert "Kernel Ridge" in s
        assert "alpha" in s
        assert "kernel" in s

    def test_protocol_conformance(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.kernel_ridge_fit(X, y)
        assert isinstance(r, FitResultBase)

    def test_validation_alpha(self, xy_rng42):
        X, y = xy_rng42
        with pytest.raises(ValueError, match="alpha"):
            mf.functions.kernel_ridge_fit(X, y, alpha=0.0)

    def test_namespace_wiring(self):
        assert "kernel_ridge_fit" in mf.functions.__all__
        assert "KernelRidgeFitResult" in mf.functions.__all__


# ===========================================================================
# TestMARSFit
# ===========================================================================

@pytest.mark.skipif(not _pyearth_available(), reason="pyearth not installed")
class TestMARSFit:
    """mars_fit: correctness (pyearth available), predict, summary, protocol."""

    def test_returns_result(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.mars_fit(X, y)
        assert r.n_features_in_ == 3
        assert r.n_terms >= 0

    def test_bit_exact_with_recipe(self, xy_rng42):
        X_arr, y_arr = xy_rng42
        r = mf.functions.mars_fit(X_arr, y_arr)
        ref = _recipe_predict("mars", {}, X_arr, y_arr)
        np.testing.assert_allclose(r.predict(X_arr), ref, rtol=1e-10)

    def test_predict_shape(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.mars_fit(X, y)
        assert r.predict(X).shape == (50,)

    def test_summary_contains_required_fields(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.mars_fit(X, y)
        s = r.summary()
        assert "MARS" in s
        assert "n_terms" in s

    def test_protocol_conformance(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.mars_fit(X, y)
        assert isinstance(r, FitResultBase)

    def test_namespace_wiring(self):
        assert "mars_fit" in mf.functions.__all__
        assert "MARSFitResult" in mf.functions.__all__
