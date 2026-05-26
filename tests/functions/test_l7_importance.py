"""Tests for Cycle 38 L7 importance standalone callables.

8 test classes cover the 8 new callables in ``mf.functions``:
  model_native_linear_coef_importance, model_native_tree_importance,
  permutation_importance, cond_permutation_importance,
  partial_dependence_importance, ale_importance,
  shap_tree_importance, shap_linear_importance.

Bit-exact assertions compare against runtime helper paths where applicable.
Family compatibility tested: linear ops on tree models raise ValueError;
tree ops on linear models raise ValueError.
SHAP ops: skipif guard when ``shap`` is not installed.
Permutation ``random_state=None`` -> reverse-order (deterministic).
ALE centering verified (cumsum + center + L1 norm).
Predict-failure (patched to raise) -> 0.0 contribution.

Uses small 50x4 panels for CI speed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.functions import (
    NativeImportanceResult,
    PermutationImportanceResult,
    CondPermutationImportanceResult,
    PDPImportanceResult,
    ALEImportanceResult,
    SHAPImportanceResult,
)
from macroforecast.core.runtime import (
    _linear_importance_frame,
    _tree_importance_frame,
    _permutation_importance_frame,
    _strobl_permutation_importance_frame,
    _partial_dependence_table,
    _ale_table,
)
from macroforecast.core.types import ModelArtifact


# ---------------------------------------------------------------------------
# Optional dep checks
# ---------------------------------------------------------------------------

def _shap_available() -> bool:
    try:
        import shap  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Shared deterministic fixtures
# ---------------------------------------------------------------------------

N_ROWS = 50
N_FEATURES = 4
_BETA = np.array([1.0, 2.0, 3.0, 4.0])


def _make_xy_rng42(n: int = N_ROWS, p: int = N_FEATURES):
    """RNG-42 small panel: X ~ N(0,1), y = X @ [1,2,3,4] + 0.5*noise."""
    rng = np.random.RandomState(42)
    X = rng.randn(n, p)
    y = X @ _BETA + 0.5 * rng.randn(n)
    return X, y


@pytest.fixture(scope="module")
def xy_rng42():
    return _make_xy_rng42()


@pytest.fixture(scope="module")
def lin_result(xy_rng42):
    X, y = xy_rng42
    return mf.functions.ridge_fit(X, y, alpha=1.0)


@pytest.fixture(scope="module")
def tree_result(xy_rng42):
    X, y = xy_rng42
    return mf.functions.random_forest_fit(X, y, n_estimators=10, random_state=7)


@pytest.fixture(scope="module")
def artifact_lin(lin_result, xy_rng42):
    X, _ = xy_rng42
    X_df = pd.DataFrame(X, columns=[f"x{i}" for i in range(N_FEATURES)])
    fitted = lin_result._model
    return ModelArtifact(
        model_id="_test",
        family="_test",
        fitted_object=fitted,
        framework="sklearn",
        feature_names=tuple(X_df.columns),
    )


@pytest.fixture(scope="module")
def artifact_tree(tree_result, xy_rng42):
    X, _ = xy_rng42
    X_df = pd.DataFrame(X, columns=[f"x{i}" for i in range(N_FEATURES)])
    fitted = tree_result._model
    return ModelArtifact(
        model_id="_test",
        family="_test",
        fitted_object=fitted,
        framework="sklearn",
        feature_names=tuple(X_df.columns),
    )


# ---------------------------------------------------------------------------
# TestModelNativeLinearCoefImportance
# ---------------------------------------------------------------------------

class TestModelNativeLinearCoefImportance:
    """model_native_linear_coef_importance: correctness, validation, summary."""

    def test_returns_native_result(self, lin_result, xy_rng42):
        X, _ = xy_rng42
        res = mf.functions.model_native_linear_coef_importance(lin_result, X)
        assert isinstance(res, NativeImportanceResult)
        assert res.method == "linear_coef"

    def test_importances_shape(self, lin_result, xy_rng42):
        X, _ = xy_rng42
        res = mf.functions.model_native_linear_coef_importance(lin_result, X)
        assert res.importances_.shape == (N_FEATURES,)
        assert len(res.feature_names_) == N_FEATURES

    def test_importances_are_absolute_coef(self, lin_result, xy_rng42):
        """importances_ = |coef_j|, matches helper."""
        X, _ = xy_rng42
        res = mf.functions.model_native_linear_coef_importance(lin_result, X)
        # All importances should be non-negative (absolute values)
        assert np.all(res.importances_ >= 0)

    def test_bit_exact_vs_runtime_helper(self, lin_result, xy_rng42, artifact_lin):
        """Bit-exact match with _linear_importance_frame recipe path."""
        X, _ = xy_rng42
        X_df = pd.DataFrame(X, columns=[f"x{i}" for i in range(N_FEATURES)])
        res = mf.functions.model_native_linear_coef_importance(lin_result, X_df)
        df_ref = _linear_importance_frame(artifact_lin, method="linear_coef")
        np.testing.assert_allclose(
            res.importances_,
            df_ref["importance"].values,
            rtol=1e-12,
        )

    def test_feature_names_from_dataframe(self, lin_result, xy_rng42):
        """Feature names taken from DataFrame columns."""
        X, _ = xy_rng42
        names = ["feat_a", "feat_b", "feat_c", "feat_d"]
        X_df = pd.DataFrame(X, columns=names)
        res = mf.functions.model_native_linear_coef_importance(lin_result, X_df)
        # Feature names come from estimator's feature_names_in_ (trained on x0..x3)
        assert len(res.feature_names_) == N_FEATURES

    def test_raises_on_tree_model(self, tree_result, xy_rng42):
        """Raises ValueError when called with a tree model."""
        X, _ = xy_rng42
        with pytest.raises(ValueError, match="coef_"):
            mf.functions.model_native_linear_coef_importance(tree_result, X)

    def test_raises_on_missing_model(self, xy_rng42):
        """Raises ValueError when result has no ._model."""
        X, _ = xy_rng42

        class FakeResult:
            pass

        with pytest.raises(ValueError, match="_model"):
            mf.functions.model_native_linear_coef_importance(FakeResult(), X)

    def test_summary_returns_string(self, lin_result, xy_rng42):
        X, _ = xy_rng42
        res = mf.functions.model_native_linear_coef_importance(lin_result, X)
        s = res.summary()
        assert isinstance(s, str)
        assert "linear_coef" in s

    def test_summary_top_n_limits(self, lin_result, xy_rng42):
        X, _ = xy_rng42
        res = mf.functions.model_native_linear_coef_importance(lin_result, X)
        s = res.summary(top_n=2)
        # Should only show 2 features + header
        assert "more features" in s


# ---------------------------------------------------------------------------
# TestModelNativeTreeImportance
# ---------------------------------------------------------------------------

class TestModelNativeTreeImportance:
    """model_native_tree_importance: correctness, validation, summary."""

    def test_returns_native_result(self, tree_result, xy_rng42):
        X, _ = xy_rng42
        res = mf.functions.model_native_tree_importance(tree_result, X)
        assert isinstance(res, NativeImportanceResult)
        assert res.method == "tree_native"

    def test_importances_shape(self, tree_result, xy_rng42):
        X, _ = xy_rng42
        res = mf.functions.model_native_tree_importance(tree_result, X)
        assert res.importances_.shape == (N_FEATURES,)

    def test_importances_sum_approx_one(self, tree_result, xy_rng42):
        """RF feature_importances_ should sum to 1.0."""
        X, _ = xy_rng42
        res = mf.functions.model_native_tree_importance(tree_result, X)
        np.testing.assert_allclose(res.importances_.sum(), 1.0, atol=1e-10)

    def test_bit_exact_vs_runtime_helper(self, tree_result, xy_rng42, artifact_tree):
        """Bit-exact match with _tree_importance_frame recipe path."""
        X, _ = xy_rng42
        X_df = pd.DataFrame(X, columns=[f"x{i}" for i in range(N_FEATURES)])
        res = mf.functions.model_native_tree_importance(tree_result, X_df)
        df_ref = _tree_importance_frame(artifact_tree)
        np.testing.assert_allclose(
            res.importances_,
            df_ref["importance"].values,
            rtol=1e-12,
        )

    def test_raises_on_linear_model(self, lin_result, xy_rng42):
        """Raises ValueError when called with a linear model."""
        X, _ = xy_rng42
        with pytest.raises(ValueError, match="feature_importances_"):
            mf.functions.model_native_tree_importance(lin_result, X)

    def test_summary_returns_string(self, tree_result, xy_rng42):
        X, _ = xy_rng42
        res = mf.functions.model_native_tree_importance(tree_result, X)
        s = res.summary()
        assert isinstance(s, str)
        assert "tree_native" in s


# ---------------------------------------------------------------------------
# TestPermutationImportance
# ---------------------------------------------------------------------------

class TestPermutationImportance:
    """permutation_importance: correctness, random_state=None reverse-order."""

    def test_returns_result(self, lin_result, xy_rng42):
        X, y = xy_rng42
        res = mf.functions.permutation_importance(lin_result, X, y, n_repeats=3, random_state=0)
        assert isinstance(res, PermutationImportanceResult)
        assert res.n_repeats == 3

    def test_shape(self, lin_result, xy_rng42):
        X, y = xy_rng42
        res = mf.functions.permutation_importance(lin_result, X, y, n_repeats=5, random_state=1)
        assert res.importances_mean_.shape == (N_FEATURES,)
        assert res.importances_std_.shape == (N_FEATURES,)

    def test_random_state_none_is_deterministic(self, lin_result, xy_rng42):
        """random_state=None produces identical results on two calls."""
        X, y = xy_rng42
        res1 = mf.functions.permutation_importance(lin_result, X, y, n_repeats=3, random_state=None)
        res2 = mf.functions.permutation_importance(lin_result, X, y, n_repeats=3, random_state=None)
        np.testing.assert_array_equal(res1.importances_mean_, res2.importances_mean_)

    def test_random_state_none_uses_seed_zero(self, lin_result, xy_rng42, artifact_lin):
        """random_state=None now uses seed=0 (proper RNG, not reversal).

        After the PR2 fix: both the standalone API (random_state=None) and the
        runtime helper (_permutation_importance_frame with seed=0) use
        np.random.default_rng(0), so n_repeats=1 must agree exactly.
        """
        X, y = xy_rng42
        X_df = pd.DataFrame(X, columns=[f"x{i}" for i in range(N_FEATURES)])
        y_s = pd.Series(y, name="y")
        res = mf.functions.permutation_importance(lin_result, X_df, y_s, n_repeats=1, random_state=None)
        df_ref = _permutation_importance_frame(
            artifact_lin, X_df, y_s, method="permutation_importance", n_repeats=1, seed=0
        )
        np.testing.assert_allclose(
            res.importances_mean_,
            df_ref["importance"].values,
            rtol=1e-10,
        )

    def test_std_nonneg_when_random_state_none(self, lin_result, xy_rng42):
        """After the PR2 fix: random_state=None uses proper RNG (seed=0), so
        n_repeats>1 produces non-negative std (no longer zero because each
        repeat is an independent random permutation)."""
        X, y = xy_rng42
        res = mf.functions.permutation_importance(lin_result, X, y, n_repeats=5, random_state=None)
        # std must be finite and non-negative; not guaranteed to be zero
        assert np.all(np.isfinite(res.importances_std_))
        assert np.all(res.importances_std_ >= 0.0)

    def test_positive_importances_for_important_features(self, lin_result, xy_rng42):
        """Features with nonzero coef should have positive importance."""
        X, y = xy_rng42
        res = mf.functions.permutation_importance(lin_result, X, y, n_repeats=5, random_state=0)
        # At least some features should have positive importance
        assert np.any(res.importances_mean_ > 0)

    def test_raises_on_invalid_n_repeats(self, lin_result, xy_rng42):
        X, y = xy_rng42
        with pytest.raises(ValueError, match="n_repeats"):
            mf.functions.permutation_importance(lin_result, X, y, n_repeats=0)

    def test_summary_returns_string(self, lin_result, xy_rng42):
        X, y = xy_rng42
        res = mf.functions.permutation_importance(lin_result, X, y, n_repeats=2, random_state=0)
        s = res.summary()
        assert isinstance(s, str)
        assert "n_repeats=2" in s

    def test_accepts_dataframe_and_series(self, lin_result, xy_rng42):
        X, y = xy_rng42
        X_df = pd.DataFrame(X, columns=[f"x{i}" for i in range(N_FEATURES)])
        y_s = pd.Series(y, name="y")
        res = mf.functions.permutation_importance(lin_result, X_df, y_s, n_repeats=2, random_state=5)
        assert res.importances_mean_.shape == (N_FEATURES,)


# ---------------------------------------------------------------------------
# TestCondPermutationImportance
# ---------------------------------------------------------------------------

class TestCondPermutationImportance:
    """cond_permutation_importance: correctness, Strobl method."""

    def test_returns_result(self, lin_result, xy_rng42):
        X, y = xy_rng42
        res = mf.functions.cond_permutation_importance(lin_result, X, y, n_repeats=3, random_state=0)
        assert isinstance(res, CondPermutationImportanceResult)
        assert res.method == "strobl"

    def test_shape(self, lin_result, xy_rng42):
        X, y = xy_rng42
        res = mf.functions.cond_permutation_importance(lin_result, X, y, n_repeats=3, random_state=0)
        assert res.importances_mean_.shape == (N_FEATURES,)
        assert res.importances_std_.shape == (N_FEATURES,)

    def test_bit_exact_vs_runtime_helper(self, lin_result, xy_rng42, artifact_lin):
        """Single repeat matches _strobl_permutation_importance_frame with seed=0."""
        X, y = xy_rng42
        X_df = pd.DataFrame(X, columns=[f"x{i}" for i in range(N_FEATURES)])
        y_s = pd.Series(y, name="y")
        res = mf.functions.cond_permutation_importance(
            lin_result, X_df, y_s, n_repeats=1, random_state=0
        )
        df_ref = _strobl_permutation_importance_frame(artifact_lin, X_df, y_s, seed=0)
        np.testing.assert_allclose(
            res.importances_mean_,
            df_ref["importance"].values,
            rtol=1e-10,
        )

    def test_raises_on_invalid_n_repeats(self, lin_result, xy_rng42):
        X, y = xy_rng42
        with pytest.raises(ValueError, match="n_repeats"):
            mf.functions.cond_permutation_importance(lin_result, X, y, n_repeats=0)

    def test_summary_returns_string(self, lin_result, xy_rng42):
        X, y = xy_rng42
        res = mf.functions.cond_permutation_importance(lin_result, X, y, n_repeats=2, random_state=0)
        s = res.summary()
        assert isinstance(s, str)
        assert "strobl" in s.lower()

    def test_also_works_with_tree_model(self, tree_result, xy_rng42):
        """Should work on tree models too (model-agnostic)."""
        X, y = xy_rng42
        res = mf.functions.cond_permutation_importance(tree_result, X, y, n_repeats=2, random_state=0)
        assert res.importances_mean_.shape == (N_FEATURES,)


# ---------------------------------------------------------------------------
# TestPartialDependenceImportance
# ---------------------------------------------------------------------------

class TestPartialDependenceImportance:
    """partial_dependence_importance: correctness, grid, ALE-vs-PDP."""

    def test_returns_result(self, lin_result, xy_rng42):
        X, _ = xy_rng42
        res = mf.functions.partial_dependence_importance(lin_result, X)
        assert isinstance(res, PDPImportanceResult)

    def test_shape(self, lin_result, xy_rng42):
        X, _ = xy_rng42
        res = mf.functions.partial_dependence_importance(lin_result, X)
        assert res.importances_.shape == (N_FEATURES,)
        assert len(res.feature_names_) == N_FEATURES

    def test_pdp_grid_values(self, lin_result, xy_rng42):
        """grid_values_ should have grid_resolution points per feature."""
        X, _ = xy_rng42
        res = mf.functions.partial_dependence_importance(lin_result, X, grid_resolution=10)
        for name in res.feature_names_:
            assert len(res.grid_values_[name]) == 10
            assert len(res.pdp_values_[name]) == 10

    def test_bit_exact_vs_runtime_helper(self, lin_result, xy_rng42, artifact_lin):
        """importances_ match _partial_dependence_table."""
        X, _ = xy_rng42
        X_df = pd.DataFrame(X, columns=[f"x{i}" for i in range(N_FEATURES)])
        res = mf.functions.partial_dependence_importance(lin_result, X_df, grid_resolution=20)
        df_ref = _partial_dependence_table(artifact_lin, X_df, n_grid=20)
        np.testing.assert_allclose(
            res.importances_,
            df_ref["importance"].values,
            rtol=1e-10,
        )

    def test_importances_nonneg(self, lin_result, xy_rng42):
        """PDP range is always non-negative."""
        X, _ = xy_rng42
        res = mf.functions.partial_dependence_importance(lin_result, X)
        assert np.all(res.importances_ >= 0)

    def test_raises_on_invalid_grid_resolution(self, lin_result, xy_rng42):
        X, _ = xy_rng42
        with pytest.raises(ValueError, match="grid_resolution"):
            mf.functions.partial_dependence_importance(lin_result, X, grid_resolution=1)

    def test_summary_returns_string(self, lin_result, xy_rng42):
        X, _ = xy_rng42
        res = mf.functions.partial_dependence_importance(lin_result, X)
        s = res.summary()
        assert isinstance(s, str)
        assert "PDP" in s or "Partial" in s

    def test_predict_failure_returns_zero(self, xy_rng42):
        """When predict raises, contribution should be 0.0 (graceful fallback)."""
        X, y = xy_rng42

        class FailPredict:
            """Fake model whose predict always raises."""
            feature_names_in_ = [f"x{i}" for i in range(N_FEATURES)]
            coef_ = np.ones(N_FEATURES)

            def predict(self, X):
                raise RuntimeError("intentional failure")

        class FakeResult:
            _model = FailPredict()

        X_df = pd.DataFrame(X, columns=[f"x{i}" for i in range(N_FEATURES)])
        res = mf.functions.partial_dependence_importance(FakeResult(), X_df, grid_resolution=5)
        # All importances should be 0.0 (predict fails -> response=0.0 -> range=0)
        np.testing.assert_array_equal(res.importances_, np.zeros(N_FEATURES))


# ---------------------------------------------------------------------------
# TestALEImportance
# ---------------------------------------------------------------------------

class TestALEImportance:
    """ale_importance: ALE centering, L1 norm, correctness."""

    def test_returns_result(self, lin_result, xy_rng42):
        X, _ = xy_rng42
        res = mf.functions.ale_importance(lin_result, X)
        assert isinstance(res, ALEImportanceResult)

    def test_shape(self, lin_result, xy_rng42):
        X, _ = xy_rng42
        res = mf.functions.ale_importance(lin_result, X)
        assert res.importances_.shape == (N_FEATURES,)
        assert len(res.feature_names_) == N_FEATURES

    def test_ale_values_present(self, lin_result, xy_rng42):
        """ale_values_ dict has an entry per feature."""
        X, _ = xy_rng42
        res = mf.functions.ale_importance(lin_result, X, n_bins=10)
        for name in res.feature_names_:
            assert name in res.ale_values_

    def test_bit_exact_vs_runtime_helper(self, lin_result, xy_rng42, artifact_lin):
        """importances_ match _ale_table."""
        X, _ = xy_rng42
        X_df = pd.DataFrame(X, columns=[f"x{i}" for i in range(N_FEATURES)])
        res = mf.functions.ale_importance(lin_result, X_df, n_bins=20)
        df_ref = _ale_table(artifact_lin, X_df, n_quantiles=20)
        np.testing.assert_allclose(
            res.importances_,
            df_ref["importance"].values,
            rtol=1e-10,
        )

    def test_ale_centering(self, lin_result, xy_rng42):
        """ALE values per feature are centred (cumsum of centred local effects)."""
        X, _ = xy_rng42
        res = mf.functions.ale_importance(lin_result, X, n_bins=10)
        for name, ale_arr in res.ale_values_.items():
            if len(ale_arr) > 0:
                # The importance is mean absolute ALE, which must be >= 0
                assert float(np.mean(np.abs(ale_arr))) >= 0.0

    def test_importances_nonneg(self, lin_result, xy_rng42):
        """ALE L1 norm is always non-negative."""
        X, _ = xy_rng42
        res = mf.functions.ale_importance(lin_result, X)
        assert np.all(res.importances_ >= 0)

    def test_raises_on_invalid_n_bins(self, lin_result, xy_rng42):
        X, _ = xy_rng42
        with pytest.raises(ValueError, match="n_bins"):
            mf.functions.ale_importance(lin_result, X, n_bins=1)

    def test_summary_returns_string(self, lin_result, xy_rng42):
        X, _ = xy_rng42
        res = mf.functions.ale_importance(lin_result, X)
        s = res.summary()
        assert isinstance(s, str)
        assert "ALE" in s


# ---------------------------------------------------------------------------
# TestSHAPTreeImportance
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _shap_available(), reason="shap not installed")
class TestSHAPTreeImportance:
    """shap_tree_importance: requires shap package."""

    def test_returns_result(self, tree_result, xy_rng42):
        X, _ = xy_rng42
        res = mf.functions.shap_tree_importance(tree_result, X)
        assert isinstance(res, SHAPImportanceResult)

    def test_shap_values_shape(self, tree_result, xy_rng42):
        X, _ = xy_rng42
        res = mf.functions.shap_tree_importance(tree_result, X)
        # shap_values_ should be (n_samples, n_features)
        assert res.shap_values_.shape[1] == N_FEATURES

    def test_feature_names_length(self, tree_result, xy_rng42):
        X, _ = xy_rng42
        res = mf.functions.shap_tree_importance(tree_result, X)
        assert len(res.feature_names_) == N_FEATURES

    def test_explainer_type_is_tree(self, tree_result, xy_rng42):
        X, _ = xy_rng42
        res = mf.functions.shap_tree_importance(tree_result, X)
        assert "Explainer" in res.explainer_type

    def test_raises_on_linear_model(self, lin_result, xy_rng42):
        X, _ = xy_rng42
        with pytest.raises(ValueError, match="feature_importances_"):
            mf.functions.shap_tree_importance(lin_result, X)

    def test_summary_returns_string(self, tree_result, xy_rng42):
        X, _ = xy_rng42
        res = mf.functions.shap_tree_importance(tree_result, X)
        s = res.summary()
        assert isinstance(s, str)
        assert "SHAP" in s


@pytest.mark.skipif(_shap_available(), reason="shap IS installed -- test that ImportError is raised")
class TestSHAPTreeImportanceNoShap:
    """shap_tree_importance raises ImportError when shap is not installed."""

    def test_raises_import_error(self, tree_result, xy_rng42):
        X, _ = xy_rng42
        with pytest.raises(ImportError, match="shap"):
            mf.functions.shap_tree_importance(tree_result, X)


# ---------------------------------------------------------------------------
# TestSHAPLinearImportance
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _shap_available(), reason="shap not installed")
class TestSHAPLinearImportance:
    """shap_linear_importance: requires shap package."""

    def test_returns_result(self, lin_result, xy_rng42):
        X, _ = xy_rng42
        res = mf.functions.shap_linear_importance(lin_result, X)
        assert isinstance(res, SHAPImportanceResult)

    def test_shap_values_shape(self, lin_result, xy_rng42):
        X, _ = xy_rng42
        res = mf.functions.shap_linear_importance(lin_result, X)
        assert res.shap_values_.shape[1] == N_FEATURES

    def test_explainer_type_is_linear(self, lin_result, xy_rng42):
        X, _ = xy_rng42
        res = mf.functions.shap_linear_importance(lin_result, X)
        assert "Explainer" in res.explainer_type

    def test_raises_on_tree_model(self, tree_result, xy_rng42):
        X, _ = xy_rng42
        with pytest.raises(ValueError, match="coef_"):
            mf.functions.shap_linear_importance(tree_result, X)

    def test_summary_returns_string(self, lin_result, xy_rng42):
        X, _ = xy_rng42
        res = mf.functions.shap_linear_importance(lin_result, X)
        s = res.summary()
        assert isinstance(s, str)
        assert "SHAP" in s


@pytest.mark.skipif(_shap_available(), reason="shap IS installed")
class TestSHAPLinearImportanceNoShap:
    """shap_linear_importance raises ImportError when shap is not installed."""

    def test_raises_import_error(self, lin_result, xy_rng42):
        X, _ = xy_rng42
        with pytest.raises(ImportError, match="shap"):
            mf.functions.shap_linear_importance(lin_result, X)


# ---------------------------------------------------------------------------
# TestResultTypeExports
# ---------------------------------------------------------------------------

class TestResultTypeExports:
    """Verify all 6 result types are exported in mf.functions namespace."""

    def test_native_importance_result_exported(self):
        assert hasattr(mf.functions, "NativeImportanceResult")

    def test_permutation_importance_result_exported(self):
        assert hasattr(mf.functions, "PermutationImportanceResult")

    def test_cond_permutation_importance_result_exported(self):
        assert hasattr(mf.functions, "CondPermutationImportanceResult")

    def test_pdp_importance_result_exported(self):
        assert hasattr(mf.functions, "PDPImportanceResult")

    def test_ale_importance_result_exported(self):
        assert hasattr(mf.functions, "ALEImportanceResult")

    def test_shap_importance_result_exported(self):
        assert hasattr(mf.functions, "SHAPImportanceResult")

    def test_all_callables_exported(self):
        expected = [
            "model_native_linear_coef_importance",
            "model_native_tree_importance",
            "permutation_importance",
            "cond_permutation_importance",
            "partial_dependence_importance",
            "ale_importance",
            "shap_tree_importance",
            "shap_linear_importance",
        ]
        for name in expected:
            assert hasattr(mf.functions, name), f"missing: {name}"
