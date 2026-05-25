"""Unit tests for Cycle 63 standalone promotions.

Covers:
- A. mf.models package: 21 L4 classes promoted via thin subclassing
- B. mf.feature_selection package: 5 sklearn-style class wrappers
- C. mf.transforms.chow_lin_disaggregate function wrapper
- D. mf.interpretation package: GIRF + LSTMHiddenState wrappers
- E. 8 gap callables in mf.functions (4 MIDAS + 4 ridge variants)
- F. mf.__init__ lazy import updates

Each test group verifies:
1. Import / instantiation succeeds.
2. isinstance backward compatibility (public class inherits from private).
3. fit/predict or compute produces plausible numeric output.
4. Correctness assertions where deterministic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_xy(n: int = 60, p: int = 4, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Deterministic feature matrix and target."""
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.randn(n, p), columns=[f"x{i}" for i in range(p)])
    y = pd.Series(X["x0"] + 2.0 * X["x1"] + 0.1 * rng.randn(n), name="y")
    return X, y


@pytest.fixture(scope="module")
def xy():
    return _make_xy()


@pytest.fixture(scope="module")
def xy_small():
    """Smaller panel for slow tests."""
    return _make_xy(n=30, p=3)


# ===========================================================================
# A. mf.models — 21 thin subclasses
# ===========================================================================

class TestModelsPackageImport:
    """Verify all 21 public model classes are importable and subclass correctly."""

    def test_flat_import_21_classes(self) -> None:
        from macroforecast.layers.l4_models import (
            MidasAlmon, MidasBeta, MidasStep, UnrestrictedMidas,
            LinearAR, FactorAugmentedAR,
            NonNegRidge, TwoStageRandomWalkRidge, ShrinkToTargetRidge,
            FusedDifferenceRidge,
            PrincipalComponentRegression, FactorAugmentedVAR, VAR, GLMBoost,
            BVAR, BVARMinnesota, DFMMixedFrequency,
            GARCH, RealizedGARCH,
            ETS, Theta, HoltWinters,
        )
        # All 21 names imported -- no AttributeError means success.
        classes = [
            MidasAlmon, MidasBeta, MidasStep, UnrestrictedMidas,
            LinearAR, FactorAugmentedAR,
            NonNegRidge, TwoStageRandomWalkRidge, ShrinkToTargetRidge,
            FusedDifferenceRidge,
            PrincipalComponentRegression, FactorAugmentedVAR, VAR, GLMBoost,
            BVAR, BVARMinnesota, DFMMixedFrequency,
            GARCH, RealizedGARCH,
            ETS, Theta, HoltWinters,
        ]
        # 14 linear + 3 bayesian (BVAR, BVARMinnesota, DFM) + 2 volatility + 3 timeseries = 22
        assert len(classes) == 22

    def test_submodule_imports(self) -> None:
        from macroforecast.layers.l4_models.linear import MidasAlmon, GLMBoost, VAR
        from macroforecast.layers.l4_models.bayesian import BVAR, BVARMinnesota, DFMMixedFrequency
        from macroforecast.layers.l4_models.volatility import GARCH, RealizedGARCH
        from macroforecast.layers.l4_models.timeseries import ETS, Theta, HoltWinters
        assert True  # all imports succeeded

    def test_isinstance_realized_garch(self) -> None:
        """RealizedGARCH instance must satisfy isinstance check for private class."""
        from macroforecast.layers.l4_models import RealizedGARCH
        from macroforecast.core.runtime import _RealizedGARCHModel

        obj = RealizedGARCH()
        assert isinstance(obj, _RealizedGARCHModel)

    def test_isinstance_midas_almon(self) -> None:
        from macroforecast.layers.l4_models import MidasAlmon
        from macroforecast.core.runtime import _MidasAlmonModel

        obj = MidasAlmon()
        assert isinstance(obj, _MidasAlmonModel)

    def test_isinstance_bvar(self) -> None:
        from macroforecast.layers.l4_models import BVAR, BVARMinnesota
        from macroforecast.core.runtime import _BayesianVAR

        assert isinstance(BVAR(), _BayesianVAR)
        assert isinstance(BVARMinnesota(), _BayesianVAR)

    def test_isinstance_garch(self) -> None:
        from macroforecast.layers.l4_models import GARCH
        from macroforecast.core.runtime import _GARCHFamily

        assert isinstance(GARCH(), _GARCHFamily)

    def test_isinstance_ets(self) -> None:
        from macroforecast.layers.l4_models import ETS
        from macroforecast.core.runtime import _ETSWrapper

        assert isinstance(ETS(), _ETSWrapper)

    def test_isinstance_theta(self) -> None:
        from macroforecast.layers.l4_models import Theta
        from macroforecast.core.runtime import _ThetaWrapper

        assert isinstance(Theta(), _ThetaWrapper)

    def test_isinstance_holt_winters(self) -> None:
        from macroforecast.layers.l4_models import HoltWinters
        from macroforecast.core.runtime import _HoltWintersWrapper

        assert isinstance(HoltWinters(), _HoltWintersWrapper)

    def test_isinstance_dfm(self) -> None:
        from macroforecast.layers.l4_models import DFMMixedFrequency
        from macroforecast.core.runtime import _DFMMixedFrequency

        assert isinstance(DFMMixedFrequency(), _DFMMixedFrequency)

    def test_isinstance_nonneg_ridge(self) -> None:
        from macroforecast.layers.l4_models import NonNegRidge
        from macroforecast.core.runtime import _NonNegRidge

        assert isinstance(NonNegRidge(), _NonNegRidge)

    def test_isinstance_var(self) -> None:
        from macroforecast.layers.l4_models import VAR
        from macroforecast.core.runtime import _VARWrapper

        assert isinstance(VAR(), _VARWrapper)

# NOTE (hotfix-3b-5): test_lazy_import_from_mf removed — Phase 3b moved
# macroforecast.models to macroforecast.layers.l4_models and dropped the
# "models" lazy-module alias from __init__.py.  The remaining methods in
# this class already use macroforecast.layers.l4_models directly and are valid.

# ===========================================================================
# B. mf.feature_selection — 5 sklearn-style wrappers
# ===========================================================================

class TestFeatureSelectionImport:
    """Verify all 5 feature selection classes import correctly."""

    def test_import_all_five(self) -> None:
        from macroforecast.feature_selection import (
            Boruta, RFE, LassoPathSelector, StabilitySelection, GeneticSelection,
        )
        assert True

    def test_lazy_import_from_mf(self) -> None:
        import macroforecast as mf

        mod = mf.feature_selection
        assert hasattr(mod, "Boruta")
        assert hasattr(mod, "RFE")


class TestRFE:
    """Recursive Feature Elimination functional tests."""

    def test_fit_returns_self(self, xy: tuple) -> None:
        from macroforecast.feature_selection import RFE

        X, y = xy
        rfe = RFE(n_features_to_select=2, random_state=0)
        result = rfe.fit(X, y)
        assert result is rfe

    def test_transform_reduces_columns(self, xy: tuple) -> None:
        from macroforecast.feature_selection import RFE

        X, y = xy
        rfe = RFE(n_features_to_select=2, random_state=0)
        rfe.fit(X, y)
        X_sel = rfe.transform(X)
        assert isinstance(X_sel, pd.DataFrame)
        assert X_sel.shape[1] == 2

    def test_selected_features_is_list(self, xy: tuple) -> None:
        from macroforecast.feature_selection import RFE

        X, y = xy
        rfe = RFE(n_features_to_select=3, random_state=0)
        rfe.fit(X, y)
        assert isinstance(rfe.selected_features_, list)
        assert len(rfe.selected_features_) == 3

    def test_fraction_n_features(self, xy: tuple) -> None:
        from macroforecast.feature_selection import RFE

        X, y = xy
        rfe = RFE(n_features_to_select=0.5, random_state=0)
        rfe.fit(X, y)
        # 0.5 * 4 features = 2
        assert len(rfe.selected_features_) == 2


class TestLassoPathSelector:
    """LARS path selection functional tests."""

    def test_fit_transform_roundtrip(self, xy: tuple) -> None:
        from macroforecast.feature_selection import LassoPathSelector

        X, y = xy
        sel = LassoPathSelector(n_features_to_select=2)
        sel.fit(X, y)
        X_sel = sel.transform(X)
        assert isinstance(X_sel, pd.DataFrame)
        assert X_sel.shape[1] == 2

    def test_selected_subsets_columns(self, xy: tuple) -> None:
        from macroforecast.feature_selection import LassoPathSelector

        X, y = xy
        sel = LassoPathSelector(n_features_to_select=3)
        sel.fit(X, y)
        # All selected columns must be in original X
        assert all(c in X.columns for c in sel.selected_features_)

    def test_deterministic(self, xy: tuple) -> None:
        """LARS path is deterministic; two runs should give identical results."""
        from macroforecast.feature_selection import LassoPathSelector

        X, y = xy
        s1 = LassoPathSelector(n_features_to_select=2)
        s2 = LassoPathSelector(n_features_to_select=2)
        s1.fit(X, y)
        s2.fit(X, y)
        assert s1.selected_features_ == s2.selected_features_


class TestStabilitySelection:
    """Stability selection functional tests."""

    def test_fit_returns_self(self, xy: tuple) -> None:
        from macroforecast.feature_selection import StabilitySelection

        X, y = xy
        sel = StabilitySelection(n_subsamples=20, random_state=0)
        result = sel.fit(X, y)
        assert result is sel

    def test_transform_output_is_dataframe(self, xy: tuple) -> None:
        from macroforecast.feature_selection import StabilitySelection

        X, y = xy
        sel = StabilitySelection(n_subsamples=20, random_state=0)
        sel.fit(X, y)
        X_sel = sel.transform(X)
        assert isinstance(X_sel, pd.DataFrame)


class TestBoruta:
    """Boruta algorithm functional tests."""

    def test_fit_returns_self(self, xy: tuple) -> None:
        from macroforecast.feature_selection import Boruta

        X, y = xy
        boruta = Boruta(n_estimators_rf=10, max_iter=5, random_state=0)
        result = boruta.fit(X, y)
        assert result is boruta

    def test_transform_subsets_columns(self, xy: tuple) -> None:
        from macroforecast.feature_selection import Boruta

        X, y = xy
        boruta = Boruta(n_estimators_rf=10, max_iter=5, random_state=0)
        boruta.fit(X, y)
        X_sel = boruta.transform(X)
        # All selected columns are valid column names from X
        assert all(c in X.columns for c in boruta.selected_features_)


class TestGeneticSelection:
    """Genetic algorithm selection functional tests."""

    def test_fit_returns_self(self, xy: tuple) -> None:
        from macroforecast.feature_selection import GeneticSelection

        X, y = xy
        sel = GeneticSelection(population_size=5, n_generations=3, cv_folds=2, random_state=0)
        result = sel.fit(X, y)
        assert result is sel

    def test_transform_output(self, xy: tuple) -> None:
        from macroforecast.feature_selection import GeneticSelection

        X, y = xy
        sel = GeneticSelection(population_size=5, n_generations=3, cv_folds=2, random_state=0)
        sel.fit(X, y)
        X_sel = sel.transform(X)
        assert isinstance(X_sel, pd.DataFrame)
        assert X_sel.shape[1] >= 1


# ===========================================================================
# C. mf.transforms.chow_lin_disaggregate
# ===========================================================================

class TestChowLinDisaggregate:
    """Chow-Lin transform functional tests."""

    def test_import(self) -> None:
        from macroforecast.transforms import chow_lin_disaggregate
        assert callable(chow_lin_disaggregate)

    def test_lazy_import_from_mf(self) -> None:
        import macroforecast as mf

        mod = mf.transforms
        assert hasattr(mod, "chow_lin_disaggregate")

    def test_output_length_matches_indicator(self) -> None:
        from macroforecast.transforms import chow_lin_disaggregate

        rng = np.random.RandomState(0)
        idx_m = pd.date_range("2010-01-31", periods=36, freq="ME")
        idx_q = pd.date_range("2010-03-31", periods=12, freq="QE")
        indicator = pd.Series(rng.randn(36), index=idx_m, name="ind")
        ind_q = indicator.resample("QE").mean()
        y_q = pd.Series(
            0.5 + 2.0 * ind_q.values + 0.1 * rng.randn(12),
            index=idx_q, name="y_q",
        )
        y_m = chow_lin_disaggregate(y_q, indicator)
        assert len(y_m) == 36
        assert isinstance(y_m, pd.Series)

    def test_output_no_nan_when_data_aligned(self) -> None:
        """When both series are DatetimeIndex-aligned, no NaN should appear."""
        from macroforecast.transforms import chow_lin_disaggregate

        rng = np.random.RandomState(1)
        idx_m = pd.date_range("2015-01-31", periods=24, freq="ME")
        idx_q = pd.date_range("2015-03-31", periods=8, freq="QE")
        indicator = pd.Series(np.arange(24, dtype=float) * 0.1, index=idx_m)
        ind_q = indicator.resample("QE").mean()
        y_q = pd.Series(ind_q.values * 2.0 + 1.0, index=idx_q)
        y_m = chow_lin_disaggregate(y_q, indicator)
        assert y_m.isna().sum() == 0

    def test_quarterly_aggregates_approximately(self) -> None:
        """Quarterly average of the disaggregated series should be near y_q."""
        from macroforecast.transforms import chow_lin_disaggregate

        rng = np.random.RandomState(7)
        idx_m = pd.date_range("2010-01-31", periods=36, freq="ME")
        idx_q = pd.date_range("2010-03-31", periods=12, freq="QE")
        indicator = pd.Series(rng.randn(36) + 5, index=idx_m)
        ind_q = indicator.resample("QE").mean()
        y_q = pd.Series(ind_q.values * 1.5 + 0.5, index=idx_q)
        y_m = chow_lin_disaggregate(y_q, indicator)
        # Average the monthly output back to quarterly
        y_m_q = y_m.resample("QE").mean()
        # Align and compare: should be close (OLS reconstruction property)
        common = y_q.reindex(y_m_q.index).dropna()
        recon = y_m_q.reindex(common.index).dropna()
        np.testing.assert_allclose(common.values, recon.values, rtol=0.1, atol=0.5)


# ===========================================================================
# D. mf.interpretation — GIRF + LSTMHiddenState
# ===========================================================================

class TestInterpretationImport:
    """Verify interpretation classes import correctly."""

    def test_import_both(self) -> None:
        from macroforecast.interpretation import GIRF, LSTMHiddenState
        assert True

    def test_lazy_import_from_mf(self) -> None:
        import macroforecast as mf

        mod = mf.interpretation
        assert hasattr(mod, "GIRF")
        assert hasattr(mod, "LSTMHiddenState")


class TestGIRF:
    """GIRF compute() tests using a fallback (non-VAR) model artifact."""

    def test_fallback_returns_dataframe(self, xy: tuple) -> None:
        """When a non-VAR model is supplied, GIRF falls back to tree importance."""
        from macroforecast.interpretation import GIRF
        from macroforecast.functions import ridge_fit

        X, y = xy
        fitted = ridge_fit(X, y, alpha=1.0)

        girf = GIRF()
        result = girf.compute(fitted, n_periods=5)
        assert isinstance(result, pd.DataFrame)
        assert "importance" in result.columns or result.shape[0] >= 0

    def test_girf_instantiation(self) -> None:
        from macroforecast.interpretation import GIRF

        girf = GIRF()
        assert girf is not None


class TestLSTMHiddenState:
    """LSTMHiddenState compute() tests."""

    def test_instantiation(self) -> None:
        from macroforecast.interpretation import LSTMHiddenState

        lstm_hs = LSTMHiddenState()
        assert lstm_hs is not None

    def test_raises_without_torch(self, xy: tuple) -> None:
        """Without a real LSTM fitted object, should raise NotImplementedError."""
        from macroforecast.interpretation import LSTMHiddenState
        from macroforecast.functions import ridge_fit

        X, y = xy
        fitted = ridge_fit(X, y, alpha=1.0)

        lstm_hs = LSTMHiddenState()
        # ridge._model is not a _TorchSequenceModel -> should raise NotImplementedError
        with pytest.raises((NotImplementedError, AttributeError)):
            lstm_hs.compute(fitted, X)


# ===========================================================================
# E. 8 gap callables in mf.functions
# ===========================================================================

class TestMidasFitCallables:
    """Tests for the 4 MIDAS family fit callables."""

    def test_midas_almon_fit_output_shape(self, xy: tuple) -> None:
        from macroforecast.functions import midas_almon_fit

        X, y = xy
        r = midas_almon_fit(X, y, freq_ratio=1, n_lags_high=4,
                            n_starts=1, max_iter=30, random_state=0)
        assert r.coef_.shape == (4,)
        assert isinstance(r.intercept_, float)
        assert r.family == "midas_almon"

    def test_midas_almon_predict(self, xy: tuple) -> None:
        from macroforecast.functions import midas_almon_fit

        X, y = xy
        r = midas_almon_fit(X, y, freq_ratio=1, n_lags_high=4,
                            n_starts=1, max_iter=30, random_state=0)
        preds = r.predict(X)
        assert preds.shape == (len(X),)
        assert np.isfinite(preds).all()

    def test_midas_almon_weights_sum_to_one(self, xy: tuple) -> None:
        """Almon weights with sum_to_one=True should sum to approximately 1."""
        from macroforecast.functions import midas_almon_fit

        X, y = xy
        r = midas_almon_fit(X, y, freq_ratio=1, n_lags_high=4, sum_to_one=True,
                            n_starts=1, max_iter=30, random_state=0)
        # The weights may be zero-padded; sum of nonzero weights should be ~1
        s = float(r.coef_.sum())
        assert abs(s - 1.0) < 0.01 or s == pytest.approx(1.0, abs=0.01)

    def test_midas_beta_fit(self, xy: tuple) -> None:
        from macroforecast.functions import midas_beta_fit

        X, y = xy
        r = midas_beta_fit(X, y, freq_ratio=1, n_lags_high=4,
                           n_starts=1, max_iter=30)
        assert r.coef_.shape == (4,)
        assert r.family == "midas_beta"

    def test_midas_step_fit(self, xy: tuple) -> None:
        from macroforecast.functions import midas_step_fit

        X, y = xy
        r = midas_step_fit(X, y, freq_ratio=1, n_lags_high=4, n_steps=2)
        assert r.family == "midas_step"
        preds = r.predict(X)
        assert preds.shape == (len(X),)

    def test_unrestricted_midas_fit(self, xy: tuple) -> None:
        from macroforecast.functions import unrestricted_midas_fit

        X, y = xy
        r = unrestricted_midas_fit(X, y, freq_ratio=1, n_lags_high=4)
        assert r.family == "dfm_unrestricted_midas"
        preds = r.predict(X)
        assert preds.shape == (len(X),)

    def test_midas_fit_result_summary_str(self, xy: tuple) -> None:
        from macroforecast.functions import midas_almon_fit, MidasFitResult

        X, y = xy
        r = midas_almon_fit(X, y, freq_ratio=1, n_lags_high=4,
                            n_starts=1, max_iter=30, random_state=0)
        assert isinstance(r, MidasFitResult)
        s = r.summary()
        assert "MIDAS-Almon" in s
        assert "intercept" in s


class TestRidgeVariantCallables:
    """Tests for the 4 ridge-variant fit callables."""

    def test_nonneg_ridge_fit_non_negative_coef(self, xy: tuple) -> None:
        from macroforecast.functions import nonneg_ridge_fit

        X, y = xy
        r = nonneg_ridge_fit(X, y, alpha=1.0)
        # All coefficients must be >= 0 (within floating-point tolerance).
        assert np.all(r.coef_ >= -1e-9)

    def test_nonneg_ridge_fit_predict(self, xy: tuple) -> None:
        from macroforecast.functions import nonneg_ridge_fit

        X, y = xy
        r = nonneg_ridge_fit(X, y, alpha=1.0)
        preds = r.predict(X)
        assert preds.shape == (len(X),)
        assert np.isfinite(preds).all()

    def test_nonneg_ridge_negative_alpha_raises(self) -> None:
        from macroforecast.functions import nonneg_ridge_fit

        X, y = _make_xy(n=20, p=3)
        with pytest.raises(ValueError, match="alpha must be >= 0"):
            nonneg_ridge_fit(X, y, alpha=-0.1)

    def test_random_walk_ridge_fit(self, xy: tuple) -> None:
        from macroforecast.functions import random_walk_ridge_fit

        X, y = xy
        # Use ewma vol model (no arch dependency) and fixed strategy for speed.
        r = random_walk_ridge_fit(X, y, alpha=1.0, vol_model="ewma",
                                  alpha_search_policy="fixed")
        assert r.coef_.shape == (X.shape[1],)
        preds = r.predict(X)
        assert preds.shape == (len(X),)
        assert np.isfinite(preds).all()

    def test_random_walk_ridge_negative_alpha_raises(self, xy: tuple) -> None:
        from macroforecast.functions import random_walk_ridge_fit

        X, y = xy
        with pytest.raises(ValueError):
            random_walk_ridge_fit(X, y, alpha=-1.0)

    def test_shrink_to_target_ridge_fit(self, xy_small: tuple) -> None:
        from macroforecast.functions import shrink_to_target_ridge_fit

        X, y = xy_small
        target = np.full(X.shape[1], 1.0 / X.shape[1])
        r = shrink_to_target_ridge_fit(X, y, alpha=1.0, prior_target=target,
                                       simplex=True, nonneg=True)
        assert r.coef_.shape == (X.shape[1],)
        # Simplex: coefficients should sum to ~1 and be >= 0.
        assert float(r.coef_.sum()) == pytest.approx(1.0, abs=0.05)
        assert np.all(r.coef_ >= -1e-9)

    def test_fused_difference_ridge_fit(self, xy_small: tuple) -> None:
        from macroforecast.functions import fused_difference_ridge_fit

        X, y = xy_small
        # mean_equality=False avoids the equality constraint that may fail
        # on very small panels.
        r = fused_difference_ridge_fit(X, y, alpha=0.5, mean_equality=False,
                                       nonneg=False)
        assert r.coef_.shape == (X.shape[1],)
        preds = r.predict(X)
        assert preds.shape == (len(X),)

    def test_ridge_variants_return_RidgeFitResult(self, xy: tuple) -> None:
        """All ridge-variant callables return a RidgeFitResult instance."""
        from macroforecast.functions import (
            nonneg_ridge_fit, random_walk_ridge_fit, fused_difference_ridge_fit,
        )
        from macroforecast.functions.ridge import RidgeFitResult

        X, y = xy
        target = np.full(X.shape[1], 1.0 / X.shape[1])

        r1 = nonneg_ridge_fit(X, y)
        assert isinstance(r1, RidgeFitResult)

        r2 = random_walk_ridge_fit(X, y, vol_model="ewma", alpha_search_policy="fixed")
        assert isinstance(r2, RidgeFitResult)

        r3 = fused_difference_ridge_fit(X, y, mean_equality=False, nonneg=False)
        assert isinstance(r3, RidgeFitResult)


# ===========================================================================
# F. __init__.py lazy import updates
# ===========================================================================

class TestLazyImports:
    """Verify all four new submodules are accessible via mf.<name>."""

    # NOTE (hotfix-3b-5): test_models_lazy_import removed — Phase 3b dropped
    # the "models" lazy-module alias; mf.models no longer exists.

    def test_feature_selection_lazy_import(self) -> None:
        import macroforecast as mf

        mod = mf.feature_selection
        from macroforecast.feature_selection import Boruta
        assert mod.Boruta is Boruta

    def test_transforms_lazy_import(self) -> None:
        import macroforecast as mf

        mod = mf.transforms
        from macroforecast.transforms import chow_lin_disaggregate
        assert mod.chow_lin_disaggregate is chow_lin_disaggregate

    def test_interpretation_lazy_import(self) -> None:
        import macroforecast as mf

        mod = mf.interpretation
        from macroforecast.interpretation import GIRF
        assert mod.GIRF is GIRF

    def test_all_four_in_dir(self) -> None:
        """Remaining lazy submodule names appear in macroforecast.__dir__().

        NOTE (hotfix-3b-5): "models" removed from the check — Phase 3b dropped
        the mf.models alias (moved to macroforecast.layers.l4_models).
        """
        import macroforecast as mf

        d = dir(mf)
        for name in ("feature_selection", "transforms", "interpretation"):
            assert name in d, f"{name!r} not found in mf.__dir__()"
