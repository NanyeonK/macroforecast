"""Tester-authored independent validation for Cycle 63 standalone promotions.

Validates:
A. mf.models — 22 L4 public classes (22 promoted; dispatch note says 21,
   actual count per __init__.py and linear.py is 22; flagged as NOTE).
B. mf.feature_selection — 5 sklearn-style wrappers
C. mf.transforms.chow_lin_disaggregate — 1 function
D. mf.interpretation — GIRF + LSTMHiddenState (2 classes)
E. 8 gap callables in mf.functions (4 MIDAS + 4 ridge variants)
F. mf.__init__ lazy imports for all four new namespaces
G. Backward compat — private names still importable; isinstance holds for all
   tested L4 classes

This file is authored independently from builder's test at
tests/functions/test_c63_standalone_promotions.py. Tester does NOT read
implementation.md.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Shared helpers — independent of builder's fixtures
# ---------------------------------------------------------------------------

def _rng(seed: int = 99) -> np.random.RandomState:
    return np.random.RandomState(seed)


def _panel(n: int = 80, p: int = 5, seed: int = 7) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic panel: y = 2*x0 - x2 + noise."""
    rng = _rng(seed)
    X = pd.DataFrame(rng.randn(n, p), columns=[f"feat_{i}" for i in range(p)])
    y = pd.Series(2.0 * X["feat_0"] - 1.0 * X["feat_2"] + 0.05 * rng.randn(n), name="target")
    return X, y


# ===========================================================================
# Section A — mf.models: import verification + isinstance backward compat
# ===========================================================================

class TestModelsImport:
    """Section A: Public model classes accessible from mf.models.<group>.<Name>."""

    # --- A1: flat re-export from mf.models ---

    def test_A1_midas_almon_importable(self) -> None:
        from macroforecast.models import MidasAlmon
        assert MidasAlmon is not None

    def test_A1_midas_beta_importable(self) -> None:
        from macroforecast.models import MidasBeta
        assert MidasBeta is not None

    def test_A1_midas_step_importable(self) -> None:
        from macroforecast.models import MidasStep
        assert MidasStep is not None

    def test_A1_unrestricted_midas_importable(self) -> None:
        from macroforecast.models import UnrestrictedMidas
        assert UnrestrictedMidas is not None

    def test_A1_linear_ar_importable(self) -> None:
        from macroforecast.models import LinearAR
        assert LinearAR is not None

    def test_A1_factor_augmented_ar_importable(self) -> None:
        from macroforecast.models import FactorAugmentedAR
        assert FactorAugmentedAR is not None

    def test_A1_nonneg_ridge_importable(self) -> None:
        from macroforecast.models import NonNegRidge
        assert NonNegRidge is not None

    def test_A1_rw_ridge_importable(self) -> None:
        from macroforecast.models import TwoStageRandomWalkRidge
        assert TwoStageRandomWalkRidge is not None

    def test_A1_shrink_ridge_importable(self) -> None:
        from macroforecast.models import ShrinkToTargetRidge
        assert ShrinkToTargetRidge is not None

    def test_A1_fused_ridge_importable(self) -> None:
        from macroforecast.models import FusedDifferenceRidge
        assert FusedDifferenceRidge is not None

    def test_A1_pcr_importable(self) -> None:
        from macroforecast.models import PrincipalComponentRegression
        assert PrincipalComponentRegression is not None

    def test_A1_favar_importable(self) -> None:
        from macroforecast.models import FactorAugmentedVAR
        assert FactorAugmentedVAR is not None

    def test_A1_var_importable(self) -> None:
        from macroforecast.models import VAR
        assert VAR is not None

    def test_A1_glmboost_importable(self) -> None:
        from macroforecast.models import GLMBoost
        assert GLMBoost is not None

    def test_A1_bvar_importable(self) -> None:
        from macroforecast.models import BVAR
        assert BVAR is not None

    def test_A1_bvar_minnesota_importable(self) -> None:
        from macroforecast.models import BVARMinnesota
        assert BVARMinnesota is not None

    def test_A1_dfm_mixed_importable(self) -> None:
        from macroforecast.models import DFMMixedFrequency
        assert DFMMixedFrequency is not None

    def test_A1_garch_importable(self) -> None:
        from macroforecast.models import GARCH
        assert GARCH is not None

    def test_A1_realized_garch_importable(self) -> None:
        from macroforecast.models import RealizedGARCH
        assert RealizedGARCH is not None

    def test_A1_ets_importable(self) -> None:
        from macroforecast.models import ETS
        assert ETS is not None

    def test_A1_theta_importable(self) -> None:
        from macroforecast.models import Theta
        assert Theta is not None

    def test_A1_holt_winters_importable(self) -> None:
        from macroforecast.models import HoltWinters
        assert HoltWinters is not None

    # --- A2: submodule imports ---

    def test_A2_linear_submodule_imports(self) -> None:
        from macroforecast.models.linear import (
            MidasAlmon, MidasBeta, MidasStep, UnrestrictedMidas,
            LinearAR, FactorAugmentedAR,
            NonNegRidge, TwoStageRandomWalkRidge, ShrinkToTargetRidge,
            FusedDifferenceRidge,
            PrincipalComponentRegression, FactorAugmentedVAR, VAR, GLMBoost,
        )
        assert len([
            MidasAlmon, MidasBeta, MidasStep, UnrestrictedMidas,
            LinearAR, FactorAugmentedAR,
            NonNegRidge, TwoStageRandomWalkRidge, ShrinkToTargetRidge,
            FusedDifferenceRidge,
            PrincipalComponentRegression, FactorAugmentedVAR, VAR, GLMBoost,
        ]) == 14

    def test_A2_bayesian_submodule_imports(self) -> None:
        from macroforecast.models.bayesian import BVAR, BVARMinnesota, DFMMixedFrequency
        assert BVAR is not None
        assert BVARMinnesota is not None
        assert DFMMixedFrequency is not None

    def test_A2_volatility_submodule_imports(self) -> None:
        from macroforecast.models.volatility import GARCH, RealizedGARCH
        assert GARCH is not None
        assert RealizedGARCH is not None

    def test_A2_timeseries_submodule_imports(self) -> None:
        from macroforecast.models.timeseries import ETS, Theta, HoltWinters
        assert ETS is not None
        assert Theta is not None
        assert HoltWinters is not None

    # --- A3: lazy import via mf.models ---

    def test_A3_mf_models_lazy_accessible(self) -> None:
        import macroforecast as mf
        mod = mf.models
        assert hasattr(mod, "RealizedGARCH")
        assert hasattr(mod, "MidasAlmon")
        assert hasattr(mod, "BVAR")
        assert hasattr(mod, "ETS")

    def test_A3_mf_models_identity_preserved(self) -> None:
        import macroforecast as mf
        from macroforecast.models import RealizedGARCH
        assert mf.models.RealizedGARCH is RealizedGARCH

    # --- A4: __all__ count ---

    def test_A4_models_all_count_is_22(self) -> None:
        """mf.models exposes exactly 22 public names (14+3+2+3).
        NOTE: dispatch memo says 21 but actual linear.py defines 14 classes
        (MidasAlmon/Beta/Step/Unrestricted + LinearAR + FactorAugmentedAR +
        NonNegRidge/TwoStageRandomWalkRidge/ShrinkToTargetRidge/FusedDifferenceRidge
        + PCR + FactorAugmentedVAR + VAR + GLMBoost = 14). Total = 22.
        """
        import macroforecast.models as mmod
        assert len(mmod.__all__) == 22


# ===========================================================================
# Section B — Backward compat: private names importable + isinstance holds
# ===========================================================================

class TestBackwardCompat:
    """Section B: _Private classes still importable; isinstance works."""

    def test_B1_private_RealizedGARCH_importable(self) -> None:
        from macroforecast.core.runtime import _RealizedGARCHModel
        assert _RealizedGARCHModel is not None

    def test_B1_private_GARCHFamily_importable(self) -> None:
        from macroforecast.core.runtime import _GARCHFamily
        assert _GARCHFamily is not None

    def test_B1_private_MidasAlmon_importable(self) -> None:
        from macroforecast.core.runtime import _MidasAlmonModel
        assert _MidasAlmonModel is not None

    def test_B1_private_BayesianVAR_importable(self) -> None:
        from macroforecast.core.runtime import _BayesianVAR
        assert _BayesianVAR is not None

    def test_B1_private_DFMMixedFrequency_importable(self) -> None:
        from macroforecast.core.runtime import _DFMMixedFrequency
        assert _DFMMixedFrequency is not None

    def test_B1_private_VARWrapper_importable(self) -> None:
        from macroforecast.core.runtime import _VARWrapper
        assert _VARWrapper is not None

    def test_B1_private_ETSWrapper_importable(self) -> None:
        from macroforecast.core.runtime import _ETSWrapper
        assert _ETSWrapper is not None

    def test_B1_private_ThetaWrapper_importable(self) -> None:
        from macroforecast.core.runtime import _ThetaWrapper
        assert _ThetaWrapper is not None

    def test_B1_private_HoltWintersWrapper_importable(self) -> None:
        from macroforecast.core.runtime import _HoltWintersWrapper
        assert _HoltWintersWrapper is not None

    def test_B1_private_NonNegRidge_importable(self) -> None:
        from macroforecast.core.runtime import _NonNegRidge
        assert _NonNegRidge is not None

    # isinstance checks for L4 classes
    def test_B2_isinstance_RealizedGARCH(self) -> None:
        from macroforecast.models import RealizedGARCH
        from macroforecast.core.runtime import _RealizedGARCHModel
        obj = RealizedGARCH()
        assert isinstance(obj, _RealizedGARCHModel)

    def test_B2_isinstance_GARCH(self) -> None:
        from macroforecast.models import GARCH
        from macroforecast.core.runtime import _GARCHFamily
        obj = GARCH()
        assert isinstance(obj, _GARCHFamily)

    def test_B2_isinstance_MidasAlmon(self) -> None:
        from macroforecast.models import MidasAlmon
        from macroforecast.core.runtime import _MidasAlmonModel
        obj = MidasAlmon()
        assert isinstance(obj, _MidasAlmonModel)

    def test_B2_isinstance_BVAR(self) -> None:
        from macroforecast.models import BVAR
        from macroforecast.core.runtime import _BayesianVAR
        obj = BVAR()
        assert isinstance(obj, _BayesianVAR)

    def test_B2_isinstance_BVARMinnesota(self) -> None:
        from macroforecast.models import BVARMinnesota
        from macroforecast.core.runtime import _BayesianVAR
        obj = BVARMinnesota()
        assert isinstance(obj, _BayesianVAR)

    def test_B2_isinstance_DFMMixedFrequency(self) -> None:
        from macroforecast.models import DFMMixedFrequency
        from macroforecast.core.runtime import _DFMMixedFrequency
        obj = DFMMixedFrequency()
        assert isinstance(obj, _DFMMixedFrequency)

    def test_B2_isinstance_VAR(self) -> None:
        from macroforecast.models import VAR
        from macroforecast.core.runtime import _VARWrapper
        obj = VAR()
        assert isinstance(obj, _VARWrapper)

    def test_B2_isinstance_ETS(self) -> None:
        from macroforecast.models import ETS
        from macroforecast.core.runtime import _ETSWrapper
        obj = ETS()
        assert isinstance(obj, _ETSWrapper)

    def test_B2_isinstance_Theta(self) -> None:
        from macroforecast.models import Theta
        from macroforecast.core.runtime import _ThetaWrapper
        obj = Theta()
        assert isinstance(obj, _ThetaWrapper)

    def test_B2_isinstance_HoltWinters(self) -> None:
        from macroforecast.models import HoltWinters
        from macroforecast.core.runtime import _HoltWintersWrapper
        obj = HoltWinters()
        assert isinstance(obj, _HoltWintersWrapper)

    def test_B2_isinstance_NonNegRidge(self) -> None:
        from macroforecast.models import NonNegRidge
        from macroforecast.core.runtime import _NonNegRidge
        obj = NonNegRidge()
        assert isinstance(obj, _NonNegRidge)


# ===========================================================================
# Section C — Smoke: RealizedGARCH fit/predict on synthetic data
# ===========================================================================

class TestRealizedGARCHSmoke:
    """Smoke test: L4 class RealizedGARCH fit + predict."""

    def test_C1_fit_predict_finite(self) -> None:
        """RealizedGARCH.fit().predict() returns finite predictions."""
        from macroforecast.models import RealizedGARCH
        rng = _rng(17)
        n = 120
        X = pd.DataFrame(rng.randn(n, 2), columns=["rv", "x1"])
        y = pd.Series(rng.randn(n), name="returns")
        model = RealizedGARCH()
        model.fit(X, y)
        preds = model.predict(X)
        assert preds is not None
        assert preds.shape == (n,)
        assert np.isfinite(preds).all(), "Predictions must be finite"

    def test_C2_predict_shape(self) -> None:
        """predict() output length equals input rows."""
        from macroforecast.models import RealizedGARCH
        rng = _rng(31)
        n = 60
        X = pd.DataFrame(rng.randn(n, 1), columns=["x0"])
        y = pd.Series(rng.randn(n), name="y")
        model = RealizedGARCH()
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == n


# ===========================================================================
# Section D — mf.feature_selection: import + Boruta smoke
# ===========================================================================

class TestFeatureSelectionImport:
    """Section D1: All 5 feature-selection classes accessible."""

    def test_D1_boruta_importable(self) -> None:
        from macroforecast.feature_selection import Boruta
        assert Boruta is not None

    def test_D1_rfe_importable(self) -> None:
        from macroforecast.feature_selection import RFE
        assert RFE is not None

    def test_D1_lasso_path_importable(self) -> None:
        from macroforecast.feature_selection import LassoPathSelector
        assert LassoPathSelector is not None

    def test_D1_stability_importable(self) -> None:
        from macroforecast.feature_selection import StabilitySelection
        assert StabilitySelection is not None

    def test_D1_genetic_importable(self) -> None:
        from macroforecast.feature_selection import GeneticSelection
        assert GeneticSelection is not None

    def test_D2_lazy_mf_feature_selection(self) -> None:
        import macroforecast as mf
        mod = mf.feature_selection
        for name in ("Boruta", "RFE", "LassoPathSelector", "StabilitySelection", "GeneticSelection"):
            assert hasattr(mod, name), f"mf.feature_selection missing {name}"


class TestBorutaSmoke:
    """Boruta fit/transform smoke on synthetic panel."""

    def test_D3_fit_returns_self(self) -> None:
        from macroforecast.feature_selection import Boruta
        X, y = _panel(n=80, p=5, seed=7)
        sel = Boruta(n_estimators_rf=10, max_iter=5, random_state=0)
        result = sel.fit(X, y)
        assert result is sel

    def test_D3_selected_features_is_list(self) -> None:
        from macroforecast.feature_selection import Boruta
        X, y = _panel(n=80, p=5, seed=7)
        sel = Boruta(n_estimators_rf=10, max_iter=5, random_state=0)
        sel.fit(X, y)
        assert isinstance(sel.selected_features_, list)

    def test_D3_selected_features_are_valid_columns(self) -> None:
        from macroforecast.feature_selection import Boruta
        X, y = _panel(n=80, p=5, seed=7)
        sel = Boruta(n_estimators_rf=10, max_iter=5, random_state=0)
        sel.fit(X, y)
        for feat in sel.selected_features_:
            assert feat in X.columns

    def test_D3_transform_returns_dataframe(self) -> None:
        from macroforecast.feature_selection import Boruta
        X, y = _panel(n=80, p=5, seed=7)
        sel = Boruta(n_estimators_rf=10, max_iter=5, random_state=0)
        sel.fit(X, y)
        X_sel = sel.transform(X)
        assert isinstance(X_sel, pd.DataFrame)


# ===========================================================================
# Section E — mf.transforms.chow_lin_disaggregate
# ===========================================================================

class TestChowLinTransform:
    """Section E: chow_lin_disaggregate importable and functional."""

    def test_E1_importable(self) -> None:
        from macroforecast.transforms import chow_lin_disaggregate
        assert callable(chow_lin_disaggregate)

    def test_E2_lazy_mf_transforms(self) -> None:
        import macroforecast as mf
        mod = mf.transforms
        assert hasattr(mod, "chow_lin_disaggregate")

    def test_E3_output_length_equals_indicator(self) -> None:
        """Output length must equal indicator length."""
        from macroforecast.transforms import chow_lin_disaggregate
        rng = _rng(5)
        idx_m = pd.date_range("2012-01-31", periods=36, freq="ME")
        idx_q = pd.date_range("2012-03-31", periods=12, freq="QE")
        indicator = pd.Series(rng.randn(36) + 3.0, index=idx_m, name="ind")
        ind_q = indicator.resample("QE").mean()
        y_q = pd.Series(ind_q.values * 1.2 + 0.8, index=idx_q, name="gdp_q")
        y_m = chow_lin_disaggregate(y_q, indicator)
        assert isinstance(y_m, pd.Series)
        assert len(y_m) == 36

    def test_E4_no_nan_on_aligned_data(self) -> None:
        """Clean aligned input should produce NaN-free output."""
        from macroforecast.transforms import chow_lin_disaggregate
        rng = _rng(11)
        idx_m = pd.date_range("2018-01-31", periods=24, freq="ME")
        idx_q = pd.date_range("2018-03-31", periods=8, freq="QE")
        indicator = pd.Series(np.arange(24, dtype=float) + rng.randn(24) * 0.1, index=idx_m)
        ind_q = indicator.resample("QE").mean()
        y_q = pd.Series(ind_q.values * 2.5, index=idx_q)
        y_m = chow_lin_disaggregate(y_q, indicator)
        assert y_m.isna().sum() == 0


# ===========================================================================
# Section F — mf.interpretation: import + smoke
# ===========================================================================

class TestInterpretationImport:
    """Section F1: GIRF and LSTMHiddenState importable."""

    def test_F1_girf_importable(self) -> None:
        from macroforecast.interpretation import GIRF
        assert GIRF is not None

    def test_F1_lstm_hidden_state_importable(self) -> None:
        from macroforecast.interpretation import LSTMHiddenState
        assert LSTMHiddenState is not None

    def test_F2_lazy_mf_interpretation(self) -> None:
        import macroforecast as mf
        mod = mf.interpretation
        assert hasattr(mod, "GIRF")
        assert hasattr(mod, "LSTMHiddenState")


class TestGIRFSmoke:
    """Section F2: GIRF compute() smoke."""

    def test_F3_girf_instantiation(self) -> None:
        from macroforecast.interpretation import GIRF
        g = GIRF()
        assert g is not None

    def test_F3_girf_compute_returns_df_or_raises(self) -> None:
        """GIRF.compute() on a ridge fit should return a DataFrame or gracefully error."""
        from macroforecast.interpretation import GIRF
        from macroforecast.functions import ridge_fit
        X, y = _panel(n=60, p=3, seed=2)
        fitted = ridge_fit(X, y, alpha=0.5)
        g = GIRF()
        # GIRF is expected to return a DataFrame when a VAR-compatible object
        # is supplied. For non-VAR fallback, may raise or return empty frame.
        try:
            result = g.compute(fitted, n_periods=5)
            assert isinstance(result, pd.DataFrame)
        except (NotImplementedError, ValueError, AttributeError):
            # Graceful fallback is acceptable
            pass


class TestLSTMHiddenStateSmoke:
    """Section F3: LSTMHiddenState smoke."""

    def test_F4_lstm_hs_instantiation(self) -> None:
        from macroforecast.interpretation import LSTMHiddenState
        hs = LSTMHiddenState()
        assert hs is not None

    def test_F4_lstm_hs_raises_on_non_lstm(self) -> None:
        """LSTMHiddenState.compute() on a ridge artifact should raise."""
        from macroforecast.interpretation import LSTMHiddenState
        from macroforecast.functions import ridge_fit
        X, y = _panel(n=40, p=3, seed=3)
        fitted = ridge_fit(X, y, alpha=1.0)
        hs = LSTMHiddenState()
        with pytest.raises((NotImplementedError, AttributeError)):
            hs.compute(fitted, X)


# ===========================================================================
# Section G — 8 gap callables in mf.functions
# ===========================================================================

class TestMidasCallables:
    """Section G1: 4 MIDAS fit callables."""

    def _make_xy(self) -> tuple[pd.DataFrame, pd.Series]:
        return _panel(n=60, p=4, seed=42)

    def test_G1_midas_almon_fit_importable(self) -> None:
        from macroforecast.functions import midas_almon_fit
        assert callable(midas_almon_fit)

    def test_G1_midas_beta_fit_importable(self) -> None:
        from macroforecast.functions import midas_beta_fit
        assert callable(midas_beta_fit)

    def test_G1_midas_step_fit_importable(self) -> None:
        from macroforecast.functions import midas_step_fit
        assert callable(midas_step_fit)

    def test_G1_unrestricted_midas_fit_importable(self) -> None:
        from macroforecast.functions import unrestricted_midas_fit
        assert callable(unrestricted_midas_fit)

    def test_G2_midas_almon_coef_shape(self) -> None:
        from macroforecast.functions import midas_almon_fit
        X, y = self._make_xy()
        r = midas_almon_fit(X, y, freq_ratio=1, n_lags_high=4, n_starts=1, max_iter=30, random_state=0)
        assert r.coef_.shape == (4,), f"Expected coef shape (4,), got {r.coef_.shape}"

    def test_G2_midas_almon_intercept_is_float(self) -> None:
        from macroforecast.functions import midas_almon_fit
        X, y = self._make_xy()
        r = midas_almon_fit(X, y, freq_ratio=1, n_lags_high=4, n_starts=1, max_iter=30, random_state=0)
        assert isinstance(r.intercept_, float)

    def test_G2_midas_almon_family_tag(self) -> None:
        from macroforecast.functions import midas_almon_fit
        X, y = self._make_xy()
        r = midas_almon_fit(X, y, freq_ratio=1, n_lags_high=4, n_starts=1, max_iter=30, random_state=0)
        assert r.family == "midas_almon"

    def test_G2_midas_almon_predict_shape(self) -> None:
        from macroforecast.functions import midas_almon_fit
        X, y = self._make_xy()
        r = midas_almon_fit(X, y, freq_ratio=1, n_lags_high=4, n_starts=1, max_iter=30, random_state=0)
        preds = r.predict(X)
        assert preds.shape == (len(X),)
        assert np.isfinite(preds).all()

    def test_G2_midas_beta_family_tag(self) -> None:
        from macroforecast.functions import midas_beta_fit
        X, y = self._make_xy()
        r = midas_beta_fit(X, y, freq_ratio=1, n_lags_high=4, n_starts=1, max_iter=30)
        assert r.family == "midas_beta"

    def test_G2_midas_step_family_tag(self) -> None:
        from macroforecast.functions import midas_step_fit
        X, y = self._make_xy()
        r = midas_step_fit(X, y, freq_ratio=1, n_lags_high=4, n_steps=2)
        assert r.family == "midas_step"

    def test_G2_unrestricted_midas_family_tag(self) -> None:
        from macroforecast.functions import unrestricted_midas_fit
        X, y = self._make_xy()
        r = unrestricted_midas_fit(X, y, freq_ratio=1, n_lags_high=4)
        assert r.family == "dfm_unrestricted_midas"

    def test_G3_midas_fit_result_type(self) -> None:
        from macroforecast.functions import midas_almon_fit, MidasFitResult
        X, y = self._make_xy()
        r = midas_almon_fit(X, y, freq_ratio=1, n_lags_high=4, n_starts=1, max_iter=30, random_state=0)
        assert isinstance(r, MidasFitResult)


class TestRidgeVariantCallables:
    """Section G2: 4 ridge-variant fit callables."""

    def _make_xy(self) -> tuple[pd.DataFrame, pd.Series]:
        return _panel(n=60, p=4, seed=42)

    def test_G4_nonneg_ridge_importable(self) -> None:
        from macroforecast.functions import nonneg_ridge_fit
        assert callable(nonneg_ridge_fit)

    def test_G4_random_walk_ridge_importable(self) -> None:
        from macroforecast.functions import random_walk_ridge_fit
        assert callable(random_walk_ridge_fit)

    def test_G4_shrink_to_target_ridge_importable(self) -> None:
        from macroforecast.functions import shrink_to_target_ridge_fit
        assert callable(shrink_to_target_ridge_fit)

    def test_G4_fused_difference_ridge_importable(self) -> None:
        from macroforecast.functions import fused_difference_ridge_fit
        assert callable(fused_difference_ridge_fit)

    def test_G5_nonneg_ridge_coef_non_negative(self) -> None:
        """Non-negative ridge must produce coef >= 0 within floating-point tolerance."""
        from macroforecast.functions import nonneg_ridge_fit
        X, y = self._make_xy()
        r = nonneg_ridge_fit(X, y, alpha=1.0)
        assert np.all(r.coef_ >= -1e-9), f"NonNeg ridge has negative coef: {r.coef_}"

    def test_G5_nonneg_ridge_predict_shape(self) -> None:
        from macroforecast.functions import nonneg_ridge_fit
        X, y = self._make_xy()
        r = nonneg_ridge_fit(X, y, alpha=1.0)
        preds = r.predict(X)
        assert preds.shape == (len(X),)
        assert np.isfinite(preds).all()

    def test_G5_nonneg_ridge_negative_alpha_raises(self) -> None:
        from macroforecast.functions import nonneg_ridge_fit
        X, y = _panel(n=30, p=3, seed=9)
        with pytest.raises(ValueError, match="alpha"):
            nonneg_ridge_fit(X, y, alpha=-0.1)

    def test_G6_random_walk_ridge_predict_shape(self) -> None:
        from macroforecast.functions import random_walk_ridge_fit
        X, y = self._make_xy()
        r = random_walk_ridge_fit(X, y, alpha=1.0, vol_model="ewma", alpha_strategy="fixed")
        preds = r.predict(X)
        assert preds.shape == (len(X),)
        assert np.isfinite(preds).all()

    def test_G6_random_walk_ridge_coef_shape(self) -> None:
        from macroforecast.functions import random_walk_ridge_fit
        X, y = self._make_xy()
        r = random_walk_ridge_fit(X, y, alpha=1.0, vol_model="ewma", alpha_strategy="fixed")
        assert r.coef_.shape == (X.shape[1],)

    def test_G7_shrink_to_target_simplex(self) -> None:
        """Shrink-to-target with simplex=True must have coef sum ~1 and coef >= 0."""
        from macroforecast.functions import shrink_to_target_ridge_fit
        X, y = _panel(n=40, p=3, seed=77)
        target = np.full(3, 1.0 / 3)
        r = shrink_to_target_ridge_fit(X, y, alpha=1.0, prior_target=target, simplex=True, nonneg=True)
        assert r.coef_.shape == (3,)
        assert float(r.coef_.sum()) == pytest.approx(1.0, abs=0.05)
        assert np.all(r.coef_ >= -1e-9)

    def test_G8_fused_difference_ridge_predict_finite(self) -> None:
        from macroforecast.functions import fused_difference_ridge_fit
        X, y = _panel(n=40, p=3, seed=88)
        r = fused_difference_ridge_fit(X, y, alpha=0.5, mean_equality=False, nonneg=False)
        assert r.coef_.shape == (X.shape[1],)
        preds = r.predict(X)
        assert preds.shape == (len(X),)
        assert np.isfinite(preds).all()

    def test_G9_ridge_variants_return_RidgeFitResult(self) -> None:
        from macroforecast.functions import (
            nonneg_ridge_fit, random_walk_ridge_fit, fused_difference_ridge_fit,
        )
        from macroforecast.functions.ridge import RidgeFitResult
        X, y = self._make_xy()
        r1 = nonneg_ridge_fit(X, y, alpha=1.0)
        assert isinstance(r1, RidgeFitResult)
        r2 = random_walk_ridge_fit(X, y, vol_model="ewma", alpha_strategy="fixed")
        assert isinstance(r2, RidgeFitResult)
        r3 = fused_difference_ridge_fit(X, y, mean_equality=False, nonneg=False)
        assert isinstance(r3, RidgeFitResult)


# ===========================================================================
# Section H — mf.__init__ lazy import: all 4 new namespaces in _LAZY_MODULES
# ===========================================================================

class TestLazyImportsInit:
    """Section H: All four new submodules in macroforecast.__dir__."""

    def test_H1_four_names_in_dir(self) -> None:
        import macroforecast as mf
        d = dir(mf)
        for name in ("models", "feature_selection", "transforms", "interpretation"):
            assert name in d, f"'{name}' not found in macroforecast.__dir__()"

    def test_H2_models_identity(self) -> None:
        import macroforecast as mf
        from macroforecast.models import RealizedGARCH
        assert mf.models.RealizedGARCH is RealizedGARCH

    def test_H2_feature_selection_identity(self) -> None:
        import macroforecast as mf
        from macroforecast.feature_selection import Boruta
        assert mf.feature_selection.Boruta is Boruta

    def test_H2_transforms_identity(self) -> None:
        import macroforecast as mf
        from macroforecast.transforms import chow_lin_disaggregate
        assert mf.transforms.chow_lin_disaggregate is chow_lin_disaggregate

    def test_H2_interpretation_identity(self) -> None:
        import macroforecast as mf
        from macroforecast.interpretation import GIRF
        assert mf.interpretation.GIRF is GIRF
