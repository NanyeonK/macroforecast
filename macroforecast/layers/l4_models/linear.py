"""Public thin-subclass promotions for the L4 linear model family.

Each class is a zero-overhead subclass of the corresponding private runtime
class. No new behaviour is introduced; the public name simply provides a
stable, importable symbol that users can reference without depending on the
internal ``_Private`` naming convention.

``isinstance`` works correctly for both names::

    from macroforecast.layers.l4_models.linear import MidasAlmon
    from macroforecast.core.runtime import _MidasAlmonModel

    model = MidasAlmon(freq_ratio=3)
    assert isinstance(model, _MidasAlmonModel)  # True via subclass

Cycle 63 -- L4 linear family class promotion (14 classes).
"""
from __future__ import annotations

from macroforecast.core.runtime import (
    _MidasAlmonModel,
    _MidasBetaModel,
    _MidasStepModel,
    _UnrestrictedMidasModel,
    _LinearARModel,
    _FactorAugmentedAR,
    _NonNegRidge,
    _TwoStageRandomWalkRidge,
    _ShrinkToTargetRidge,
    _FusedDifferenceRidge,
    _PrincipalComponentRegression,
    _FactorAugmentedVAR,
    _VARWrapper,
    _GLMBoost,
)


class MidasAlmon(_MidasAlmonModel):
    """Ghysels-Santa-Clara-Valkanov (2004) MIDAS with Almon polynomial weights.

    Public alias for :class:`~macroforecast.core.runtime._MidasAlmonModel`.

    Fits a mixed-data sampling (MIDAS) regression with Almon (exponential
    Almon) distributed lag weights estimated by Nelder-Mead NLS.

    Parameters
    ----------
    freq_ratio : int
        High-frequency periods per low-frequency period (m). Default 1.
    n_lags_high : int
        Number of high-frequency lags K to include. Default 12.
    polynomial_order : int
        Almon polynomial degree Q; weight params = Q + 1. Default 2.
    sum_to_one : bool
        Normalize Almon weights to sum to one. Default True.
    max_iter : int
        Max Nelder-Mead iterations per start. Default 200.
    n_starts : int
        Number of NLS multi-starts for robustness. Default 5.
    random_state : int
        RNG seed for perturbed NLS starts. Default 0.

    References
    ----------
    Ghysels, Santa-Clara, Valkanov (2004) "The MIDAS Touch."
    """


class MidasBeta(_MidasBetaModel):
    """MIDAS with Beta polynomial lag weights (Ghysels-Sinko-Valkanov 2007).

    Public alias for :class:`~macroforecast.core.runtime._MidasBetaModel`.

    Uses the two-parameter Beta function B(k; theta_1, theta_2) to weight
    high-frequency lags. Estimation via Nelder-Mead NLS with multi-start.

    Parameters
    ----------
    freq_ratio : int
        High-frequency periods per low-frequency period. Default 1.
    n_lags_high : int
        Number of high-frequency lags K to include. Default 12.
    sum_to_one : bool
        Normalize Beta weights to sum to one. Default True.
    max_iter : int
        Max Nelder-Mead iterations per start. Default 200.
    n_starts : int
        Number of NLS multi-starts. Default 5.
    random_state : int
        RNG seed. Default 0.

    References
    ----------
    Ghysels, Sinko, Valkanov (2007) "MIDAS Regressions: Further Results
    and New Directions." Econometric Reviews 26(1).
    """


class MidasStep(_MidasStepModel):
    """MIDAS with step-function (unrestricted group) lag weights.

    Public alias for :class:`~macroforecast.core.runtime._MidasStepModel`.

    Groups the high-frequency lags into ``n_steps`` equal-weight blocks
    (step functions). Estimation via OLS on the block-averaged design matrix.

    Parameters
    ----------
    freq_ratio : int
        High-frequency periods per low-frequency period. Default 1.
    n_lags_high : int
        Number of high-frequency lags K. Default 12.
    n_steps : int
        Number of step-function blocks. Default = ``freq_ratio``.

    References
    ----------
    Foroni, Marcellino, Schumacher (2015) JRSS-A 178(1) 57-82.
    """


class UnrestrictedMidas(_UnrestrictedMidasModel):
    """Unrestricted MIDAS (U-MIDAS) regression (Foroni et al. 2015).

    Public alias for :class:`~macroforecast.core.runtime._UnrestrictedMidasModel`.

    Fits a direct OLS regression on all high-frequency lags without imposing
    a parametric weight polynomial. Lag order K can be selected by BIC.

    Parameters
    ----------
    freq_ratio : int
        High-frequency periods per low-frequency period. Default 1.
    n_lags_high : int or "bic"
        Number of high-frequency lags K, or "bic" for automatic BIC
        selection. Default "bic".
    include_y_lag : bool
        Include one lag of the low-frequency target as a regressor.
        Default False.
    random_state : int
        RNG seed (reserved; U-MIDAS OLS is deterministic). Default 0.

    References
    ----------
    Foroni, Marcellino, Schumacher (2015) JRSS-A 178(1) 57-82.
    """


class LinearAR(_LinearARModel):
    """Linear AR(p) model with OLS estimation.

    Public alias for :class:`~macroforecast.core.runtime._LinearARModel`.

    Estimates a standard autoregressive model y_t = mu + sum_{i=1}^p phi_i
    y_{t-i} + eps_t by ordinary least squares. Lag order p is fixed.
    """


class FactorAugmentedAR(_FactorAugmentedAR):
    """Factor-augmented AR (FAR) model (Stock-Watson 2002).

    Public alias for :class:`~macroforecast.core.runtime._FactorAugmentedAR`.

    Augments an AR(p) model with principal-component factors extracted from
    a large predictor panel. Estimation is two-step OLS: PCA factors are
    extracted first, then the factors are added to the AR lag matrix.
    """


class NonNegRidge(_NonNegRidge):
    """Non-negative ridge regression (Coulombe et al. 2024 Assemblage Regression).

    Public alias for :class:`~macroforecast.core.runtime._NonNegRidge`.

    Ridge regression with the additional constraint beta >= 0 (non-negative
    coefficients) enforced via an augmented NNLS solver.

    Parameters
    ----------
    alpha : float
        L2 regularisation strength. Default 1.0.

    References
    ----------
    Coulombe et al. (2024) "Assemblage Regression."
    """


class TwoStageRandomWalkRidge(_TwoStageRandomWalkRidge):
    """Time-varying parameter ridge via Coulombe (2025 IJF) two-stage estimator.

    Public alias for :class:`~macroforecast.core.runtime._TwoStageRandomWalkRidge`.

    Implements "Time-Varying Parameters as Ridge Regressions" (Coulombe 2025
    IJF). The estimator uses a Kalman-smoother-inspired two-step procedure:
    step 1 identifies beta path via a random-walk state equation; step 2
    re-estimates lambda via CV on the smoother residuals.

    Parameters
    ----------
    alpha : float
        Initial L2 regularisation strength. Default 1.0.
    vol_model : str
        Volatility model for the Omega reconstruction: "garch11" or "ewma".
        Default "garch11".
    max_alpha_ratio : float
        Upper bound on the second-step alpha/alpha_1 ratio. Default 1e6.
    alpha_search_policy : str
        Strategy for second-step alpha: "second_cv" (default) or "fixed".
    random_state : int
        RNG seed. Default 0.

    References
    ----------
    Coulombe (2025) "Time-Varying Parameters as Ridge Regressions."
    International Journal of Forecasting.
    """


class ShrinkToTargetRidge(_ShrinkToTargetRidge):
    """Shrink-to-target ridge with simplex constraint (Albacore Variant A).

    Public alias for :class:`~macroforecast.core.runtime._ShrinkToTargetRidge`.

    Ridge regression with an explicit prior target for the coefficient vector
    plus an optional simplex (sum-to-one + non-negative) constraint. Implements
    the Maximally Forward-Looking Core Inflation Albacore_comps specification.

    Parameters
    ----------
    alpha : float
        L2 regularisation strength. Default 1.0.
    prior_target : array-like or None
        Target coefficient vector. None defaults to zero vector. Default None.
    simplex : bool
        Enforce simplex constraint (sum-to-one + non-negative). Default True.
    nonneg : bool
        Enforce non-negativity without simplex normalization. Default False.
    """


class FusedDifferenceRidge(_FusedDifferenceRidge):
    """Fused-difference ridge penalty over rank-position weights (Albacore Variant B).

    Public alias for :class:`~macroforecast.core.runtime._FusedDifferenceRidge`.

    Ridge regression with a fused-difference prior that penalizes differences
    between adjacent coefficient values in rank order. Implements the
    Maximally Forward-Looking Core Inflation Albacore_ranks specification.

    Parameters
    ----------
    alpha : float
        L2 regularisation strength. Default 1.0.
    difference_order : int
        Order of differences in the fused-difference penalty. Default 1.
    mean_equality : bool
        Enforce mean-equality constraint across coefficient groups. Default True.
    nonneg : bool
        Enforce non-negativity constraint. Default False.
    """


class PrincipalComponentRegression(_PrincipalComponentRegression):
    """Principal Component Regression (PCR) via OLS on PCA factors.

    Public alias for :class:`~macroforecast.core.runtime._PrincipalComponentRegression`.

    Extracts the top-k principal components from the predictor matrix X and
    regresses the target y on those components via OLS. The number of factors
    k is controlled by the ``n_components`` parameter.
    """


class FactorAugmentedVAR(_FactorAugmentedVAR):
    """Bernanke-Boivin-Eliasz (2005) Factor-Augmented VAR (FAVAR).

    Public alias for :class:`~macroforecast.core.runtime._FactorAugmentedVAR`.

    Augments a VAR with latent factors extracted from a large panel by PCA.
    The two-step procedure: extract PC factors from informational series,
    then estimate a VAR on (factors, target) jointly.

    References
    ----------
    Bernanke, Boivin, Eliasz (2005) "Measuring the Effects of Monetary
    Policy: A Factor-Augmented Vector Autoregressive (FAVAR) Approach."
    QJE 120(1).
    """


class VAR(_VARWrapper):
    """Vector Autoregression (VAR) via statsmodels.

    Public alias for :class:`~macroforecast.core.runtime._VARWrapper`.

    Estimates a VAR(p) model by OLS on the stacked lag matrix. Used in L4
    for multi-step forecasting and as the input for L7 impulse-response
    analysis (GIRF, FEVD, historical decomposition).
    """


class GLMBoost(_GLMBoost):
    """Gradient boosting on a GLM base learner (Buhlmann-Yu 2003 boosting).

    Public alias for :class:`~macroforecast.core.runtime._GLMBoost`.

    Implements component-wise L2-boosting with a linear base learner.
    Each boosting step fits a single predictor OLS, adding a fraction
    (nu) of the step to the current model. Shrinkage controls overfitting.

    References
    ----------
    Buhlmann, Yu (2003) "Boosting with the L2 Loss: Regression and
    Classification." JASA 98(462).
    """


__all__ = [
    "MidasAlmon",
    "MidasBeta",
    "MidasStep",
    "UnrestrictedMidas",
    "LinearAR",
    "FactorAugmentedAR",
    "NonNegRidge",
    "TwoStageRandomWalkRidge",
    "ShrinkToTargetRidge",
    "FusedDifferenceRidge",
    "PrincipalComponentRegression",
    "FactorAugmentedVAR",
    "VAR",
    "GLMBoost",
]
