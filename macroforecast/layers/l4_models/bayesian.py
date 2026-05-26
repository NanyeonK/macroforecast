"""Public thin-subclass promotions for the L4 Bayesian and DFM model family.

Each class is a zero-overhead subclass of the corresponding private runtime
class. No new behaviour is introduced.

v0.9.5 -- L4 Bayesian/DFM class promotion (3 classes: BVAR Minnesota,
BVAR NIW, DFM Mixed-Frequency).
"""
from __future__ import annotations

from typing import Any

from macroforecast.core.runtime import (
    _BayesianVAR,
    _DFMMixedFrequency,
)


class BVAR(_BayesianVAR):
    """Bayesian Vector Autoregression with Minnesota or NIW prior.

    Public alias for :class:`~macroforecast.core.runtime._BayesianVAR`.

    Estimates a BVAR via the closed-form Litterman (1986) Minnesota prior
    or the Normal-Inverse-Wishart (NIW) conjugate posterior mean. The prior
    type is controlled by the ``prior`` parameter.

    Parameters
    ----------
    prior : str
        Prior type: "minnesota" (Litterman 1986) or "normal_inverse_wishart"
        (NIW). Default "minnesota".
    lambda1 : float
        Overall tightness hyperparameter (Minnesota prior). Default 0.1.
    lambda2 : float
        Cross-variable tightness (Minnesota prior). Default 1.0.
    lambda3 : float
        Lag decay hyperparameter (Minnesota prior). Default 1.0.
    n_lags : int
        Number of VAR lags p. Default 1.

    References
    ----------
    Litterman (1986) "Forecasting with Bayesian Vector Autoregressions."
    Journal of Business and Economic Statistics 4(1).
    """


class BVARMinnesota(_BayesianVAR):
    """Bayesian VAR with Minnesota prior (Litterman 1986).

    This is a convenience subclass that constrains ``prior='minnesota'``.
    For arbitrary BVAR priors, use :class:`BVAR` directly.

    The Minnesota prior treats each variable's own lags as most informative
    and imposes increasing shrinkage on higher-order lags and cross-variable
    predictors.

    Parameters
    ----------
    prior : str
        Must be ``'minnesota'`` (default). Any other value raises
        ``ValueError``. Use :class:`BVAR` for arbitrary priors.
    **kwargs
        Forwarded to :class:`BVAR`. See :class:`BVAR` docstring for
        available hyperparameters (``lambda1``, ``lambda2``, ``lambda3``,
        ``n_lags``, ``n_draws``, etc.).

    Raises
    ------
    ValueError
        If ``prior`` is not ``'minnesota'``.

    References
    ----------
    Litterman (1986) "Forecasting with Bayesian Vector Autoregressions."
    Journal of Business and Economic Statistics 4(1).
    """

    def __init__(self, *, prior: str = "minnesota", **kwargs: Any) -> None:
        if prior != "minnesota":
            raise ValueError(
                f"BVARMinnesota requires prior='minnesota'; got prior={prior!r}. "
                "Use BVAR for arbitrary priors."
            )
        # Translate user-facing 'minnesota' to internal 'bvar_minnesota' for _BayesianVAR.
        # _BayesianVAR.__init__ dispatches on 'bvar_minnesota' (not 'minnesota') to activate
        # the closed-form Litterman Minnesota posterior mean branch.
        super().__init__(prior="bvar_minnesota", **kwargs)


class DFMMixedFrequency(_DFMMixedFrequency):
    """Mariano-Murasawa (2003) Dynamic Factor Model for mixed-frequency data.

    Public alias for :class:`~macroforecast.core.runtime._DFMMixedFrequency`.

    Fits a DFM via the statsmodels Kalman state-space MLE
    (``DynamicFactorMQ`` when ``mixed_frequency=True``). Handles mixed
    monthly/quarterly panel data using the Mariano-Murasawa aggregation
    equation (Eq. 4) to bridge the frequency mismatch.

    Parameters
    ----------
    k_factors : int
        Number of latent common factors. Default 1.
    factor_order : int
        Factor VAR order. Default 1.
    mixed_frequency : bool
        Use the mixed-frequency Kalman filter route. Default True.
    idiosyncratic_ar1 : bool
        Model idiosyncratic component as AR(1) per Mariano-Murasawa Eq. (4).
        Default True.

    References
    ----------
    Mariano, Murasawa (2003) "A New Coincident Index of Business Cycles
    Based on Monthly and Quarterly Series." Journal of Applied Econometrics.
    """


__all__ = [
    "BVAR",
    "BVARMinnesota",
    "DFMMixedFrequency",
]
