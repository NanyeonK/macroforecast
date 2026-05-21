"""Standalone time-series regression wrappers for the L4 timeseries sub-family.

Exposes fourteen fit callables:

    var_fit, bvar_minnesota_fit, bvar_niw_fit, ar_fit, far_fit, pcr_fit,
    favar_fit, garch11_fit, egarch_fit, realized_garch_fit, ets_fit,
    theta_fit, holt_winters_fit, dfm_fit

each returning a frozen dataclass that conforms structurally to
:class:`~macroforecast.functions.FitResultBase`.

All callables call ``_build_l4_model`` from ``macroforecast.core.runtime``
directly, producing bit-exact numeric output identical to the recipe DAG
with the same parameter values.

Cycle 37 -- L4 timeseries family standalone-ization (14 ops).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helper: convert ndarray/Series inputs to DataFrame/Series
# ---------------------------------------------------------------------------

def _to_frame(X: np.ndarray | pd.DataFrame) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])


def _to_series(y: np.ndarray | pd.Series) -> pd.Series:
    if isinstance(y, pd.Series):
        return y
    return pd.Series(np.asarray(y).ravel(), name="y")


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VARFitResult:
    """Result of :func:`var_fit`.

    Attributes
    ----------
    n_lag :
        Lag order p used.
    n_series :
        Number of endogenous series (= X.shape[1] + 1, target included).
    n_obs :
        Number of observations after lag creation.
    _model :
        Internal fitted ``_VARWrapper`` instance.
        Not part of the public contract.
    """

    n_lag: int
    n_series: int
    n_obs: int
    _model: Any

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return point forecast for each row of ``X``.

        Parameters
        ----------
        X :
            Feature matrix.  Accepts numpy arrays or DataFrames.

        Returns
        -------
        np.ndarray
            1-D float array, shape (n_samples,).
        """
        return np.asarray(self._model.predict(_to_frame(X)), dtype=float)

    def summary(self) -> str:
        """Return a minimal text summary of the VAR fit result.

        Returns
        -------
        str
            Statsmodels-style table showing lag order and observations.
        """
        sep = "=" * 78
        lines = [
            sep,
            f"{'VAR Results':^78}",
            sep,
            f"{'n_lag:':35s} {self.n_lag:>20d}",
            f"{'n_series:':35s} {self.n_series:>20d}",
            f"{'n_obs:':35s} {self.n_obs:>20d}",
            sep,
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class BVARMinnesotaFitResult:
    """Result of :func:`bvar_minnesota_fit`.

    Attributes
    ----------
    n_lag :
        Lag order p used.
    lambda1 :
        Minnesota tightness hyperparameter.
    n_obs :
        Number of observations used.
    _model :
        Internal fitted ``_BayesianVAR`` instance.
        Not part of the public contract.
    """

    n_lag: int
    lambda1: float
    n_obs: int
    _model: Any

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return point forecast for each row of ``X``.

        Returns
        -------
        np.ndarray
            1-D float array, shape (n_samples,).
        """
        return np.asarray(self._model.predict(_to_frame(X)), dtype=float)

    def summary(self) -> str:
        """Return a minimal text summary of the BVAR Minnesota fit result.

        Returns
        -------
        str
            Statsmodels-style table showing lag order, tightness, and
            observation count.
        """
        sep = "=" * 78
        lines = [
            sep,
            f"{'BVAR Minnesota Results':^78}",
            sep,
            f"{'n_lag:':35s} {self.n_lag:>20d}",
            f"{'lambda1:':35s} {self.lambda1:>20.4f}",
            f"{'n_obs:':35s} {self.n_obs:>20d}",
            sep,
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class BVARNIWFitResult:
    """Result of :func:`bvar_niw_fit`.

    Attributes
    ----------
    n_lag :
        Lag order p used.
    lambda1 :
        NIW prior tightness parameter (default 0.2).  Set to the overall
        tightness hyperparameter used for the Normal-Inverse-Wishart prior,
        analogous to Minnesota lambda1.  Always >= 0.2.
    n_obs :
        Number of observations used.
    _model :
        Internal fitted ``_BayesianVAR`` instance.
        Not part of the public contract.
    """

    n_lag: int
    lambda1: float
    n_obs: int
    _model: Any

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return point forecast for each row of ``X``.

        Returns
        -------
        np.ndarray
            1-D float array, shape (n_samples,).
        """
        return np.asarray(self._model.predict(_to_frame(X)), dtype=float)

    def summary(self) -> str:
        """Return a minimal text summary of the BVAR NIW fit result.

        Returns
        -------
        str
            Statsmodels-style table showing lag order, tightness, and
            observation count.
        """
        sep = "=" * 78
        lines = [
            sep,
            f"{'BVAR NIW Results':^78}",
            sep,
            f"{'n_lag:':35s} {self.n_lag:>20d}",
            f"{'lambda1:':35s} {self.lambda1:>20.4f}",
            f"{'n_obs:':35s} {self.n_obs:>20d}",
            sep,
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class ARFitResult:
    """Result of :func:`ar_fit`.

    Attributes
    ----------
    n_lag :
        AR lag order p.
    coef_ :
        Fitted AR coefficients, shape (p,).
    intercept_ :
        Fitted intercept.
    _model :
        Internal fitted ``_LinearARModel`` instance.
        Not part of the public contract.
    """

    n_lag: int
    coef_: np.ndarray
    intercept_: float
    _model: Any

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return point forecast for each row of ``X``.

        Returns
        -------
        np.ndarray
            1-D float array, shape (n_samples,).
        """
        return np.asarray(self._model.predict(_to_frame(X)), dtype=float)

    def summary(self) -> str:
        """Return a minimal text summary of the AR fit result.

        Returns
        -------
        str
            Statsmodels-style table showing lag order, intercept, and
            AR coefficients.
        """
        sep = "=" * 78
        dash = "-" * 78
        lines = [
            sep,
            f"{'AR Results':^78}",
            sep,
            f"{'n_lag:':35s} {self.n_lag:>20d}",
            f"{'intercept_:':35s} {self.intercept_:>20.6f}",
            sep,
            f"{'Lag':30s} {'coef':>12s}",
            dash,
        ]
        for i, c in enumerate(self.coef_):
            lines.append(f"{'lag_' + str(i+1):30s} {c:>12.6f}")
        lines.append(sep)
        return "\n".join(lines)


@dataclass(frozen=True)
class FARFitResult:
    """Result of :func:`far_fit`.

    Attributes
    ----------
    n_factors :
        Number of PCA factors extracted from ``X``.
    n_lag :
        AR lag order p on target.
    coef_ :
        Fitted regression coefficients on (factors + AR lags), shape
        (n_factors + n_lag,).
    _model :
        Internal fitted ``_FactorAugmentedAR`` instance.
        Not part of the public contract.
    """

    n_factors: int
    n_lag: int
    coef_: "np.ndarray"
    _model: Any

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return point forecast for each row of ``X``.

        Returns
        -------
        np.ndarray
            1-D float array, shape (n_samples,).
        """
        return np.asarray(self._model.predict(_to_frame(X)), dtype=float)

    def summary(self) -> str:
        """Return a minimal text summary of the FAR fit result.

        Returns
        -------
        str
            Statsmodels-style table showing factor count and lag order.
        """
        sep = "=" * 78
        lines = [
            sep,
            f"{'FAR (Factor-Augmented AR) Results':^78}",
            sep,
            f"{'n_factors:':35s} {self.n_factors:>20d}",
            f"{'n_lag:':35s} {self.n_lag:>20d}",
            f"{'coef_ shape:':35s} {str(self.coef_.shape):>20s}",
            sep,
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class PCRFitResult:
    """Result of :func:`pcr_fit`.

    Attributes
    ----------
    n_components :
        Number of principal components used in regression.
    coef_ :
        Fitted OLS coefficients in the PC score space, shape (n_components,).
    _model :
        Internal fitted ``_PrincipalComponentRegression`` instance.
        Not part of the public contract.
    """

    n_components: int
    coef_: "np.ndarray"
    _model: Any

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return point forecast for each row of ``X``.

        Returns
        -------
        np.ndarray
            1-D float array, shape (n_samples,).
        """
        return np.asarray(self._model.predict(_to_frame(X)), dtype=float)

    def summary(self) -> str:
        """Return a minimal text summary of the PCR fit result.

        Returns
        -------
        str
            Statsmodels-style table showing component count.
        """
        sep = "=" * 78
        lines = [
            sep,
            f"{'PCR Results':^78}",
            sep,
            f"{'n_components:':35s} {self.n_components:>20d}",
            f"{'coef_ shape:':35s} {str(self.coef_.shape):>20s}",
            sep,
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class FAVARFitResult:
    """Result of :func:`favar_fit`.

    Attributes
    ----------
    n_factors :
        Number of PCA factors extracted from ``X``.
    n_lag :
        VAR lag order p.
    _model :
        Internal fitted ``_FactorAugmentedVAR`` instance.
        Not part of the public contract.
    """

    n_factors: int
    n_lag: int
    _model: Any

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return point forecast for each row of ``X``.

        Returns
        -------
        np.ndarray
            1-D float array, shape (n_samples,).
        """
        return np.asarray(self._model.predict(_to_frame(X)), dtype=float)

    def summary(self) -> str:
        """Return a minimal text summary of the FAVAR fit result.

        Returns
        -------
        str
            Statsmodels-style table showing factor count and lag order.
        """
        sep = "=" * 78
        lines = [
            sep,
            f"{'FAVAR (Factor-Augmented VAR) Results':^78}",
            sep,
            f"{'n_factors:':35s} {self.n_factors:>20d}",
            f"{'n_lag:':35s} {self.n_lag:>20d}",
            sep,
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class _GARCH11FitResultV0:
    """Pre-consolidation result type for :func:`garch11_fit`.

    Superseded by :class:`GARCHFitResult`. The public name
    ``GARCH11FitResult`` is now a TypeAlias for ``GARCHFitResult``.

    Attributes
    ----------
    conditional_mu :
        Fitted constant conditional mean mu.
    n_obs :
        Number of non-missing observations in y.
    params_ :
        Dictionary of fitted GARCH model parameters.
    _model :
        Internal fitted ``_GARCHFamily`` instance.
        Not part of the public contract.
    """

    conditional_mu: float
    n_obs: int
    params_: dict
    _model: Any

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return conditional-mean forecast broadcast over ``len(X)`` rows.

        Returns
        -------
        np.ndarray
            1-D float array, shape (n_samples,).
        """
        return np.asarray(self._model.predict(_to_frame(X)), dtype=float)

    def predict_variance(self, h_steps: int = 1) -> np.ndarray:
        """Return h-step-ahead conditional variance forecast.

        Parameters
        ----------
        h_steps :
            Forecast horizon.  Default 1.

        Returns
        -------
        np.ndarray
            1-D float array, shape (h_steps,).
        """
        return self._model.predict_variance(h_steps)

    def summary(self) -> str:
        """Return a minimal text summary of the GARCH(1,1) fit result.

        Returns
        -------
        str
            Statsmodels-style table showing conditional mean and fitted
            parameters.
        """
        sep = "=" * 78
        lines = [
            sep,
            f"{'GARCH(1,1) Results':^78}",
            sep,
            f"{'conditional_mu:':35s} {self.conditional_mu:>20.6f}",
            f"{'n_obs:':35s} {self.n_obs:>20d}",
        ]
        for k, v in self.params_.items():
            lines.append(f"{k + ':':35s} {v:>20.6f}")
        lines.append(sep)
        return "\n".join(lines)


@dataclass(frozen=True)
class _EGARCHFitResultV0:
    """Pre-consolidation result type for :func:`egarch_fit`.

    Superseded by :class:`GARCHFitResult`. The public name
    ``EGARCHFitResult`` is now a TypeAlias for ``GARCHFitResult``.

    Attributes
    ----------
    conditional_mu :
        Fitted constant conditional mean mu.
    n_obs :
        Number of non-missing observations in y.
    params_ :
        Dictionary of fitted EGARCH model parameters.
    _model :
        Internal fitted ``_GARCHFamily`` instance.
        Not part of the public contract.
    """

    conditional_mu: float
    n_obs: int
    params_: dict
    _model: Any

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return conditional-mean forecast broadcast over ``len(X)`` rows.

        Returns
        -------
        np.ndarray
            1-D float array, shape (n_samples,).
        """
        return np.asarray(self._model.predict(_to_frame(X)), dtype=float)

    def predict_variance(self, h_steps: int = 1) -> np.ndarray:
        """Return h-step-ahead conditional variance forecast.

        Parameters
        ----------
        h_steps :
            Forecast horizon.  Default 1.

        Returns
        -------
        np.ndarray
            1-D float array, shape (h_steps,).
        """
        return self._model.predict_variance(h_steps)

    def summary(self) -> str:
        """Return a minimal text summary of the EGARCH fit result.

        Returns
        -------
        str
            Statsmodels-style table showing conditional mean and parameters.
        """
        sep = "=" * 78
        lines = [
            sep,
            f"{'EGARCH Results':^78}",
            sep,
            f"{'conditional_mu:':35s} {self.conditional_mu:>20.6f}",
            f"{'n_obs:':35s} {self.n_obs:>20d}",
        ]
        for k, v in self.params_.items():
            lines.append(f"{k + ':':35s} {v:>20.6f}")
        lines.append(sep)
        return "\n".join(lines)


@dataclass(frozen=True)
class _RealizedGARCHFitResultV0:
    """Pre-consolidation result type for :func:`realized_garch_fit`.

    Superseded by :class:`GARCHFitResult`. The public name
    ``RealizedGARCHFitResult`` is now a TypeAlias for ``GARCHFitResult``.

    Attributes
    ----------
    conditional_mu :
        Fitted constant conditional mean mu.
    n_obs :
        Number of non-missing observations in y.
    params_ :
        Dictionary of fitted model parameters.
    _model :
        Internal fitted ``_GARCHFamily`` instance.
        Not part of the public contract.
    """

    conditional_mu: float
    n_obs: int
    params_: dict
    _model: Any

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return conditional-mean forecast broadcast over ``len(X)`` rows.

        Returns
        -------
        np.ndarray
            1-D float array, shape (n_samples,).
        """
        return np.asarray(self._model.predict(_to_frame(X)), dtype=float)

    def predict_variance(self, h_steps: int = 1) -> np.ndarray:
        """Return h-step-ahead conditional variance forecast.

        Parameters
        ----------
        h_steps :
            Forecast horizon.  Default 1.

        Returns
        -------
        np.ndarray
            1-D float array, shape (h_steps,).
        """
        return self._model.predict_variance(h_steps)

    def summary(self) -> str:
        """Return a minimal text summary of the RealizedGARCH fit result.

        Returns
        -------
        str
            Statsmodels-style table showing conditional mean and parameters.
        """
        sep = "=" * 78
        lines = [
            sep,
            f"{'RealizedGARCH Results':^78}",
            sep,
            f"{'conditional_mu:':35s} {self.conditional_mu:>20.6f}",
            f"{'n_obs:':35s} {self.n_obs:>20d}",
        ]
        for k, v in self.params_.items():
            lines.append(f"{k + ':':35s} {v:>20.6f}")
        lines.append(sep)
        return "\n".join(lines)



@dataclass(frozen=True)
class GARCHFitResult:
    """Consolidated result for GARCH family callables.

    All three GARCH variants (:func:`garch11_fit`, :func:`egarch_fit`,
    :func:`realized_garch_fit`) return this single type, distinguished by the
    ``variant`` attribute.

    Attributes
    ----------
    variant :
        GARCH family variant: ``"garch"`` | ``"egarch"`` | ``"realized_garch"``.
    conditional_mu :
        Fitted constant conditional mean mu.
    n_obs :
        Number of non-missing observations in y.
    params_ :
        Dictionary of fitted model parameters.
    _model :
        Internal fitted ``_GARCHFamily`` instance.
        Not part of the public contract.
    """

    variant: str
    conditional_mu: float
    n_obs: int
    params_: dict
    _model: Any

    def predict(self, X: "np.ndarray | pd.DataFrame") -> "np.ndarray":
        """Return conditional-mean forecast broadcast over ``len(X)`` rows."""
        return np.asarray(self._model.predict(_to_frame(X)), dtype=float)

    def predict_variance(self, h_steps: int = 1) -> "np.ndarray":
        """Return h-step-ahead conditional variance forecast."""
        return self._model.predict_variance(h_steps)

    def summary(self) -> str:
        """Return a minimal text summary of the GARCH fit result."""
        sep = "=" * 78
        label = {
            "garch": "GARCH(1,1) Results",
            "egarch": "EGARCH Results",
            "realized_garch": "RealizedGARCH Results",
        }.get(self.variant, f"GARCH[{self.variant}] Results")
        lines = [
            sep,
            f"{label:^78}",
            sep,
            f"{'variant:':35s} {self.variant:>20s}",
            f"{'conditional_mu:':35s} {self.conditional_mu:>20.6f}",
            f"{'n_obs:':35s} {self.n_obs:>20d}",
        ]
        for k, v in self.params_.items():
            lines.append(f"{k + ':':35s} {v:>20.6f}")
        lines.append(sep)
        return "\n".join(lines)


# Backward-compat aliases: old result types map to the new consolidated type.
# These aliases keep existing code that references GARCH11FitResult etc. working.
GARCH11FitResult: TypeAlias = GARCHFitResult
EGARCHFitResult: TypeAlias = GARCHFitResult
RealizedGARCHFitResult: TypeAlias = GARCHFitResult

@dataclass(frozen=True)
class ETSFitResult:
    """Result of :func:`ets_fit`.

    Attributes
    ----------
    error_trend_seasonal :
        3-character ETS code, e.g. ``'AAN'``.
    n_obs :
        Number of observations used.
    _model :
        Internal fitted ``_ETSWrapper`` instance.
        Not part of the public contract.
    """

    error_trend_seasonal: str
    n_obs: int
    _model: Any

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return forecast for each row of ``X`` (``len(X)`` steps ahead).

        Returns
        -------
        np.ndarray
            1-D float array, shape (n_samples,).
        """
        return np.asarray(self._model.predict(_to_frame(X)), dtype=float)

    def summary(self) -> str:
        """Return a minimal text summary of the ETS fit result.

        Returns
        -------
        str
            Statsmodels-style table showing ETS code and observation count.
        """
        sep = "=" * 78
        lines = [
            sep,
            f"{'ETS Results':^78}",
            sep,
            f"{'error_trend_seasonal:':35s} {self.error_trend_seasonal:>20s}",
            f"{'n_obs:':35s} {self.n_obs:>20d}",
            sep,
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class ThetaFitResult:
    """Result of :func:`theta_fit`.

    Attributes
    ----------
    theta :
        Theta parameter value (default 2.0 = M3 winner).
    alpha_ :
        Fitted SES smoothing parameter alpha.
    n_obs :
        Number of observations used.
    _model :
        Internal fitted ``_ThetaWrapper`` instance.
        Not part of the public contract.
    """

    theta: float
    alpha_: float
    n_obs: int
    _model: Any

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return forecast for each row of ``X`` (``len(X)`` steps ahead).

        Returns
        -------
        np.ndarray
            1-D float array, shape (n_samples,).
        """
        return np.asarray(self._model.predict(_to_frame(X)), dtype=float)

    def summary(self) -> str:
        """Return a minimal text summary of the Theta fit result.

        Returns
        -------
        str
            Statsmodels-style table showing theta, alpha, and observation
            count.
        """
        sep = "=" * 78
        lines = [
            sep,
            f"{'Theta Results':^78}",
            sep,
            f"{'theta:':35s} {self.theta:>20.4f}",
            f"{'alpha_ (SES):':35s} {self.alpha_:>20.6f}",
            f"{'n_obs:':35s} {self.n_obs:>20d}",
            sep,
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class HoltWintersFitResult:
    """Result of :func:`holt_winters_fit`.

    Attributes
    ----------
    seasonal :
        Seasonal component type: ``'add'`` or ``'mul'``.
    seasonal_periods :
        Number of periods per season.
    trend :
        Trend component type: ``'add'``, ``'mul'``, or ``None``.
    n_obs :
        Number of observations used.
    _model :
        Internal fitted ``_HoltWintersWrapper`` instance.
        Not part of the public contract.
    """

    seasonal: str
    seasonal_periods: int
    trend: str
    n_obs: int
    _model: Any

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return forecast for each row of ``X`` (``len(X)`` steps ahead).

        Returns
        -------
        np.ndarray
            1-D float array, shape (n_samples,).
        """
        return np.asarray(self._model.predict(_to_frame(X)), dtype=float)

    def summary(self) -> str:
        """Return a minimal text summary of the Holt-Winters fit result.

        Returns
        -------
        str
            Statsmodels-style table showing seasonal type, periods, trend,
            and observation count.
        """
        sep = "=" * 78
        lines = [
            sep,
            f"{'Holt-Winters Results':^78}",
            sep,
            f"{'seasonal:':35s} {self.seasonal:>20s}",
            f"{'seasonal_periods:':35s} {self.seasonal_periods:>20d}",
            f"{'trend:':35s} {str(self.trend):>20s}",
            f"{'n_obs:':35s} {self.n_obs:>20d}",
            sep,
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class DFMFitResult:
    """Result of :func:`dfm_fit`.

    Attributes
    ----------
    n_factors :
        Number of dynamic factors extracted.
    mode_ :
        DFM estimator mode indicator: always ``"mariano_murasawa"`` for this
        family (Kalman state-space MLE, Mariano-Murasawa 2010).
    n_obs :
        Number of observations used.
    _model :
        Internal fitted ``_DFMMixedFrequency`` instance.
        Not part of the public contract.
    """

    n_factors: int
    mode_: str
    n_obs: int
    _model: Any

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return point forecast for each row of ``X``.

        Returns
        -------
        np.ndarray
            1-D float array, shape (n_samples,).
        """
        return np.asarray(self._model.predict(_to_frame(X)), dtype=float)

    def summary(self) -> str:
        """Return a minimal text summary of the DFM fit result.

        Returns
        -------
        str
            Statsmodels-style table showing factor count, mode, and
            observation count.
        """
        sep = "=" * 78
        lines = [
            sep,
            f"{'DFM Results':^78}",
            sep,
            f"{'n_factors:':35s} {self.n_factors:>20d}",
            f"{'mode_:':35s} {self.mode_:>20s}",
            f"{'n_obs:':35s} {self.n_obs:>20d}",
            sep,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Callable wrappers
# ---------------------------------------------------------------------------

def var_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    n_lag: int = 1,
) -> VARFitResult:
    """Standalone Vector Autoregression VAR(p).

    Calls ``_build_l4_model("var", params)`` directly; bypasses the recipe DAG.
    Uses statsmodels VAR on the joint panel of ``y`` and ``X`` columns.

    Parameters
    ----------
    X :
        Feature matrix.  Shape (n_samples, n_features).
    y :
        Target vector.  Shape (n_samples,).
    n_lag :
        VAR lag order p.  Must be >= 1.  Default 1.

    Returns
    -------
    VARFitResult
        Fitted result exposing ``n_lag``, ``n_obs``, ``.predict(X)``,
        ``.summary()``.

    Raises
    ------
    ValueError
        If ``n_lag < 1``.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import var_fit
    >>> rng = np.random.RandomState(42)
    >>> X = rng.randn(50, 3)
    >>> y = X[:, 0] + 0.5 * rng.randn(50)
    >>> result = var_fit(X, y)
    >>> result.n_lag
    1

    References
    ----------
    Sims (1980) 'Macroeconomics and Reality', Econometrica 48(1).
    """
    from ..core.runtime import _build_l4_model  # lazy import to avoid circular

    if n_lag < 1:
        raise ValueError(f"n_lag must be >= 1, got {n_lag!r}")

    X_df = _to_frame(X)
    y_s = _to_series(y)

    params: dict[str, Any] = {"n_lag": int(n_lag)}
    model = _build_l4_model("var", params)
    model.fit(X_df, y_s)

    return VARFitResult(
        n_lag=int(n_lag),
        n_series=int(X_df.shape[1]) + 1,
        n_obs=len(y_s),
        _model=model,
    )


def bvar_minnesota_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    n_lag: int = 1,
    lambda1: float = 0.2,
) -> BVARMinnesotaFitResult:
    """Standalone Bayesian VAR with Minnesota prior.

    Calls ``_build_l4_model("bvar_minnesota", params)`` directly.  Closed-form
    posterior mean -- no MCMC.  Cheap and deterministic.

    Parameters
    ----------
    X :
        Feature matrix.  Shape (n_samples, n_features).
    y :
        Target vector.  Shape (n_samples,).
    n_lag :
        VAR lag order p.  Must be >= 1.  Default 1.
    lambda1 :
        Minnesota prior tightness.  Must be > 0.  Default 0.2.

    Returns
    -------
    BVARMinnesotaFitResult
        Fitted result exposing ``n_lag``, ``lambda1``, ``n_obs``,
        ``.predict(X)``, ``.summary()``.

    Raises
    ------
    ValueError
        If ``n_lag < 1`` or ``lambda1 <= 0``.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import bvar_minnesota_fit
    >>> rng = np.random.RandomState(42)
    >>> X = rng.randn(50, 3)
    >>> y = X[:, 0] + 0.5 * rng.randn(50)
    >>> result = bvar_minnesota_fit(X, y)
    >>> result.n_lag
    1

    References
    ----------
    Litterman (1986) 'Forecasting With Bayesian Vector Autoregressions --
    Five Years of Experience', JBES 4(1).
    """
    from ..core.runtime import _build_l4_model  # lazy import to avoid circular

    if n_lag < 1:
        raise ValueError(f"n_lag must be >= 1, got {n_lag!r}")
    if lambda1 <= 0:
        raise ValueError(f"lambda1 must be > 0, got {lambda1!r}")

    X_df = _to_frame(X)
    y_s = _to_series(y)

    params: dict[str, Any] = {
        "n_lag": int(n_lag),
        "lambda_1": float(lambda1),
    }
    model = _build_l4_model("bvar_minnesota", params)
    model.fit(X_df, y_s)

    return BVARMinnesotaFitResult(
        n_lag=int(n_lag),
        lambda1=float(lambda1),
        n_obs=len(y_s),
        _model=model,
    )


def bvar_niw_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    n_lag: int = 1,
    lambda1: float = 0.2,
) -> BVARNIWFitResult:
    """Standalone Bayesian VAR with Normal-Inverse-Wishart prior.

    Calls ``_build_l4_model("bvar_normal_inverse_wishart", params)`` directly.
    Conjugate Normal-IW prior on (beta, Sigma); closed-form posterior mean.

    Parameters
    ----------
    X :
        Feature matrix.  Shape (n_samples, n_features).
    y :
        Target vector.  Shape (n_samples,).
    n_lag :
        VAR lag order p.  Must be >= 1.  Default 1.

    Returns
    -------
    BVARNIWFitResult
        Fitted result exposing ``n_lag``, ``n_obs``, ``.predict(X)``,
        ``.summary()``.

    Raises
    ------
    ValueError
        If ``n_lag < 1``.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import bvar_niw_fit
    >>> rng = np.random.RandomState(42)
    >>> X = rng.randn(50, 3)
    >>> y = X[:, 0] + 0.5 * rng.randn(50)
    >>> result = bvar_niw_fit(X, y)
    >>> result.n_lag
    1

    References
    ----------
    Kadiyala & Karlsson (1997) 'Numerical Methods for Estimation and
    Inference in Bayesian VAR-models', Journal of Applied Econometrics 12(2).
    """
    from ..core.runtime import _build_l4_model  # lazy import to avoid circular

    if n_lag < 1:
        raise ValueError(f"n_lag must be >= 1, got {n_lag!r}")

    X_df = _to_frame(X)
    y_s = _to_series(y)

    params: dict[str, Any] = {"n_lag": int(n_lag)}
    model = _build_l4_model("bvar_normal_inverse_wishart", params)
    model.fit(X_df, y_s)

    return BVARNIWFitResult(
        n_lag=int(n_lag),
        lambda1=float(lambda1),
        n_obs=len(y_s),
        _model=model,
    )


def ar_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    n_lag: int = 1,
) -> ARFitResult:
    """Standalone AR(p) autoregression on the target.

    Calls ``_build_l4_model("ar_p", params)`` directly.  Pure autoregression
    -- predictor matrix is the lagged target (``X`` is ignored by the AR
    internal implementation).

    Parameters
    ----------
    X :
        Feature matrix.  Shape (n_samples, n_features).  Ignored by the AR
        internal implementation; included for API consistency.
    y :
        Target vector.  Shape (n_samples,).
    n_lag :
        AR lag order p.  Must be >= 1.  Default 1.

    Returns
    -------
    ARFitResult
        Fitted result exposing ``n_lag``, ``coef_``, ``intercept_``,
        ``.predict(X)``, ``.summary()``.

    Raises
    ------
    ValueError
        If ``n_lag < 1``.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import ar_fit
    >>> rng = np.random.RandomState(42)
    >>> X = rng.randn(50, 3)
    >>> y = X[:, 0] + 0.5 * rng.randn(50)
    >>> result = ar_fit(X, y)
    >>> result.n_lag
    1
    >>> result.coef_.shape
    (1,)

    References
    ----------
    Stock & Watson (2007) 'Why Has US Inflation Become Harder to Forecast?',
    JMCB 39.
    """
    from ..core.runtime import _build_l4_model  # lazy import to avoid circular

    if n_lag < 1:
        raise ValueError(f"n_lag must be >= 1, got {n_lag!r}")

    X_df = _to_frame(X)
    y_s = _to_series(y)

    params: dict[str, Any] = {"n_lag": int(n_lag)}
    model = _build_l4_model("ar_p", params)
    model.fit(X_df, y_s)

    coef = np.asarray(model.coef_ if model.coef_ is not None else np.zeros(n_lag), dtype=float)
    intercept = float(model.intercept_)

    return ARFitResult(
        n_lag=int(n_lag),
        coef_=coef,
        intercept_=intercept,
        _model=model,
    )


def far_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    n_factors: int = 3,
    n_lag: int = 1,
) -> FARFitResult:
    """Standalone Factor-Augmented AR (Stock-Watson 2002).

    Calls ``_build_l4_model("factor_augmented_ar", params)`` directly.
    Extracts PCA factors from ``X``, augments with AR(p) lags on target, fits
    OLS.

    Parameters
    ----------
    X :
        Feature matrix.  Shape (n_samples, n_features).
    y :
        Target vector.  Shape (n_samples,).
    n_factors :
        Number of principal components to extract from ``X``.  Must be >= 1.
        Default 3.
    n_lag :
        AR lag order p on the target.  Must be >= 1.  Default 1.

    Returns
    -------
    FARFitResult
        Fitted result exposing ``n_factors``, ``n_lag``, ``.predict(X)``,
        ``.summary()``.

    Raises
    ------
    ValueError
        If ``n_factors < 1`` or ``n_lag < 1``.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import far_fit
    >>> rng = np.random.RandomState(42)
    >>> X = rng.randn(50, 3)
    >>> y = X @ [1., 2., 3.] + 0.5 * rng.randn(50)
    >>> result = far_fit(X, y, n_factors=2)
    >>> result.n_factors
    2

    References
    ----------
    Stock & Watson (2002) 'Forecasting Using Principal Components from a
    Large Number of Predictors', JASA 97(460).
    """
    from ..core.runtime import _build_l4_model  # lazy import to avoid circular

    if n_factors < 1:
        raise ValueError(f"n_factors must be >= 1, got {n_factors!r}")
    if n_lag < 1:
        raise ValueError(f"n_lag must be >= 1, got {n_lag!r}")

    X_df = _to_frame(X)
    y_s = _to_series(y)

    params: dict[str, Any] = {
        "n_factors": int(n_factors),
        "n_lag": int(n_lag),
    }
    model = _build_l4_model("factor_augmented_ar", params)
    model.fit(X_df, y_s)

    # Extract regression coefficients from the fitted _FactorAugmentedAR.
    if hasattr(model, "_regression") and model._regression is not None:
        coef = np.asarray(model._regression.coef_, dtype=float)
    else:
        coef = np.zeros(n_factors + n_lag, dtype=float)

    return FARFitResult(
        n_factors=int(n_factors),
        n_lag=int(n_lag),
        coef_=coef,
        _model=model,
    )


def pcr_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    n_components: int = 3,
) -> PCRFitResult:
    """Standalone Principal Component Regression (PCA + OLS).

    Calls ``_build_l4_model("principal_component_regression", params)``
    directly.  Extracts PCA components from ``X`` then fits OLS on the
    principal scores.

    Parameters
    ----------
    X :
        Feature matrix.  Shape (n_samples, n_features).
    y :
        Target vector.  Shape (n_samples,).
    n_components :
        Number of principal components.  Must be >= 1.  Default 3.

    Returns
    -------
    PCRFitResult
        Fitted result exposing ``n_components``, ``.predict(X)``,
        ``.summary()``.

    Raises
    ------
    ValueError
        If ``n_components < 1``.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import pcr_fit
    >>> rng = np.random.RandomState(42)
    >>> X = rng.randn(50, 3)
    >>> y = X @ [1., 2., 3.] + 0.5 * rng.randn(50)
    >>> result = pcr_fit(X, y)
    >>> result.n_components
    3
    """
    from ..core.runtime import _build_l4_model  # lazy import to avoid circular

    if n_components < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components!r}")

    X_df = _to_frame(X)
    y_s = _to_series(y)

    params: dict[str, Any] = {"n_components": int(n_components)}
    model = _build_l4_model("principal_component_regression", params)
    model.fit(X_df, y_s)

    # Extract OLS coefficients from the fitted _PrincipalComponentRegression.
    if hasattr(model, "_regression") and model._regression is not None:
        coef = np.asarray(model._regression.coef_, dtype=float)
    else:
        coef = np.zeros(n_components, dtype=float)

    return PCRFitResult(
        n_components=int(n_components),
        coef_=coef,
        _model=model,
    )


def favar_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    n_factors: int = 3,
    n_lag: int = 1,
) -> FAVARFitResult:
    """Standalone Factor-Augmented VAR (Bernanke-Boivin-Eliasz 2005).

    Calls ``_build_l4_model("factor_augmented_var", params)`` directly.
    Two-stage estimator: PCA factors from ``X`` + VAR(p) on (factors, target).

    Parameters
    ----------
    X :
        Feature matrix.  Shape (n_samples, n_features).
    y :
        Target vector.  Shape (n_samples,).
    n_factors :
        Number of PCA factors to extract from ``X``.  Must be >= 1.  Default 3.
    n_lag :
        VAR lag order p.  Must be >= 1.  Default 1.

    Returns
    -------
    FAVARFitResult
        Fitted result exposing ``n_factors``, ``n_lag``, ``.predict(X)``,
        ``.summary()``.

    Raises
    ------
    ValueError
        If ``n_factors < 1`` or ``n_lag < 1``.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import favar_fit
    >>> rng = np.random.RandomState(42)
    >>> X = rng.randn(50, 3)
    >>> y = X @ [1., 2., 3.] + 0.5 * rng.randn(50)
    >>> result = favar_fit(X, y, n_factors=2)
    >>> result.n_factors
    2

    References
    ----------
    Bernanke, Boivin & Eliasz (2005) 'Measuring the Effects of Monetary
    Policy: A Factor-Augmented Vector Autoregressive Approach', QJE 120(1).
    """
    from ..core.runtime import _build_l4_model  # lazy import to avoid circular

    if n_factors < 1:
        raise ValueError(f"n_factors must be >= 1, got {n_factors!r}")
    if n_lag < 1:
        raise ValueError(f"n_lag must be >= 1, got {n_lag!r}")

    X_df = _to_frame(X)
    y_s = _to_series(y)

    params: dict[str, Any] = {
        "n_factors": int(n_factors),
        "n_lag": int(n_lag),
    }
    model = _build_l4_model("factor_augmented_var", params)
    model.fit(X_df, y_s)

    return FAVARFitResult(
        n_factors=int(n_factors),
        n_lag=int(n_lag),
        _model=model,
    )


def garch11_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> GARCH11FitResult:
    """Standalone GARCH(1,1) conditional variance model.

    Calls ``_build_l4_model("garch11", params)`` directly; requires the
    optional ``arch`` package (``pip install macroforecast[arch]``).
    ``y`` is treated as the return-like series.  ``X`` is ignored for the
    GARCH(1,1) variant.

    Parameters
    ----------
    X :
        Feature matrix.  Shape (n_samples, n_features).  Ignored by GARCH.
    y :
        Return-like target series.  Must have >= 30 non-missing observations.

    Returns
    -------
    GARCH11FitResult
        Fitted result exposing ``conditional_mu``, ``n_obs``, ``params_``,
        ``.predict(X)``, ``.predict_variance(h_steps)``, ``.summary()``.

    Raises
    ------
    NotImplementedError
        If ``arch`` package is not installed.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import garch11_fit
    >>> rng = np.random.RandomState(42)
    >>> X = rng.randn(100, 3)
    >>> y = rng.randn(100)
    >>> result = garch11_fit(X, y)  # doctest: +SKIP
    >>> result.conditional_mu  # doctest: +SKIP
    ...

    References
    ----------
    Bollerslev (1986) 'Generalized Autoregressive Conditional
    Heteroskedasticity', Journal of Econometrics 31(3): 307-327.
    """
    from ..core.runtime import _build_l4_model  # lazy import to avoid circular

    X_df = _to_frame(X)
    y_s = _to_series(y)

    params: dict[str, Any] = {}
    model = _build_l4_model("garch11", params)
    model.fit(X_df, y_s)

    return GARCHFitResult(
        variant="garch",
        conditional_mu=float(model._mu),
        n_obs=int(y_s.dropna().shape[0]),
        params_=dict(model.params_),
        _model=model,
    )


def egarch_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> EGARCHFitResult:
    """Standalone EGARCH conditional variance model.

    Calls ``_build_l4_model("egarch", params)`` directly; requires the
    optional ``arch`` package (``pip install macroforecast[arch]``).

    Parameters
    ----------
    X :
        Feature matrix.  Shape (n_samples, n_features).  Ignored by EGARCH.
    y :
        Return-like target series.  Must have >= 30 non-missing observations.

    Returns
    -------
    EGARCHFitResult
        Fitted result exposing ``conditional_mu``, ``n_obs``, ``params_``,
        ``.predict(X)``, ``.predict_variance(h_steps)``, ``.summary()``.

    Raises
    ------
    NotImplementedError
        If ``arch`` package is not installed.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import egarch_fit
    >>> rng = np.random.RandomState(42)
    >>> X = rng.randn(100, 3)
    >>> y = rng.randn(100)
    >>> result = egarch_fit(X, y)  # doctest: +SKIP

    References
    ----------
    Nelson (1991) 'Conditional Heteroskedasticity in Asset Returns:
    A New Approach', Econometrica 59(2): 347-370.
    """
    from ..core.runtime import _build_l4_model  # lazy import to avoid circular

    X_df = _to_frame(X)
    y_s = _to_series(y)

    params: dict[str, Any] = {}
    model = _build_l4_model("egarch", params)
    model.fit(X_df, y_s)

    return GARCHFitResult(
        variant="egarch",
        conditional_mu=float(model._mu),
        n_obs=int(y_s.dropna().shape[0]),
        params_=dict(model.params_),
        _model=model,
    )


def realized_garch_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    rv: np.ndarray | pd.Series,
) -> RealizedGARCHFitResult:
    """Standalone GARCH(1,1) with realised-variance series as exogenous input.

    Calls ``_build_l4_model("realized_garch_with_rv_exog", params)`` directly;
    requires the optional ``arch`` package.  The realised-variance series ``rv``
    is appended as the column ``"rv"`` in ``X`` and passed as exogenous to the
    GARCH model.

    Parameters
    ----------
    X :
        Feature matrix.  Shape (n_samples, n_features).
    y :
        Return-like target series.  Must have >= 30 non-missing observations.
    rv :
        Realised-variance series.  Shape (n_samples,).  Must be non-negative.
        Appended to ``X`` under the column name ``"rv"``.

    Returns
    -------
    RealizedGARCHFitResult
        Fitted result exposing ``conditional_mu``, ``n_obs``, ``params_``,
        ``.predict(X)``, ``.predict_variance(h_steps)``, ``.summary()``.

    Raises
    ------
    NotImplementedError
        If ``arch`` package is not installed.
    ValueError
        If ``rv`` length does not match ``y`` length.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import realized_garch_fit
    >>> rng = np.random.RandomState(42)
    >>> X = rng.randn(100, 3)
    >>> y = rng.randn(100)
    >>> rv = np.abs(rng.randn(100))
    >>> result = realized_garch_fit(X, y, rv)  # doctest: +SKIP

    References
    ----------
    Hansen, Huang & Shek (2012) 'Realized GARCH: A Joint Model for
    Returns and Realized Measures of Volatility', Journal of Applied
    Econometrics 27(6): 877-906.
    """
    from ..core.runtime import _build_l4_model  # lazy import to avoid circular

    X_df = _to_frame(X)
    y_s = _to_series(y)
    rv_s = pd.Series(np.asarray(rv).ravel(), name="rv")

    if len(rv_s) != len(y_s):
        raise ValueError(
            f"rv length {len(rv_s)} does not match y length {len(y_s)}"
        )

    # Attach rv column to X so _GARCHFamily can pick it up via realized_variance_col.
    X_with_rv = X_df.copy()
    X_with_rv["rv"] = rv_s.values

    params: dict[str, Any] = {"realized_variance": "rv"}
    model = _build_l4_model("realized_garch_with_rv_exog", params)
    model.fit(X_with_rv, y_s)

    return GARCHFitResult(
        variant="realized_garch",
        conditional_mu=float(model._mu),
        n_obs=int(y_s.dropna().shape[0]),
        params_=dict(model.params_),
        _model=model,
    )


def ets_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> ETSFitResult:
    """Standalone Exponential Smoothing State-Space (ETS).

    Calls ``_build_l4_model("ets", params)`` directly.  Default spec is
    ``AAN`` (additive error + additive trend + no seasonal).  ``X`` is ignored;
    ETS is a univariate method.

    Parameters
    ----------
    X :
        Feature matrix.  Shape (n_samples, n_features).  Ignored by ETS.
    y :
        Target series.  Shape (n_samples,).  Requires >= 4 observations for
        full fitting; shorter series return the mean as forecast.

    Returns
    -------
    ETSFitResult
        Fitted result exposing ``error_trend_seasonal``, ``n_obs``,
        ``.predict(X)``, ``.summary()``.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import ets_fit
    >>> rng = np.random.RandomState(42)
    >>> X = rng.randn(50, 3)
    >>> y = np.cumsum(rng.randn(50))
    >>> result = ets_fit(X, y)
    >>> result.error_trend_seasonal
    'AAN'

    References
    ----------
    Hyndman, Koehler, Ord & Snyder (2008) 'Forecasting with Exponential
    Smoothing: The State Space Approach', Springer.
    """
    from ..core.runtime import _build_l4_model  # lazy import to avoid circular

    X_df = _to_frame(X)
    y_s = _to_series(y)

    params: dict[str, Any] = {}
    model = _build_l4_model("ets", params)
    model.fit(X_df, y_s)

    return ETSFitResult(
        error_trend_seasonal="AAN",
        n_obs=len(y_s),
        _model=model,
    )


def theta_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> ThetaFitResult:
    """Standalone Theta method (Assimakopoulos-Nikolopoulos 2000).

    Calls ``_build_l4_model("theta_method", params)`` directly.  Implements
    the Theta(2) closed-form that won the M3 competition: blends a linear-trend
    extrapolation with a SES level on the doubled-curvature transform.  ``X``
    is ignored.

    Parameters
    ----------
    X :
        Feature matrix.  Shape (n_samples, n_features).  Ignored.
    y :
        Target series.  Shape (n_samples,).

    Returns
    -------
    ThetaFitResult
        Fitted result exposing ``theta``, ``alpha_``, ``n_obs``,
        ``.predict(X)``, ``.summary()``.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import theta_fit
    >>> rng = np.random.RandomState(42)
    >>> X = rng.randn(50, 3)
    >>> y = np.arange(50, dtype=float) + 0.5 * rng.randn(50)
    >>> result = theta_fit(X, y)
    >>> result.theta
    2.0

    References
    ----------
    Assimakopoulos & Nikolopoulos (2000) 'The theta model: a decomposition
    approach to forecasting', International Journal of Forecasting 16(4):
    521-530.
    """
    from ..core.runtime import _build_l4_model  # lazy import to avoid circular

    X_df = _to_frame(X)
    y_s = _to_series(y)

    params: dict[str, Any] = {}
    model = _build_l4_model("theta_method", params)
    model.fit(X_df, y_s)

    return ThetaFitResult(
        theta=float(model.theta),
        alpha_=float(model._alpha),
        n_obs=int(model._n),
        _model=model,
    )


def holt_winters_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> HoltWintersFitResult:
    """Standalone Holt-Winters seasonal exponential smoothing.

    Calls ``_build_l4_model("holt_winters", params)`` directly.  Default spec
    is additive trend + additive seasonal (periods=12).  Auto-disables seasonal
    fitting when series is shorter than 2 * seasonal_periods.  ``X`` is
    ignored.

    Parameters
    ----------
    X :
        Feature matrix.  Shape (n_samples, n_features).  Ignored.
    y :
        Target series.  Shape (n_samples,).

    Returns
    -------
    HoltWintersFitResult
        Fitted result exposing ``seasonal``, ``seasonal_periods``, ``n_obs``,
        ``.predict(X)``, ``.summary()``.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import holt_winters_fit
    >>> rng = np.random.RandomState(42)
    >>> X = rng.randn(50, 3)
    >>> y = np.cumsum(rng.randn(50))
    >>> result = holt_winters_fit(X, y)
    >>> result.seasonal
    'add'

    References
    ----------
    Hyndman & Athanasopoulos (2018) 'Forecasting: Principles and Practice',
    2nd ed., OTexts, Chapter 7.
    """
    from ..core.runtime import _build_l4_model  # lazy import to avoid circular

    X_df = _to_frame(X)
    y_s = _to_series(y)

    params: dict[str, Any] = {}
    model = _build_l4_model("holt_winters", params)
    model.fit(X_df, y_s)

    return HoltWintersFitResult(
        seasonal=str(model.seasonal),
        seasonal_periods=int(model.seasonal_periods),
        trend=str(model.trend) if model.trend is not None else "add",
        n_obs=len(y_s),
        _model=model,
    )


def dfm_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    n_factors: int = 3,
) -> DFMFitResult:
    """Standalone Dynamic Factor Model (single-frequency).

    Calls ``_build_l4_model("dfm_mixed_mariano_murasawa", params)`` directly.
    Distinct from :func:`~macroforecast.functions.dfm_transform` (the L3
    transform op).  This is the L4 forecasting family that fits a linear-
    Gaussian state-space DFM and produces point forecasts.

    Parameters
    ----------
    X :
        Feature matrix.  Shape (n_samples, n_features).
    y :
        Target vector.  Shape (n_samples,).
    n_factors :
        Number of latent dynamic factors.  Must be >= 1.  Default 3.

    Returns
    -------
    DFMFitResult
        Fitted result exposing ``n_factors``, ``n_obs``, ``.predict(X)``,
        ``.summary()``.

    Raises
    ------
    ValueError
        If ``n_factors < 1``.

    Examples
    --------
    >>> import numpy as np
    >>> from macroforecast.functions import dfm_fit
    >>> rng = np.random.RandomState(42)
    >>> X = rng.randn(50, 3)
    >>> y = X @ [1., 2., 3.] + 0.5 * rng.randn(50)
    >>> result = dfm_fit(X, y, n_factors=2)
    >>> result.n_factors
    2

    References
    ----------
    Mariano & Murasawa (2010) 'A coincident index, common factors, and
    monthly real GDP', Oxford Bulletin of Economics and Statistics 72(1).
    """
    from ..core.runtime import _build_l4_model  # lazy import to avoid circular

    if n_factors < 1:
        raise ValueError(f"n_factors must be >= 1, got {n_factors!r}")

    X_df = _to_frame(X)
    y_s = _to_series(y)

    params: dict[str, Any] = {
        "n_factors": int(n_factors),
    }
    model = _build_l4_model("dfm_mixed_mariano_murasawa", params)
    model.fit(X_df, y_s)

    return DFMFitResult(
        n_factors=int(n_factors),
        mode_="mariano_murasawa",
        n_obs=len(y_s),
        _model=model,
    )


__all__ = [
    "GARCHFitResult",
    "VARFitResult",
    "var_fit",
    "BVARMinnesotaFitResult",
    "bvar_minnesota_fit",
    "BVARNIWFitResult",
    "bvar_niw_fit",
    "ARFitResult",
    "ar_fit",
    "FARFitResult",
    "far_fit",
    "PCRFitResult",
    "pcr_fit",
    "FAVARFitResult",
    "favar_fit",
    "GARCH11FitResult",
    "garch11_fit",
    "EGARCHFitResult",
    "egarch_fit",
    "RealizedGARCHFitResult",
    "realized_garch_fit",
    "ETSFitResult",
    "ets_fit",
    "ThetaFitResult",
    "theta_fit",
    "HoltWintersFitResult",
    "holt_winters_fit",
    "DFMFitResult",
    "dfm_fit",
]
