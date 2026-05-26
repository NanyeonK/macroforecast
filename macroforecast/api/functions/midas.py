"""Standalone MIDAS family fit callables.

Exposes four MIDAS model fit functions:
- ``midas_almon_fit``        -- Almon polynomial MIDAS (Ghysels et al. 2004)
- ``midas_beta_fit``         -- Beta polynomial MIDAS (Ghysels et al. 2007)
- ``midas_step_fit``         -- Step-function MIDAS (Foroni et al. 2015)
- ``unrestricted_midas_fit`` -- U-MIDAS OLS (Foroni et al. 2015)

Each callable builds the corresponding private runtime class directly and
returns a frozen FitResult exposing ``coef_``, ``intercept_``, and a
``.predict(X)`` method. The result is bit-exact with recipe-based MIDAS
when using the same parameter values.

Cycle 63 -- L4 MIDAS family standalone-ization (4 ops).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Shared FitResult for MIDAS family
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MidasFitResult:
    """Result of any MIDAS fit callable.

    Attributes
    ----------
    coef_ :
        Final weight vector applied to high-frequency lags (shape varies by
        model: for Almon/Beta it is the normalized polynomial weights;
        for Step it is the group-average weights; for U-MIDAS it is the
        OLS coefficient vector on the lag design matrix).
    intercept_ :
        Fitted intercept scalar.
    model :
        Descriptor string: one of "midas_almon", "midas_beta", "midas_step",
        "dfm_unrestricted_midas".
    _model :
        Internal fitted estimator instance. Not part of the public contract.
    """

    coef_: np.ndarray
    intercept_: float
    family: str
    _model: Any

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return predictions for new data.

        Parameters
        ----------
        X :
            Feature matrix. Accepts numpy arrays or DataFrames. For MIDAS
            models with ``freq_ratio > 1``, X should be the high-frequency
            DataFrame (the model applies lag-stacking internally).

        Returns
        -------
        np.ndarray
            1-D float array of predictions.
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
        return np.asarray(self._model.predict(X), dtype=float)

    def summary(self) -> str:
        """Return a human-readable text summary of the MIDAS fit.

        Returns
        -------
        str
            Table showing model family, intercept, and weight vector.
        """
        k = len(self.coef_)
        sep = "=" * 78
        dash = "-" * 78
        title = {
            "midas_almon": "MIDAS-Almon Results",
            "midas_beta": "MIDAS-Beta Results",
            "midas_step": "MIDAS-Step Results",
            "dfm_unrestricted_midas": "U-MIDAS Results",
        }.get(self.family, "MIDAS Results")
        lines = [
            sep,
            f"{title:^78}",
            sep,
            f"{'Family:':35s} {self.family:>20s}",
            f"{'No. Weight Parameters:':35s} {k:>20d}",
            sep,
            f"{'':30s} {'weight':>12s}",
            dash,
            f"{'intercept':30s} {self.intercept_:>12.6f}",
        ]
        for i, w in enumerate(self.coef_):
            lines.append(f"{'w' + str(i):30s} {w:>12.6f}")
        lines.append(sep)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _prep_xy(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Convert X/y to DataFrame/Series with default column names."""
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    if isinstance(y, np.ndarray):
        y = pd.Series(y.ravel(), name="y")
    return X, y


def _extract_midas_coef(model: Any, family: str) -> tuple[np.ndarray, float]:
    """Extract fitted weight vector and intercept from a MIDAS model."""
    # Almon / Beta models expose _w_hat (normalized polynomial weights)
    if hasattr(model, "_w_hat") and model._w_hat is not None:
        coef = np.asarray(model._w_hat, dtype=float)
    # Step model may expose _step_weights
    elif hasattr(model, "_step_weights") and model._step_weights is not None:
        coef = np.asarray(model._step_weights, dtype=float)
    # U-MIDAS exposes _coef (OLS vector)
    elif hasattr(model, "_coef") and model._coef is not None:
        coef = np.asarray(model._coef, dtype=float)
    else:
        coef = np.zeros(1, dtype=float)

    intercept = float(getattr(model, "_intercept", 0.0))
    return coef, intercept


# ---------------------------------------------------------------------------
# Callable wrappers
# ---------------------------------------------------------------------------

def midas_almon_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    freq_ratio: int = 1,
    n_lags_high: int = 12,
    polynomial_order: int = 2,
    sum_to_one: bool = True,
    max_iter: int = 200,
    n_starts: int = 5,
    random_state: int = 0,
) -> MidasFitResult:
    """Standalone Almon-polynomial MIDAS regression (Ghysels et al. 2004).

    Fits a MIDAS regression with Almon exponential distributed-lag weights
    estimated by Nelder-Mead NLS with multi-start. Calls the
    ``_MidasAlmonModel`` class from ``macroforecast.core.runtime`` directly;
    bit-exact with recipe-based MIDAS-Almon.

    Parameters
    ----------
    X :
        Feature matrix. Shape (n_samples, n_features). When ``freq_ratio > 1``,
        X should contain the high-frequency columns; the model applies
        lag-stacking internally. When ``freq_ratio == 1`` (default), X is
        already a low-frequency-aligned design matrix.
    y :
        Target vector. Shape (n_samples,).
    freq_ratio : int
        High-frequency periods per low-frequency period (m). Default 1.
    n_lags_high : int
        Number of high-frequency lags K. Default 12.
    polynomial_order : int
        Almon polynomial degree Q; weight parameters = Q + 1. Default 2.
    sum_to_one : bool
        Normalize Almon weights to sum to one. Default True.
    max_iter : int
        Max Nelder-Mead iterations per start. Default 200.
    n_starts : int
        Number of NLS multi-starts. Default 5.
    random_state : int
        RNG seed for perturbed NLS starts. Default 0.

    Returns
    -------
    MidasFitResult
        Fitted result exposing ``coef_`` (Almon weights), ``intercept_``,
        ``family``, and a ``.predict(X)`` method.

    References
    ----------
    Ghysels, Santa-Clara, Valkanov (2004) "The MIDAS Touch." Working paper.
    """
    from macroforecast.core.runtime import _MidasAlmonModel

    X, y = _prep_xy(X, y)
    model = _MidasAlmonModel(
        freq_ratio=freq_ratio,
        n_lags_high=n_lags_high,
        polynomial_order=polynomial_order,
        sum_to_one=sum_to_one,
        max_iter=max_iter,
        n_starts=n_starts,
        random_state=random_state,
    )
    model.fit(X, y)
    coef, intercept = _extract_midas_coef(model, "midas_almon")
    return MidasFitResult(
        coef_=coef,
        intercept_=intercept,
        family="midas_almon",
        _model=model,
    )


def midas_beta_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    freq_ratio: int = 1,
    n_lags_high: int = 12,
    sum_to_one: bool = True,
    max_iter: int = 200,
    n_starts: int = 5,
    random_state: int = 0,
) -> MidasFitResult:
    """Standalone Beta-polynomial MIDAS regression (Ghysels-Sinko-Valkanov 2007).

    Fits a MIDAS regression with two-parameter Beta function distributed-lag
    weights. Estimation via Nelder-Mead NLS with multi-start. Bit-exact with
    recipe-based MIDAS-Beta.

    Parameters
    ----------
    X :
        Feature matrix.
    y :
        Target vector.
    freq_ratio : int
        High-frequency periods per low-frequency period. Default 1.
    n_lags_high : int
        Number of high-frequency lags K. Default 12.
    sum_to_one : bool
        Normalize Beta weights to sum to one. Default True.
    max_iter : int
        Max Nelder-Mead iterations per start. Default 200.
    n_starts : int
        Number of NLS multi-starts. Default 5.
    random_state : int
        RNG seed. Default 0.

    Returns
    -------
    MidasFitResult
        Fitted result exposing ``coef_`` (Beta weights), ``intercept_``,
        ``family="midas_beta"``, and a ``.predict(X)`` method.

    References
    ----------
    Ghysels, Sinko, Valkanov (2007) "MIDAS Regressions: Further Results
    and New Directions." Econometric Reviews 26(1).
    """
    from macroforecast.core.runtime import _MidasBetaModel

    X, y = _prep_xy(X, y)
    model = _MidasBetaModel(
        freq_ratio=freq_ratio,
        n_lags_high=n_lags_high,
        sum_to_one=sum_to_one,
        max_iter=max_iter,
        n_starts=n_starts,
        random_state=random_state,
    )
    model.fit(X, y)
    coef, intercept = _extract_midas_coef(model, "midas_beta")
    return MidasFitResult(
        coef_=coef,
        intercept_=intercept,
        family="midas_beta",
        _model=model,
    )


def midas_step_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    freq_ratio: int = 1,
    n_lags_high: int = 12,
    n_steps: int | None = None,
) -> MidasFitResult:
    """Standalone Step-function MIDAS regression (Foroni et al. 2015).

    Groups high-frequency lags into equal-weight step-function blocks and
    estimates via OLS on the block-averaged design matrix. Bit-exact with
    recipe-based MIDAS-Step.

    Parameters
    ----------
    X :
        Feature matrix.
    y :
        Target vector.
    freq_ratio : int
        High-frequency periods per low-frequency period. Default 1.
    n_lags_high : int
        Number of high-frequency lags K. Default 12.
    n_steps : int or None
        Number of step-function blocks. Defaults to ``freq_ratio``.

    Returns
    -------
    MidasFitResult
        Fitted result exposing ``coef_`` (step weights), ``intercept_``,
        ``family="midas_step"``, and a ``.predict(X)`` method.

    References
    ----------
    Foroni, Marcellino, Schumacher (2015) JRSS-A 178(1) 57-82.
    """
    from macroforecast.core.runtime import _MidasStepModel

    X, y = _prep_xy(X, y)
    n_steps_resolved = n_steps if n_steps is not None else max(1, freq_ratio)
    model = _MidasStepModel(
        freq_ratio=freq_ratio,
        n_lags_high=n_lags_high,
        n_steps=n_steps_resolved,
    )
    model.fit(X, y)
    coef, intercept = _extract_midas_coef(model, "midas_step")
    return MidasFitResult(
        coef_=coef,
        intercept_=intercept,
        family="midas_step",
        _model=model,
    )


def unrestricted_midas_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    freq_ratio: int = 1,
    n_lags_high: int | str = "bic",
    include_y_lag: bool = False,
    random_state: int = 0,
) -> MidasFitResult:
    """Standalone U-MIDAS (Unrestricted MIDAS) regression (Foroni et al. 2015).

    Direct OLS on all high-frequency lags without a parametric weight
    polynomial. Lag order K can be selected automatically by BIC.
    Bit-exact with recipe-based dfm_unrestricted_midas.

    Parameters
    ----------
    X :
        Feature matrix.
    y :
        Target vector.
    freq_ratio : int
        High-frequency periods per low-frequency period. Default 1.
    n_lags_high : int or "bic"
        Number of high-frequency lags K, or "bic" for automatic BIC
        selection. Default "bic".
    include_y_lag : bool
        Include one lag of the target as a regressor. Default False.
    random_state : int
        RNG seed (reserved; U-MIDAS OLS is deterministic). Default 0.

    Returns
    -------
    MidasFitResult
        Fitted result exposing ``coef_`` (OLS coefficients on the lag
        design matrix), ``intercept_``, ``family="dfm_unrestricted_midas"``,
        and a ``.predict(X)`` method.

    References
    ----------
    Foroni, Marcellino, Schumacher (2015) JRSS-A 178(1) 57-82.
    """
    from macroforecast.core.runtime import _UnrestrictedMidasModel

    X, y = _prep_xy(X, y)
    model = _UnrestrictedMidasModel(
        freq_ratio=freq_ratio,
        n_lags_high=n_lags_high,
        include_y_lag=include_y_lag,
        random_state=random_state,
    )
    model.fit(X, y)
    coef, intercept = _extract_midas_coef(model, "dfm_unrestricted_midas")
    return MidasFitResult(
        coef_=coef,
        intercept_=intercept,
        family="dfm_unrestricted_midas",
        _model=model,
    )


__all__ = [
    "MidasFitResult",
    "midas_almon_fit",
    "midas_beta_fit",
    "midas_step_fit",
    "unrestricted_midas_fit",
]
