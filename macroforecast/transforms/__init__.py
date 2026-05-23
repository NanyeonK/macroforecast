"""Public transform function wrappers.

Exposes the Chow-Lin (1971) temporal disaggregation function as a standalone
public callable. Implements the canonical GLS algorithm directly; does not
delegate to the private runtime implementation in ``macroforecast.core.runtime``
(which remains a separate AR(0) helper for the recipe-path L2 preprocessing).

Functions
---------
- :func:`chow_lin_disaggregate` -- Chow-Lin (1971) canonical GLS disaggregation.

Usage::

    import pandas as pd
    from macroforecast.transforms import chow_lin_disaggregate

    # Quarterly GDP disaggregated to monthly using retail sales indicator.
    monthly = chow_lin_disaggregate(
        low_freq=quarterly_gdp,
        indicator_high_freq=monthly_retail_sales,
        aggregation='mean',
    )

Cycle 63.1 -- Chow-Lin (1971) canonical GLS implementation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Named constants for numerical safety
_RHO_GRID_LOW: float = 0.0
_RHO_GRID_HIGH: float = 0.98
_RHO_GRID_N: int = 50
_MIN_N_L: int = 3  # minimum low-frequency obs to run GLS (2 df after intercept+slope)
_CONSERVATION_ATOL: float = 1e-8


def chow_lin_disaggregate(
    low_freq: pd.Series,
    indicator_high_freq: pd.DataFrame | pd.Series,
    *,
    aggregation: str = "mean",
    rho: float | None = None,
    rho_method: str = "min_chi_squared",
) -> pd.Series:
    """Chow-Lin (1971) canonical GLS regression-based temporal disaggregation.

    Disaggregates a low-frequency series (e.g. quarterly) to a higher
    frequency (e.g. monthly) using a related high-frequency indicator via
    the Chow-Lin (1971) Generalized Least Squares method with AR(1) errors.

    The implementation follows Chow & Lin (1971) Steps 0-6:

    - Build aggregation matrix C (n_l x n_h) based on aggregation mode.
    - Build AR(1) variance matrix V_h (Toeplitz: V_h[i,j] = rho^|i-j|).
    - Form aggregated covariance Omega_l = C V_h C'.
    - Estimate GLS beta_hat solving the weighted system.
    - Disaggregate via BLUE correction: y_h = X_h beta_hat + V_h C' Omega_l^{-1} resid_l.
    - Optionally estimate rho via min_chi_squared or max_likelihood criteria.

    Parameters
    ----------
    low_freq : pd.Series
        Observed low-frequency series (e.g. quarterly). Must have a
        ``DatetimeIndex`` for temporal aggregation. Non-DatetimeIndex inputs
        receive a bfill/ffill fallback.
    indicator_high_freq : pd.DataFrame or pd.Series
        High-frequency indicator series (e.g. monthly). The output is aligned
        to this series's index. If a DataFrame is passed, the first column is
        used as the indicator.
    aggregation : str, optional
        Aggregation mode: ``'sum'`` or ``'mean'`` (default ``'mean'``). Controls
        how high-frequency periods aggregate to low-frequency observations.
        The aggregation matrix C is built accordingly:

        - ``'sum'``: C[i, i*m:(i+1)*m] = 1.0
        - ``'mean'``: C[i, i*m:(i+1)*m] = 1.0 / m
    rho : float or None, optional
        AR(1) autocorrelation parameter for the error process. Must be in
        the open interval (-1, 1) if provided. If ``None`` (default), rho is
        estimated from data using ``rho_method``.
    rho_method : str, optional
        Estimation method for rho when ``rho is None``. Accepted values:

        - ``'min_chi_squared'`` (default): minimize the chi-squared criterion
          from Chow & Lin (1971) over a grid of 50 rho values in [0, 0.98].
        - ``'max_likelihood'``: maximize the concentrated log-likelihood.
        - ``'fixed'``: treat rho as 0.0 (AR(0) / OLS special case).

    Returns
    -------
    pd.Series
        Disaggregated high-frequency series aligned with
        ``indicator_high_freq.index[:n_l*m]``. Index and name inherited
        from ``indicator_high_freq``. If ``n_l * m < len(indicator_high_freq)``,
        trailing periods with no low-frequency coverage are excluded (the
        series is trimmed to length ``n_l * m``).

    Raises
    ------
    ValueError
        If ``aggregation`` not in ``{'sum', 'mean'}``.
        If ``rho`` is provided and outside the open interval (-1, 1).
        If ``rho_method`` not in ``{'min_chi_squared', 'max_likelihood', 'fixed'}``.

    Notes
    -----
    This function implements the canonical GLS algorithm directly and is
    self-contained. It does not call the private
    :func:`~macroforecast.core.runtime._chow_lin_disaggregate` helper, which
    is retained separately for the recipe-path L2 preprocessing.

    Conservation property: for ``aggregation='sum'``, the disaggregated series
    satisfies ``result.resample(<low_freq>).sum() == low_freq`` to numerical
    tolerance (atol=1e-8). For ``aggregation='mean'``,
    ``result.resample(<low_freq>).mean() == low_freq`` holds similarly.

    The V_h AR(1) Toeplitz matrix is positive definite for rho in (-1, 1),
    guaranteeing invertibility. The rho estimation grid uses [0, 0.98] (not 1.0)
    to maintain positive definiteness at the boundary.

    References
    ----------
    Chow, G.C. and Lin, A.L. (1971) "Best Linear Unbiased Interpolation,
    Distribution, and Extrapolation of Time Series by Related Series."
    Review of Economics and Statistics 53(4): 372-375.
    doi:10.2307/1928739

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> idx_m = pd.date_range("2010-01-31", periods=36, freq="ME")
    >>> idx_q = pd.date_range("2010-03-31", periods=12, freq="QE")
    >>> rng = np.random.RandomState(0)
    >>> indicator = pd.Series(rng.randn(36), index=idx_m, name="ind")
    >>> ind_q = indicator.resample("QE").mean()
    >>> y_q = pd.Series(0.5 + 2.0 * ind_q.values + 0.1 * rng.randn(12),
    ...                 index=idx_q, name="y_q")
    >>> y_m = chow_lin_disaggregate(y_q, indicator)
    >>> len(y_m) == 36
    True
    """
    # ------------------------------------------------------------------
    # Step 0 — Input validation
    # ------------------------------------------------------------------
    if aggregation not in {"sum", "mean"}:
        raise ValueError(f"aggregation must be 'sum' or 'mean'; got {aggregation!r}")

    if rho is not None and not (-1 < rho < 1):
        raise ValueError(f"rho must be in the open interval (-1, 1); got {rho}")

    if rho_method not in {"min_chi_squared", "max_likelihood", "fixed"}:
        raise ValueError(
            f"rho_method must be 'min_chi_squared', 'max_likelihood', or 'fixed';"
            f" got {rho_method!r}"
        )

    # Normalize indicator_high_freq to a pd.Series
    if isinstance(indicator_high_freq, pd.DataFrame):
        indicator_high_freq = indicator_high_freq.iloc[:, 0]

    # Fallback for non-DatetimeIndex inputs
    if not isinstance(indicator_high_freq.index, pd.DatetimeIndex):
        fallback = low_freq.bfill().ffill()
        return fallback.reindex(indicator_high_freq.index)

    # ------------------------------------------------------------------
    # Step 1 — Temporal alignment and aggregation ratio inference
    # ------------------------------------------------------------------
    # Infer the low-frequency period from low_freq.index
    lf_index = low_freq.index
    hf_index = indicator_high_freq.index

    # Resample indicator to the low-frequency grid to find common index
    # Detect low-frequency offset from low_freq index
    try:
        lf_freq = pd.infer_freq(lf_index)
    except Exception:
        lf_freq = None

    if lf_freq is None:
        # Try QE as fallback (quarterly-to-monthly is the primary use case)
        lf_freq = "QE"

    # Resample indicator to low frequency using aggregation mode
    try:
        if aggregation == "sum":
            ind_lf = indicator_high_freq.resample(lf_freq).sum()
        else:
            ind_lf = indicator_high_freq.resample(lf_freq).mean()
    except Exception:
        # Final fallback: bfill/ffill
        fallback = low_freq.bfill().ffill()
        return fallback.reindex(indicator_high_freq.index)

    # Align low_freq and resampled indicator on common index (inner join)
    common_idx = lf_index.intersection(ind_lf.index)
    if len(common_idx) < _MIN_N_L:
        # Fewer than 3 aligned low-frequency observations — use fallback
        fallback = low_freq.bfill().ffill()
        return fallback.reindex(indicator_high_freq.index)

    # n_l: number of low-frequency periods after alignment
    n_l: int = len(common_idx)

    # y_l: observed low-frequency values as a numpy vector
    y_l_vec: np.ndarray = low_freq.reindex(common_idx).values.astype(float)

    # 1c. Infer m (aggregation ratio)
    n_h_full: int = len(indicator_high_freq)
    m: int = int(round(n_h_full / n_l))
    if m < 1:
        m = 1

    # Total high-frequency periods used (trimmed to n_l * m)
    n_h: int = n_l * m

    # ------------------------------------------------------------------
    # Step 2 — Build aggregation matrix C and design matrices X_h, X_l
    # ------------------------------------------------------------------
    # C: (n_l x n_h) aggregation matrix
    C: np.ndarray = np.zeros((n_l, n_h), dtype=float)
    weight: float = 1.0 if aggregation == "sum" else 1.0 / m
    for i in range(n_l):
        C[i, i * m : (i + 1) * m] = weight

    # X_h: (n_h, 2) design matrix — column 0 = ones, column 1 = indicator values
    ind_vals: np.ndarray = indicator_high_freq.iloc[:n_h].values.astype(float)
    X_h = np.column_stack([np.ones(n_h, dtype=float), ind_vals])

    # X_l: (n_l, 2) aggregated design matrix
    X_l: np.ndarray = C @ X_h

    # ------------------------------------------------------------------
    # Step 3 — Estimate rho if not provided
    # ------------------------------------------------------------------
    if rho is not None:
        # Use caller-supplied rho directly
        rho_final: float = rho
    elif rho_method == "fixed":
        # AR(0) / OLS special case
        rho_final = 0.0
    else:
        # Build rho grid over [0, 0.98] — upper bound keeps V_h positive definite
        rho_grid = np.linspace(_RHO_GRID_LOW, _RHO_GRID_HIGH, _RHO_GRID_N)

        best_rho: float | None = None
        if rho_method == "min_chi_squared":
            best_criterion: float = np.inf
            for rho_c in rho_grid:
                try:
                    criterion_val = _chi_squared_criterion(rho_c, n_h, C, X_l, y_l_vec, n_l)
                except np.linalg.LinAlgError:
                    continue
                if criterion_val < best_criterion:
                    best_criterion = criterion_val
                    best_rho = float(rho_c)
        else:
            # max_likelihood
            best_ll: float = -np.inf
            for rho_c in rho_grid:
                try:
                    ll_val = _log_likelihood(rho_c, n_h, C, X_l, y_l_vec, n_l)
                except np.linalg.LinAlgError:
                    continue
                if ll_val > best_ll:
                    best_ll = ll_val
                    best_rho = float(rho_c)

        rho_final = 0.0 if best_rho is None else best_rho

    # ------------------------------------------------------------------
    # Step 4 — GLS estimation of beta_hat with final rho
    # ------------------------------------------------------------------
    V_h = _ar1_covariance(rho_final, n_h)
    Omega_l: np.ndarray = C @ V_h @ C.T

    try:
        Omega_l_inv: np.ndarray = np.linalg.inv(Omega_l)
    except np.linalg.LinAlgError:
        Omega_l_inv = np.linalg.pinv(Omega_l)

    # GLS normal equations: (X_l' Omega_l^{-1} X_l) beta = X_l' Omega_l^{-1} y_l
    A = X_l.T @ Omega_l_inv @ X_l          # (2 x 2)
    b_rhs = X_l.T @ Omega_l_inv @ y_l_vec  # (2,)

    try:
        beta_hat: np.ndarray = np.linalg.solve(A, b_rhs)
    except np.linalg.LinAlgError:
        beta_hat, _, _, _ = np.linalg.lstsq(A, b_rhs, rcond=None)

    # ------------------------------------------------------------------
    # Step 5 — Disaggregation (BLUE correction)
    # ------------------------------------------------------------------
    # High-frequency fitted values from the regression
    y_fit_h: np.ndarray = X_h @ beta_hat                          # (n_h,)

    # Low-frequency GLS residual
    resid_l: np.ndarray = y_l_vec - C @ y_fit_h                   # (n_l,)

    # BLUE correction: distribute low-frequency residual back to high frequency
    # via the AR(1) covariance structure
    VhCt: np.ndarray = V_h @ C.T                                   # (n_h x n_l)
    correction: np.ndarray = VhCt @ Omega_l_inv @ resid_l          # (n_h,)

    # Final disaggregated series
    y_h: np.ndarray = y_fit_h + correction                         # (n_h,)

    # ------------------------------------------------------------------
    # Step 6 — Build and return pd.Series
    # ------------------------------------------------------------------
    out_index = indicator_high_freq.index[:n_h]
    out_name = indicator_high_freq.name if hasattr(indicator_high_freq, "name") else None
    return pd.Series(y_h, index=out_index, name=out_name)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _ar1_covariance(rho: float, n: int) -> np.ndarray:
    """Build the n x n AR(1) Toeplitz covariance matrix.

    V[i, j] = rho^|i-j| for i, j in 0..n-1.
    Positive definite for rho in (-1, 1).
    """
    lags = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    return rho ** lags


def _gls_beta(
    Omega_l_inv: np.ndarray,
    X_l: np.ndarray,
    y_l_vec: np.ndarray,
) -> np.ndarray:
    """Solve for GLS beta_hat given Omega_l^{-1}, X_l, and y_l."""
    A = X_l.T @ Omega_l_inv @ X_l
    b_rhs = X_l.T @ Omega_l_inv @ y_l_vec
    try:
        return np.linalg.solve(A, b_rhs)
    except np.linalg.LinAlgError:
        result, _, _, _ = np.linalg.lstsq(A, b_rhs, rcond=None)
        return result


def _chi_squared_criterion(
    rho_c: float,
    n_h: int,
    C: np.ndarray,
    X_l: np.ndarray,
    y_l_vec: np.ndarray,
    n_l: int,
) -> float:
    """Chow-Lin (1971) chi-squared criterion for a given rho candidate.

    criterion = y_l' Omega_l^{-1} M_l y_l
    where M_l = I - X_l (X_l' Omega_l^{-1} X_l)^{-1} X_l' Omega_l^{-1}
    is the GLS annihilator matrix.
    Raises np.linalg.LinAlgError on singular Omega_l.
    """
    V_h = _ar1_covariance(rho_c, n_h)
    Omega_l = C @ V_h @ C.T
    Omega_l_inv = np.linalg.inv(Omega_l)  # raises LinAlgError if singular

    # GLS annihilator matrix M_l = I - X_l (X_l' Omega_l^{-1} X_l)^{-1} X_l' Omega_l^{-1}
    A = X_l.T @ Omega_l_inv @ X_l          # (2 x 2)
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        A_inv = np.linalg.pinv(A)
    M_l = np.eye(n_l) - X_l @ A_inv @ X_l.T @ Omega_l_inv

    return float(y_l_vec @ Omega_l_inv @ M_l @ y_l_vec)


def _log_likelihood(
    rho_c: float,
    n_h: int,
    C: np.ndarray,
    X_l: np.ndarray,
    y_l_vec: np.ndarray,
    n_l: int,
) -> float:
    """Concentrated Gaussian log-likelihood for a given rho candidate.

    ll = -0.5 * (n_l * log(2*pi) + log|Omega_l| + e' Omega_l^{-1} e)
    where e = y_l - X_l beta_hat (GLS residual).
    Raises np.linalg.LinAlgError on singular Omega_l.
    """
    V_h = _ar1_covariance(rho_c, n_h)
    Omega_l = C @ V_h @ C.T
    Omega_l_inv = np.linalg.inv(Omega_l)  # raises LinAlgError if singular

    # Compute sign-safe log-determinant
    sign, log_det = np.linalg.slogdet(Omega_l)
    if sign <= 0:
        raise np.linalg.LinAlgError("Omega_l not positive definite")

    beta_hat = _gls_beta(Omega_l_inv, X_l, y_l_vec)
    e = y_l_vec - X_l @ beta_hat

    ll = -0.5 * (n_l * np.log(2 * np.pi) + log_det + e @ Omega_l_inv @ e)
    return float(ll)


__all__ = ["chow_lin_disaggregate"]
