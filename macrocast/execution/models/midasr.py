from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence

import numpy as np


MIDASR_NEALMON_CONTRACT_VERSION = "midasr_nealmon_direct_v1"
MIDASR_REFERENCE_PACKAGE = "midasr"
MIDASR_REFERENCE_FUNCTION = "midas_r + nealmon"


@dataclass(frozen=True)
class MidasrNealmonFitResult:
    prediction: float
    intercept: float
    params_by_term: tuple[tuple[float, ...], ...]
    weights_by_term: tuple[tuple[float, ...], ...]
    residual_ss: float
    success: bool
    message: str
    nfev: int


def _as_positive_int(value: int, *, name: str) -> int:
    out = int(value)
    if out < 1:
        raise ValueError(f"{name} must be >= 1")
    return out


def almonp_weights(params: Sequence[float], lag_count: int) -> np.ndarray:
    """R midasr almonp-compatible polynomial lag weights."""

    d = _as_positive_int(lag_count, name="lag_count")
    p = np.asarray(params, dtype=float).reshape(-1)
    if p.size == 0:
        raise ValueError("params must contain at least an intercept")
    positions = np.arange(1, d + 1, dtype=float)
    weights = np.full(d, float(p[0]), dtype=float)
    for degree, coef in enumerate(p[1:], start=1):
        weights += float(coef) * (positions**degree)
    return weights


def nealmon_weights(params: Sequence[float], lag_count: int) -> np.ndarray:
    """R midasr nealmon-compatible normalized exponential Almon weights."""

    d = _as_positive_int(lag_count, name="lag_count")
    p = np.asarray(params, dtype=float).reshape(-1)
    if p.size < 2:
        raise ValueError("nealmon params must contain delta and at least one lambda")
    delta = float(p[0])
    lambdas = p[1:]
    positions = np.arange(1, d + 1, dtype=float)
    eta = np.zeros(d, dtype=float)
    for degree, coef in enumerate(lambdas, start=1):
        eta += float(coef) * (positions**degree)
    eta -= float(np.max(eta))
    exp_eta = np.exp(eta)
    denom = float(np.sum(exp_eta))
    if not math.isfinite(denom) or denom <= 0.0:
        return np.full(d, delta / d, dtype=float)
    return delta * exp_eta / denom


def fit_midasr_nealmon_direct(
    lag_tensor: np.ndarray,
    y_train: Sequence[float],
    pred_lag_tensor: np.ndarray,
    *,
    degree: int = 2,
    max_nfev: int = 500,
) -> MidasrNealmonFitResult:
    """Fit a restricted MIDAS direct forecast with nealmon weights.

    lag_tensor has shape (n_obs, n_terms, n_lags). pred_lag_tensor accepts
    either (n_terms, n_lags) or (1, n_terms, n_lags).
    """

    try:
        from scipy.optimize import least_squares
    except ModuleNotFoundError as exc:  # pragma: no cover - scipy is required by pyproject
        raise RuntimeError("scipy is required for model_family='midasr_nealmon'") from exc

    X = np.asarray(lag_tensor, dtype=float)
    if X.ndim != 3:
        raise ValueError("lag_tensor must be a 3-D array with shape (n_obs, n_terms, n_lags)")
    y = np.asarray(y_train, dtype=float).reshape(-1)
    if X.shape[0] != y.shape[0]:
        raise ValueError("lag_tensor row count must match y_train")
    if X.shape[0] < 2:
        raise ValueError("midasr_nealmon requires at least two training rows")
    term_count = int(X.shape[1])
    lag_count = int(X.shape[2])
    if term_count < 1 or lag_count < 1:
        raise ValueError("lag_tensor must include at least one term and one lag")
    degree = int(degree)
    if degree < 1:
        raise ValueError("degree must be >= 1 for nealmon")

    X_pred = np.asarray(pred_lag_tensor, dtype=float)
    if X_pred.ndim == 3:
        if X_pred.shape[0] != 1:
            raise ValueError("pred_lag_tensor with 3 dimensions must have one prediction row")
        X_pred = X_pred[0]
    if X_pred.shape != (term_count, lag_count):
        raise ValueError(
            "pred_lag_tensor must have shape (n_terms, n_lags); "
            f"got {X_pred.shape}, expected {(term_count, lag_count)}"
        )

    finite_rows = np.isfinite(y)
    finite_rows &= np.isfinite(X).all(axis=(1, 2))
    if int(finite_rows.sum()) < 2:
        raise ValueError("midasr_nealmon has fewer than two finite training rows")
    X = X[finite_rows]
    y = y[finite_rows]
    if not np.isfinite(X_pred).all():
        raise ValueError("pred_lag_tensor contains non-finite values")

    param_width = degree + 1
    start = np.zeros(1 + term_count * param_width, dtype=float)
    start[0] = float(np.mean(y))
    if term_count:
        y_scale = float(np.std(y)) if y.size > 1 else 1.0
        if not math.isfinite(y_scale) or y_scale <= 0.0:
            y_scale = 1.0
        for term_idx in range(term_count):
            offset = 1 + term_idx * param_width
            start[offset] = y_scale / max(term_count, 1)

    def _predict(params: np.ndarray, tensor: np.ndarray) -> np.ndarray:
        out = np.full(tensor.shape[0], float(params[0]), dtype=float)
        for term_idx in range(term_count):
            offset = 1 + term_idx * param_width
            weights = nealmon_weights(params[offset : offset + param_width], lag_count)
            out += tensor[:, term_idx, :] @ weights
        return out

    def _residuals(params: np.ndarray) -> np.ndarray:
        return _predict(params, X) - y

    opt = least_squares(
        _residuals,
        start,
        max_nfev=int(max_nfev),
        method="trf",
    )
    params = np.asarray(opt.x, dtype=float)
    pred = float(_predict(params, X_pred.reshape(1, term_count, lag_count))[0])
    resid = _residuals(params)
    params_by_term = []
    weights_by_term = []
    for term_idx in range(term_count):
        offset = 1 + term_idx * param_width
        term_params = tuple(float(v) for v in params[offset : offset + param_width])
        params_by_term.append(term_params)
        weights_by_term.append(tuple(float(v) for v in nealmon_weights(term_params, lag_count)))
    return MidasrNealmonFitResult(
        prediction=pred,
        intercept=float(params[0]),
        params_by_term=tuple(params_by_term),
        weights_by_term=tuple(weights_by_term),
        residual_ss=float(np.dot(resid, resid)),
        success=bool(opt.success),
        message=str(opt.message),
        nfev=int(opt.nfev),
    )

