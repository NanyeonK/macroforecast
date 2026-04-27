from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence

import numpy as np


MIDASR_RESTRICTED_CONTRACT_VERSION = "midasr_restricted_direct_v1"
MIDASR_NEALMON_CONTRACT_VERSION = "midasr_nealmon_direct_v1"
MIDASR_REFERENCE_PACKAGE = "midasr"
MIDASR_REFERENCE_FUNCTION = "midas_r"
MIDASR_SUPPORTED_WEIGHT_FAMILIES = ("nealmon", "almonp", "nbeta", "genexp", "harstep")
MIDASR_WEIGHT_REFERENCE_FUNCTIONS = {
    "nealmon": "midas_r + nealmon",
    "almonp": "midas_r + almonp",
    "nbeta": "midas_r + nbeta",
    "genexp": "midas_r + genexp",
    "harstep": "midas_r + harstep",
}
MIDASR_WEIGHT_BASIS_NAMES = {
    "nealmon": "midasr_nealmon_normalized_exponential_almon",
    "almonp": "midasr_almonp_raw_polynomial",
    "nbeta": "midasr_nbeta_normalized_beta",
    "genexp": "midasr_genexp_generalized_exponential",
    "harstep": "midasr_harstep_har_rv_step",
}


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


def normalize_midasr_weight_family(value: str) -> str:
    family = str(value).strip().lower().replace("-", "_")
    if family not in MIDASR_SUPPORTED_WEIGHT_FAMILIES:
        allowed = ", ".join(MIDASR_SUPPORTED_WEIGHT_FAMILIES)
        raise ValueError(f"unsupported midasr_weight_family={value!r}; expected one of: {allowed}")
    return family


def midasr_weight_param_width(weight_family: str, degree: int) -> int:
    family = normalize_midasr_weight_family(weight_family)
    degree = int(degree)
    if family == "nealmon":
        if degree < 1:
            raise ValueError("degree must be >= 1 for nealmon")
        return degree + 1
    if family == "almonp":
        if degree < 0:
            raise ValueError("degree must be >= 0 for almonp")
        return degree + 1
    if family in {"nbeta", "harstep"}:
        return 3
    if family == "genexp":
        return 4
    raise AssertionError(f"unhandled midasr_weight_family={family!r}")


def _require_param_width(params: Sequence[float], *, expected: int, family: str) -> np.ndarray:
    p = np.asarray(params, dtype=float).reshape(-1)
    if p.size != expected:
        raise ValueError(f"{family} params must contain exactly {expected} values")
    return p


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


def nbeta_weights(params: Sequence[float], lag_count: int) -> np.ndarray:
    """R midasr nbeta-compatible normalized beta lag weights."""

    d = _as_positive_int(lag_count, name="lag_count")
    p = _require_param_width(params, expected=3, family="nbeta")
    if d == 1:
        return np.asarray([float(p[0])], dtype=float)
    eps = np.finfo(float).eps
    xi = (np.arange(1, d + 1, dtype=float) - 1.0) / float(d - 1)
    xi[0] += eps
    xi[-1] -= eps
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        beta_shape = (xi ** (float(p[1]) - 1.0)) * ((1.0 - xi) ** (float(p[2]) - 1.0))
    total = float(np.sum(beta_shape))
    if not math.isfinite(total) or total <= eps:
        return np.zeros(d, dtype=float)
    return float(p[0]) * beta_shape / total


def genexp_weights(params: Sequence[float], lag_count: int) -> np.ndarray:
    """R midasr genexp-compatible generalized exponential lag weights."""

    d = _as_positive_int(lag_count, name="lag_count")
    p = _require_param_width(params, expected=4, family="genexp")
    positions = (np.arange(1, d + 1, dtype=float) - 1.0) / 100.0
    exponent = float(p[2]) * positions + float(p[3]) * (positions**2)
    return (float(p[0]) + float(p[1]) * positions) * np.exp(np.clip(exponent, -700.0, 700.0))


def harstep_weights(params: Sequence[float], lag_count: int) -> np.ndarray:
    """R midasr harstep-compatible HAR(3)-RV lag weights."""

    d = _as_positive_int(lag_count, name="lag_count")
    if d != 20:
        raise ValueError("harstep requires lag_count=20, matching R midasr HAR(3)-RV weights")
    p = _require_param_width(params, expected=3, family="harstep")
    weights = np.zeros(20, dtype=float)
    weights[0] = float(p[0]) + float(p[1]) / 5.0 + float(p[2]) / 20.0
    weights[1:5] = float(p[1]) / 5.0 + float(p[2]) / 20.0
    weights[5:20] = float(p[2]) / 20.0
    return weights


def midasr_weights(weight_family: str, params: Sequence[float], lag_count: int) -> np.ndarray:
    family = normalize_midasr_weight_family(weight_family)
    if family == "nealmon":
        return nealmon_weights(params, lag_count)
    if family == "almonp":
        return almonp_weights(params, lag_count)
    if family == "nbeta":
        return nbeta_weights(params, lag_count)
    if family == "genexp":
        return genexp_weights(params, lag_count)
    return harstep_weights(params, lag_count)


def fit_midasr_restricted_direct(
    lag_tensor: np.ndarray,
    y_train: Sequence[float],
    pred_lag_tensor: np.ndarray,
    *,
    weight_family: str = "nealmon",
    degree: int = 2,
    max_nfev: int = 500,
) -> MidasrNealmonFitResult:
    """Fit a restricted MIDAS direct forecast with an R midasr weight family.

    lag_tensor has shape (n_obs, n_terms, n_lags). pred_lag_tensor accepts
    either (n_terms, n_lags) or (1, n_terms, n_lags).
    """

    family = normalize_midasr_weight_family(weight_family)
    try:
        from scipy.optimize import least_squares
    except ModuleNotFoundError as exc:  # pragma: no cover - scipy is required by pyproject
        raise RuntimeError("scipy is required for model_family='midasr'") from exc

    X = np.asarray(lag_tensor, dtype=float)
    if X.ndim != 3:
        raise ValueError("lag_tensor must be a 3-D array with shape (n_obs, n_terms, n_lags)")
    y = np.asarray(y_train, dtype=float).reshape(-1)
    if X.shape[0] != y.shape[0]:
        raise ValueError("lag_tensor row count must match y_train")
    if X.shape[0] < 2:
        raise ValueError("midasr requires at least two training rows")
    term_count = int(X.shape[1])
    lag_count = int(X.shape[2])
    if term_count < 1 or lag_count < 1:
        raise ValueError("lag_tensor must include at least one term and one lag")
    param_width = midasr_weight_param_width(family, int(degree))

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
        raise ValueError("midasr has fewer than two finite training rows")
    X = X[finite_rows]
    y = y[finite_rows]
    if not np.isfinite(X_pred).all():
        raise ValueError("pred_lag_tensor contains non-finite values")

    start = np.zeros(1 + term_count * param_width, dtype=float)
    start[0] = float(np.mean(y))
    if term_count:
        y_scale = float(np.std(y)) if y.size > 1 else 1.0
        if not math.isfinite(y_scale) or y_scale <= 0.0:
            y_scale = 1.0
        for term_idx in range(term_count):
            offset = 1 + term_idx * param_width
            if family == "nealmon":
                start[offset] = y_scale / max(term_count, 1)
            elif family == "almonp":
                start[offset] = y_scale / max(term_count * lag_count, 1)
            elif family == "nbeta":
                start[offset] = y_scale / max(term_count, 1)
                start[offset + 1] = 1.0
                start[offset + 2] = 1.0
            elif family == "genexp":
                start[offset] = y_scale / max(term_count * lag_count, 1)
            else:
                scale = y_scale / max(term_count * 3, 1)
                start[offset : offset + 3] = scale

    lower = np.full(start.shape, -np.inf, dtype=float)
    upper = np.full(start.shape, np.inf, dtype=float)
    if family == "nbeta":
        for term_idx in range(term_count):
            offset = 1 + term_idx * param_width
            lower[offset + 1] = np.finfo(float).eps
            lower[offset + 2] = np.finfo(float).eps

    def _predict(params: np.ndarray, tensor: np.ndarray) -> np.ndarray:
        out = np.full(tensor.shape[0], float(params[0]), dtype=float)
        for term_idx in range(term_count):
            offset = 1 + term_idx * param_width
            weights = midasr_weights(family, params[offset : offset + param_width], lag_count)
            out += tensor[:, term_idx, :] @ weights
        return out

    def _residuals(params: np.ndarray) -> np.ndarray:
        return _predict(params, X) - y

    opt = least_squares(
        _residuals,
        start,
        bounds=(lower, upper),
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
        weights_by_term.append(tuple(float(v) for v in midasr_weights(family, term_params, lag_count)))
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

    return fit_midasr_restricted_direct(
        lag_tensor,
        y_train,
        pred_lag_tensor,
        weight_family="nealmon",
        degree=degree,
        max_nfev=max_nfev,
    )
