"""Shared forecast/estimator combination weight solvers.

Pure numpy/scipy weight estimators used by both ``forecasting.combination``
(post-hoc forecast combination) and, going forward, ``model_ensemble`` (so the
constrained-least-squares / NNLS logic is not duplicated). Each solver takes a
design matrix ``F`` (rows = observations, columns = component forecasts) and a
target ``y`` and returns a weight vector (and, where relevant, an intercept).
"""
from __future__ import annotations

import numpy as np


def min_variance_weights(errors: np.ndarray) -> np.ndarray:
    """Bates-Granger minimum-variance weights from an error matrix.

    ``errors`` has rows = observations, columns = models (e_i = f_i - y). Returns
    ``w = Sigma^{-1} 1 / (1' Sigma^{-1} 1)`` where ``Sigma`` is the error
    second-moment (covariance) matrix. Weights may be negative (classic
    Bates-Granger); falls back to equal weights if ``Sigma`` is singular.
    """
    e = np.asarray(errors, dtype=float)
    n_models = e.shape[1]
    if e.shape[0] < 2:
        return np.full(n_models, 1.0 / n_models)
    sigma = np.cov(e, rowvar=False)
    sigma = np.atleast_2d(sigma)
    ones = np.ones(n_models)
    try:
        inv_ones = np.linalg.solve(sigma + 1e-12 * np.eye(n_models), ones)
    except np.linalg.LinAlgError:
        return np.full(n_models, 1.0 / n_models)
    denom = float(ones @ inv_ones)
    if not np.isfinite(denom) or abs(denom) < 1e-15:
        return np.full(n_models, 1.0 / n_models)
    return inv_ones / denom


def regression_weights(
    F: np.ndarray, y: np.ndarray, *, intercept: bool = True, sum_to_one: bool = False
) -> tuple[np.ndarray, float]:
    """Granger-Ramanathan regression weights.

    Returns ``(weights, intercept)`` from regressing ``y`` on the forecasts.
    ``intercept`` adds a constant (method A); ``sum_to_one`` imposes the
    weights-sum-to-one constraint with no intercept (method C).
    """
    F = np.asarray(F, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n_models = F.shape[1]
    if sum_to_one:
        # Constrained OLS: min ||y - F w||^2 s.t. 1'w = 1, closed form via the
        # equality-constrained least squares normal equations.
        G = F.T @ F
        try:
            g_inv = np.linalg.inv(G + 1e-12 * np.eye(n_models))
        except np.linalg.LinAlgError:
            return np.full(n_models, 1.0 / n_models), 0.0
        ones = np.ones(n_models)
        b = F.T @ y
        lam = (float(ones @ g_inv @ b) - 1.0) / float(ones @ g_inv @ ones)
        w = g_inv @ (b - lam * ones)
        return w, 0.0
    design = np.column_stack([np.ones(len(F)), F]) if intercept else F
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    if intercept:
        return coef[1:], float(coef[0])
    return coef, 0.0


def constrained_ls_weights(F: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Non-negative weights summing to one minimising ``||y - F w||^2`` (NNLS+simplex).

    Solved by SLSQP over the probability simplex; this is the shared kernel that
    super-learner-style stacking also needs.
    """
    from scipy.optimize import minimize

    F = np.asarray(F, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n_models = F.shape[1]
    w0 = np.full(n_models, 1.0 / n_models)

    def objective(w: np.ndarray) -> float:
        resid = y - F @ w
        return float(resid @ resid)

    def gradient(w: np.ndarray) -> np.ndarray:
        return -2.0 * F.T @ (y - F @ w)

    constraints = ({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},)
    bounds = [(0.0, 1.0)] * n_models
    result = minimize(
        objective, w0, jac=gradient, bounds=bounds, constraints=constraints,
        method="SLSQP", options={"maxiter": 200, "ftol": 1e-10},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        return w0
    w = np.clip(result.x, 0.0, None)
    total = float(w.sum())
    return w / total if total > 0 else w0


def eigenvector_weights(errors: np.ndarray) -> np.ndarray:
    """Eigenvector (PC) combination weights (Hsiao-Wan).

    Weights are the eigenvector of the error second-moment matrix with the
    smallest eigenvalue, normalised to sum to one.
    """
    e = np.asarray(errors, dtype=float)
    n_models = e.shape[1]
    if e.shape[0] < 2:
        return np.full(n_models, 1.0 / n_models)
    moment = (e.T @ e) / e.shape[0]
    try:
        values, vectors = np.linalg.eigh(moment)
    except np.linalg.LinAlgError:
        return np.full(n_models, 1.0 / n_models)
    v = vectors[:, int(np.argmin(values))]
    denom = float(np.sum(v))
    if abs(denom) < 1e-15:
        return np.full(n_models, 1.0 / n_models)
    return v / denom


def shrink_weights(weights: np.ndarray, shrinkage: float) -> np.ndarray:
    """Shrink estimated weights toward equal weights (combination-puzzle fix)."""
    w = np.asarray(weights, dtype=float)
    n = len(w)
    lam = float(np.clip(shrinkage, 0.0, 1.0))
    return (1.0 - lam) * w + lam * (1.0 / n)


def regularized_weights(
    F: np.ndarray, y: np.ndarray, *, penalty: str = "ridge", alpha: float = 1.0,
    intercept: bool = True,
) -> tuple[np.ndarray, float]:
    """Ridge/Lasso-penalised regression combination weights.

    Useful when combining many forecasts (high-dimensional weight estimation),
    where unpenalised Granger-Ramanathan overfits. ``penalty`` is ``"ridge"`` or
    ``"lasso"``; ``alpha`` is the regularisation strength.
    """
    from sklearn.linear_model import Lasso, Ridge

    F = np.asarray(F, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    key = str(penalty).lower()
    if key == "ridge":
        model = Ridge(alpha=float(alpha), fit_intercept=bool(intercept))
    elif key == "lasso":
        model = Lasso(alpha=float(alpha), fit_intercept=bool(intercept), max_iter=5000)
    else:
        raise ValueError("penalty must be 'ridge' or 'lasso'")
    model.fit(F, y)
    return np.asarray(model.coef_, dtype=float).ravel(), float(getattr(model, "intercept_", 0.0))
