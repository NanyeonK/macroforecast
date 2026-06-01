from __future__ import annotations

from typing import Any

from macroforecast.models.types import ModelFit
from macroforecast.models.utils import fit_estimator


def _feature_kernel(kernel: str) -> str:
    value = str(kernel)
    if value == "precomputed":
        raise ValueError(
            "kernel='precomputed' is not supported by macroforecast's "
            "feature-matrix ModelFit contract"
        )
    return value


def svr(
    X: Any,
    y: Any | None = None,
    *,
    kernel: str = "rbf",
    C: float = 1.0,
    epsilon: float = 0.1,
    gamma: str | float = "scale",
    degree: int = 3,
    coef0: float = 0.0,
    shrinking: bool = True,
    tol: float = 1e-3,
    cache_size: float = 200.0,
    max_iter: int = -1,
    **kwargs: Any,
) -> ModelFit:
    """Fit support-vector regression."""

    from sklearn.svm import SVR

    # R comparison is intentionally skipped for this callable. This is a thin
    # sklearn backend wrapper, so the algorithmic contract is sklearn SVR
    # behavior plus macroforecast ModelFit metadata, diagnostics, and
    # feature-matrix IO. kernel="precomputed" is blocked because it violates
    # that feature-matrix contract, not because of an R-source mismatch.
    params = {
        "kernel": _feature_kernel(kernel),
        "C": float(C),
        "epsilon": float(epsilon),
        "gamma": gamma,
        "degree": int(degree),
        "coef0": float(coef0),
        "shrinking": bool(shrinking),
        "tol": float(tol),
        "cache_size": float(cache_size),
        "max_iter": int(max_iter),
        **kwargs,
    }
    return fit_estimator(SVR(**params), X, y, model="svr", metadata=params)


def linear_svr(
    X: Any,
    y: Any | None = None,
    *,
    C: float = 1.0,
    epsilon: float = 0.0,
    loss: str = "epsilon_insensitive",
    tol: float = 1e-4,
    max_iter: int = 10000,
    random_state: int | None = 0,
    **kwargs: Any,
) -> ModelFit:
    """Fit linear support-vector regression."""

    from sklearn.svm import LinearSVR

    # R comparison is intentionally skipped for this callable. This is a thin
    # sklearn backend wrapper, so the algorithmic contract is sklearn LinearSVR
    # behavior plus macroforecast ModelFit metadata, diagnostics, and
    # feature-matrix IO. No package-native support-vector objective is
    # implemented here.
    params = {
        "C": float(C),
        "epsilon": float(epsilon),
        "loss": str(loss),
        "tol": float(tol),
        "max_iter": int(max_iter),
        "random_state": None if random_state is None else int(random_state),
        **kwargs,
    }
    return fit_estimator(LinearSVR(**params), X, y, model="linear_svr", metadata=params)


def nu_svr(
    X: Any,
    y: Any | None = None,
    *,
    kernel: str = "rbf",
    C: float = 1.0,
    nu: float = 0.5,
    gamma: str | float = "scale",
    degree: int = 3,
    coef0: float = 0.0,
    shrinking: bool = True,
    tol: float = 1e-3,
    cache_size: float = 200.0,
    max_iter: int = -1,
    **kwargs: Any,
) -> ModelFit:
    """Fit nu-support-vector regression."""

    from sklearn.svm import NuSVR

    # R comparison is intentionally skipped for this callable. This is a thin
    # sklearn backend wrapper, so the algorithmic contract is sklearn NuSVR
    # behavior plus macroforecast ModelFit metadata, diagnostics, and
    # feature-matrix IO. kernel="precomputed" is blocked because it violates
    # that feature-matrix contract, not because of an R-source mismatch.
    params = {
        "kernel": _feature_kernel(kernel),
        "C": float(C),
        "nu": float(nu),
        "gamma": gamma,
        "degree": int(degree),
        "coef0": float(coef0),
        "shrinking": bool(shrinking),
        "tol": float(tol),
        "cache_size": float(cache_size),
        "max_iter": int(max_iter),
        **kwargs,
    }
    return fit_estimator(NuSVR(**params), X, y, model="nu_svr", metadata=params)


__all__ = ["linear_svr", "nu_svr", "svr"]
