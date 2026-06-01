from __future__ import annotations

from typing import Any

from macroforecast.models.types import ModelFit
from macroforecast.models.utils import fit_estimator, resolve_xy


def kernel_ridge(
    X: Any,
    y: Any | None = None,
    *,
    alpha: float = 1.0,
    kernel: str = "linear",
    gamma: float | None = None,
    degree: int = 3,
    coef0: float = 1.0,
    **kwargs: Any,
) -> ModelFit:
    """Fit sklearn kernel ridge regression."""

    from sklearn.kernel_ridge import KernelRidge

    # R comparison is intentionally skipped for this callable. This is a thin
    # sklearn backend wrapper, so the algorithmic contract is sklearn
    # KernelRidge behavior plus macroforecast ModelFit metadata, diagnostics,
    # and feature-matrix IO. There is no package-native objective here to align
    # against an R implementation.
    params = {
        "alpha": float(alpha),
        "kernel": str(kernel),
        "gamma": gamma,
        "degree": int(degree),
        "coef0": float(coef0),
        "implementation_note": (
            "Thin sklearn.kernel_ridge.KernelRidge wrapper; macroforecast owns "
            "the pandas X/y contract, ModelFit metadata, and diagnostics."
        ),
        **kwargs,
    }
    estimator_params = {key: value for key, value in params.items() if key != "implementation_note"}
    return fit_estimator(
        KernelRidge(**estimator_params),
        X,
        y,
        model="kernel_ridge",
        metadata=params,
    )


def knn(
    X: Any,
    y: Any | None = None,
    *,
    n_neighbors: int = 5,
    weights: str = "uniform",
    metric: str = "minkowski",
    p: int = 2,
    **kwargs: Any,
) -> ModelFit:
    """Fit sklearn k-nearest-neighbor regression."""

    from sklearn.neighbors import KNeighborsRegressor

    # R comparison is intentionally skipped for this callable. This is a thin
    # sklearn backend wrapper, so the algorithmic contract is sklearn
    # KNeighborsRegressor behavior plus macroforecast ModelFit metadata,
    # diagnostics, and feature-matrix IO. The small-window n_neighbors
    # resolution below is macroforecast API hardening, not an R-alignment step.
    frame, target = resolve_xy(X, y)
    requested_n_neighbors = max(1, int(n_neighbors))
    resolved_n_neighbors = min(requested_n_neighbors, len(frame))
    params = {
        "n_neighbors": resolved_n_neighbors,
        "weights": str(weights),
        "metric": str(metric),
        "p": int(p),
        "implementation_note": (
            "Thin sklearn.neighbors.KNeighborsRegressor wrapper; macroforecast "
            "owns small-window n_neighbors resolution, ModelFit metadata, and "
            "diagnostics."
        ),
        **kwargs,
    }
    metadata: dict[str, Any] = dict(params)
    if resolved_n_neighbors != requested_n_neighbors:
        metadata["requested_n_neighbors"] = requested_n_neighbors
    estimator_params = {key: value for key, value in params.items() if key != "implementation_note"}
    return fit_estimator(
        KNeighborsRegressor(**estimator_params),
        frame,
        target,
        model="knn",
        metadata=metadata,
    )


__all__ = ["kernel_ridge", "knn"]
