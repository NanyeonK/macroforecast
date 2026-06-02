from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

from macroforecast.feature_engineering.aggregation import align_reference_weights
from macroforecast.models.types import ModelFit
from macroforecast.models.utils import fit_estimator, resolve_xy

AggregationSpace = Literal["component", "rank"]
AggregationPenalty = Literal["ridge", "target_shrinkage", "fused_difference"]

ASSEMBLAGE_SOURCE = (
    "Goulet Coulombe, Klieber, Barrette, and Goebel, "
    "Maximally Forward-Looking Core Inflation; R package assemblage."
)


class SupervisedAggregationRegressor:
    """Constrained supervised aggregation estimator.

    This is the generic, inflation-free primitive behind Albacore-style
    assemblage regression. It learns nonnegative, optionally simplex or
    mean-matched weights that map a component panel to a future aggregate
    target. Source cue: R ``assemblage`` functions ``nonneg.ridge``,
    ``nonneg.ridge.sum1``, ``nonneg.ridge.mean``, and ``nonneg.ridge.meanD``.
    """

    def __init__(
        self,
        *,
        space: AggregationSpace = "component",
        penalty: AggregationPenalty = "ridge",
        alpha: float = 1.0,
        reference_weights: Mapping[str, float] | Sequence[float] | pd.Series | None = None,
        nonneg: bool = True,
        simplex: bool = False,
        mean_match: bool = False,
        difference_order: int = 1,
        fit_intercept: bool = False,
        penalty_scale: Literal["none", "feature_std"] = "feature_std",
        max_iter: int = 1000,
        tol: float = 1e-9,
    ) -> None:
        if float(alpha) < 0.0:
            raise ValueError("alpha must be non-negative")
        if int(difference_order) < 1:
            raise ValueError("difference_order must be at least 1")
        self.space = _normalize_space(space)
        self.penalty = _normalize_penalty(penalty)
        self.alpha = float(alpha)
        self.reference_weights = reference_weights
        self.nonneg = bool(nonneg)
        self.simplex = bool(simplex)
        self.mean_match = bool(mean_match)
        self.difference_order = int(difference_order)
        self.fit_intercept = bool(fit_intercept)
        self.penalty_scale = str(penalty_scale)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.feature_names_in_: np.ndarray | None = None
        self.diagnostic_feature_names_: tuple[str, ...] | None = None
        self.reference_weights_: pd.Series | None = None
        self.penalty_scale_: np.ndarray | None = None
        self.difference_matrix_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.x_mean_: np.ndarray | None = None
        self.y_mean_: float = 0.0
        self.solver_success_: bool = False
        self.solver_message_: str = ""
        self.weights_: pd.Series | None = None
        self.rank_weight_curve_: pd.DataFrame | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SupervisedAggregationRegressor":
        from scipy.optimize import minimize

        frame = X.astype(float).copy()
        target = pd.Series(y, index=frame.index).astype(float)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        work_frame = _rank_frame(frame) if self.space == "rank" else frame
        self.diagnostic_feature_names_ = tuple(str(column) for column in work_frame.columns)
        values = _filled_float_values(work_frame)
        y_values = target.to_numpy(dtype=float)
        self.reference_weights_ = _reference_weight_series(
            self.reference_weights,
            columns=work_frame.columns,
            default="uniform" if self.simplex else "zero",
        )
        self.penalty_scale_ = _penalty_scale(values, mode=self.penalty_scale)
        self.difference_matrix_ = _difference_matrix(values.shape[1], self.difference_order)
        if self.fit_intercept and not (self.simplex or self.mean_match):
            self.x_mean_ = values.mean(axis=0)
            self.y_mean_ = float(y_values.mean()) if y_values.size else 0.0
            x_work = values - self.x_mean_
            y_work = y_values - self.y_mean_
        else:
            self.x_mean_ = np.zeros(values.shape[1], dtype=float)
            self.y_mean_ = 0.0
            x_work = values
            y_work = y_values

        def objective(coef: np.ndarray) -> float:
            # R alignment notes:
            # - nonneg.ridge: glmnet ridge with lower.limits=0.
            # - nonneg.ridge.sum1: CVXR objective
            #   SSE + lambda * sum(sd(x) * (b - w0)^2), b >= 0, sum(b)=1.
            # - nonneg.ridge.mean: SSE + lambda * sum(sd(x) * b^2),
            #   b >= 0, mean(X b)=mean(y).
            # - nonneg.ridge.meanD: SSE + lambda * sum(diff(sd(x) * b)^2),
            #   b >= 0, mean(X b)=mean(y).
            # Macroforecast implements the same objective families with one
            # selected alpha. Cross-validation over alpha belongs to
            # model_selection/forecasting rather than this low-level fit.
            residual = y_work - x_work @ coef
            return float(residual @ residual + self.alpha * self._penalty_value(coef))

        constraints: list[dict[str, Any]] = []
        if self.simplex:
            constraints.append({"type": "eq", "fun": lambda coef: float(np.sum(coef) - 1.0)})
        if self.mean_match:
            x_mean = values.mean(axis=0)
            y_mean = float(y_values.mean()) if y_values.size else 0.0
            constraints.append({"type": "eq", "fun": lambda coef: float(x_mean @ coef - y_mean)})
        bounds = [(0.0, None) if self.nonneg else (None, None)] * values.shape[1]
        start = _feasible_start(
            x_work,
            y_work,
            reference=self.reference_weights_.to_numpy(dtype=float),
            simplex=self.simplex,
            mean_match=self.mean_match,
            nonneg=self.nonneg,
            alpha=max(self.alpha, 1e-12),
            values=values,
            y_values=y_values,
        )
        result = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )
        if not result.success:
            raise RuntimeError(f"supervised aggregation solver failed: {result.message}")
        self.coef_ = np.asarray(result.x, dtype=float)
        self.intercept_ = (
            self.y_mean_ - float(self.x_mean_ @ self.coef_)
            if self.fit_intercept and not (self.simplex or self.mean_match)
            else 0.0
        )
        self.solver_success_ = bool(result.success)
        self.solver_message_ = str(result.message)
        self.weights_ = pd.Series(
            self.coef_,
            index=list(self.diagnostic_feature_names_),
            name="weight",
        )
        if self.space == "rank":
            n_features = len(self.weights_)
            self.rank_weight_curve_ = pd.DataFrame(
                {
                    "rank": np.arange(1, n_features + 1),
                    "percentile": np.arange(1, n_features + 1) / n_features,
                    "weight": self.coef_,
                }
            )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.coef_ is None or self.feature_names_in_ is None:
            return np.zeros(len(X), dtype=float)
        frame = X.reindex(columns=list(self.feature_names_in_), fill_value=0.0).astype(float)
        work_frame = _rank_frame(frame) if self.space == "rank" else frame
        values = _filled_float_values(work_frame)
        return values @ self.coef_ + self.intercept_

    def _penalty_value(self, coef: np.ndarray) -> float:
        scale = self.penalty_scale_
        if scale is None:
            scale = np.ones_like(coef, dtype=float)
        if self.penalty == "target_shrinkage":
            if self.reference_weights_ is None:
                target = np.zeros_like(coef, dtype=float)
            else:
                target = self.reference_weights_.to_numpy(dtype=float)
            diff = coef - target
            return float(np.sum(scale * diff * diff))
        if self.penalty == "fused_difference":
            if self.difference_matrix_ is None:
                return 0.0
            smooth = self.difference_matrix_ @ (scale * coef)
            return float(smooth @ smooth)
        return float(np.sum(scale * coef * coef))


def solve_nonnegative_ridge(
    X: Any,
    y: Any,
    *,
    alpha: float = 1.0,
    penalty_scale: Literal["none", "feature_std"] = "feature_std",
) -> pd.Series:
    """Return nonnegative ridge weights from the assemblage solver primitive."""

    return _solve_weights(
        X,
        y,
        alpha=alpha,
        penalty="ridge",
        nonneg=True,
        simplex=False,
        mean_match=False,
        penalty_scale=penalty_scale,
    )


def solve_simplex_ridge(
    X: Any,
    y: Any,
    *,
    alpha: float = 1.0,
    penalty_scale: Literal["none", "feature_std"] = "feature_std",
) -> pd.Series:
    """Return nonnegative sum-to-one ridge weights."""

    return _solve_weights(
        X,
        y,
        alpha=alpha,
        penalty="ridge",
        nonneg=True,
        simplex=True,
        mean_match=False,
        penalty_scale=penalty_scale,
    )


def solve_target_shrinkage_ridge(
    X: Any,
    y: Any,
    *,
    reference_weights: Mapping[str, float] | Sequence[float] | pd.Series,
    alpha: float = 1.0,
    simplex: bool = True,
    penalty_scale: Literal["none", "feature_std"] = "feature_std",
) -> pd.Series:
    """Return weights for Albacore-style target-shrinkage ridge."""

    return _solve_weights(
        X,
        y,
        alpha=alpha,
        penalty="target_shrinkage",
        reference_weights=reference_weights,
        nonneg=True,
        simplex=simplex,
        mean_match=False,
        penalty_scale=penalty_scale,
    )


def solve_mean_aligned_ridge(
    X: Any,
    y: Any,
    *,
    alpha: float = 1.0,
    penalty_scale: Literal["none", "feature_std"] = "feature_std",
) -> pd.Series:
    """Return nonnegative weights constrained to match target mean."""

    return _solve_weights(
        X,
        y,
        alpha=alpha,
        penalty="ridge",
        nonneg=True,
        simplex=False,
        mean_match=True,
        penalty_scale=penalty_scale,
    )


def solve_fused_difference_ridge(
    X: Any,
    y: Any,
    *,
    alpha: float = 1.0,
    difference_order: int = 1,
    mean_match: bool = True,
    penalty_scale: Literal["none", "feature_std"] = "feature_std",
) -> pd.Series:
    """Return nonnegative fused-difference weights for rank aggregation."""

    return _solve_weights(
        X,
        y,
        alpha=alpha,
        penalty="fused_difference",
        nonneg=True,
        simplex=False,
        mean_match=mean_match,
        difference_order=difference_order,
        penalty_scale=penalty_scale,
    )


def supervised_aggregation(
    X: Any,
    y: Any | None = None,
    *,
    space: AggregationSpace = "component",
    penalty: AggregationPenalty = "ridge",
    alpha: float = 1.0,
    reference_weights: Mapping[str, float] | Sequence[float] | pd.Series | None = None,
    nonneg: bool = True,
    simplex: bool = False,
    mean_match: bool = False,
    difference_order: int = 1,
    fit_intercept: bool = False,
    penalty_scale: Literal["none", "feature_std"] = "feature_std",
    max_iter: int = 1000,
    tol: float = 1e-9,
) -> ModelFit:
    """Fit a generic supervised component-to-aggregate weighting model."""

    params = {
        "space": _normalize_space(space),
        "penalty": _normalize_penalty(penalty),
        "alpha": float(alpha),
        "reference_weights": reference_weights,
        "nonneg": bool(nonneg),
        "simplex": bool(simplex),
        "mean_match": bool(mean_match),
        "difference_order": int(difference_order),
        "fit_intercept": bool(fit_intercept),
        "penalty_scale": str(penalty_scale),
        "max_iter": int(max_iter),
        "tol": float(tol),
        "source_reference": ASSEMBLAGE_SOURCE,
    }
    return fit_estimator(
        SupervisedAggregationRegressor(
            space=params["space"],  # type: ignore[arg-type]
            penalty=params["penalty"],  # type: ignore[arg-type]
            alpha=float(alpha),
            reference_weights=reference_weights,
            nonneg=bool(nonneg),
            simplex=bool(simplex),
            mean_match=bool(mean_match),
            difference_order=int(difference_order),
            fit_intercept=bool(fit_intercept),
            penalty_scale=str(penalty_scale),  # type: ignore[arg-type]
            max_iter=int(max_iter),
            tol=float(tol),
        ),
        X,
        y,
        model="supervised_aggregation",
        metadata=params,
    )


def component_aggregation(
    X: Any,
    y: Any | None = None,
    *,
    alpha: float = 1.0,
    reference_weights: Mapping[str, float] | Sequence[float] | pd.Series | None = None,
    penalty: AggregationPenalty | None = None,
    simplex: bool = True,
    nonneg: bool = True,
    penalty_scale: Literal["none", "feature_std"] = "feature_std",
    max_iter: int = 1000,
    tol: float = 1e-9,
) -> ModelFit:
    """Fit component-space supervised aggregation weights."""

    penalty_value = penalty or ("target_shrinkage" if reference_weights is not None else "ridge")
    fit = supervised_aggregation(
        X,
        y,
        space="component",
        penalty=penalty_value,
        alpha=alpha,
        reference_weights=reference_weights,
        nonneg=nonneg,
        simplex=simplex,
        mean_match=False,
        fit_intercept=False,
        penalty_scale=penalty_scale,
        max_iter=max_iter,
        tol=tol,
    )
    fit.model = "component_aggregation"
    fit.metadata["wrapper"] = "component_aggregation"
    return fit


def rank_aggregation(
    X: Any,
    y: Any | None = None,
    *,
    alpha: float = 1.0,
    penalty: AggregationPenalty = "fused_difference",
    mean_match: bool = True,
    nonneg: bool = True,
    difference_order: int = 1,
    penalty_scale: Literal["none", "feature_std"] = "feature_std",
    max_iter: int = 1000,
    tol: float = 1e-9,
) -> ModelFit:
    """Fit rank-space supervised aggregation weights."""

    fit = supervised_aggregation(
        X,
        y,
        space="rank",
        penalty=penalty,
        alpha=alpha,
        reference_weights=None,
        nonneg=nonneg,
        simplex=False,
        mean_match=mean_match,
        difference_order=difference_order,
        fit_intercept=False,
        penalty_scale=penalty_scale,
        max_iter=max_iter,
        tol=tol,
    )
    fit.model = "rank_aggregation"
    fit.metadata["wrapper"] = "rank_aggregation"
    return fit


def assemblage_regression(
    X: Any,
    y: Any | None = None,
    *,
    space: AggregationSpace = "component",
    alpha: float = 1.0,
    reference_weights: Mapping[str, float] | Sequence[float] | pd.Series | None = None,
    penalty: AggregationPenalty | None = None,
    max_iter: int = 1000,
    tol: float = 1e-9,
) -> ModelFit:
    """Fit the generic assemblage regression family.

    ``space="component"`` corresponds to the component basket logic used by
    Albacorecomps. ``space="rank"`` corresponds to the sorted order-statistic
    logic used by Albacoreranks. The callable remains generic and does not
    require inflation data.
    """

    if _normalize_space(space) == "rank":
        fit = rank_aggregation(
            X,
            y,
            alpha=alpha,
            penalty=penalty or "fused_difference",
            mean_match=True,
            max_iter=max_iter,
            tol=tol,
        )
    else:
        fit = component_aggregation(
        X,
        y,
        alpha=alpha,
        reference_weights=reference_weights,
        penalty=penalty,
        simplex=True,
        max_iter=max_iter,
        tol=tol,
    )
    fit.model = "assemblage_regression"
    fit.metadata["wrapper"] = "assemblage_regression"
    return fit


def albacore_components(
    X: Any,
    y: Any | None = None,
    *,
    reference_weights: Mapping[str, float] | Sequence[float] | pd.Series | None = None,
    alpha: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-9,
) -> ModelFit:
    """Fit the inflation-specific component-space Albacore wrapper."""

    fit = component_aggregation(
        X,
        y,
        alpha=alpha,
        reference_weights=reference_weights,
        penalty="target_shrinkage",
        simplex=True,
        nonneg=True,
        penalty_scale="feature_std",
        max_iter=max_iter,
        tol=tol,
    )
    fit.model = "albacore_components"
    fit.metadata["wrapper"] = "albacore_components"
    fit.metadata["paper_specific"] = "Albacorecomps"
    return fit


def albacore_ranks(
    X: Any,
    y: Any | None = None,
    *,
    alpha: float = 1.0,
    difference_order: int = 1,
    max_iter: int = 1000,
    tol: float = 1e-9,
) -> ModelFit:
    """Fit the inflation-specific rank-space Albacore wrapper."""

    fit = rank_aggregation(
        X,
        y,
        alpha=alpha,
        penalty="fused_difference",
        mean_match=True,
        nonneg=True,
        difference_order=difference_order,
        penalty_scale="feature_std",
        max_iter=max_iter,
        tol=tol,
    )
    fit.model = "albacore_ranks"
    fit.metadata["wrapper"] = "albacore_ranks"
    fit.metadata["paper_specific"] = "Albacoreranks"
    return fit


def _solve_weights(
    X: Any,
    y: Any,
    *,
    alpha: float,
    penalty: AggregationPenalty,
    reference_weights: Mapping[str, float] | Sequence[float] | pd.Series | None = None,
    nonneg: bool,
    simplex: bool,
    mean_match: bool,
    difference_order: int = 1,
    penalty_scale: Literal["none", "feature_std"] = "feature_std",
) -> pd.Series:
    frame, target = resolve_xy(X, y)
    estimator = SupervisedAggregationRegressor(
        space="component",
        penalty=penalty,
        alpha=alpha,
        reference_weights=reference_weights,
        nonneg=nonneg,
        simplex=simplex,
        mean_match=mean_match,
        difference_order=difference_order,
        fit_intercept=False,
        penalty_scale=penalty_scale,
    ).fit(frame, target)
    if estimator.weights_ is None:
        raise RuntimeError("solver did not produce weights")
    return estimator.weights_


def _normalize_space(value: str) -> AggregationSpace:
    key = str(value).lower().replace("-", "_")
    aliases = {
        "component": "component",
        "components": "component",
        "comp": "component",
        "rank": "rank",
        "ranks": "rank",
        "rank_space": "rank",
    }
    if key not in aliases:
        raise ValueError(f"space must be one of {sorted(aliases)}; got {value!r}")
    return aliases[key]  # type: ignore[return-value]


def _normalize_penalty(value: str) -> AggregationPenalty:
    key = str(value).lower().replace("-", "_")
    aliases = {
        "ridge": "ridge",
        "l2": "ridge",
        "target": "target_shrinkage",
        "target_shrinkage": "target_shrinkage",
        "shrink_to_target": "target_shrinkage",
        "fused": "fused_difference",
        "fused_difference": "fused_difference",
        "difference": "fused_difference",
    }
    if key not in aliases:
        raise ValueError(f"penalty must be one of {sorted(aliases)}; got {value!r}")
    return aliases[key]  # type: ignore[return-value]


def _filled_float_values(frame: pd.DataFrame) -> np.ndarray:
    return frame.fillna(frame.mean(axis=0)).fillna(0.0).to_numpy(dtype=float)


def _rank_frame(frame: pd.DataFrame) -> pd.DataFrame:
    values = np.sort(frame.to_numpy(dtype=float), axis=1)
    columns = [f"rank_{idx}" for idx in range(1, frame.shape[1] + 1)]
    return pd.DataFrame(values, index=frame.index, columns=columns)


def _reference_weight_series(
    weights: Mapping[str, float] | Sequence[float] | pd.Series | None,
    *,
    columns: pd.Index,
    default: Literal["zero", "uniform"],
) -> pd.Series:
    names = tuple(str(column) for column in columns)
    if weights is None:
        if default == "uniform":
            values = np.full(len(names), 1.0 / max(len(names), 1), dtype=float)
        else:
            values = np.zeros(len(names), dtype=float)
        return pd.Series(values, index=names, name="reference_weight")
    return align_reference_weights(weights, names, normalize=False)


def _penalty_scale(values: np.ndarray, *, mode: str) -> np.ndarray:
    key = str(mode).lower()
    if key == "none":
        return np.ones(values.shape[1], dtype=float)
    if key != "feature_std":
        raise ValueError("penalty_scale must be 'none' or 'feature_std'")
    scale = np.nanstd(values, axis=0, ddof=1)
    scale[~np.isfinite(scale) | (scale <= 1e-12)] = 1.0
    return scale


def _difference_matrix(n_features: int, order: int) -> np.ndarray:
    if n_features <= int(order):
        return np.zeros((0, n_features), dtype=float)
    matrix: np.ndarray = np.eye(n_features, dtype=float)
    for _ in range(int(order)):
        matrix = np.diff(matrix, axis=0)
    return matrix


def _feasible_start(
    X: np.ndarray,
    y: np.ndarray,
    *,
    reference: np.ndarray,
    simplex: bool,
    mean_match: bool,
    nonneg: bool,
    alpha: float,
    values: np.ndarray,
    y_values: np.ndarray,
) -> np.ndarray:
    if simplex:
        start = reference.copy()
        start[~np.isfinite(start)] = 0.0
        if nonneg:
            start = np.maximum(start, 0.0)
        total = float(start.sum())
        if total <= 1e-12:
            start = np.full(values.shape[1], 1.0 / max(values.shape[1], 1), dtype=float)
        else:
            start = start / total
        return start
    if mean_match:
        x_mean = values.mean(axis=0)
        y_mean = float(y_values.mean()) if y_values.size else 0.0
        denom = float(np.sum(x_mean))
        if abs(denom) > 1e-12:
            start = np.full(values.shape[1], y_mean / denom, dtype=float)
            if nonneg and np.all(start >= 0.0):
                return start
    start = _ridge_start(X, y, alpha=alpha)
    if nonneg:
        start = np.maximum(start, 0.0)
    start[~np.isfinite(start)] = 0.0
    return start


def _ridge_start(X: np.ndarray, y: np.ndarray, *, alpha: float) -> np.ndarray:
    n_features = X.shape[1]
    lhs = X.T @ X + float(alpha) * np.eye(n_features, dtype=float)
    rhs = X.T @ y
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(lhs) @ rhs


__all__ = [
    "ASSEMBLAGE_SOURCE",
    "SupervisedAggregationRegressor",
    "albacore_components",
    "albacore_ranks",
    "assemblage_regression",
    "component_aggregation",
    "rank_aggregation",
    "solve_fused_difference_ridge",
    "solve_mean_aligned_ridge",
    "solve_nonnegative_ridge",
    "solve_simplex_ridge",
    "solve_target_shrinkage_ridge",
    "supervised_aggregation",
]
