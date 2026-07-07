from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from macroforecast.feature_engineering.screening import marginal_t_stats as _screen_marginal_t_stats
from macroforecast.models.types import ModelFit
from macroforecast.models.utils import fit_estimator, resolve_xy


def ols(X: Any, y: Any | None = None, **kwargs: Any) -> ModelFit:
    """Fit ordinary least squares."""

    from sklearn.linear_model import LinearRegression

    return fit_estimator(LinearRegression(**kwargs), X, y, model="ols", metadata=dict(kwargs))


def ridge(X: Any, y: Any | None = None, *, alpha: float = 1.0, **kwargs: Any) -> ModelFit:
    """Fit ridge regression."""

    from sklearn.linear_model import Ridge

    params = {"alpha": float(alpha), **kwargs}
    return fit_estimator(Ridge(**params), X, y, model="ridge", metadata=params)


class _NonNegativeRidge:
    """Ridge regression with non-negative coefficient constraints."""

    def __init__(self, *, alpha: float = 1.0, fit_intercept: bool = True) -> None:
        if float(alpha) < 0.0:
            raise ValueError("alpha must be non-negative")
        self.alpha = float(alpha)
        self.fit_intercept = bool(fit_intercept)
        self.feature_names_in_: np.ndarray | None = None
        self.x_mean_: np.ndarray | None = None
        self.y_mean_: float = 0.0
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_NonNegativeRidge":
        from scipy.optimize import nnls

        frame = X.astype(float).copy()
        target = pd.Series(y, index=frame.index).astype(float)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        values = _filled_float_values(frame)
        y_values = target.to_numpy(dtype=float)
        if self.fit_intercept:
            self.x_mean_ = values.mean(axis=0)
            self.y_mean_ = float(y_values.mean()) if y_values.size else 0.0
            x_work = values - self.x_mean_
            y_work = y_values - self.y_mean_
        else:
            self.x_mean_ = np.zeros(values.shape[1], dtype=float)
            self.y_mean_ = 0.0
            x_work = values
            y_work = y_values
        if self.alpha > 0.0:
            penalty = np.sqrt(self.alpha) * np.eye(values.shape[1], dtype=float)
            x_aug = np.vstack([x_work, penalty])
            y_aug = np.concatenate([y_work, np.zeros(values.shape[1], dtype=float)])
        else:
            x_aug = x_work
            y_aug = y_work
        # R source cue, nnls/R/nnls.R:
        #   sol <- .Fortran("nnls", A = as.numeric(A), ..., B = as.numeric(b))
        #   nnls.out <- list(x = sol$X, deviance = sol$RNORM^2, ...)
        # This aligns with the same mathematical problem, not with the exact
        # Fortran/SciPy backend: solve min ||A b_hat - b||^2 subject to
        # b_hat >= 0. The ridge penalty is represented by the standard
        # augmented design [X; sqrt(alpha) I] and response [y; 0]. When
        # fit_intercept=True, centering is done before the NNLS call and the
        # intercept is recovered after solving the constrained coefficient
        # problem, matching how an R user would manually build A and b.
        coef, _ = nnls(x_aug, y_aug)
        self.coef_ = np.asarray(coef, dtype=float)
        self.intercept_ = (
            self.y_mean_ - float(self.x_mean_ @ self.coef_)
            if self.fit_intercept
            else 0.0
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.coef_ is None or self.feature_names_in_ is None:
            return np.zeros(len(X), dtype=float)
        frame = X.reindex(columns=list(self.feature_names_in_), fill_value=0.0).astype(float)
        return _filled_float_values(frame) @ self.coef_ + self.intercept_


def nonneg_ridge(
    X: Any,
    y: Any | None = None,
    *,
    alpha: float = 1.0,
    fit_intercept: bool = True,
) -> ModelFit:
    """Fit ridge regression with non-negative coefficients."""

    params = {"alpha": float(alpha), "fit_intercept": bool(fit_intercept)}
    return fit_estimator(
        _NonNegativeRidge(alpha=float(alpha), fit_intercept=bool(fit_intercept)),
        X,
        y,
        model="nonneg_ridge",
        metadata=params,
    )


class _ShrinkToTargetRidge:
    """Ridge regression that shrinks coefficients toward a target vector."""

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        prior_target: float | Sequence[float] | dict[str, float] | None = None,
        simplex: bool = False,
        nonneg: bool = False,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-9,
    ) -> None:
        if float(alpha) < 0.0:
            raise ValueError("alpha must be non-negative")
        self.alpha = float(alpha)
        self.prior_target = prior_target
        self.simplex = bool(simplex)
        self.nonneg = bool(nonneg)
        self.fit_intercept = bool(fit_intercept)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.feature_names_in_: np.ndarray | None = None
        self.prior_target_: np.ndarray | None = None
        self.x_mean_: np.ndarray | None = None
        self.y_mean_: float = 0.0
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.solver_success_: bool = False
        self.solver_message_: str = ""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_ShrinkToTargetRidge":
        from scipy.optimize import minimize

        frame = X.astype(float).copy()
        target = pd.Series(y, index=frame.index).astype(float)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        values = _filled_float_values(frame)
        y_values = target.to_numpy(dtype=float)
        self.prior_target_ = _resolve_prior_target(
            self.prior_target,
            columns=frame.columns,
            n_features=values.shape[1],
            default="uniform" if self.simplex else "zero",
        )
        if self.simplex:
            self.x_mean_ = np.zeros(values.shape[1], dtype=float)
            self.y_mean_ = 0.0
            x_work = values
            y_work = y_values
        elif self.fit_intercept:
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
            # R comparison:
            # - R source cue, rags2ridges/R/rags2ridges.R:
            #     ridgeP <- function(S, lambda, type = "Alt",
            #                        target = default.target(S)) { ... }
            #     P <- .armaRidgeP(S, target, lambda)
            # - rags2ridges implements target-ridge shrinkage for covariance /
            #   precision matrices, not this regression API. The shared logic
            #   is the Tikhonov target penalty: shrink parameters toward a
            #   user-specified target rather than toward zero.
            # - Alignment here is the regression objective below, not identical
            #   rags2ridges API, domain, or solver output.
            # - In unconstrained regression form, this objective is
            #   ||y - X beta||^2 + alpha ||beta - beta0||^2. The closed-form
            #   normal equation is
            #   (X'X + alpha I) beta = X'y + alpha beta0.
            # - simplex=True is a forecasting-combination variant: coefficients
            #   sum to one, no intercept is fitted, and the uniform prior target
            #   is used when prior_target is omitted.
            residual = y_work - x_work @ coef
            shrink = coef - self.prior_target_
            return float(residual @ residual + self.alpha * (shrink @ shrink))

        constraints = []
        if self.simplex:
            constraints.append({"type": "eq", "fun": lambda coef: float(np.sum(coef) - 1.0)})
        bounds = [(0.0, None) if self.nonneg else (None, None)] * values.shape[1]
        start = _constrained_start(
            self.prior_target_,
            simplex=self.simplex,
            nonneg=self.nonneg,
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
            raise RuntimeError(f"shrink_to_target_ridge solver failed: {result.message}")
        self.coef_ = np.asarray(result.x, dtype=float)
        self.intercept_ = (
            self.y_mean_ - float(self.x_mean_ @ self.coef_)
            if self.fit_intercept and not self.simplex
            else 0.0
        )
        self.solver_success_ = bool(result.success)
        self.solver_message_ = str(result.message)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.coef_ is None or self.feature_names_in_ is None:
            return np.zeros(len(X), dtype=float)
        frame = X.reindex(columns=list(self.feature_names_in_), fill_value=0.0).astype(float)
        return _filled_float_values(frame) @ self.coef_ + self.intercept_


def shrink_to_target_ridge(
    X: Any,
    y: Any | None = None,
    *,
    alpha: float = 1.0,
    prior_target: float | Sequence[float] | dict[str, float] | None = None,
    simplex: bool = False,
    nonneg: bool = False,
    fit_intercept: bool = True,
    max_iter: int = 1000,
    tol: float = 1e-9,
) -> ModelFit:
    """Fit ridge regression with a coefficient prior target."""

    params = {
        "alpha": float(alpha),
        "prior_target": prior_target,
        "simplex": bool(simplex),
        "nonneg": bool(nonneg),
        "fit_intercept": bool(fit_intercept),
        "max_iter": int(max_iter),
        "tol": float(tol),
    }
    return fit_estimator(
        _ShrinkToTargetRidge(
            alpha=float(alpha),
            prior_target=prior_target,
            simplex=bool(simplex),
            nonneg=bool(nonneg),
            fit_intercept=bool(fit_intercept),
            max_iter=int(max_iter),
            tol=float(tol),
        ),
        X,
        y,
        model="shrink_to_target_ridge",
        metadata=params,
    )


class _FusedDifferenceRidge:
    """Ridge regression with a smoothness penalty on adjacent coefficients."""

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        difference_order: int = 1,
        mean_equality: bool = False,
        nonneg: bool = False,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-9,
    ) -> None:
        if float(alpha) < 0.0:
            raise ValueError("alpha must be non-negative")
        if int(difference_order) < 1:
            raise ValueError("difference_order must be at least 1")
        self.alpha = float(alpha)
        self.difference_order = int(difference_order)
        self.mean_equality = bool(mean_equality)
        self.nonneg = bool(nonneg)
        self.fit_intercept = bool(fit_intercept)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.feature_names_in_: np.ndarray | None = None
        self.difference_matrix_: np.ndarray | None = None
        self.x_mean_: np.ndarray | None = None
        self.y_mean_: float = 0.0
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.solver_success_: bool = False
        self.solver_message_: str = ""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_FusedDifferenceRidge":
        from scipy.optimize import minimize

        frame = X.astype(float).copy()
        target = pd.Series(y, index=frame.index).astype(float)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        values = _filled_float_values(frame)
        y_values = target.to_numpy(dtype=float)
        self.difference_matrix_ = _difference_matrix(values.shape[1], self.difference_order)
        if self.mean_equality:
            self.x_mean_ = np.zeros(values.shape[1], dtype=float)
            self.y_mean_ = 0.0
            x_work = values
            y_work = y_values
        elif self.fit_intercept:
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
            # R comparison:
            # - R source cue, rags2ridges/R/rags2ridgesFused.R:
            #     ridgeP.fused <- function(Slist, ns, Tlist, lambda, ...)
            #     Plist <- .armaRidgeP.fused(Slist = Slist, ns = ns,
            #                                Tlist = Tlist, lambda = lambda)
            # - rags2ridges::ridgeP.fused applies L2 fusion to precision
            #   matrices across related groups. This callable applies the same
            #   L2 fusion idea to an ordered regression-coefficient vector.
            # - Alignment here is the L2 fusion/smoothness objective below,
            #   not identical rags2ridges matrix-estimation API or solver.
            # - D is a finite-difference matrix over adjacent coefficients. In
            #   the unconstrained regression case, the objective is
            #   ||y - X beta||^2 + alpha ||D beta||^2 and the normal equation is
            #   (X'X + alpha D'D) beta = X'y.
            # - mean_equality=True is a macro forecasting mass-conservation
            #   variant and has no direct rags2ridges analogue.
            residual = y_work - x_work @ coef
            smooth = self.difference_matrix_ @ coef
            return float(residual @ residual + self.alpha * (smooth @ smooth))

        constraints = []
        if self.mean_equality:
            target_sum = float(np.sum(y_work))
            constraints.append(
                {"type": "eq", "fun": lambda coef: float(np.sum(x_work @ coef) - target_sum)}
            )
        bounds = [(0.0, None) if self.nonneg else (None, None)] * values.shape[1]
        start = _ridge_start(x_work, y_work, alpha=max(self.alpha, 1e-12))
        if self.nonneg:
            start = np.maximum(start, 0.0)
        result = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )
        if not result.success:
            raise RuntimeError(f"fused_difference_ridge solver failed: {result.message}")
        self.coef_ = np.asarray(result.x, dtype=float)
        self.intercept_ = (
            self.y_mean_ - float(self.x_mean_ @ self.coef_)
            if self.fit_intercept and not self.mean_equality
            else 0.0
        )
        self.solver_success_ = bool(result.success)
        self.solver_message_ = str(result.message)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.coef_ is None or self.feature_names_in_ is None:
            return np.zeros(len(X), dtype=float)
        frame = X.reindex(columns=list(self.feature_names_in_), fill_value=0.0).astype(float)
        return _filled_float_values(frame) @ self.coef_ + self.intercept_


def fused_difference_ridge(
    X: Any,
    y: Any | None = None,
    *,
    alpha: float = 1.0,
    difference_order: int = 1,
    mean_equality: bool = False,
    nonneg: bool = False,
    fit_intercept: bool = True,
    max_iter: int = 1000,
    tol: float = 1e-9,
) -> ModelFit:
    """Fit ridge regression with adjacent-coefficient smoothness."""

    params = {
        "alpha": float(alpha),
        "difference_order": int(difference_order),
        "mean_equality": bool(mean_equality),
        "nonneg": bool(nonneg),
        "fit_intercept": bool(fit_intercept),
        "max_iter": int(max_iter),
        "tol": float(tol),
    }
    return fit_estimator(
        _FusedDifferenceRidge(
            alpha=float(alpha),
            difference_order=int(difference_order),
            mean_equality=bool(mean_equality),
            nonneg=bool(nonneg),
            fit_intercept=bool(fit_intercept),
            max_iter=int(max_iter),
            tol=float(tol),
        ),
        X,
        y,
        model="fused_difference_ridge",
        metadata=params,
    )


class _RandomWalkRidge:
    """Time-varying ridge where adjacent coefficient vectors follow a random walk."""

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        initial_alpha: float = 1.0,
        fit_intercept: bool = True,
    ) -> None:
        if float(alpha) < 0.0:
            raise ValueError("alpha must be non-negative")
        if float(initial_alpha) < 0.0:
            raise ValueError("initial_alpha must be non-negative")
        self.alpha = float(alpha)
        self.initial_alpha = float(initial_alpha)
        self.fit_intercept = bool(fit_intercept)
        self.feature_names_in_: np.ndarray | None = None
        self.x_mean_: np.ndarray | None = None
        self.y_mean_: float = 0.0
        self.coef_: np.ndarray | None = None
        self.coef_path_: pd.DataFrame | None = None
        self.intercept_: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_RandomWalkRidge":
        frame = X.astype(float).copy()
        target = pd.Series(y, index=frame.index).astype(float)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        values = _filled_float_values(frame)
        y_values = target.to_numpy(dtype=float)
        if self.fit_intercept:
            self.x_mean_ = values.mean(axis=0)
            self.y_mean_ = float(y_values.mean()) if y_values.size else 0.0
            x_work = values - self.x_mean_
            y_work = y_values - self.y_mean_
        else:
            self.x_mean_ = np.zeros(values.shape[1], dtype=float)
            self.y_mean_ = 0.0
            x_work = values
            y_work = y_values
        n_obs, n_features = x_work.shape
        # R comparison:
        # - R source cue, walker/R/walker_rw1.R:
        #     xreg <- model.matrix(attr(mf, "terms"), mf)
        #     stan_data <- list(k = k, n = n, y = y, xreg = t(xreg), ...)
        #     do.call(sampling, list(object = stanmodels$rw1_model, ...))
        # - walker::walker_rw1 models beta_t as random-walk state variables and
        #   estimates a Bayesian posterior with Stan / state-space smoothing.
        # - This callable keeps the same RW1 prior idea but uses the MAP-style
        #   penalized least-squares objective:
        #   sum_t (y_t - x_t beta_t)^2 + initial_alpha ||beta_1||^2
        #   + alpha sum_t ||beta_t - beta_{t-1}||^2.
        # - Alignment here is the RW1 prior / MAP objective, not posterior
        #   sampling, credible intervals, or a Kalman/Stan implementation.
        # - The objective is encoded as one augmented least-squares system over
        #   vec(beta_1, ..., beta_T). Predictions intentionally use the final
        #   coefficient vector, not posterior simulations or a Kalman smoother.
        design = np.zeros((n_obs, n_obs * n_features), dtype=float)
        for row in range(n_obs):
            start = row * n_features
            design[row, start : start + n_features] = x_work[row]
        penalty_rows: list[np.ndarray] = []
        penalty_targets: list[np.ndarray] = []
        if self.initial_alpha > 0.0:
            block = np.zeros((n_features, n_obs * n_features), dtype=float)
            block[:, :n_features] = np.sqrt(self.initial_alpha) * np.eye(n_features)
            penalty_rows.append(block)
            penalty_targets.append(np.zeros(n_features, dtype=float))
        if self.alpha > 0.0 and n_obs > 1:
            scale = np.sqrt(self.alpha)
            block = np.zeros(((n_obs - 1) * n_features, n_obs * n_features), dtype=float)
            cursor = 0
            for row in range(1, n_obs):
                previous = (row - 1) * n_features
                current = row * n_features
                block[cursor : cursor + n_features, previous : previous + n_features] = (
                    -scale * np.eye(n_features)
                )
                block[cursor : cursor + n_features, current : current + n_features] = (
                    scale * np.eye(n_features)
                )
                cursor += n_features
            penalty_rows.append(block)
            penalty_targets.append(np.zeros((n_obs - 1) * n_features, dtype=float))
        if penalty_rows:
            design = np.vstack([design, *penalty_rows])
            y_aug = np.concatenate([y_work, *penalty_targets])
        else:
            y_aug = y_work
        coef_vector = np.linalg.lstsq(design, y_aug, rcond=None)[0]
        coef_path = coef_vector.reshape(n_obs, n_features)
        self.coef_path_ = pd.DataFrame(
            coef_path,
            index=frame.index,
            columns=[str(column) for column in frame.columns],
        )
        self.coef_ = coef_path[-1].copy()
        self.intercept_ = (
            self.y_mean_ - float(self.x_mean_ @ self.coef_)
            if self.fit_intercept
            else 0.0
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.coef_ is None or self.feature_names_in_ is None:
            return np.zeros(len(X), dtype=float)
        frame = X.reindex(columns=list(self.feature_names_in_), fill_value=0.0).astype(float)
        return _filled_float_values(frame) @ self.coef_ + self.intercept_


def random_walk_ridge(
    X: Any,
    y: Any | None = None,
    *,
    alpha: float = 1.0,
    initial_alpha: float = 1.0,
    fit_intercept: bool = True,
) -> ModelFit:
    """Fit a random-walk coefficient ridge model and predict with the final coefficient."""

    params = {
        "alpha": float(alpha),
        "initial_alpha": float(initial_alpha),
        "fit_intercept": bool(fit_intercept),
    }
    return fit_estimator(
        _RandomWalkRidge(
            alpha=float(alpha),
            initial_alpha=float(initial_alpha),
            fit_intercept=bool(fit_intercept),
        ),
        X,
        y,
        model="random_walk_ridge",
        metadata=params,
    )


def lasso(
    X: Any,
    y: Any | None = None,
    *,
    alpha: float = 1.0,
    max_iter: int = 20000,
    standardize: bool = False,
    **kwargs: Any,
) -> ModelFit:
    """Fit lasso regression with a user-supplied alpha."""

    from sklearn.linear_model import Lasso
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    estimator_params = {"alpha": float(alpha), "max_iter": int(max_iter), **kwargs}
    estimator = Lasso(**estimator_params)
    if standardize:
        estimator = make_pipeline(StandardScaler(), estimator)
    params = {
        **estimator_params,
        "standardize": bool(standardize),
    }
    return fit_estimator(
        estimator,
        X,
        y,
        model="lasso",
        metadata=params,
    )


def elastic_net(
    X: Any,
    y: Any | None = None,
    *,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    max_iter: int = 20000,
    standardize: bool = False,
    **kwargs: Any,
) -> ModelFit:
    """Fit elastic net regression."""

    from sklearn.linear_model import ElasticNet
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    estimator_params = {
        "alpha": float(alpha),
        "l1_ratio": float(l1_ratio),
        "max_iter": int(max_iter),
        **kwargs,
    }
    estimator = ElasticNet(**estimator_params)
    if standardize:
        estimator = make_pipeline(StandardScaler(), estimator)
    params = {
        **estimator_params,
        "standardize": bool(standardize),
    }
    return fit_estimator(
        estimator,
        X,
        y,
        model="elastic_net",
        metadata=params,
    )


class _AdaptiveLinear:
    """Adaptive lasso/elastic-net via initial coefficient weights."""

    def __init__(
        self,
        *,
        kind: str = "lasso",
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        gamma: float = 1.0,
        initial: str = "ridge",
        initial_alpha: float = 1.0,
        eps: float = 1e-4,
        normalize_weights: bool = True,
        max_iter: int = 20000,
        tol: float = 1e-4,
        random_state: int | None = None,
    ) -> None:
        if kind not in {"lasso", "elastic_net"}:
            raise ValueError("kind must be 'lasso' or 'elastic_net'")
        if initial not in {"ridge", "ols"}:
            raise ValueError("initial must be 'ridge' or 'ols'")
        self.kind = kind
        self.alpha = float(alpha)
        self.l1_ratio = float(l1_ratio)
        self.gamma = float(gamma)
        self.initial = initial
        self.initial_alpha = float(initial_alpha)
        self.eps = max(float(eps), 1e-12)
        self.normalize_weights = bool(normalize_weights)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state
        self.feature_names_in_: np.ndarray | None = None
        self.x_mean_: np.ndarray | None = None
        self.x_scale_: np.ndarray | None = None
        self.y_mean_: float = 0.0
        self.initial_coef_: np.ndarray | None = None
        self.raw_adaptive_weights_: np.ndarray | None = None
        self.adaptive_weights_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.estimator_: Any = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_AdaptiveLinear":
        from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge

        frame = X.astype(float).copy()
        target = pd.Series(y, index=frame.index).astype(float)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        values = frame.fillna(frame.mean(axis=0)).fillna(0.0).to_numpy(dtype=float)
        y_values = target.to_numpy(dtype=float)
        self.x_mean_ = values.mean(axis=0)
        self.x_scale_ = _safe_array_scale(values.std(axis=0, ddof=1))
        self.y_mean_ = float(y_values.mean()) if y_values.size else 0.0
        x_scaled = (values - self.x_mean_) / self.x_scale_
        y_centered = y_values - self.y_mean_

        if self.initial == "ridge":
            initial_estimator = Ridge(alpha=self.initial_alpha, fit_intercept=False)
        else:
            initial_estimator = LinearRegression(fit_intercept=False)
        initial_estimator.fit(x_scaled, y_centered)
        initial_coef = np.asarray(initial_estimator.coef_, dtype=float).reshape(-1)
        raw_weights = 1.0 / np.power(np.abs(initial_coef) + self.eps, self.gamma)
        weights = raw_weights.copy()
        if self.normalize_weights:
            # R source cue, glmnet/R/glmnetFlex.R:
            #   vp = pmax(0, penalty.factor)
            #   vp = as.double(vp * nvars / sum(vp))
            # This aligns with glmnet's weighted lasso/elastic-net objective,
            # not glmnet's full path solver. glmnet(penalty.factor=...) treats
            # penalty factors as relative multipliers and rescales them
            # internally to sum to the number of predictors. Mean-one
            # normalization keeps alpha on the same scale while preserving
            # adaptive-lasso relative weights.
            weight_sum = float(np.sum(weights))
            if np.isfinite(weight_sum) and weight_sum > 1e-12:
                weights = weights * (len(weights) / weight_sum)
        # sklearn Lasso/ElasticNet do not expose per-coordinate penalty
        # factors. Rescaling each standardized predictor by 1 / weight_j gives
        # the same weighted L1 penalty after mapping coefficients back:
        # beta_j = theta_j / weight_j.
        weighted_x = x_scaled / weights

        if self.kind == "lasso":
            estimator = Lasso(
                alpha=self.alpha,
                fit_intercept=False,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
            )
        else:
            estimator = ElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                fit_intercept=False,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
            )
        estimator.fit(weighted_x, y_centered)
        weighted_coef = np.asarray(estimator.coef_, dtype=float).reshape(-1)
        scaled_coef = weighted_coef / weights
        self.initial_coef_ = initial_coef
        self.raw_adaptive_weights_ = raw_weights
        self.adaptive_weights_ = weights
        self.coef_ = scaled_coef / self.x_scale_
        self.intercept_ = self.y_mean_ - float(self.x_mean_ @ self.coef_)
        self.estimator_ = estimator
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.coef_ is None or self.feature_names_in_ is None:
            return np.zeros(len(X), dtype=float)
        frame = X.reindex(columns=list(self.feature_names_in_), fill_value=0.0).astype(float)
        return frame.to_numpy(dtype=float) @ self.coef_ + self.intercept_


def adaptive_lasso(
    X: Any,
    y: Any | None = None,
    *,
    alpha: float = 1.0,
    gamma: float = 1.0,
    initial: str = "ridge",
    initial_alpha: float = 1.0,
    eps: float = 1e-4,
    normalize_weights: bool = True,
    max_iter: int = 20000,
    tol: float = 1e-4,
    random_state: int | None = None,
) -> ModelFit:
    """Fit adaptive lasso using initial coefficient-based penalty weights."""

    params = {
        "alpha": float(alpha),
        "gamma": float(gamma),
        "initial": initial,
        "initial_alpha": float(initial_alpha),
        "eps": float(eps),
        "normalize_weights": bool(normalize_weights),
        "max_iter": int(max_iter),
        "tol": float(tol),
        "random_state": random_state,
    }
    return fit_estimator(
        _AdaptiveLinear(
            kind="lasso",
            alpha=float(alpha),
            gamma=float(gamma),
            initial=initial,
            initial_alpha=float(initial_alpha),
            eps=float(eps),
            normalize_weights=bool(normalize_weights),
            max_iter=int(max_iter),
            tol=float(tol),
            random_state=random_state,
        ),
        X,
        y,
        model="adaptive_lasso",
        metadata=params,
    )


def adaptive_elastic_net(
    X: Any,
    y: Any | None = None,
    *,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    gamma: float = 1.0,
    initial: str = "ridge",
    initial_alpha: float = 1.0,
    eps: float = 1e-4,
    normalize_weights: bool = True,
    max_iter: int = 20000,
    tol: float = 1e-4,
    random_state: int | None = None,
) -> ModelFit:
    """Fit adaptive elastic net using initial coefficient-based column weights."""

    params = {
        "alpha": float(alpha),
        "l1_ratio": float(l1_ratio),
        "gamma": float(gamma),
        "initial": initial,
        "initial_alpha": float(initial_alpha),
        "eps": float(eps),
        "normalize_weights": bool(normalize_weights),
        "max_iter": int(max_iter),
        "tol": float(tol),
        "random_state": random_state,
    }
    return fit_estimator(
        _AdaptiveLinear(
            kind="elastic_net",
            alpha=float(alpha),
            l1_ratio=float(l1_ratio),
            gamma=float(gamma),
            initial=initial,
            initial_alpha=float(initial_alpha),
            eps=float(eps),
            normalize_weights=bool(normalize_weights),
            max_iter=int(max_iter),
            tol=float(tol),
            random_state=random_state,
        ),
        X,
        y,
        model="adaptive_elastic_net",
        metadata=params,
    )


class _GroupLinear:
    """Proximal-gradient group lasso or sparse group lasso."""

    def __init__(
        self,
        *,
        groups: Sequence[str | int] | None = None,
        alpha: float = 1.0,
        l1_ratio: float = 0.0,
        group_weights: dict[str, float] | None = None,
        max_iter: int = 5000,
        tol: float = 1e-5,
        scale: bool = True,
    ) -> None:
        self.groups = None if groups is None else tuple(groups)
        self.alpha = float(alpha)
        self.l1_ratio = float(np.clip(l1_ratio, 0.0, 1.0))
        self.group_weights = dict(group_weights or {})
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.scale = bool(scale)
        self.feature_names_in_: np.ndarray | None = None
        self.groups_: tuple[str, ...] = ()
        self.group_index_: dict[str, np.ndarray] = {}
        self.resolved_group_weights_: dict[str, float] = {}
        self.x_mean_: np.ndarray | None = None
        self.x_scale_: np.ndarray | None = None
        self.y_mean_: float = 0.0
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.n_iter_: int = 0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_GroupLinear":
        frame = X.astype(float).copy()
        target = pd.Series(y, index=frame.index).astype(float)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        values = frame.fillna(frame.mean(axis=0)).fillna(0.0).to_numpy(dtype=float)
        y_values = target.to_numpy(dtype=float)
        self.y_mean_ = float(y_values.mean()) if y_values.size else 0.0
        y_centered = y_values - self.y_mean_
        if self.scale:
            self.x_mean_ = values.mean(axis=0)
            self.x_scale_ = _safe_array_scale(values.std(axis=0, ddof=1))
            x_work = (values - self.x_mean_) / self.x_scale_
        else:
            self.x_mean_ = np.zeros(values.shape[1], dtype=float)
            self.x_scale_ = np.ones(values.shape[1], dtype=float)
            x_work = values
        groups = self._resolve_groups(values.shape[1])
        self.groups_ = tuple(groups)
        self.group_index_ = {
            group: np.flatnonzero(np.asarray(groups, dtype=object) == group)
            for group in dict.fromkeys(groups)
        }
        self.resolved_group_weights_ = {
            group: float(self.group_weights.get(group, np.sqrt(len(index))))
            for group, index in self.group_index_.items()
        }
        coef_scaled = self._solve(x_work, y_centered)
        self.coef_ = coef_scaled / self.x_scale_
        self.intercept_ = self.y_mean_ - float(self.x_mean_ @ self.coef_)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.coef_ is None or self.feature_names_in_ is None:
            return np.zeros(len(X), dtype=float)
        frame = X.reindex(columns=list(self.feature_names_in_), fill_value=0.0).astype(float)
        return frame.to_numpy(dtype=float) @ self.coef_ + self.intercept_

    def _resolve_groups(self, n_features: int) -> tuple[str, ...]:
        if self.groups is None:
            return tuple(f"g{i}" for i in range(n_features))
        if len(self.groups) != n_features:
            raise ValueError("groups must have one entry per X column")
        return tuple(str(group) for group in self.groups)

    def _solve(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n_obs, n_features = X.shape
        if n_features == 0:
            return np.empty(0, dtype=float)
        spectral = float(np.linalg.norm(X, ord=2) ** 2 / max(1, n_obs))
        step = 1.0 / max(spectral, 1e-8)
        coef = np.zeros(n_features, dtype=float)
        l1_penalty = self.alpha * self.l1_ratio
        group_penalty = self.alpha * (1.0 - self.l1_ratio)
        # R source cues:
        #   grpreg/R/newXG.R:
        #     std <- .Call("standardize", X)
        #     if (all(is.na(m))) m <- sqrt(table(g[g != 0]))
        #   sparsegl/R/sparsegl.R:
        #     lambda = NULL, pf_group = sqrt(bs), pf_sparse = rep(1, nvars)
        #     pf_sparse <- pf_sparse / sum(pf_sparse) * nvars
        #   grpreg/R/grpreg.R objective:
        #     Q(beta|X,y) = (1/n) * L(beta|X,y) + P_lambda(beta)
        # This aligns with the Gaussian group-lasso / sparse-group-lasso
        # objective and the default sqrt(group-size) group weights. It does not
        # reproduce grpreg's full lambda path, GLM families, C backend, or its
        # within-group orthogonalization step for group penalties. l1_ratio=0
        # gives the grLasso-style group penalty; l1_ratio>0 adds sparsegl-style
        # feature-level soft-thresholding before group shrinkage.
        for iteration in range(1, self.max_iter + 1):
            previous = coef.copy()
            residual = X @ coef - y
            coef = coef - step * (X.T @ residual) / max(1, n_obs)
            if l1_penalty > 0.0:
                coef = _soft_threshold(coef, step * l1_penalty)
            if group_penalty > 0.0:
                for group, index in self.group_index_.items():
                    block = coef[index]
                    norm = float(np.linalg.norm(block, ord=2))
                    weight = self.resolved_group_weights_.get(group, np.sqrt(len(index)))
                    threshold = step * group_penalty * float(weight)
                    if norm <= threshold:
                        coef[index] = 0.0
                    else:
                        coef[index] = block * (1.0 - threshold / norm)
            delta = float(np.linalg.norm(coef - previous, ord=2))
            self.n_iter_ = iteration
            if delta <= self.tol * max(1.0, float(np.linalg.norm(previous, ord=2))):
                break
        return coef


def group_lasso(
    X: Any,
    y: Any | None = None,
    *,
    groups: Sequence[str | int] | None = None,
    alpha: float = 1.0,
    group_weights: dict[str, float] | None = None,
    max_iter: int = 5000,
    tol: float = 1e-5,
    scale: bool = True,
) -> ModelFit:
    """Fit group lasso with one group label per predictor."""

    params = {
        "groups": None if groups is None else tuple(groups),
        "alpha": float(alpha),
        "group_weights": group_weights,
        "max_iter": int(max_iter),
        "tol": float(tol),
        "scale": bool(scale),
    }
    return fit_estimator(
        _GroupLinear(
            groups=groups,
            alpha=float(alpha),
            l1_ratio=0.0,
            group_weights=group_weights,
            max_iter=int(max_iter),
            tol=float(tol),
            scale=bool(scale),
        ),
        X,
        y,
        model="group_lasso",
        metadata=params,
    )


def sparse_group_lasso(
    X: Any,
    y: Any | None = None,
    *,
    groups: Sequence[str | int] | None = None,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    group_weights: dict[str, float] | None = None,
    max_iter: int = 5000,
    tol: float = 1e-5,
    scale: bool = True,
) -> ModelFit:
    """Fit sparse group lasso with group and feature-level sparsity."""

    params = {
        "groups": None if groups is None else tuple(groups),
        "alpha": float(alpha),
        "l1_ratio": float(l1_ratio),
        "group_weights": group_weights,
        "max_iter": int(max_iter),
        "tol": float(tol),
        "scale": bool(scale),
    }
    return fit_estimator(
        _GroupLinear(
            groups=groups,
            alpha=float(alpha),
            l1_ratio=float(l1_ratio),
            group_weights=group_weights,
            max_iter=int(max_iter),
            tol=float(tol),
            scale=bool(scale),
        ),
        X,
        y,
        model="sparse_group_lasso",
        metadata=params,
    )


def bayesian_ridge(X: Any, y: Any | None = None, **kwargs: Any) -> ModelFit:
    """Fit empirical-Bayes Bayesian ridge regression."""

    from sklearn.linear_model import BayesianRidge

    return fit_estimator(BayesianRidge(**kwargs), X, y, model="bayesian_ridge", metadata=dict(kwargs))


def huber(
    X: Any,
    y: Any | None = None,
    *,
    epsilon: float = 1.35,
    max_iter: int = 1000,
    **kwargs: Any,
) -> ModelFit:
    """Fit robust Huber regression."""

    from sklearn.linear_model import HuberRegressor

    params = {"epsilon": float(epsilon), "max_iter": int(max_iter), **kwargs}
    return fit_estimator(
        HuberRegressor(**params),
        X,
        y,
        model="huber",
        metadata=params,
    )


class _GLMBoost:
    """Componentwise L2 boosting with linear base learners."""

    def __init__(
        self,
        *,
        n_iter: int = 100,
        learning_rate: float = 0.1,
        center: bool = True,
        candidate_sampling: str = "all",
        candidate_count: int | None = None,
        candidate_fraction: float | None = None,
        candidate_cap: int | None = None,
        candidate_min: int = 1,
        candidate_rounding: str = "floor",
        random_state: int | None = None,
    ) -> None:
        self.n_iter = max(1, int(n_iter))
        self.learning_rate = float(learning_rate)
        self.center = bool(center)
        self.candidate_sampling = candidate_sampling
        self.candidate_count = candidate_count
        self.candidate_fraction = candidate_fraction
        self.candidate_cap = candidate_cap
        self.candidate_min = candidate_min
        self.candidate_rounding = candidate_rounding
        self.random_state = random_state
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.feature_names_in_: np.ndarray | None = None
        self.x_mean_: np.ndarray | None = None
        self.n_iter_: int = 0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_GLMBoost":
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        x_raw = X.fillna(0.0).to_numpy(dtype=float)
        if self.center:
            self.x_mean_ = x_raw.mean(axis=0)
            x = x_raw - self.x_mean_
        else:
            self.x_mean_ = np.zeros(x_raw.shape[1], dtype=float)
            x = x_raw
        target = np.asarray(y, dtype=float)
        self.intercept_ = float(np.mean(target)) if target.size else 0.0
        residual = target - self.intercept_
        self.coef_ = np.zeros(x.shape[1], dtype=float)
        rng = np.random.default_rng(self.random_state)
        n_candidates = _resolve_glmboost_candidate_count(
            self.candidate_sampling,
            n_features=x.shape[1],
            candidate_count=self.candidate_count,
            candidate_fraction=self.candidate_fraction,
            candidate_cap=self.candidate_cap,
            candidate_min=self.candidate_min,
            candidate_rounding=self.candidate_rounding,
        )
        # R source cues, mboost/R/mboost.R and mboost/R/bolscw.R:
        #   glmboost.matrix <- function(x, y, center = TRUE, ...)
        #   if (center) X <- scale(X, center = cm, scale = FALSE)
        #   xtx <- colSums(X^2 * weights); sxtx <- sqrt(xtx)
        #   mu <- crossprod(X, y * weights) / sxtx
        #   xselect <- which(abs(mu) == max(abs(mu)))[1]
        #   coef <- mu[xselect] / sxtx[xselect]
        # This aligns with mboost's Gaussian componentwise L2 update for a
        # matrix input: center predictors by default, select the best one-step
        # base learner by normalized correlation, then apply the shrinkage
        # update. When candidate_sampling="random", each boosting step samples
        # a candidate subset before the best-base-learner search. Goulet
        # Coulombe et al. (2021) Appendix A.6 is expressed without a magic
        # preset as candidate_fraction=1/3, candidate_cap=200,
        # candidate_rounding="floor", candidate_min=1.
        # It does not reproduce mboost's formula handling, weights,
        # alternative families, hat values, or stopping machinery.
        for iteration in range(1, self.n_iter + 1):
            covariances = x.T @ residual
            denominators = np.sum(x * x, axis=0)
            usable = denominators > 1e-12
            if not bool(np.any(usable)):
                break
            candidates = np.flatnonzero(usable)
            if n_candidates and n_candidates < len(candidates):
                candidates = np.sort(
                    rng.choice(candidates, size=n_candidates, replace=False)
                )
            scores = np.zeros_like(covariances, dtype=float)
            scores[candidates] = covariances[candidates] / np.sqrt(denominators[candidates])
            best = int(candidates[np.argmax(np.abs(scores[candidates]))])
            denom = float(denominators[best])
            if denom <= 1e-12:
                break
            step = self.learning_rate * float(covariances[best]) / denom
            self.coef_[best] += step
            residual = residual - step * x[:, best]
            self.n_iter_ = iteration
        self.intercept_ = self.intercept_ - float(self.x_mean_ @ self.coef_)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.coef_ is None or self.feature_names_in_ is None:
            return np.zeros(len(X), dtype=float)
        frame = X.reindex(columns=list(self.feature_names_in_), fill_value=0.0)
        return frame.fillna(0.0).to_numpy(dtype=float) @ self.coef_ + self.intercept_


def glmboost(
    X: Any,
    y: Any | None = None,
    *,
    n_iter: int = 100,
    learning_rate: float = 0.1,
    center: bool = True,
    candidate_sampling: str = "all",
    candidate_count: int | None = None,
    candidate_fraction: float | None = None,
    candidate_cap: int | None = None,
    candidate_min: int = 1,
    candidate_rounding: str = "floor",
    random_state: int | None = None,
) -> ModelFit:
    """Fit componentwise linear boosting."""

    return fit_estimator(
        _GLMBoost(
            n_iter=n_iter,
            learning_rate=learning_rate,
            center=center,
            candidate_sampling=candidate_sampling,
            candidate_count=candidate_count,
            candidate_fraction=candidate_fraction,
            candidate_cap=candidate_cap,
            candidate_min=candidate_min,
            candidate_rounding=candidate_rounding,
            random_state=random_state,
        ),
        X,
        y,
        model="glmboost",
        metadata={
            "n_iter": int(n_iter),
            "learning_rate": float(learning_rate),
            "center": bool(center),
            "candidate_sampling": candidate_sampling,
            "candidate_count": candidate_count,
            "candidate_fraction": candidate_fraction,
            "candidate_cap": candidate_cap,
            "candidate_min": int(candidate_min),
            "candidate_rounding": candidate_rounding,
            "random_state": random_state,
        },
    )


def _resolve_glmboost_candidate_count(
    candidate_sampling: str,
    *,
    n_features: int,
    candidate_count: int | None,
    candidate_fraction: float | None,
    candidate_cap: int | None,
    candidate_min: int,
    candidate_rounding: str,
) -> int | None:
    key = str(candidate_sampling).lower().replace("-", "_")
    if key in {"all", "none", "full"}:
        if candidate_count is not None or candidate_fraction is not None or candidate_cap is not None:
            raise ValueError(
                "candidate_count, candidate_fraction, and candidate_cap require "
                "candidate_sampling='random'"
            )
        return None
    if key not in {"random", "random_subset", "subsample"}:
        raise ValueError("candidate_sampling must be 'all' or 'random'")
    if candidate_count is not None and candidate_fraction is not None:
        raise ValueError("use either candidate_count or candidate_fraction, not both")
    if candidate_count is None and candidate_fraction is None:
        raise ValueError(
            "candidate_sampling='random' requires candidate_count or candidate_fraction"
        )
    lower = int(candidate_min)
    if lower < 1:
        raise ValueError("candidate_min must be at least 1")
    cap: int | None = None
    if candidate_cap is not None:
        cap = int(candidate_cap)
        if cap < 1:
            raise ValueError("candidate_cap must be at least 1")
        if lower > cap:
            raise ValueError("candidate_min must be smaller than or equal to candidate_cap")
    if candidate_count is not None:
        raw = int(candidate_count)
        if raw < 1:
            raise ValueError("candidate_count must be at least 1")
    else:
        # candidate_count is None here, so the earlier both-None guard ensures
        # candidate_fraction is not None.
        assert candidate_fraction is not None
        fraction = float(candidate_fraction)
        if not 0 < fraction <= 1:
            raise ValueError("candidate_fraction must be in (0, 1]")
        rounding_key = str(candidate_rounding).lower()
        value = fraction * float(n_features)
        if rounding_key == "floor":
            raw = int(np.floor(value))
        elif rounding_key == "ceil":
            raw = int(np.ceil(value))
        elif rounding_key == "round":
            raw = int(round(value))
        else:
            raise ValueError("candidate_rounding must be 'floor', 'ceil', or 'round'")
    if cap is not None:
        raw = min(raw, cap)
    return max(1, min(n_features, max(lower, raw)))


class _PLSCompositeRegressor:
    """PLS regression with optional Hounyo-Li-style control residualization."""

    # Source alignment:
    # - Hounyo-Li's MATLAB baseline PLS_emp002.m first residualizes ytplush on
    #   wt, then calls plsregress() on standardized predictors and the residual,
    #   extracts stats.W, builds factors, re-estimates factor coefficients, and
    #   adds the control forecast back.
    # - macroforecast keeps that forecasting contract but delegates the PLS
    #   component extraction to sklearn.cross_decomposition.PLSRegression. The
    #   optional control block mirrors wt. With no controls, include_constant=True
    #   gives the usual intercept-only residualization.

    def __init__(
        self,
        *,
        n_components: int = 3,
        scale: bool = True,
        max_iter: int = 500,
        tol: float = 1e-6,
        control_columns: Sequence[str] | None = None,
        include_constant: bool = True,
        drop_control_columns: bool = True,
        quadratic_factors: bool = False,
        **kwargs: Any,
    ) -> None:
        self.n_components = max(1, int(n_components))
        self.scale = bool(scale)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.control_columns = tuple(str(column) for column in (control_columns or ()))
        self.include_constant = bool(include_constant)
        self.drop_control_columns = bool(drop_control_columns)
        self.quadratic_factors = bool(quadratic_factors)
        self.kwargs = dict(kwargs)
        self.x_mean_: pd.Series | None = None
        self.x_scale_: pd.Series | None = None
        self.factor_features_: tuple[str, ...] = ()
        self.factor_loadings_: np.ndarray | None = None
        self.factor_coefs_: np.ndarray | None = None
        self.factor_square_coefs_: np.ndarray | None = None
        self.control_coef_: np.ndarray = np.empty(0, dtype=float)
        self.control_names_: tuple[str, ...] = ()
        self.n_components_: int = 0
        self.pls_model_: Any | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_PLSCompositeRegressor":
        from sklearn.cross_decomposition import PLSRegression

        frame = X.astype(float).copy()
        target = pd.Series(y, index=frame.index).astype(float)
        if frame.shape[1] == 0:
            raise ValueError("pls requires at least one predictor")
        if self.scale:
            self.x_mean_ = frame.mean(axis=0)
            self.x_scale_ = _safe_series_scale(frame.std(axis=0, ddof=1))
            standardized_values = ((frame - self.x_mean_) / self.x_scale_).to_numpy(dtype=float)
        else:
            self.x_mean_ = pd.Series(0.0, index=frame.columns)
            self.x_scale_ = pd.Series(1.0, index=frame.columns)
            standardized_values = frame.to_numpy(dtype=float)
        standardized = pd.DataFrame(standardized_values, index=frame.index, columns=frame.columns)

        control_frame = _control_matrix(
            standardized,
            self.control_columns,
            include_constant=self.include_constant,
        )
        self.control_names_ = tuple(str(column) for column in control_frame.columns)
        factor_columns = _factor_columns(
            standardized.columns,
            self.control_columns,
            drop_controls=self.drop_control_columns,
        )
        factor_frame = standardized.loc[:, factor_columns]
        if factor_frame.shape[1] == 0:
            raise ValueError("pls requires at least one factor predictor")

        y_values = target.to_numpy(dtype=float)
        control_values = control_frame.to_numpy(dtype=float)
        self.control_coef_ = _least_squares_coef(control_values, y_values)
        residual = y_values - control_values @ self.control_coef_ if control_values.size else y_values.copy()
        resolved_components = min(self.n_components, factor_frame.shape[1], len(factor_frame))
        resolved_components = max(1, int(resolved_components))
        model = PLSRegression(
            n_components=resolved_components,
            scale=False,
            max_iter=self.max_iter,
            tol=self.tol,
            **self.kwargs,
        )
        model.fit(factor_frame.to_numpy(dtype=float), residual)
        factors = np.asarray(model.transform(factor_frame.to_numpy(dtype=float)), dtype=float)
        self.factor_coefs_ = _least_squares_coef(factors, residual)
        self.factor_square_coefs_ = (
            _least_squares_coef(factors**2, residual)
            if self.quadratic_factors
            else np.zeros(resolved_components, dtype=float)
        )
        self.factor_features_ = tuple(str(column) for column in factor_frame.columns)
        self.factor_loadings_ = np.asarray(model.x_weights_, dtype=float)
        self.n_components_ = resolved_components
        self.pls_model_ = model
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.pls_model_ is None:
            raise ValueError("pls is not fitted")
        frame = X.astype(float).copy()
        if self.x_mean_ is None or self.x_scale_ is None:
            raise ValueError("pls is missing fitted scaling state")
        frame = frame.reindex(columns=list(self.x_mean_.index), fill_value=0.0)
        values = ((frame - self.x_mean_) / self.x_scale_).to_numpy(dtype=float) if self.scale else frame.to_numpy(dtype=float)
        standardized = pd.DataFrame(values, index=frame.index, columns=self.x_mean_.index)
        factor_values = standardized.reindex(columns=list(self.factor_features_), fill_value=0.0).to_numpy(dtype=float)
        factors = np.asarray(self.pls_model_.transform(factor_values), dtype=float)
        columns = [f"pls_factor{i}" for i in range(1, self.n_components_ + 1)]
        return pd.DataFrame(factors, index=frame.index, columns=columns)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.factor_coefs_ is None:
            raise ValueError("pls is not fitted")
        factors = self.transform(X).to_numpy(dtype=float)
        frame = X.astype(float).copy()
        if self.x_mean_ is None or self.x_scale_ is None:
            raise ValueError("pls is missing fitted scaling state")
        frame = frame.reindex(columns=list(self.x_mean_.index), fill_value=0.0)
        values = ((frame - self.x_mean_) / self.x_scale_).to_numpy(dtype=float) if self.scale else frame.to_numpy(dtype=float)
        standardized = pd.DataFrame(values, index=frame.index, columns=self.x_mean_.index)
        control_values = _control_matrix(
            standardized,
            self.control_columns,
            include_constant=self.include_constant,
        ).to_numpy(dtype=float)
        control_part = control_values @ self.control_coef_ if control_values.size else 0.0
        factor_part = factors @ self.factor_coefs_
        if self.quadratic_factors and self.factor_square_coefs_ is not None:
            factor_part = factor_part + (factors**2) @ self.factor_square_coefs_
        return np.asarray(control_part + factor_part, dtype=float)


def pls(
    X: Any,
    y: Any | None = None,
    *,
    n_components: int = 3,
    scale: bool = True,
    max_iter: int = 500,
    tol: float = 1e-6,
    control_columns: Sequence[str] | None = None,
    include_constant: bool = True,
    drop_control_columns: bool = True,
    quadratic_factors: bool = False,
    **kwargs: Any,
) -> ModelFit:
    """Fit partial least squares regression."""

    frame, target = resolve_xy(X, y)
    requested_components = max(1, int(n_components))
    factor_columns = _factor_columns(
        frame.columns,
        tuple(str(column) for column in (control_columns or ())),
        drop_controls=bool(drop_control_columns),
    )
    max_components = max(1, min(len(factor_columns), len(frame)))
    resolved_components = min(requested_components, max_components)
    params = {
        "n_components": resolved_components,
        "scale": bool(scale),
        "max_iter": int(max_iter),
        "tol": float(tol),
        "control_columns": tuple(str(column) for column in (control_columns or ())),
        "include_constant": bool(include_constant),
        "drop_control_columns": bool(drop_control_columns),
        "quadratic_factors": bool(quadratic_factors),
        "source": "sklearn PLSRegression with optional Hounyo-Li PLS_emp002-style controls",
        **kwargs,
    }
    metadata = {
        **params,
        "requested_n_components": requested_components,
        "resolved_n_components": resolved_components,
        "backend": "sklearn.cross_decomposition.PLSRegression",
        "implementation_note": (
            "Uses sklearn PLS latent components; when control_columns or "
            "include_constant are set, y is residualized on the control block "
            "and the control forecast is added back, matching the Hounyo-Li "
            "PLS_emp002.m forecasting contract. quadratic_factors=True adds "
            "the PLS_PC2.m squared-factor forecast head."
        ),
    }
    return fit_estimator(
        _PLSCompositeRegressor(
            n_components=resolved_components,
            scale=bool(scale),
            max_iter=int(max_iter),
            tol=float(tol),
            control_columns=control_columns,
            include_constant=bool(include_constant),
            drop_control_columns=bool(drop_control_columns),
            quadratic_factors=bool(quadratic_factors),
            **kwargs,
        ),
        frame,
        target,
        model="pls",
        metadata=metadata,
    )


class ScaledPCARegressor:
    """Huang et al. scaled PCA factor extraction with a linear forecast head."""

    # Checked against Huang's spcaest.m scaling/factor extraction in tests.
    def __init__(
        self,
        *,
        n_components: int = 3,
        scale: bool = True,
        control_columns: Sequence[str] | None = None,
        include_constant: bool = True,
        drop_control_columns: bool = True,
        winsorize_slopes: tuple[float, float] | None = None,
        quadratic_factors: bool = False,
    ) -> None:
        self.n_components = max(1, int(n_components))
        self.scale = bool(scale)
        self.control_columns = tuple(str(column) for column in (control_columns or ()))
        self.include_constant = bool(include_constant)
        self.drop_control_columns = bool(drop_control_columns)
        self.winsorize_slopes = winsorize_slopes
        self.quadratic_factors = bool(quadratic_factors)
        self.x_mean_: pd.Series | None = None
        self.x_scale_: pd.Series | None = None
        self.factor_features_: tuple[str, ...] = ()
        self.scaling_slopes_: pd.Series | None = None
        self.factor_scores_: np.ndarray | None = None
        self.factor_loadings_: np.ndarray | None = None
        self.factor_projection_: np.ndarray | None = None
        self.factor_coefs_: np.ndarray = np.empty(0, dtype=float)
        self.factor_square_coefs_: np.ndarray = np.empty(0, dtype=float)
        self.control_coef_: np.ndarray = np.empty(0, dtype=float)
        self.control_names_: tuple[str, ...] = ()
        self.n_components_: int = 0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ScaledPCARegressor":
        frame = X.astype(float).copy()
        target = pd.Series(y, index=frame.index).astype(float)
        if frame.shape[1] == 0:
            raise ValueError("scaled_pca requires at least one predictor")

        if self.scale:
            self.x_mean_ = frame.mean(axis=0)
            self.x_scale_ = _safe_series_scale(frame.std(axis=0, ddof=1))
            standardized_values = ((frame - self.x_mean_) / self.x_scale_).to_numpy(dtype=float)
        else:
            self.x_mean_ = pd.Series(0.0, index=frame.columns)
            self.x_scale_ = pd.Series(1.0, index=frame.columns)
            standardized_values = frame.to_numpy(dtype=float)
        standardized = pd.DataFrame(standardized_values, index=frame.index, columns=frame.columns)

        control_frame = _control_matrix(
            standardized,
            self.control_columns,
            include_constant=self.include_constant,
        )
        self.control_names_ = tuple(str(column) for column in control_frame.columns)
        factor_columns = _factor_columns(
            standardized.columns,
            self.control_columns,
            drop_controls=self.drop_control_columns,
        )
        factor_frame = standardized.loc[:, factor_columns]
        factor_values = factor_frame.to_numpy(dtype=float)
        slopes = _marginal_slopes(factor_values, target.to_numpy(dtype=float))
        if self.winsorize_slopes is not None:
            slopes = _winsorize(slopes, self.winsorize_slopes)
        scaled_values = factor_values * slopes
        state = _huang_scaled_pca_state(scaled_values, self.n_components)

        control_values = control_frame.to_numpy(dtype=float)
        target_values = target.to_numpy(dtype=float)
        self.control_coef_ = _least_squares_coef(control_values, target_values)
        residual = target_values - control_values @ self.control_coef_ if control_values.size else target_values.copy()
        self.factor_coefs_ = _least_squares_coef(state["factors"], residual)
        self.factor_square_coefs_ = (
            _least_squares_coef(state["factors"] ** 2, residual)
            if self.quadratic_factors
            else np.zeros(state["n_components"], dtype=float)
        )
        self.factor_features_ = tuple(str(column) for column in factor_frame.columns)
        self.scaling_slopes_ = pd.Series(slopes, index=factor_frame.columns)
        self.factor_scores_ = state["factors"]
        self.factor_loadings_ = state["loadings"]
        self.factor_projection_ = state["projection"]
        self.n_components_ = state["n_components"]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.factor_projection_ is None or self.scaling_slopes_ is None:
            raise ValueError("scaled_pca is not fitted")
        frame = X.astype(float).copy()
        if self.x_mean_ is None or self.x_scale_ is None:
            raise ValueError("scaled_pca is missing fitted scaling state")
        frame = frame.reindex(columns=list(self.x_mean_.index), fill_value=0.0)
        values = ((frame - self.x_mean_) / self.x_scale_).to_numpy(dtype=float) if self.scale else frame.to_numpy(dtype=float)
        standardized = pd.DataFrame(values, index=frame.index, columns=self.x_mean_.index)
        factor_values = standardized.reindex(columns=list(self.factor_features_), fill_value=0.0).to_numpy(dtype=float)
        slopes = self.scaling_slopes_.reindex(self.factor_features_).to_numpy(dtype=float)
        factors = (factor_values * slopes) @ self.factor_projection_
        columns = [f"scaled_pc{i}" for i in range(1, self.n_components_ + 1)]
        return pd.DataFrame(factors, index=frame.index, columns=columns)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        factors = self.transform(X).to_numpy(dtype=float)
        frame = X.astype(float).copy()
        if self.x_mean_ is None or self.x_scale_ is None:
            raise ValueError("scaled_pca is missing fitted scaling state")
        frame = frame.reindex(columns=list(self.x_mean_.index), fill_value=0.0)
        values = ((frame - self.x_mean_) / self.x_scale_).to_numpy(dtype=float) if self.scale else frame.to_numpy(dtype=float)
        standardized = pd.DataFrame(values, index=frame.index, columns=self.x_mean_.index)
        control_values = _control_matrix(
            standardized,
            self.control_columns,
            include_constant=self.include_constant,
        ).to_numpy(dtype=float)
        control_part = control_values @ self.control_coef_ if control_values.size else 0.0
        factor_part = factors @ self.factor_coefs_
        if self.quadratic_factors:
            factor_part = factor_part + (factors**2) @ self.factor_square_coefs_
        return np.asarray(control_part + factor_part, dtype=float)


class SupervisedPCARegressor:
    """Original-style SPCA with iterative screening, PCA, and projection."""

    # Checked against the MATLAB-style recursive SPCA specification in tests.
    def __init__(
        self,
        *,
        n_components: int = 3,
        n_selected: int | None = 50,
        min_abs_corr: float = 0.0,
        scale: bool = True,
        control_columns: Sequence[str] | None = None,
        include_constant: bool = True,
        drop_control_columns: bool = True,
        preselect: str = "none",
        t_threshold: float = 1.28,
        elastic_net_alpha: float = 0.0002,
        elastic_net_l1_ratio: float = 0.5,
        slope_scale: bool = False,
        quadratic_factors: bool = False,
        random_state: int = 0,
    ) -> None:
        self.n_components = max(1, int(n_components))
        self.n_selected = None if n_selected is None else max(1, int(n_selected))
        self.min_abs_corr = float(max(0.0, min_abs_corr))
        self.scale = bool(scale)
        self.control_columns = tuple(str(column) for column in (control_columns or ()))
        self.include_constant = bool(include_constant)
        self.drop_control_columns = bool(drop_control_columns)
        self.preselect = _normalize_preselect(preselect)
        self.t_threshold = float(max(0.0, t_threshold))
        self.elastic_net_alpha = float(max(0.0, elastic_net_alpha))
        self.elastic_net_l1_ratio = float(min(1.0, max(0.0, elastic_net_l1_ratio)))
        self.slope_scale = bool(slope_scale)
        self.quadratic_factors = bool(quadratic_factors)
        self.random_state = int(random_state)
        self.selected_features_: tuple[str, ...] = ()
        self.screening_scores_: dict[str, float] = {}
        self.component_selected_features_: list[tuple[str, ...]] = []
        self.n_components_: int = 0
        self.x_mean_: pd.Series | None = None
        self.x_scale_: pd.Series | None = None
        self.y_mean_: float = 0.0
        self.y_scale_: float = 1.0
        self.factor_features_: tuple[str, ...] = ()
        self.scaling_slopes_: pd.Series | None = None
        self.loadings_: np.ndarray | None = None
        self.factor_coefs_: np.ndarray | None = None
        self.factor_square_coefs_: np.ndarray | None = None
        self.control_coef_: np.ndarray = np.empty(0, dtype=float)
        self.control_names_: tuple[str, ...] = ()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SupervisedPCARegressor":
        frame = X.astype(float).copy()
        target = pd.Series(y, index=frame.index).astype(float)
        if frame.shape[1] == 0:
            raise ValueError("supervised_pca requires at least one predictor")

        x_values = frame.to_numpy(dtype=float)
        y_values = target.to_numpy(dtype=float)
        if self.scale:
            self.x_mean_ = frame.mean(axis=0)
            self.x_scale_ = _safe_series_scale(frame.std(axis=0, ddof=1))
            x_values = ((frame - self.x_mean_) / self.x_scale_).to_numpy(dtype=float)
            self.y_mean_ = float(np.nanmean(y_values))
            y_std = float(np.nanstd(y_values, ddof=1))
            self.y_scale_ = y_std if np.isfinite(y_std) and y_std > 1e-12 else 1.0
            y_values = (y_values - self.y_mean_) / self.y_scale_
        else:
            self.x_mean_ = pd.Series(0.0, index=frame.columns)
            self.x_scale_ = pd.Series(1.0, index=frame.columns)
            self.y_mean_ = 0.0
            self.y_scale_ = 1.0

        standardized = pd.DataFrame(x_values, index=frame.index, columns=frame.columns)
        control_frame = _control_matrix(
            standardized,
            self.control_columns,
            include_constant=self.include_constant,
        )
        self.control_names_ = tuple(str(column) for column in control_frame.columns)
        factor_columns = _factor_columns(
            standardized.columns,
            self.control_columns,
            drop_controls=self.drop_control_columns,
        )
        factor_frame = standardized.loc[:, factor_columns]
        preselected = _preselect_columns(
            factor_frame,
            y_values,
            method=self.preselect,
            t_threshold=self.t_threshold,
            elastic_net_alpha=self.elastic_net_alpha,
            elastic_net_l1_ratio=self.elastic_net_l1_ratio,
            random_state=self.random_state,
        )
        factor_frame = factor_frame.loc[:, preselected]
        if factor_frame.empty:
            raise ValueError("supervised_pca selected no predictors")

        factor_values = factor_frame.to_numpy(dtype=float)
        if self.slope_scale:
            slopes = _marginal_slopes(factor_values, y_values)
            factor_values = factor_values * slopes
            self.scaling_slopes_ = pd.Series(slopes, index=factor_frame.columns)
        else:
            self.scaling_slopes_ = pd.Series(1.0, index=factor_frame.columns)

        control_values = control_frame.to_numpy(dtype=float)
        self.control_coef_ = _least_squares_coef(control_values, y_values)
        residual = y_values - control_values @ self.control_coef_ if control_values.size else y_values.copy()
        extracted = _extract_supervised_components(
            factor_values,
            residual,
            columns=tuple(str(column) for column in factor_frame.columns),
            n_components=self.n_components,
            n_selected=self.n_selected,
            min_abs_corr=self.min_abs_corr,
            quadratic_factors=self.quadratic_factors,
        )
        self.n_components_ = extracted["n_components"]
        self.factor_features_ = tuple(str(column) for column in factor_frame.columns)
        self.selected_features_ = self.factor_features_
        self.screening_scores_ = extracted["first_scores"]
        self.component_selected_features_ = extracted["component_features"]
        self.loadings_ = extracted["loadings"]
        self.factor_coefs_ = extracted["factor_coefs"]
        self.factor_square_coefs_ = extracted["factor_square_coefs"]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.loadings_ is None or self.factor_coefs_ is None:
            return np.zeros(len(X), dtype=float)
        frame = X.astype(float).copy()
        if self.x_mean_ is None or self.x_scale_ is None:
            raise ValueError("supervised_pca is missing fitted scaling state")
        frame = frame.reindex(columns=list(self.x_mean_.index), fill_value=0.0)
        values = ((frame - self.x_mean_) / self.x_scale_).to_numpy(dtype=float) if self.scale else frame.to_numpy(dtype=float)
        standardized = pd.DataFrame(values, index=frame.index, columns=self.x_mean_.index)
        control_values = _control_matrix(
            standardized,
            self.control_columns,
            include_constant=self.include_constant,
        ).to_numpy(dtype=float)
        factor_values = standardized.reindex(columns=list(self.factor_features_), fill_value=0.0).to_numpy(dtype=float)
        if self.scaling_slopes_ is not None:
            factor_values = factor_values * self.scaling_slopes_.reindex(self.factor_features_).to_numpy(dtype=float)
        factors = factor_values @ self.loadings_.T
        factor_part = factors @ self.factor_coefs_
        if self.quadratic_factors and self.factor_square_coefs_ is not None:
            factor_part = factor_part + (factors**2) @ self.factor_square_coefs_
        control_part = control_values @ self.control_coef_ if control_values.size else 0.0
        y_scaled = np.asarray(factor_part + control_part, dtype=float)
        return y_scaled * self.y_scale_ + self.y_mean_


class SupervisedScaledPCARegressor(SupervisedPCARegressor):
    """Hounyo-Li supervised scaled PCA: predictive-slope scaling plus SPCA."""

    # Same recursion as SupervisedPCARegressor, with predictive-slope scaling.
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(slope_scale=True, **kwargs)


def _safe_series_scale(scale: pd.Series) -> pd.Series:
    return scale.where(np.isfinite(scale) & (scale > 1e-12), 1.0)


def _filled_float_values(frame: pd.DataFrame) -> np.ndarray:
    return frame.fillna(frame.mean(axis=0)).fillna(0.0).to_numpy(dtype=float)


def _resolve_prior_target(
    prior_target: float | Sequence[float] | dict[str, float] | None,
    *,
    columns: pd.Index,
    n_features: int,
    default: str,
) -> np.ndarray:
    if prior_target is None:
        if default == "uniform":
            return np.full(n_features, 1.0 / max(n_features, 1), dtype=float)
        return np.zeros(n_features, dtype=float)
    if isinstance(prior_target, dict):
        return np.asarray([float(prior_target.get(str(column), 0.0)) for column in columns], dtype=float)
    if isinstance(prior_target, (int, float, np.integer, np.floating)):
        return np.full(n_features, float(prior_target), dtype=float)
    out = np.asarray(tuple(float(value) for value in prior_target), dtype=float).reshape(-1)
    if len(out) != n_features:
        raise ValueError("prior_target must be scalar, mapping by column, or one value per X column")
    return out


def _constrained_start(values: np.ndarray, *, simplex: bool, nonneg: bool) -> np.ndarray:
    start = np.asarray(values, dtype=float).copy()
    if nonneg:
        start = np.maximum(start, 0.0)
    if simplex:
        total = float(np.sum(start))
        if not np.isfinite(total) or abs(total) <= 1e-12:
            start = np.full(len(start), 1.0 / max(len(start), 1), dtype=float)
        else:
            start = start / total
        if nonneg and np.any(start < 0.0):
            start = np.maximum(start, 0.0)
            total = float(np.sum(start))
            start = (
                start / total
                if total > 1e-12
                else np.full(len(start), 1.0 / max(len(start), 1), dtype=float)
            )
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


def _difference_matrix(n_features: int, order: int) -> np.ndarray:
    if n_features <= int(order):
        return np.zeros((0, n_features), dtype=float)
    matrix: np.ndarray = np.eye(n_features, dtype=float)
    for _ in range(int(order)):
        matrix = np.diff(matrix, axis=0)
    return matrix


def _safe_array_scale(scale: np.ndarray) -> np.ndarray:
    out = np.asarray(scale, dtype=float).copy()
    out[~np.isfinite(out) | (out <= 1e-12)] = 1.0
    return out


def _soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(values) * np.maximum(np.abs(values) - float(threshold), 0.0)


def _absolute_correlations(values: np.ndarray, target: np.ndarray) -> np.ndarray:
    x_centered = values - np.nanmean(values, axis=0)
    y_centered = target - float(np.nanmean(target))
    x_scale = np.sqrt(np.nansum(x_centered * x_centered, axis=0))
    y_scale = float(np.sqrt(np.nansum(y_centered * y_centered)))
    denom = x_scale * y_scale
    numer = np.nansum(x_centered * y_centered[:, None], axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.abs(numer / denom)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _normalize_preselect(value: str) -> str:
    key = str(value).lower().replace("-", "_")
    aliases = {
        "none": "none",
        "off": "none",
        "hard": "hard_tstat",
        "hard_tstat": "hard_tstat",
        "tstat": "hard_tstat",
        "soft": "elastic_net",
        "elasticnet": "elastic_net",
        "elastic_net": "elastic_net",
    }
    if key not in aliases:
        raise ValueError("preselect must be one of: none, hard_tstat, elastic_net")
    return aliases[key]


def _factor_columns(
    columns: pd.Index,
    controls: tuple[str, ...],
    *,
    drop_controls: bool,
) -> list[str]:
    control_set = set(controls) if drop_controls else set()
    out = [str(column) for column in columns if str(column) not in control_set]
    if not out:
        raise ValueError("no factor columns remain after dropping controls")
    return out


def _control_matrix(
    frame: pd.DataFrame,
    controls: tuple[str, ...],
    *,
    include_constant: bool,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    if controls:
        missing = [column for column in controls if column not in frame.columns]
        if missing:
            raise ValueError(f"control columns are not in X: {missing}")
        parts.append(frame.loc[:, list(controls)].copy())
    if include_constant:
        parts.append(pd.DataFrame({"const": np.ones(len(frame), dtype=float)}, index=frame.index))
    if not parts:
        return pd.DataFrame(index=frame.index)
    return pd.concat(parts, axis=1)


def _least_squares_coef(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    if X.size == 0 or X.shape[1] == 0:
        return np.empty(0, dtype=float)
    return np.linalg.pinv(X) @ y


def _marginal_slopes(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    centered_x = X - np.nanmean(X, axis=0)
    centered_y = y - float(np.nanmean(y))
    denom = np.nansum(centered_x * centered_x, axis=0)
    numer = np.nansum(centered_x * centered_y[:, None], axis=0)
    out = np.zeros(X.shape[1], dtype=float)
    np.divide(numer, denom, out=out, where=denom > 1e-12)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _winsorize(values: np.ndarray, percentiles: tuple[float, float]) -> np.ndarray:
    low, high = (float(percentiles[0]), float(percentiles[1]))
    if low < 0 or high > 100 or low > high:
        raise ValueError("winsorize_slopes must contain percentiles between 0 and 100")
    lower, upper = np.nanpercentile(values, [low, high])
    return np.clip(values, lower, upper)


def _huang_scaled_pca_state(values: np.ndarray, n_components: int) -> dict[str, Any]:
    n_samples, n_features = values.shape
    n_out = min(max(1, int(n_components)), n_samples, n_features)
    u, _, _ = np.linalg.svd(values, full_matrices=False)
    # Huang et al. normalize estimated factors so F'F/T = I.
    factors = u[:, :n_out] * np.sqrt(float(n_samples))
    loadings = values.T @ factors / float(n_samples)
    projection = loadings @ np.linalg.pinv(loadings.T @ loadings)
    return {
        "n_components": n_out,
        "factors": factors,
        "loadings": loadings,
        "projection": projection,
    }


def _preselect_columns(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    method: str,
    t_threshold: float,
    elastic_net_alpha: float,
    elastic_net_l1_ratio: float,
    random_state: int,
) -> list[str]:
    if method == "none":
        return [str(column) for column in X.columns]
    values = X.to_numpy(dtype=float)
    if method == "hard_tstat":
        keep = np.abs(_marginal_t_stats(values, y)) > t_threshold
    else:
        from sklearn.linear_model import ElasticNet

        model = ElasticNet(
            alpha=elastic_net_alpha,
            l1_ratio=elastic_net_l1_ratio,
            max_iter=20000,
            random_state=random_state,
        )
        model.fit(values, y)
        keep = np.abs(np.asarray(model.coef_, dtype=float)) > 1e-12
    columns = [str(column) for column, use in zip(X.columns, keep, strict=True) if bool(use)]
    return columns or [str(column) for column in X.columns]


def _marginal_t_stats(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return _screen_marginal_t_stats(X, y)


def _extract_supervised_components(
    X: np.ndarray,
    y_residual: np.ndarray,
    *,
    columns: tuple[str, ...],
    n_components: int,
    n_selected: int | None,
    min_abs_corr: float,
    quadratic_factors: bool,
) -> dict[str, Any]:
    n_samples, n_features = X.shape
    n_out = min(max(1, int(n_components)), n_features, n_samples)
    work_x = X.copy()
    work_y = y_residual.copy()
    loadings = np.zeros((n_out, n_features), dtype=float)
    factor_coefs = np.zeros(n_out, dtype=float)
    factor_square_coefs = np.zeros(n_out, dtype=float)
    component_features: list[tuple[str, ...]] = []
    first_scores: dict[str, float] = {}
    for component in range(n_out):
        scores = _absolute_correlations(work_x, work_y)
        if component == 0:
            first_scores = {column: float(score) for column, score in zip(columns, scores, strict=True)}
        selected = _selected_indices(
            scores,
            n_features=n_features,
            n_selected=n_selected,
            min_abs_corr=min_abs_corr,
        )
        selected_x = work_x[:, selected]
        _, _, vt = np.linalg.svd(selected_x, full_matrices=False)
        loading_selected = np.asarray(vt[0], dtype=float)
        loading = np.zeros(n_features, dtype=float)
        loading[selected] = loading_selected
        factor = work_x @ loading
        denom = float(factor @ factor)
        if denom <= 1e-12:
            break
        alpha = float(work_y @ factor / denom)
        alpha2 = 0.0
        if quadratic_factors:
            squared = factor**2
            denom2 = float(squared @ squared)
            alpha2 = 0.0 if denom2 <= 1e-12 else float(work_y @ squared / denom2)
        lambdas = work_x.T @ factor / denom
        factor_coefs[component] = alpha
        factor_square_coefs[component] = alpha2
        loadings[component, :] = loading
        component_features.append(tuple(columns[int(idx)] for idx in selected))
        work_y = work_y - alpha * factor - alpha2 * (factor**2)
        work_x = work_x - np.outer(factor, lambdas)
    active = int(sum(np.linalg.norm(row) > 1e-12 for row in loadings))
    if active == 0:
        raise ValueError("supervised_pca could not extract a non-zero component")
    return {
        "n_components": active,
        "loadings": loadings[:active],
        "factor_coefs": factor_coefs[:active],
        "factor_square_coefs": factor_square_coefs[:active],
        "component_features": component_features[:active],
        "first_scores": first_scores,
    }


def _selected_indices(
    scores: np.ndarray,
    *,
    n_features: int,
    n_selected: int | None,
    min_abs_corr: float,
) -> np.ndarray:
    order = np.argsort(-scores)
    selected = [int(idx) for idx in order if scores[int(idx)] >= min_abs_corr]
    if n_selected is not None:
        selected = selected[: min(n_selected, n_features)]
    if not selected:
        fallback_n = n_features if n_selected is None else min(n_selected, n_features)
        selected = [int(idx) for idx in order[: max(1, fallback_n)]]
    return np.asarray(selected, dtype=int)


def supervised_pca(
    X: Any,
    y: Any | None = None,
    *,
    n_components: int = 3,
    n_selected: int | None = 50,
    min_abs_corr: float = 0.0,
    scale: bool = True,
    control_columns: Sequence[str] | None = None,
    include_constant: bool = True,
    drop_control_columns: bool = True,
    preselect: str = "none",
    t_threshold: float = 1.28,
    elastic_net_alpha: float = 0.0002,
    elastic_net_l1_ratio: float = 0.5,
    quadratic_factors: bool = False,
    random_state: int = 0,
) -> ModelFit:
    """Fit original-style supervised PCA regression."""

    params = {
        "n_components": int(n_components),
        "n_selected": None if n_selected is None else int(n_selected),
        "min_abs_corr": float(min_abs_corr),
        "scale": bool(scale),
        "control_columns": tuple(str(column) for column in (control_columns or ())),
        "include_constant": bool(include_constant),
        "drop_control_columns": bool(drop_control_columns),
        "preselect": _normalize_preselect(preselect),
        "t_threshold": float(t_threshold),
        "elastic_net_alpha": float(elastic_net_alpha),
        "elastic_net_l1_ratio": float(elastic_net_l1_ratio),
        "quadratic_factors": bool(quadratic_factors),
        "random_state": int(random_state),
    }
    return fit_estimator(
        SupervisedPCARegressor(
            n_components=int(n_components),
            n_selected=None if n_selected is None else int(n_selected),
            min_abs_corr=float(min_abs_corr),
            scale=bool(scale),
            control_columns=control_columns,
            include_constant=bool(include_constant),
            drop_control_columns=bool(drop_control_columns),
            preselect=preselect,
            t_threshold=float(t_threshold),
            elastic_net_alpha=float(elastic_net_alpha),
            elastic_net_l1_ratio=float(elastic_net_l1_ratio),
            quadratic_factors=bool(quadratic_factors),
            random_state=int(random_state),
        ),
        X,
        y,
        model="supervised_pca",
        metadata=params,
    )


def scaled_pca(
    X: Any,
    y: Any | None = None,
    *,
    n_components: int = 3,
    scale: bool = True,
    control_columns: Sequence[str] | None = None,
    include_constant: bool = True,
    drop_control_columns: bool = True,
    winsorize_slopes: tuple[float, float] | None = None,
    quadratic_factors: bool = False,
) -> ModelFit:
    """Fit Huang et al. scaled PCA with a linear forecast head."""

    params = {
        "n_components": int(n_components),
        "scale": bool(scale),
        "control_columns": tuple(str(column) for column in (control_columns or ())),
        "include_constant": bool(include_constant),
        "drop_control_columns": bool(drop_control_columns),
        "winsorize_slopes": winsorize_slopes,
        "quadratic_factors": bool(quadratic_factors),
        "source": "Huang et al. Management Science 2022 scaled PCA spcaest.m",
    }
    return fit_estimator(
        ScaledPCARegressor(
            n_components=int(n_components),
            scale=bool(scale),
            control_columns=control_columns,
            include_constant=bool(include_constant),
            drop_control_columns=bool(drop_control_columns),
            winsorize_slopes=winsorize_slopes,
            quadratic_factors=bool(quadratic_factors),
        ),
        X,
        y,
        model="scaled_pca",
        metadata=params,
    )


def supervised_scaled_pca(
    X: Any,
    y: Any | None = None,
    *,
    n_components: int = 3,
    n_selected: int | None = 50,
    min_abs_corr: float = 0.0,
    scale: bool = True,
    control_columns: Sequence[str] | None = None,
    include_constant: bool = True,
    drop_control_columns: bool = True,
    preselect: str = "none",
    t_threshold: float = 1.28,
    elastic_net_alpha: float = 0.0002,
    elastic_net_l1_ratio: float = 0.5,
    quadratic_factors: bool = False,
    random_state: int = 0,
) -> ModelFit:
    """Fit Hounyo-Li supervised scaled PCA regression."""

    params = {
        "n_components": int(n_components),
        "n_selected": None if n_selected is None else int(n_selected),
        "min_abs_corr": float(min_abs_corr),
        "scale": bool(scale),
        "control_columns": tuple(str(column) for column in (control_columns or ())),
        "include_constant": bool(include_constant),
        "drop_control_columns": bool(drop_control_columns),
        "preselect": _normalize_preselect(preselect),
        "t_threshold": float(t_threshold),
        "elastic_net_alpha": float(elastic_net_alpha),
        "elastic_net_l1_ratio": float(elastic_net_l1_ratio),
        "quadratic_factors": bool(quadratic_factors),
        "random_state": int(random_state),
        "source": "Hounyo and Li IJF 2026 SsPCA MATLAB reproducibility package",
    }
    return fit_estimator(
        SupervisedScaledPCARegressor(
            n_components=int(n_components),
            n_selected=None if n_selected is None else int(n_selected),
            min_abs_corr=float(min_abs_corr),
            scale=bool(scale),
            control_columns=control_columns,
            include_constant=bool(include_constant),
            drop_control_columns=bool(drop_control_columns),
            preselect=preselect,
            t_threshold=float(t_threshold),
            elastic_net_alpha=float(elastic_net_alpha),
            elastic_net_l1_ratio=float(elastic_net_l1_ratio),
            quadratic_factors=bool(quadratic_factors),
            random_state=int(random_state),
        ),
        X,
        y,
        model="supervised_scaled_pca",
        metadata=params,
    )


__all__ = [
    "ScaledPCARegressor",
    "SupervisedPCARegressor",
    "SupervisedScaledPCARegressor",
    "adaptive_elastic_net",
    "adaptive_lasso",
    "bayesian_ridge",
    "elastic_net",
    "fused_difference_ridge",
    "glmboost",
    "group_lasso",
    "huber",
    "lasso",
    "nonneg_ridge",
    "ols",
    "pls",
    "random_walk_ridge",
    "ridge",
    "scaled_pca",
    "shrink_to_target_ridge",
    "sparse_group_lasso",
    "supervised_pca",
    "supervised_scaled_pca",
]
