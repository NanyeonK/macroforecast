from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from macroforecast.models.types import VolatilityFit
from macroforecast.models.utils import as_frame, as_series, optional_import


class GARCHEstimator:
    """GARCH/EGARCH wrapper around the optional `arch` package."""

    # Source alignment:
    # - R rugarch implements sGARCH/eGARCH through ugarchspec() + ugarchfit()
    #   and native C/R likelihood/filter code. macroforecast does not duplicate
    #   that likelihood. It delegates these two models to Python's arch package,
    #   mapping the public orders onto arch.arch_model(vol="GARCH", p=p, q=q)
    #   and arch.arch_model(vol="EGARCH", p=p, o=o, q=q).
    # - The statistical target is the same GARCH-family conditional variance
    #   contract, while solver controls, parameter naming, and some distribution
    #   aliases are backend-defined by arch rather than rugarch.

    def __init__(
        self,
        *,
        variant: str = "garch11",
        p: int = 1,
        o: int = 0,
        q: int = 1,
        mean_model: str = "constant",
        dist: str = "normal",
        rescale: bool = False,
        realized_variance: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.variant = variant
        self.p = _validate_positive_int("p", p)
        self.o = _validate_nonnegative_int("o", o)
        self.q = _validate_positive_int("q", q)
        self.mean_model = mean_model
        self.dist = dist
        self.rescale = bool(rescale)
        self.realized_variance = realized_variance
        self.kwargs = dict(kwargs)
        self._fitted: Any = None
        self._mu = 0.0
        self._last_variance = 1.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GARCHEstimator":
        arch = optional_import("arch", extra="arch")
        r = pd.Series(y).astype(float).dropna()
        if len(r) < 30:
            raise ValueError(f"{self.variant} requires at least 30 non-missing observations")
        arch_model = arch.arch_model
        kwargs: dict[str, Any] = {
            "mean": self.mean_model,
            "dist": self.dist,
            "rescale": self.rescale,
            **self.kwargs,
        }
        if self.variant == "garch11":
            model = arch_model(r, vol="GARCH", p=self.p, q=self.q, **kwargs)
        elif self.variant == "egarch":
            model = arch_model(r, vol="EGARCH", p=self.p, o=self.o, q=self.q, **kwargs)
        else:
            rv = _realized_measure(X, r, self.realized_variance)
            model = arch_model(r, vol="GARCH", p=1, q=1, x=rv.to_frame(), **kwargs)
        self._fitted = model.fit(disp="off", show_warning=False)
        params = self._fitted.params.to_dict()
        self._mu = float(params.get("mu", r.mean()))
        try:
            self._last_variance = float(self._fitted.conditional_volatility.iloc[-1] ** 2)
        except Exception:
            self._last_variance = float(r.var(ddof=1))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self._mu, dtype=float)

    def predict_variance(self, horizon: int = 1) -> np.ndarray:
        # Conditional-expectation forecast of the log-linear realized GARCH.
        # Returns exp(E[log h_{t+k}]) -- the conditional MEDIAN of the log-normal
        # variance. For multi-step horizons this is biased low relative to the
        # mean E[h] by the Jensen gap 0.5*Var[log h]; treat the output as the
        # conditional median, not the mean variance.
        horizon = _validate_positive_int("horizon", horizon)
        if self._fitted is None:
            return np.full(horizon, self._last_variance, dtype=float)
        try:
            forecast = self._fitted.forecast(horizon=horizon, reindex=False)
            return np.asarray(forecast.variance.iloc[-1].to_numpy(), dtype=float)
        except Exception:
            return np.full(horizon, self._last_variance, dtype=float)

    @property
    def conditional_volatility_(self) -> np.ndarray | None:
        if self._fitted is None:
            return None
        return np.asarray(self._fitted.conditional_volatility, dtype=float)


def garch11(
    y: Any,
    *,
    X: Any | None = None,
    p: int = 1,
    q: int = 1,
    mean_model: str = "constant",
    dist: str = "normal",
    rescale: bool = False,
    **kwargs: Any,
) -> VolatilityFit:
    """Fit GARCH(p, q), default GARCH(1, 1)."""

    target = as_series(y)
    frame = _vol_frame(X, target)
    estimator = GARCHEstimator(
        variant="garch11",
        p=p,
        q=q,
        mean_model=mean_model,
        dist=dist,
        rescale=rescale,
        **kwargs,
    )
    estimator.fit(frame, target)
    return VolatilityFit(
        estimator=estimator,
        model="garch11",
        feature_names=tuple(frame.columns),
        target_name=str(target.name),
        metadata={
            "n_obs": int(target.dropna().shape[0]),
            "p": int(p),
            "q": int(q),
            "mean_model": mean_model,
            "dist": dist,
            "rescale": bool(rescale),
            "backend": "arch.arch_model",
            "implementation_note": (
                "Delegates likelihood estimation and variance forecasting to "
                "arch.arch_model(vol='GARCH'); comparable rugarch surface is "
                "ugarchspec(variance.model=list(model='sGARCH'))."
            ),
            **kwargs,
        },
        diagnostics=_volatility_diagnostics(estimator),
    )


def egarch(
    y: Any,
    *,
    X: Any | None = None,
    p: int = 1,
    o: int = 1,
    q: int = 1,
    mean_model: str = "constant",
    dist: str = "normal",
    rescale: bool = False,
    **kwargs: Any,
) -> VolatilityFit:
    """Fit EGARCH.

    ``o`` defaults to 1 so the asymmetric leverage term that defines Nelson's
    EGARCH is present (matching rugarch's eGARCH), deliberately overriding the
    ``arch`` backend's own ``o=0`` default (which gives a symmetric log-GARCH).
    """

    target = as_series(y)
    frame = _vol_frame(X, target)
    estimator = GARCHEstimator(
        variant="egarch",
        p=p,
        o=o,
        q=q,
        mean_model=mean_model,
        dist=dist,
        rescale=rescale,
        **kwargs,
    )
    estimator.fit(frame, target)
    return VolatilityFit(
        estimator=estimator,
        model="egarch",
        feature_names=tuple(frame.columns),
        target_name=str(target.name),
        metadata={
            "n_obs": int(target.dropna().shape[0]),
            "p": int(p),
            "o": int(o),
            "q": int(q),
            "mean_model": mean_model,
            "dist": dist,
            "rescale": bool(rescale),
            "backend": "arch.arch_model",
            "implementation_note": (
                "Delegates likelihood estimation and variance forecasting to "
                "arch.arch_model(vol='EGARCH'); comparable rugarch surface is "
                "ugarchspec(variance.model=list(model='eGARCH'))."
            ),
            **kwargs,
        },
        diagnostics=_volatility_diagnostics(estimator),
    )


class RealizedGARCHEstimator:
    """Compact Hansen-Huang-Shek-style realized GARCH joint MLE."""

    # Source comparison:
    # - R rugarch realGARCH uses Hansen, Huang, and Shek's log-linear realized
    #   GARCH skeleton. In rugarch/src/filters.c::realgarchfilter(), the C
    #   recursion adds omega, lagged log realized volatility through alpha,
    #   and lagged log variance through beta, then applies the measurement
    #   equation u_t = log(x_t) - xi - delta log(h_t) - tau(z_t).
    # - This estimator intentionally implements the same compact p=q=1 Gaussian
    #   skeleton:
    #     log h_t = omega + alpha log x_{t-1} + beta log h_{t-1}
    #     z_t = (r_t - mu) / sqrt(h_t)
    #     tau(z_t) = eta_1 z_t + eta_2 (z_t^2 - 1)
    #     log x_t = xi + delta log h_t + tau(z_t) + u_t
    #   with r_t = mu + sqrt(h_t) z_t and Gaussian u_t.
    # - This is not a full rugarch backend clone. It omits ARMA/ARFIMA mean
    #   dynamics, variance regressors, non-Gaussian distributions, fixed/se
    #   parameter machinery, robust inference, simulation/path/roll helpers,
    #   and xts-specific realizedVol validation. The multi-step variance
    #   forecast below is the conditional-expectation recursion with future
    #   tau(z_t) and measurement shocks set to zero.
    _bounds = [
        (None, None),
        (None, None),
        (1e-8, 1.0 - 1e-8),
        (1e-8, 1.0 - 1e-8),
        (None, None),
        (1e-8, 10.0),
        (None, None),
        (None, None),
        (-30.0, 5.0),
    ]

    def __init__(
        self,
        *,
        realized_variance: str | None = None,
        max_iter: int = 2000,
        n_starts: int = 5,
        random_state: int = 0,
    ) -> None:
        self.realized_variance = realized_variance
        self.max_iter = _validate_positive_int("max_iter", max_iter)
        self.n_starts = _validate_positive_int("n_starts", n_starts)
        self.random_state = int(random_state)
        self.params_: dict[str, float] = {}
        self._h: np.ndarray | None = None
        self._mu = 0.0
        self._last_h = 1.0
        self._last_log_x = 0.0

    @staticmethod
    def _forward(theta: np.ndarray, r: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mu, omega, alpha, beta, xi, delta, eta1, eta2, _ = theta
        h = np.empty_like(r, dtype=float)
        z = np.empty_like(r, dtype=float)
        u = np.empty_like(r, dtype=float)
        log_h_prev = float(np.log(max(np.var(r), 1e-6)))
        log_x_prev = float(np.log(max(np.mean(x), 1e-12)))
        for i in range(len(r)):
            # rugarch realgarchfilter() equivalent for p=q=1: lagged log
            # realized volatility enters through alpha and lagged log variance
            # enters through beta. The first lag is initialized at the sample
            # mean realized measure, matching the compact callable contract.
            log_h = omega + alpha * log_x_prev + beta * log_h_prev
            h_i = float(np.exp(np.clip(log_h, -30.0, 30.0)))
            h_i = max(h_i, 1e-8)
            z_i = (r[i] - mu) / np.sqrt(h_i)
            tau_i = eta1 * z_i + eta2 * (z_i**2 - 1.0)
            # Measurement equation aligned with rugarch:
            # u_t = log(x_t) - xi - delta log(h_t) - tau(z_t). If the caller
            # does not supply a realized measure, _realized_measure supplies
            # r_t^2 as an explicit proxy rather than silently changing the
            # model family.
            log_x_i = np.log(max(x[i], 1e-12))
            u_i = log_x_i - xi - delta * np.log(h_i) - tau_i
            h[i] = h_i
            z[i] = z_i
            u[i] = u_i
            log_h_prev = np.log(h_i)
            log_x_prev = log_x_i
        return h, z, u

    @classmethod
    def _neg_loglike(cls, theta: np.ndarray, r: np.ndarray, x: np.ndarray) -> float:
        alpha = float(theta[2])
        beta = float(theta[3])
        delta = float(theta[5])
        persistence = beta + delta * alpha
        if not np.isfinite(persistence) or persistence >= 1.0:
            return float("inf")
        h, z, u = cls._forward(theta, r, x)
        sigma_u = float(np.exp(theta[-1]))
        ll_return = 0.5 * (np.log(2.0 * np.pi) + np.log(h) + z**2)
        ll_measure = 0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(sigma_u) + (u / sigma_u) ** 2)
        value = float(np.sum(ll_return + ll_measure))
        return value if np.isfinite(value) else float("inf")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RealizedGARCHEstimator":
        r = pd.Series(y).astype(float).dropna()
        x = _realized_measure(X, r, self.realized_variance).reindex(r.index).astype(float).clip(lower=1e-12)
        joined = pd.concat([r.rename("r"), x.rename("x")], axis=1).dropna()
        if len(joined) < 30:
            raise ValueError(f"realized_garch requires at least 30 aligned observations; got {len(joined)}")
        r_arr = joined["r"].to_numpy(dtype=float)
        x_arr = joined["x"].to_numpy(dtype=float)
        self._mu = float(r_arr.mean())
        theta0 = np.asarray([
            self._mu,
            np.log(max(np.var(r_arr), 1e-6)) * 0.05,
            0.05,
            0.7,
            float(np.log(np.mean(x_arr))),
            0.5,
            0.0,
            0.0,
            float(np.log(max(np.std(np.log(x_arr)), 1e-3))),
        ])
        rng = np.random.default_rng(self.random_state)
        starts = [theta0]
        for i in range(max(0, self.n_starts - 1)):
            starts.append(theta0 + rng.normal(scale=0.1 + 0.05 * i, size=theta0.shape))
        best = None
        for start in starts:
            start[2] = float(np.clip(start[2], 1e-6, 0.5))
            start[3] = float(np.clip(start[3], 1e-6, 0.95))
            start[5] = float(np.clip(start[5], 1e-6, 2.0))
            res = minimize(
                self._neg_loglike,
                start,
                args=(r_arr, x_arr),
                method="L-BFGS-B",
                bounds=self._bounds,
                options={"maxiter": self.max_iter},
            )
            if best is None or float(res.fun) < float(best.fun):
                best = res
        if best is None or not np.isfinite(float(best.fun)):
            raise RuntimeError("realized_garch optimization did not find a finite solution")
        theta = np.asarray(best.x, dtype=float)
        names = ("mu", "omega", "alpha", "beta", "xi", "delta", "eta_1", "eta_2", "log_sigma_u")
        self.params_ = {name: float(value) for name, value in zip(names, theta)}
        self.params_["persistence"] = float(self.params_["beta"] + self.params_["delta"] * self.params_["alpha"])
        self._mu = self.params_["mu"]
        self._h, z, u = self._forward(theta, r_arr, x_arr)
        self._last_h = float(self._h[-1])
        self._last_log_x = float(np.log(max(x_arr[-1], 1e-12)))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self._mu, dtype=float)

    def predict_variance(self, horizon: int = 1) -> np.ndarray:
        # Conditional-expectation forecast of the log-linear realized GARCH.
        # Returns exp(E[log h_{t+k}]) -- the conditional MEDIAN of the log-normal
        # variance. For multi-step horizons this is biased low relative to the
        # mean E[h] by the Jensen gap 0.5*Var[log h]; treat the output as the
        # conditional median, not the mean variance.
        horizon = _validate_positive_int("horizon", horizon)
        if not self.params_:
            return np.full(horizon, self._last_h, dtype=float)
        omega = self.params_["omega"]
        alpha = self.params_["alpha"]
        beta = self.params_["beta"]
        xi = self.params_["xi"]
        delta = self.params_["delta"]
        out: np.ndarray = np.empty(horizon, dtype=float)
        log_h = np.log(max(self._last_h, 1e-12))
        log_x = self._last_log_x
        for i in range(horizon):
            # Conditional-expectation forecast: use the latest observed
            # realized measure for the first step, then propagate E[log x_t |
            # h_t] = xi + delta log h_t after setting future tau(z_t) and
            # measurement shocks to zero.
            log_h = omega + alpha * log_x + beta * log_h
            out[i] = float(np.exp(np.clip(log_h, -30.0, 30.0)))
            log_x = xi + delta * log_h
        return out

    @property
    def conditional_volatility_(self) -> np.ndarray | None:
        if self._h is None:
            return None
        return np.sqrt(np.asarray(self._h, dtype=float))


def realized_garch(
    y: Any,
    *,
    X: Any | None = None,
    rv: Any | None = None,
    realized_variance: str | None = None,
    max_iter: int = 2000,
    n_starts: int = 5,
    random_state: int = 0,
) -> VolatilityFit:
    """Fit realized GARCH with a realized-measurement equation."""

    target = as_series(y)
    realized_column: str | None
    if rv is not None:
        frame = pd.DataFrame({"rv": as_series(rv).reindex(target.index)})
        realized_column = "rv"
    else:
        frame = _vol_frame(X, target)
        realized_column = realized_variance
    estimator = RealizedGARCHEstimator(
        realized_variance=realized_column,
        max_iter=max_iter,
        n_starts=n_starts,
        random_state=random_state,
    )
    estimator.fit(frame, target)
    return VolatilityFit(
        estimator=estimator,
        model="realized_garch",
        feature_names=tuple(frame.columns),
        target_name=str(target.name),
        metadata={
            "n_obs": int(target.dropna().shape[0]),
            "realized_variance": realized_column,
            "max_iter": int(max_iter),
            "n_starts": int(n_starts),
            "random_state": int(random_state),
            "implementation_note": (
                "Compact Hansen-Huang-Shek/rugarch-style p=q=1 Gaussian "
                "log-linear realized GARCH likelihood; simplified "
                "conditional-expectation variance forecasting."
            ),
        },
        diagnostics=_volatility_diagnostics(estimator),
    )


def _vol_frame(X: Any | None, y: pd.Series) -> pd.DataFrame:
    if X is None:
        return pd.DataFrame(index=y.index)
    return as_frame(X).reindex(y.index)


def _realized_measure(X: pd.DataFrame, r: pd.Series, column: str | None) -> pd.Series:
    if column and column in X.columns:
        return pd.to_numeric(X[column], errors="coerce").reindex(r.index)
    if "rv" in X.columns:
        return pd.to_numeric(X["rv"], errors="coerce").reindex(r.index)
    return (r**2).rename("rv_proxy")


def _volatility_diagnostics(estimator: Any) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    params = getattr(estimator, "params_", None)
    if params is None:
        fitted = getattr(estimator, "_fitted", None)
        if fitted is not None:
            params = getattr(fitted, "params", None)
    if params is not None:
        if hasattr(params, "to_dict"):
            diagnostics["params"] = params.to_dict()
        else:
            diagnostics["params"] = dict(params)
    conditional = getattr(estimator, "conditional_volatility_", None)
    if conditional is not None:
        diagnostics["conditional_volatility"] = pd.Series(
            np.asarray(conditional, dtype=float),
            name="conditional_volatility",
        )
    return diagnostics


def _validate_positive_int(name: str, value: Any) -> int:
    out = int(value)
    if out < 1:
        raise ValueError(f"{name} must be positive")
    return out


def _validate_nonnegative_int(name: str, value: Any) -> int:
    out = int(value)
    if out < 0:
        raise ValueError(f"{name} must be non-negative")
    return out


__all__ = [
    "GARCHEstimator",
    "RealizedGARCHEstimator",
    "egarch",
    "garch11",
    "realized_garch",
]
