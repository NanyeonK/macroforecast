from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from macroforecast.models.types import VolatilityFit
from macroforecast.models.utils import as_frame, as_series, optional_import, resolve_xy


class GARCHEstimator:
    """GARCH/EGARCH wrapper around the optional `arch` package."""

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
    ) -> None:
        self.variant = variant
        self.p = int(p)
        self.o = int(o)
        self.q = int(q)
        self.mean_model = mean_model
        self.dist = dist
        self.rescale = bool(rescale)
        self.realized_variance = realized_variance
        self._fitted = None
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
        if self._fitted is None:
            return np.full(int(horizon), self._last_variance, dtype=float)
        try:
            forecast = self._fitted.forecast(horizon=int(horizon), reindex=False)
            return np.asarray(forecast.variance.iloc[-1].to_numpy(), dtype=float)
        except Exception:
            return np.full(int(horizon), self._last_variance, dtype=float)

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
) -> VolatilityFit:
    """Fit GARCH(p, q), default GARCH(1, 1)."""

    target = as_series(y)
    frame = _vol_frame(X, target)
    estimator = GARCHEstimator(variant="garch11", p=p, q=q, mean_model=mean_model, dist=dist, rescale=rescale)
    estimator.fit(frame, target)
    return VolatilityFit(estimator=estimator, model="garch11", feature_names=tuple(frame.columns), target_name=str(target.name), metadata={"n_obs": int(target.dropna().shape[0]), "p": int(p), "q": int(q)})


def egarch(
    y: Any,
    *,
    X: Any | None = None,
    p: int = 1,
    o: int = 0,
    q: int = 1,
    mean_model: str = "constant",
    dist: str = "normal",
    rescale: bool = False,
) -> VolatilityFit:
    """Fit EGARCH."""

    target = as_series(y)
    frame = _vol_frame(X, target)
    estimator = GARCHEstimator(variant="egarch", p=p, o=o, q=q, mean_model=mean_model, dist=dist, rescale=rescale)
    estimator.fit(frame, target)
    return VolatilityFit(estimator=estimator, model="egarch", feature_names=tuple(frame.columns), target_name=str(target.name), metadata={"n_obs": int(target.dropna().shape[0]), "p": int(p), "o": int(o), "q": int(q)})


class RealizedGARCHEstimator:
    """Compact Hansen-Huang-Shek-style realized GARCH joint MLE."""

    _bounds = [
        (None, None),
        (None, None),
        (1e-6, 1.0 - 1e-6),
        (None, None),
        (None, None),
        (None, None),
        (None, None),
        (1e-6, 1.0),
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
        self.max_iter = int(max_iter)
        self.n_starts = int(n_starts)
        self.random_state = int(random_state)
        self.params_: dict[str, float] = {}
        self._h: np.ndarray | None = None
        self._mu = 0.0
        self._last_h = 1.0

    @staticmethod
    def _forward(theta: np.ndarray, r: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mu, omega, beta, tau1, tau2, gamma, xi, phi, delta1, delta2, _ = theta
        h = np.empty_like(r, dtype=float)
        z = np.empty_like(r, dtype=float)
        u = np.empty_like(r, dtype=float)
        log_h_prev = float(np.log(max(np.var(r), 1e-6)))
        z_prev = 0.0
        u_prev = 0.0
        for i in range(len(r)):
            log_h = omega + beta * log_h_prev + tau1 * z_prev + tau2 * (z_prev**2 - 1.0) + gamma * u_prev
            h_i = float(np.exp(np.clip(log_h, -30.0, 30.0)))
            h_i = max(h_i, 1e-8)
            z_i = (r[i] - mu) / np.sqrt(h_i)
            u_i = np.log(max(x[i], 1e-12)) - (xi + phi * np.log(h_i) + delta1 * z_i + delta2 * (z_i**2 - 1.0))
            h[i] = h_i
            z[i] = z_i
            u[i] = u_i
            log_h_prev = np.log(h_i)
            z_prev = z_i
            u_prev = u_i
        return h, z, u

    @classmethod
    def _neg_loglike(cls, theta: np.ndarray, r: np.ndarray, x: np.ndarray) -> float:
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
            0.7,
            0.0,
            0.0,
            0.1,
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
        if best is None:
            raise RuntimeError("realized_garch optimization did not run")
        theta = np.asarray(best.x, dtype=float)
        names = ("mu", "omega", "beta", "tau_1", "tau_2", "gamma", "xi", "phi", "delta_1", "delta_2", "log_sigma_u")
        self.params_ = {name: float(value) for name, value in zip(names, theta)}
        self._mu = self.params_["mu"]
        self._h, z, u = self._forward(theta, r_arr, x_arr)
        self._last_h = float(self._h[-1])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self._mu, dtype=float)

    def predict_variance(self, horizon: int = 1) -> np.ndarray:
        if not self.params_:
            return np.full(int(horizon), self._last_h, dtype=float)
        omega = self.params_["omega"]
        beta = self.params_["beta"]
        out = np.empty(int(horizon), dtype=float)
        log_h = np.log(max(self._last_h, 1e-12))
        for i in range(int(horizon)):
            log_h = omega + beta * log_h
            out[i] = float(np.exp(np.clip(log_h, -30.0, 30.0)))
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
    return VolatilityFit(estimator=estimator, model="realized_garch", feature_names=tuple(frame.columns), target_name=str(target.name), metadata={"n_obs": int(target.dropna().shape[0]), "realized_variance": realized_column})


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


__all__ = [
    "GARCHEstimator",
    "RealizedGARCHEstimator",
    "egarch",
    "garch11",
    "realized_garch",
]
