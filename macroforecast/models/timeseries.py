from __future__ import annotations

import warnings

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from macroforecast.models.types import ModelFit
from macroforecast.models.utils import as_frame, as_series, fit_estimator, resolve_xy


class _AR:
    def __init__(self, *, n_lag: int = 1) -> None:
        self.n_lag = max(1, int(n_lag))
        self._coef: np.ndarray | None = None
        self._history: np.ndarray | None = None
        self._fallback: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_AR":
        series = pd.Series(y).astype(float).dropna()
        self._fallback = float(series.mean()) if not series.empty else 0.0
        if len(series) <= self.n_lag:
            self._history = series.to_numpy(dtype=float)
            return self
        rows = []
        target = []
        values = series.to_numpy(dtype=float)
        for i in range(self.n_lag, len(values)):
            rows.append([1.0, *values[i - self.n_lag : i][::-1]])
            target.append(values[i])
        design = np.asarray(rows, dtype=float)
        response = np.asarray(target, dtype=float)
        self._coef = np.linalg.lstsq(design, response, rcond=None)[0]
        self._history = values[-self.n_lag :]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._coef is None or self._history is None or len(self._history) == 0:
            return np.full(len(X), self._fallback, dtype=float)
        history = list(np.asarray(self._history, dtype=float))
        preds: list[float] = []
        for _ in range(len(X)):
            row = np.asarray([1.0, *history[-self.n_lag :][::-1]], dtype=float)
            pred = float(row @ self._coef)
            preds.append(pred)
            history.append(pred)
        return np.asarray(preds, dtype=float)


def ar(y: Any, *, n_lag: int = 1) -> ModelFit:
    """Fit an autoregression on a single target series."""

    target = as_series(y)
    dummy = pd.DataFrame({"__origin__": np.arange(len(target), dtype=float)}, index=target.index)
    return fit_estimator(_AR(n_lag=n_lag), dummy, target, model="ar", metadata={"n_lag": int(n_lag)})


class _VAR:
    # R source alignment, vars/R/VAR.R and vars/R/predict.varest.R:
    # - build ylags with lag order 1..p, append deterministic terms according
    #   to type in {"const", "trend", "both", "none"}, then fit one OLS
    #   equation per endogenous variable with no extra intercept in the formula.
    # - forecast recursion uses the last stacked endogenous lags and future
    #   deterministic rows exactly as predict.varest does.
    # - macroforecast does not reproduce the full S3 varest object, confidence
    #   intervals, diagnostics, IRF, FEVD, or restriction machinery here. It
    #   does reproduce the reduced-form equation and point-forecast recursion.

    def __init__(
        self,
        *,
        n_lag: int = 1,
        target: str | None = None,
        type: str = "const",
        season: int | None = None,
    ) -> None:
        self.n_lag = max(1, int(n_lag))
        self.target = target
        self.type = _normalize_vars_type(type)
        self.season = None if season is None else max(2, int(season))
        self.coef_: np.ndarray | None = None
        self.names_: tuple[str, ...] = ()
        self.rhs_names_: tuple[str, ...] = ()
        self.datamat_: pd.DataFrame | None = None
        self.y_values_: np.ndarray | None = None
        self._target_name: str | None = None
        self._fallback: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "_VAR":
        if y is None:
            data = X.astype(float).copy()
            target_name = self.target or str(data.columns[0])
        else:
            data = pd.concat([pd.Series(y).rename("__target__"), X], axis=1).astype(float)
            target_name = "__target__"
        if data.empty:
            return self
        if data.isna().any().any():
            raise ValueError("vars::VAR-compatible fitting does not allow missing values")
        if data.shape[1] < 2:
            raise ValueError("VAR panel must contain at least two variables")
        if target_name not in data.columns:
            raise ValueError(f"target {target_name!r} is not in the VAR panel")
        self._target_name = target_name
        self.names_ = tuple(str(column) for column in data.columns)
        self._fallback = float(pd.to_numeric(data[target_name], errors="coerce").mean())
        if data.shape[0] <= self.n_lag + 1 or data.shape[1] < 2:
            self.y_values_ = data.to_numpy(dtype=float)
            return self
        values = data.to_numpy(dtype=float)
        yend, rhs, rhs_names = _vars_rhs(
            values,
            self.names_,
            self.n_lag,
            type=self.type,
            season=self.season,
        )
        self.coef_ = np.linalg.lstsq(rhs, yend, rcond=None)[0].T
        self.rhs_names_ = tuple(rhs_names)
        self.y_values_ = values
        self.datamat_ = pd.DataFrame(
            np.column_stack([yend, rhs]),
            index=data.index[self.n_lag :],
            columns=[*self.names_, *self.rhs_names_],
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.coef_ is None or self.y_values_ is None or self._target_name is None:
            return np.full(len(X), self._fallback, dtype=float)
        forecast = _vars_forecast(
            self.coef_,
            self.y_values_,
            len(X),
            self.n_lag,
            type=self.type,
            season=self.season,
        )
        target_index = self.names_.index(self._target_name)
        return np.asarray(forecast[:, target_index], dtype=float)[: len(X)]


def var(
    panel: Any,
    *,
    target: str | None = None,
    n_lag: int = 1,
    type: str = "const",
    season: int | None = None,
) -> ModelFit:
    """Fit a vector autoregression on a multivariate panel."""

    frame = as_frame(panel)
    estimator = _VAR(n_lag=n_lag, target=target, type=type, season=season)
    estimator.fit(frame)
    return ModelFit(
        estimator=estimator,
        model="var",
        feature_names=tuple(str(c) for c in frame.columns),
        target_name=target or str(frame.columns[0]),
        metadata={
            "n_obs": len(frame.dropna()),
            "n_lag": int(n_lag),
            "type": estimator.type,
            "season": estimator.season,
            "backend": "internal vars::VAR-aligned OLS",
            "implementation_note": (
                "R vars::VAR-aligned OLS design and predict.varest-style "
                "recursive point forecasts."
            ),
        },
    )


class _BayesianVAR:
    """FAVAR::BVAR-aligned Bayesian VAR posterior sampler."""

    # R source alignment, FAVAR/R/BVAR.R plus bvartools::minnesota_prior:
    # - construct VAR data with deterministic='none';
    # - draw VAR coefficients from the conditional normal posterior;
    # - draw Sigma^{-1} from a Wishart posterior and store Sigma draws;
    # - summarize coefficients by posterior means, standard errors, and
    #   2.5/97.5 percentiles.
    # The Python RNG cannot be bit-identical to R's rWishart/MCMC stream, but
    # the model, prior shapes, stored draw arrays, and point-forecast coefficient
    # summary follow the R package logic rather than a compact ridge surrogate.

    def __init__(
        self,
        *,
        n_lag: int = 1,
        target: str | None = None,
        prior: str = "normal_inverse_wishart",
        b0: float = 0.0,
        vb0: float = 0.0,
        nu0: float = 0.0,
        s0: float | Sequence[Sequence[float]] | None = 0.0,
        kappa0: float | None = None,
        kappa1: float | None = None,
        iter: int = 10000,
        burnin: int = 5000,
        random_state: int = 0,
    ) -> None:
        self.n_lag = max(1, int(n_lag))
        self.target = target
        self.prior = str(prior)
        self.b0 = float(b0)
        self.vb0 = float(vb0)
        self.nu0 = float(nu0)
        self.s0 = s0
        self.kappa0 = None if kappa0 is None else float(kappa0)
        self.kappa1 = None if kappa1 is None else float(kappa1)
        self.iter = max(1, int(iter))
        self.burnin = max(0, int(burnin))
        if self.burnin >= self.iter:
            raise ValueError("burnin must be smaller than iter")
        self.random_state = int(random_state)
        self.names_: tuple[str, ...] = ()
        self.target_name_: str | None = None
        self.coef_: np.ndarray | None = None
        self.coef_draws_: np.ndarray | None = None
        self.sigma_draws_: np.ndarray | None = None
        self.summary_: dict[str, np.ndarray] = {}
        self.last_values_: np.ndarray | None = None
        self.fallback_: float = 0.0
        self.column_means_: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "_BayesianVAR":
        if y is None:
            data = X.astype(float).copy()
            target_name = self.target or str(data.columns[0])
        else:
            data = pd.concat([pd.Series(y).rename("__target__"), X], axis=1).astype(float)
            target_name = "__target__"
        if data.empty:
            return self
        if data.isna().any().any():
            raise ValueError("FAVAR::BVAR-compatible fitting does not allow missing values")
        self.names_ = tuple(str(column) for column in data.columns)
        if target_name not in data.columns:
            raise ValueError(f"target {target_name!r} is not in the VAR panel")
        self.target_name_ = target_name
        self.fallback_ = float(data[target_name].mean())
        # The FAVAR::BVAR design is intentionally intercept-free, so the panel
        # must be demeaned before fitting; otherwise a non-zero series mean is
        # absorbed by an inflated (near-unit-root) own-lag coefficient and the
        # forecast cannot revert to the unconditional mean. The column means are
        # added back when forecasting (mirroring how _FAVAR standardizes).
        self.column_means_ = data.mean().to_numpy(dtype=float)
        data = data - data.mean()
        values = data.to_numpy(dtype=float)
        if len(values) <= self.n_lag:
            self.last_values_ = values[-self.n_lag :]
            return self
        draws = _favar_bvar_draws(
            values,
            self.n_lag,
            prior=self.prior,
            b0=self.b0,
            vb0=self.vb0,
            nu0=self.nu0,
            s0=self.s0,
            kappa0=self.kappa0,
            kappa1=self.kappa1,
            n_iter=self.iter,
            burnin=self.burnin,
            random_state=self.random_state,
        )
        self.coef_draws_ = draws["coef_draws"]
        self.sigma_draws_ = draws["sigma_draws"]
        self.summary_ = draws["summary"]
        self.coef_ = self.summary_["mean"]
        self.last_values_ = values[-self.n_lag :]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.coef_ is None or self.last_values_ is None or self.target_name_ is None:
            return np.full(len(X), self.fallback_, dtype=float)
        forecast = _vars_forecast(
            self.coef_,
            np.asarray(self.last_values_, dtype=float),
            len(X),
            self.n_lag,
            type="none",
            season=None,
        )
        target_pos = self.names_.index(self.target_name_)
        centered = np.asarray(forecast[:, target_pos], dtype=float)
        if self.column_means_ is not None:
            centered = centered + float(self.column_means_[target_pos])
        return centered


def bvar_minnesota(
    panel: Any,
    *,
    target: str | None = None,
    n_lag: int = 1,
    kappa0: float = 2.0,
    kappa1: float = 0.5,
    nu0: float = 0.0,
    s0: float | Sequence[Sequence[float]] | None = 0.0,
    iter: int = 10000,
    burnin: int = 5000,
    random_state: int = 0,
) -> ModelFit:
    """Fit a FAVAR::BVAR-style Bayesian VAR with Minnesota prior variances."""

    frame = as_frame(panel)
    estimator = _BayesianVAR(
        n_lag=n_lag,
        target=target,
        prior="minnesota",
        nu0=nu0,
        s0=s0,
        kappa0=kappa0,
        kappa1=kappa1,
        iter=iter,
        burnin=burnin,
        random_state=random_state,
    )
    estimator.fit(frame)
    return ModelFit(
        estimator=estimator,
        model="bvar_minnesota",
        feature_names=tuple(str(c) for c in frame.columns),
        target_name=target or str(frame.columns[0]),
        metadata={
            "n_obs": len(frame.dropna()),
            "n_lag": int(n_lag),
            "kappa0": float(kappa0),
            "kappa1": float(kappa1),
            "nu0": float(nu0),
            "s0": _jsonable_prior_matrix(s0),
            "iter": int(iter),
            "burnin": int(burnin),
            "n_saved": int(iter) - int(burnin),
            "random_state": int(random_state),
            "backend": "internal FAVAR::BVAR-aligned Gibbs sampler",
            "implementation_note": (
                "FAVAR::BVAR / bvartools::minnesota_prior-aligned posterior "
                "draw sampler; point forecasts use posterior mean coefficients."
            ),
        },
        diagnostics=_bvar_diagnostics(estimator),
    )


def bvar_normal_inverse_wishart(
    panel: Any,
    *,
    target: str | None = None,
    n_lag: int = 1,
    b0: float = 0.0,
    vb0: float = 0.0,
    nu0: float = 0.0,
    s0: float | Sequence[Sequence[float]] | None = 0.0,
    iter: int = 10000,
    burnin: int = 5000,
    random_state: int = 0,
) -> ModelFit:
    """Fit a FAVAR::BVAR-style Bayesian VAR with normal-Wishart priors."""

    frame = as_frame(panel)
    estimator = _BayesianVAR(
        n_lag=n_lag,
        target=target,
        prior="normal_inverse_wishart",
        b0=b0,
        vb0=vb0,
        nu0=nu0,
        s0=s0,
        iter=iter,
        burnin=burnin,
        random_state=random_state,
    )
    estimator.fit(frame)
    return ModelFit(
        estimator=estimator,
        model="bvar_normal_inverse_wishart",
        feature_names=tuple(str(c) for c in frame.columns),
        target_name=target or str(frame.columns[0]),
        metadata={
            "n_obs": len(frame.dropna()),
            "n_lag": int(n_lag),
            "b0": float(b0),
            "vb0": float(vb0),
            "nu0": float(nu0),
            "s0": _jsonable_prior_matrix(s0),
            "iter": int(iter),
            "burnin": int(burnin),
            "n_saved": int(iter) - int(burnin),
            "random_state": int(random_state),
            "backend": "internal FAVAR::BVAR-aligned Gibbs sampler",
            "implementation_note": (
                "FAVAR::BVAR-aligned normal-Wishart posterior draw sampler; "
                "point forecasts use posterior mean coefficients."
            ),
        },
        diagnostics=_bvar_diagnostics(estimator),
    )


class _FAR:
    def __init__(self, *, n_factors: int = 3, n_lag: int = 1, random_state: int = 0) -> None:
        self.n_factors = max(1, int(n_factors))
        self.n_lag = max(1, int(n_lag))
        self.random_state = int(random_state)
        self._pca: Any = None
        self._regression: Any = None
        self._x_mean: pd.Series | None = None
        self._y_history: np.ndarray | None = None
        self._fallback: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_FAR":
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LinearRegression

        joined = pd.concat([X, y.rename("__target__")], axis=1).dropna()
        if joined.empty:
            return self
        X_clean = joined.drop(columns="__target__")
        y_clean = joined["__target__"]
        self._fallback = float(y_clean.mean())
        self._x_mean = X_clean.mean(axis=0)
        n_factors = min(self.n_factors, X_clean.shape[1], max(1, X_clean.shape[0] - 1))
        self._pca = PCA(n_components=n_factors, random_state=self.random_state)
        factors = self._pca.fit_transform((X_clean - self._x_mean).fillna(0.0))
        values = y_clean.to_numpy(dtype=float)
        rows = []
        target = []
        for i in range(self.n_lag, len(values)):
            rows.append([*factors[i], *values[i - self.n_lag : i][::-1]])
            target.append(values[i])
        if not rows:
            self._y_history = values[-self.n_lag :]
            return self
        self._regression = LinearRegression().fit(np.asarray(rows), np.asarray(target))
        self._y_history = values[-self.n_lag :]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._pca is None or self._regression is None or self._x_mean is None or self._y_history is None:
            return np.full(len(X), self._fallback, dtype=float)
        frame = X.reindex(columns=self._x_mean.index, fill_value=0.0)
        factors = self._pca.transform((frame - self._x_mean).fillna(0.0))
        history = list(np.asarray(self._y_history, dtype=float))
        preds = []
        for i in range(len(X)):
            row = np.asarray([*factors[i], *history[-self.n_lag :][::-1]], dtype=float)
            pred = float(self._regression.predict(row.reshape(1, -1))[0])
            preds.append(pred)
            history.append(pred)
        return np.asarray(preds, dtype=float)


def far(
    X: Any,
    y: Any | None = None,
    *,
    n_factors: int = 3,
    n_lag: int = 1,
    random_state: int = 0,
) -> ModelFit:
    """Fit factor-augmented autoregression."""

    return fit_estimator(
        _FAR(n_factors=n_factors, n_lag=n_lag, random_state=random_state),
        X,
        y,
        model="far",
        metadata={"n_factors": int(n_factors), "n_lag": int(n_lag), "random_state": int(random_state)},
    )


class _FAVAR:
    """FAVAR::FAVAR-aligned Bayesian FAVAR sampler."""

    # R source alignment, FAVAR/R/FAVAR.R:
    # - optionally standardize X and Y with R scale() semantics;
    # - extract principal components with ExtrPC();
    # - use BBE facrot() or BGM() to remove fast-variable information;
    # - draw factor loading equations with the same conjugate regression logic
    #   as MCMCpack::MCMCregress;
    # - estimate the VAR block by the FAVAR::BVAR-aligned sampler above.
    #
    # BVAR forecasting itself is standard; for example CRAN BVAR exposes
    # predict.bvar over posterior beta/sigma draws. CRAN FAVAR, however, exposes
    # summary/coef/irf for favar objects and uses its internal BVAR() as a
    # posterior state-draw engine rather than exposing predict.favar.
    # macroforecast predict() is therefore only the ModelFit forecast wrapper
    # over the FAVAR posterior VAR state, using posterior-mean coefficients.

    def __init__(
        self,
        *,
        n_factors: int = 2,
        n_lag: int = 2,
        fctmethod: str = "BBE",
        slowcode: Sequence[bool] | None = None,
        factorprior: Mapping[str, Any] | None = None,
        varprior: Mapping[str, Any] | None = None,
        nburn: int = 5000,
        nrep: int = 15000,
        standardize: bool = True,
        random_state: int = 0,
    ) -> None:
        self.n_factors = max(1, int(n_factors))
        self.n_lag = max(1, int(n_lag))
        self.fctmethod = str(fctmethod).upper()
        if self.fctmethod not in {"BBE", "BGM"}:
            raise ValueError("fctmethod must be 'BBE' or 'BGM'")
        self.slowcode = None if slowcode is None else tuple(bool(value) for value in slowcode)
        self.factorprior = dict(factorprior or {})
        self.varprior = dict(varprior or {})
        self.nburn = max(0, int(nburn))
        self.nrep = max(1, int(nrep))
        self.standardize = bool(standardize)
        self.random_state = int(random_state)
        self.feature_names_in_: tuple[str, ...] = ()
        self.target_name_: str | None = None
        self.x_mean_: np.ndarray | None = None
        self.x_scale_: np.ndarray | None = None
        self.y_mean_: float = 0.0
        self.y_scale_: float = 1.0
        self.factorx_: pd.DataFrame | None = None
        self.loading_draws_: np.ndarray | None = None
        self.var_coef_draws_: np.ndarray | None = None
        self.var_sigma_draws_: np.ndarray | None = None
        self.var_summary_: dict[str, np.ndarray] = {}
        self.coef_: np.ndarray | None = None
        self.last_fy_: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_FAVAR":
        frame = X.astype(float).copy()
        target = pd.Series(y, index=frame.index).astype(float)
        if frame.isna().any().any() or target.isna().any():
            raise ValueError("FAVAR::FAVAR-compatible fitting does not allow missing values")
        self.feature_names_in_ = tuple(str(column) for column in frame.columns)
        self.target_name_ = str(target.name) if target.name is not None else "y"
        x_raw = frame.to_numpy(dtype=float)
        y_raw = target.to_numpy(dtype=float).reshape(-1, 1)
        if self.standardize:
            self.x_mean_, self.x_scale_, x_work = _r_scale_matrix(x_raw)
            y_mean, y_scale, y_work = _r_scale_matrix(y_raw)
            self.y_mean_ = float(y_mean[0])
            self.y_scale_ = float(y_scale[0])
        else:
            self.x_mean_ = np.zeros(x_raw.shape[1], dtype=float)
            self.x_scale_ = np.ones(x_raw.shape[1], dtype=float)
            self.y_mean_ = 0.0
            self.y_scale_ = 1.0
            x_work = x_raw
            y_work = y_raw
        factors0, loadings0 = _favar_extr_pc(x_work, self.n_factors)
        if self.fctmethod == "BBE":
            if self.slowcode is None:
                raise ValueError("slowcode is required when fctmethod='BBE', matching FAVAR::FAVAR")
            if len(self.slowcode) != x_work.shape[1]:
                raise ValueError("slowcode must have one boolean value per X column")
            slow_x = x_work[:, np.asarray(self.slowcode, dtype=bool)]
            if slow_x.shape[1] < self.n_factors:
                raise ValueError("slowcode must select at least n_factors slow variables")
            slow_factors, _ = _favar_extr_pc(slow_x, self.n_factors)
            factorx = _favar_facrot(factors0, y_work[:, [-1]], slow_factors)
        else:
            factorx = _favar_bgm(x_work, y_work[:, -1], self.n_factors)
        fy = np.column_stack([factorx, y_work])
        loading_matrix = _favar_olssvd(np.column_stack([x_work, y_work]), fy).T
        self.loading_draws_ = _favar_loading_draws(
            x_work,
            fy,
            loading_matrix,
            self.n_factors,
            self.factorprior,
            self.nburn,
            self.nrep,
            self.random_state,
        )
        var_kwargs = _parse_favar_varprior(self.varprior)
        draws = _favar_bvar_draws(
            fy,
            self.n_lag,
            n_iter=self.nrep + self.nburn,
            burnin=self.nburn,
            random_state=self.random_state,
            **var_kwargs,
        )
        self.var_coef_draws_ = draws["coef_draws"]
        self.var_sigma_draws_ = draws["sigma_draws"]
        self.var_summary_ = draws["summary"]
        self.coef_ = self.var_summary_["mean"]
        self.last_fy_ = fy[-self.n_lag :, :]
        self.factorx_ = pd.DataFrame(
            factorx,
            index=frame.index,
            columns=[f"factor_{i + 1}" for i in range(factorx.shape[1])],
        )
        # Keep loadings for diagnostics; future-X projection is not used by
        # predict() because R FAVAR forecasts the VAR state, not new X rows.
        self.pc_loadings_ = loadings0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.coef_ is None or self.last_fy_ is None:
            return np.zeros(len(X), dtype=float)
        forecast = _vars_forecast(
            self.coef_,
            np.asarray(self.last_fy_, dtype=float),
            len(X),
            self.n_lag,
            type="none",
            season=None,
        )
        y_scaled = forecast[:, -1]
        return y_scaled * self.y_scale_ + self.y_mean_


def favar(
    X: Any,
    y: Any | None = None,
    *,
    n_factors: int = 2,
    n_lag: int = 2,
    fctmethod: str = "BBE",
    slowcode: Sequence[bool] | None = None,
    factorprior: Mapping[str, Any] | None = None,
    varprior: Mapping[str, Any] | None = None,
    nburn: int = 5000,
    nrep: int = 15000,
    standardize: bool = True,
    random_state: int = 0,
) -> ModelFit:
    """Fit a FAVAR::FAVAR-aligned Bayesian factor-augmented VAR."""

    frame, target = resolve_xy(X, y)
    estimator = _FAVAR(
        n_factors=n_factors,
        n_lag=n_lag,
        fctmethod=fctmethod,
        slowcode=slowcode,
        factorprior=factorprior,
        varprior=varprior,
        nburn=nburn,
        nrep=nrep,
        standardize=standardize,
        random_state=random_state,
    )
    estimator.fit(frame, target)
    return ModelFit(
        estimator=estimator,
        model="favar",
        feature_names=tuple(str(column) for column in frame.columns),
        target_name=str(target.name) if target.name is not None else None,
        metadata={
            "n_obs": len(frame),
            "n_factors": int(n_factors),
            "n_lag": int(n_lag),
            "fctmethod": estimator.fctmethod,
            "slowcode": None if slowcode is None else [bool(value) for value in slowcode],
            "nburn": int(nburn),
            "nrep": int(nrep),
            "standardize": bool(standardize),
            "random_state": int(random_state),
            "backend": "internal FAVAR::FAVAR-aligned Bayesian sampler",
            "implementation_note": (
                "FAVAR::FAVAR-aligned factor extraction, loading draws, and "
                "BVAR posterior draws; predict() is the ModelFit forecast "
                "wrapper over the FAVAR posterior VAR state. BVAR forecasting "
                "itself is standard, but CRAN FAVAR does not expose predict.favar."
            ),
        },
        diagnostics=_favar_diagnostics(estimator),
    )


class _MixedFrequencyDFM:
    """Statsmodels DynamicFactorMQ wrapper for monthly/quarterly panels."""

    # Source alignment:
    # - statsmodels DynamicFactorMQ implements the Banbura-Modugno /
    #   Banbura-Giannone-Reichlin EM state-space DFM and explicitly supports
    #   Mariano-Murasawa monthly/quarterly aggregation when monthly columns are
    #   ordered first and quarterly columns afterwards with k_endog_monthly set.
    # - R dfms::DFM(X, quarterly.vars=...) and archived R nowcasting::nowcast()
    #   use the same high-level contract: monthly variables first, quarterly
    #   variables last, missing quarterly observations at monthly dates, and
    #   temporal aggregation weights [1, 2, 3, 2, 1] for quarterly growth/flow
    #   variables.
    # - This class is intentionally a backend wrapper, not a reimplementation of
    #   dfms' R/C++ Kalman/EM code or nowcasting's block-DFM routines.

    def __init__(
        self,
        *,
        target: str,
        n_factors: int = 1,
        factor_order: int = 1,
        idiosyncratic_ar1: bool = True,
        standardize: bool = True,
        maxiter: int = 500,
        tolerance: float = 1e-6,
    ) -> None:
        self.target = str(target)
        self.n_factors = max(1, int(n_factors))
        self.factor_order = max(1, int(factor_order))
        self.idiosyncratic_ar1 = bool(idiosyncratic_ar1)
        self.standardize = bool(standardize)
        self.maxiter = max(1, int(maxiter))
        self.tolerance = float(tolerance)
        self.result_: Any = None
        self.feature_names_in_: tuple[str, ...] = ()
        self.monthly_columns_: tuple[str, ...] = ()
        self.quarterly_columns_: tuple[str, ...] = ()
        self.fallback_: float = 0.0
        self.fitted_target_: pd.Series | None = None
        self.factors_filtered_: pd.DataFrame | None = None
        self.params_: pd.Series | None = None
        self.llf_: float | None = None
        self.fit_error_: str | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "_MixedFrequencyDFM":
        del y
        frame = X.astype(float).copy()
        self.feature_names_in_ = tuple(str(column) for column in frame.columns)
        if self.target not in frame.columns:
            raise ValueError(f"target {self.target!r} is not in the mixed-frequency panel")
        observed_target = frame[self.target].dropna()
        self.fallback_ = float(observed_target.iloc[-1]) if len(observed_target) else 0.0
        if not self.monthly_columns_:
            raise ValueError("at least one monthly column is required for DynamicFactorMQ")
        k_monthly = len(self.monthly_columns_)
        if self.n_factors > k_monthly:
            raise ValueError("n_factors must be less than or equal to the number of monthly columns")
        try:
            from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ

            model = DynamicFactorMQ(
                frame,
                k_endog_monthly=k_monthly,
                factors=self.n_factors,
                factor_orders=self.factor_order,
                idiosyncratic_ar1=self.idiosyncratic_ar1,
                standardize=self.standardize,
            )
            self.result_ = model.fit(
                maxiter=self.maxiter,
                tolerance=self.tolerance,
                disp=False,
            )
            fitted = getattr(self.result_, "fittedvalues", None)
            if isinstance(fitted, pd.DataFrame) and self.target in fitted.columns:
                self.fitted_target_ = fitted[self.target].rename("fitted")
            factors = getattr(self.result_, "factors", None)
            filtered = getattr(factors, "filtered", None) if factors is not None else None
            if isinstance(filtered, pd.DataFrame):
                self.factors_filtered_ = filtered
            params = getattr(self.result_, "params", None)
            if isinstance(params, pd.Series):
                self.params_ = params
            llf = getattr(self.result_, "llf", None)
            self.llf_ = None if llf is None else float(llf)
        except Exception as exc:  # noqa: BLE001 - keep model callable usable on short panels.
            self.result_ = None
            self.fit_error_ = str(exc)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        steps = len(X)
        if steps <= 0:
            return np.empty(0, dtype=float)
        if self.result_ is None:
            return np.full(steps, self.fallback_, dtype=float)
        try:
            forecast = self.result_.forecast(steps=steps)
            if isinstance(forecast, pd.DataFrame) and self.target in forecast.columns:
                return forecast[self.target].to_numpy(dtype=float)[:steps]
            values = np.asarray(forecast, dtype=float)
            target_pos = self.feature_names_in_.index(self.target)
            return values[:, target_pos].reshape(-1)[:steps]
        except Exception:
            return np.full(steps, self.fallback_, dtype=float)


def dfm_mixed_mariano_murasawa(
    panel: Any,
    *,
    target: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    monthly_columns: Iterable[str] | None = None,
    quarterly_columns: Iterable[str] | None = None,
    unsupported: str = "raise",
    n_factors: int = 1,
    factor_order: int = 1,
    idiosyncratic_ar1: bool = True,
    standardize: bool = True,
    maxiter: int = 500,
    tolerance: float = 1e-6,
) -> ModelFit:
    """Fit a monthly/quarterly DynamicFactorMQ model with MM aggregation."""

    frame, meta = _coerce_panel_with_metadata(panel, metadata=metadata)
    prepared, monthly, quarterly = _prepare_mixed_frequency_panel(
        frame,
        metadata=meta,
        monthly_columns=monthly_columns,
        quarterly_columns=quarterly_columns,
        unsupported=unsupported,
    )
    target_name = target or (quarterly[0] if quarterly else prepared.columns[0])
    if target_name not in prepared.columns:
        raise ValueError(f"target {target_name!r} is not in the mixed-frequency panel")
    estimator = _MixedFrequencyDFM(
        target=target_name,
        n_factors=n_factors,
        factor_order=factor_order,
        idiosyncratic_ar1=idiosyncratic_ar1,
        standardize=standardize,
        maxiter=maxiter,
        tolerance=tolerance,
    )
    estimator.monthly_columns_ = tuple(monthly)
    estimator.quarterly_columns_ = tuple(quarterly)
    estimator.fit(prepared)
    diagnostics = _mixed_dfm_diagnostics(estimator, prepared)
    return ModelFit(
        estimator=estimator,
        model="dfm_mixed_mariano_murasawa",
        feature_names=tuple(str(column) for column in prepared.columns),
        target_name=target_name,
        metadata={
            "n_obs": int(prepared.shape[0]),
            "target": target_name,
            "n_factors": int(n_factors),
            "factor_order": int(factor_order),
            "idiosyncratic_ar1": bool(idiosyncratic_ar1),
            "standardize": bool(standardize),
            "maxiter": int(maxiter),
            "tolerance": float(tolerance),
            "monthly_columns": list(monthly),
            "quarterly_columns": list(quarterly),
            "native_frequency_counts": dict(meta.get("native_frequency_counts", {})),
            "backend": "statsmodels.tsa.statespace.dynamic_factor_mq.DynamicFactorMQ",
            "implementation_note": (
                "Backend wrapper around statsmodels DynamicFactorMQ. The input "
                "contract follows R dfms::DFM(..., quarterly.vars=...) and archived "
                "nowcasting::nowcast(method='EM'): monthly columns first, quarterly "
                "columns last, and Mariano-Murasawa [1,2,3,2,1] aggregation handled "
                "inside the state-space model."
            ),
        },
        diagnostics=diagnostics,
    )


class _DFMUnrestrictedMIDAS:
    """Composite DFM-factor plus unrestricted-MIDAS forecast head."""

    # This is the callable analogue of a common two-stage workflow in the R
    # ecosystem, not a single R package estimator: estimate a mixed-frequency DFM
    # (dfms/nowcasting style), extract monthly filtered factors, align factor and
    # observed lags to target anchors, then fit an unrestricted MIDAS head. The
    # final head matches midasr::midas_u only when alpha=0; alpha>0 is a ridge
    # extension for macroforecast model selection.

    def __init__(
        self,
        *,
        midas_estimator: Any,
        feature_names: tuple[str, ...],
        target: str,
        metadata: Mapping[str, Any],
        lag_columns: tuple[str, ...],
        lags: tuple[int, ...],
        factor_lags: tuple[int, ...],
        target_frequency: str | None,
        anchor_position: str,
        dfm_params: Mapping[str, Any],
        drop_missing: bool,
    ) -> None:
        self.midas_estimator = midas_estimator
        self.feature_names_in_ = tuple(feature_names)
        self.target = str(target)
        self.metadata = dict(metadata)
        self.lag_columns = tuple(lag_columns)
        self.lags = tuple(lags)
        self.factor_lags = tuple(factor_lags)
        self.target_frequency = target_frequency
        self.anchor_position = str(anchor_position)
        self.dfm_params = dict(dfm_params)
        self.drop_missing = bool(drop_missing)
        self.coef_ = getattr(midas_estimator, "coef_", None)
        self.intercept_ = getattr(midas_estimator, "intercept_", None)
        self.groups_ = getattr(midas_estimator, "groups_", {})
        self.design_: pd.DataFrame | None = None
        self.last_prediction_design_: pd.DataFrame | None = None

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        frame = X.reindex(columns=list(self.feature_names_in_), fill_value=np.nan).astype(float)
        return np.asarray(self.midas_estimator.predict(frame), dtype=float).reshape(-1)

    def predict_from_panel(
        self,
        panel: Any,
        *,
        metadata: Mapping[str, Any] | None = None,
        anchor_dates: Iterable[Any] | None = None,
    ) -> np.ndarray:
        """Rebuild the DFM-factor/MIDAS design from a native-frequency panel."""

        frame, meta = _coerce_panel_with_metadata(panel, metadata=metadata or self.metadata)
        design, _ = _dfm_unrestricted_midas_design(
            frame,
            metadata=meta,
            target=self.target,
            lag_columns=self.lag_columns,
            lags=self.lags,
            factor_lags=self.factor_lags,
            target_frequency=self.target_frequency,
            anchor_position=self.anchor_position,
            anchor_dates=anchor_dates,
            drop_missing=False,
            **self.dfm_params,
        )
        design = design.reindex(columns=list(self.feature_names_in_), fill_value=np.nan)
        self.last_prediction_design_ = design
        if design.isna().any().any():
            missing = sorted(str(column) for column in design.columns[design.isna().any()])
            raise ValueError(
                "DFM unrestricted MIDAS prediction design contains missing values; "
                f"missing columns: {missing}"
            )
        return self.predict(design)


def dfm_unrestricted_midas(
    panel: Any,
    *,
    target: str,
    metadata: Mapping[str, Any] | None = None,
    lag_columns: Iterable[str] | None = None,
    lags: Iterable[int] | int = (0, 1, 2),
    factor_lags: Iterable[int] | int = (0,),
    target_frequency: str | None = "quarterly",
    anchor_position: str = "period_end",
    n_factors: int = 1,
    factor_order: int = 1,
    idiosyncratic_ar1: bool = True,
    standardize: bool = True,
    maxiter: int = 500,
    tolerance: float = 1e-6,
    alpha: float = 0.0,
    fit_intercept: bool = True,
    drop_missing: bool = True,
) -> ModelFit:
    """Fit DFM factors plus unrestricted MIDAS lag coefficients."""

    frame, meta = _coerce_panel_with_metadata(panel, metadata=metadata)
    if target not in frame.columns:
        raise ValueError(f"target {target!r} is not in the mixed-frequency panel")
    lag_values = _normalize_model_lags(lags)
    factor_lag_values = _normalize_model_lags(factor_lags)
    lag_column_values = tuple(str(column) for column in lag_columns or ())
    dfm_params = {
        "n_factors": int(n_factors),
        "factor_order": int(factor_order),
        "idiosyncratic_ar1": bool(idiosyncratic_ar1),
        "standardize": bool(standardize),
        "maxiter": int(maxiter),
        "tolerance": float(tolerance),
    }
    design, dfm_fit = _dfm_unrestricted_midas_design(
        frame,
        metadata=meta,
        target=target,
        lag_columns=lag_column_values,
        lags=lag_values,
        factor_lags=factor_lag_values,
        target_frequency=target_frequency,
        anchor_position=anchor_position,
        drop_missing=False,
        **dfm_params,
    )
    anchor_index = _position_target_dates(
        frame[target].dropna().index,
        frequency=target_frequency or meta.get("frequency") or "quarterly",
        anchor_position=anchor_position,
    )
    y = pd.Series(frame[target].dropna().to_numpy(dtype=float), index=anchor_index, name=target)
    joined = pd.concat([design, y.rename("__target__")], axis=1)
    if drop_missing:
        joined = joined.dropna()
    if joined.empty:
        raise ValueError("DFM unrestricted MIDAS design has no complete rows")
    X_fit = joined.drop(columns="__target__")
    y_fit = joined["__target__"].rename(target)
    midas_fit = unrestricted_midas(
        X_fit,
        y_fit,
        alpha=alpha,
        fit_intercept=fit_intercept,
    )
    estimator = _DFMUnrestrictedMIDAS(
        midas_estimator=midas_fit.estimator,
        feature_names=tuple(str(column) for column in X_fit.columns),
        target=target,
        metadata=meta,
        lag_columns=lag_column_values,
        lags=lag_values,
        factor_lags=factor_lag_values,
        target_frequency=target_frequency,
        anchor_position=anchor_position,
        dfm_params=dfm_params,
        drop_missing=drop_missing,
    )
    estimator.design_ = X_fit
    return ModelFit(
        estimator=estimator,
        model="dfm_unrestricted_midas",
        feature_names=tuple(str(column) for column in X_fit.columns),
        target_name=target,
        metadata={
            "n_obs": int(len(X_fit)),
            "target": str(target),
            "lag_columns": list(lag_column_values),
            "lags": list(lag_values),
            "factor_lags": list(factor_lag_values),
            "target_frequency": target_frequency,
            "anchor_position": str(anchor_position),
            "n_factors": int(n_factors),
            "factor_order": int(factor_order),
            "alpha": float(alpha),
            "fit_intercept": bool(fit_intercept),
            "implementation_note": (
                "Composite model: fit DynamicFactorMQ factors from a native mixed-frequency "
                "panel, append selected observed mixed-frequency lags, then fit unrestricted MIDAS."
            ),
            "prediction_contract": (
                "predict() accepts a prepared feature matrix with fitted feature columns; "
                "predict_from_panel() rebuilds the composite design from a native panel."
            ),
        },
        diagnostics={
            "design": X_fit,
            "dfm": dfm_fit.to_dict(),
            "midas": midas_fit.to_dict(),
        },
    )


def _dfm_unrestricted_midas_design(
    frame: pd.DataFrame,
    *,
    metadata: Mapping[str, Any],
    target: str,
    lag_columns: Iterable[str],
    lags: Iterable[int] | int,
    factor_lags: Iterable[int] | int,
    target_frequency: str | None,
    anchor_position: str,
    n_factors: int,
    factor_order: int,
    idiosyncratic_ar1: bool,
    standardize: bool,
    maxiter: int,
    tolerance: float,
    anchor_dates: Iterable[Any] | None = None,
    drop_missing: bool = False,
) -> tuple[pd.DataFrame, ModelFit]:
    """Build the reusable composite DFM-factor plus observed-lag design."""

    from macroforecast.feature_engineering import mixed_frequency_lags

    dfm_fit = dfm_mixed_mariano_murasawa(
        (frame, metadata),
        target=target,
        n_factors=n_factors,
        factor_order=factor_order,
        idiosyncratic_ar1=idiosyncratic_ar1,
        standardize=standardize,
        maxiter=maxiter,
        tolerance=tolerance,
    )
    factors = dfm_fit.diagnostics.get("factors_filtered")
    if not isinstance(factors, pd.DataFrame) or factors.empty:
        raise ValueError("DFM fit did not produce filtered factors")
    raw_anchor_dates = (
        pd.DatetimeIndex(pd.to_datetime(list(anchor_dates)))
        if anchor_dates is not None
        else pd.DatetimeIndex(frame[target].dropna().index)
    )
    if raw_anchor_dates.empty:
        raise ValueError("DFM unrestricted MIDAS design has no target anchor dates")
    anchor_index = _position_target_dates(
        raw_anchor_dates,
        frequency=target_frequency or metadata.get("frequency") or "quarterly",
        anchor_position=anchor_position,
    )
    design_parts: list[pd.DataFrame] = [
        _factor_lag_design(factors, anchor_index=anchor_index, lags=factor_lags)
    ]
    lag_column_values = tuple(str(column) for column in lag_columns or ())
    if lag_column_values:
        observed_lags = mixed_frequency_lags(
            (frame, metadata),
            target=target,
            anchor_dates=raw_anchor_dates,
            columns=lag_column_values,
            lags=lags,
            target_frequency=target_frequency,
            anchor_position=anchor_position,
            drop_missing=False,
        )
        design_parts.append(observed_lags)
    design = pd.concat(design_parts, axis=1)
    if drop_missing:
        design = design.dropna()
    return design, dfm_fit


class _MIDASRegressor:
    """Linear MIDAS-style regression over lag groups."""

    # R source alignment, midasr/R/lagspec.R and midasr/R/midasreg.R:
    # - midasr::midas_r jointly estimates restricted lag-weight parameters and
    #   regression coefficients by nonlinear least squares.
    # - macroforecast's restricted MIDAS callables are fixed-shape design
    #   builders: the supplied or selected weight-shape hyperparameters compress
    #   each lag block, then a linear/ridge forecast head estimates the aggregate
    #   coefficient. This is equivalent to a midasr restricted design conditional
    #   on fixed shape parameters, not to midasr's NLS optimizer.
    # - midas_almon uses the scale-free part of midasr::nealmon, midas_beta uses
    #   midasr::nbetaMT with p=(1, a, b, 0), and midas_step uses normalized
    #   polystep-style piecewise-constant weights.

    def __init__(
        self,
        *,
        weighting: str = "almon",
        polynomial_order: int = 2,
        theta: tuple[float, ...] | None = None,
        beta_params: tuple[float, float] = (1.0, 1.0),
        n_steps: int = 3,
        step_bounds: tuple[int, ...] | None = None,
        step_weights: tuple[float, ...] | None = None,
        alpha: float = 0.0,
        fit_intercept: bool = True,
    ) -> None:
        self.weighting = str(weighting)
        self.polynomial_order = int(polynomial_order)
        self.theta = None if theta is None else tuple(float(value) for value in theta)
        self.beta_params = (float(beta_params[0]), float(beta_params[1]))
        self.n_steps = max(1, int(n_steps))
        self.step_bounds = None if step_bounds is None else tuple(int(value) for value in step_bounds)
        self.step_weights = None if step_weights is None else tuple(float(value) for value in step_weights)
        self.alpha = float(alpha)
        self.fit_intercept = bool(fit_intercept)
        self.feature_names_in_: tuple[str, ...] = ()
        self.design_columns_: tuple[str, ...] = ()
        self.diagnostic_feature_names_: tuple[str, ...] = ()
        self.groups_: dict[str, list[tuple[str, int]]] = {}
        self.weights_: dict[str, list[float]] = {}
        self.effective_lag_coefficients_: pd.Series | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self._regression: Any = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_MIDASRegressor":
        from sklearn.linear_model import LinearRegression, Ridge

        self.feature_names_in_ = tuple(str(column) for column in X.columns)
        frame = X.astype(float).copy()
        self.groups_ = _midas_groups(frame.columns)
        design = self._design(frame)
        if self.alpha > 0.0:
            self._regression = Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept)
        else:
            self._regression = LinearRegression(fit_intercept=self.fit_intercept)
        self._regression.fit(design, y.astype(float))
        self.design_columns_ = tuple(str(column) for column in design.columns)
        self.diagnostic_feature_names_ = self.design_columns_
        self.coef_ = np.asarray(getattr(self._regression, "coef_", []), dtype=float).reshape(-1)
        self.intercept_ = float(getattr(self._regression, "intercept_", 0.0))
        self.effective_lag_coefficients_ = _effective_lag_coefficients(
            groups=self.groups_,
            weights=self.weights_,
            aggregate_coefficients=self.coef_,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._regression is None:
            return np.zeros(len(X), dtype=float)
        frame = X.reindex(columns=list(self.feature_names_in_), fill_value=np.nan).astype(float)
        design = self._design(frame)
        return np.asarray(self._regression.predict(design), dtype=float).reshape(-1)

    def _design(self, frame: pd.DataFrame) -> pd.DataFrame:
        pieces: dict[str, pd.Series] = {}
        self.weights_ = {}
        for group, members in self.groups_.items():
            ordered = sorted(members, key=lambda item: item[1])
            columns = [column for column, _ in ordered]
            lags = [lag for _, lag in ordered]
            weights = _midas_weights(
                lags,
                weighting=self.weighting,
                polynomial_order=self.polynomial_order,
                theta=self.theta,
                beta_params=self.beta_params,
                n_steps=self.n_steps,
                step_bounds=self.step_bounds,
                step_weights=self.step_weights,
            )
            self.weights_[group] = weights.tolist()
            block = frame.loc[:, columns].astype(float)
            pieces[f"{group}_midas"] = block.mul(weights, axis=1).sum(
                axis=1,
                min_count=len(columns),
            )
        return pd.DataFrame(pieces, index=frame.index)


class _RestrictedMIDASRegressor:
    """Nonlinear restricted MIDAS regression over explicit lag groups."""

    # R source alignment, midasr/R/midasreg.R and midasr/R/lagspec.R:
    # - midasr::midas_r builds an explicit unrestricted lag matrix, maps each
    #   restricted term's low-dimensional parameters into full lag
    #   coefficients, then minimizes sum of squared residuals by nonlinear
    #   least squares. Its default optimizer is optim(method="BFGS"), with nls
    #   also supported.
    # - This estimator follows the same objective and the same nealmon,
    #   nbetaMT, and polystep coefficient maps for already-built lag columns.
    #   SciPy least_squares replaces R's optimizer, so iterates are not
    #   bit-identical, but the regression equation and coefficient restrictions
    #   are the same.
    # - Formula parsing, AR* common-factor terms, HAC covariance methods, model
    #   tables, and forecast.midas_r S3 utilities remain outside this callable;
    #   macroforecast receives X as an explicit lag matrix.

    def __init__(
        self,
        *,
        weighting: str = "almon",
        polynomial_order: int = 2,
        start_params: Mapping[str, Sequence[float]] | Sequence[float] | None = None,
        n_steps: int = 3,
        step_bounds: tuple[int, ...] | None = None,
        fit_intercept: bool = True,
        maxiter: int = 1000,
        tolerance: float = 1e-8,
    ) -> None:
        self.weighting = str(weighting)
        self.polynomial_order = int(polynomial_order)
        self.start_params = start_params
        self.n_steps = max(1, int(n_steps))
        self.step_bounds = None if step_bounds is None else tuple(int(value) for value in step_bounds)
        self.fit_intercept = bool(fit_intercept)
        self.maxiter = max(1, int(maxiter))
        self.tolerance = float(tolerance)
        self.feature_names_in_: tuple[str, ...] = ()
        self.groups_: dict[str, list[tuple[str, int]]] = {}
        self.group_param_slices_: dict[str, tuple[int, int]] = {}
        self.param_names_: tuple[str, ...] = ()
        self.params_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.coef_: np.ndarray | None = None
        self.effective_lag_coefficients_: pd.Series | None = None
        self.converged_: bool = False
        self.n_iter_: int = 0
        self.cost_: float | None = None
        self.message_: str = ""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_RestrictedMIDASRegressor":
        from scipy.optimize import least_squares

        self.feature_names_in_ = tuple(str(column) for column in X.columns)
        self.groups_ = _midas_groups(pd.Index(self.feature_names_in_))
        frame = X.astype(float)
        target = y.astype(float)
        start, names = self._start_vector(frame, target)
        self.param_names_ = tuple(names)

        def residual(params: np.ndarray) -> np.ndarray:
            return target.to_numpy(dtype=float) - self._predict_array(frame, params)

        opt = least_squares(
            residual,
            start,
            max_nfev=self.maxiter,
            xtol=self.tolerance,
            ftol=self.tolerance,
            gtol=self.tolerance,
        )
        self.params_ = np.asarray(opt.x, dtype=float)
        self.converged_ = bool(opt.success)
        self.n_iter_ = int(opt.nfev)
        self.cost_ = float(2.0 * opt.cost)
        self.message_ = str(opt.message)
        self.intercept_ = float(self.params_[0]) if self.fit_intercept else 0.0
        self.coef_ = self._full_coefficients(self.params_)
        self.effective_lag_coefficients_ = pd.Series(
            self.coef_,
            index=list(self.feature_names_in_),
            name="effective_lag_coefficient",
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.params_ is None:
            return np.zeros(len(X), dtype=float)
        frame = X.reindex(columns=list(self.feature_names_in_), fill_value=np.nan).astype(float)
        return self._predict_array(frame, self.params_)

    def _start_vector(self, frame: pd.DataFrame, target: pd.Series) -> tuple[np.ndarray, list[str]]:
        values: list[float] = []
        names: list[str] = []
        offset = 0
        if self.fit_intercept:
            values.append(float(target.mean()))
            names.append("intercept")
            offset = 1
        for group, members in self.groups_.items():
            ordered = sorted(members, key=lambda item: item[1])
            length = len(ordered)
            start = self._group_start(group, length)
            self.group_param_slices_[group] = (offset, offset + len(start))
            values.extend(start)
            names.extend(f"{group}_theta{i}" for i in range(len(start)))
            offset += len(start)
        return np.asarray(values, dtype=float), names

    def _group_start(self, group: str, length: int) -> list[float]:
        supplied = self.start_params
        if isinstance(supplied, Mapping) and group in supplied:
            raw = [float(value) for value in supplied[group]]
        elif supplied is not None and not isinstance(supplied, Mapping):
            raw = [float(value) for value in supplied]
        else:
            raw = _restricted_default_start(
                self.weighting,
                length,
                polynomial_order=self.polynomial_order,
                n_steps=self.n_steps,
                step_bounds=self.step_bounds,
            )
        expected = _restricted_param_count(
            self.weighting,
            length,
            polynomial_order=self.polynomial_order,
            n_steps=self.n_steps,
            step_bounds=self.step_bounds,
        )
        if len(raw) != expected:
            raise ValueError(
                f"start_params for group {group!r} must contain {expected} values; got {len(raw)}"
            )
        if not np.isfinite(raw).all():
            raise ValueError("start_params must contain finite values")
        return raw

    def _predict_array(self, frame: pd.DataFrame, params: np.ndarray) -> np.ndarray:
        values = np.full(len(frame), float(params[0]) if self.fit_intercept else 0.0, dtype=float)
        for group, members in self.groups_.items():
            start, end = self.group_param_slices_[group]
            weights = _restricted_midas_weights(
                [lag for _, lag in sorted(members, key=lambda item: item[1])],
                weighting=self.weighting,
                params=params[start:end],
                polynomial_order=self.polynomial_order,
                n_steps=self.n_steps,
                step_bounds=self.step_bounds,
            )
            columns = [column for column, _ in sorted(members, key=lambda item: item[1])]
            values += frame.loc[:, columns].to_numpy(dtype=float) @ weights
        return values

    def _full_coefficients(self, params: np.ndarray) -> np.ndarray:
        coefficients: dict[str, float] = {}
        for group, members in self.groups_.items():
            start, end = self.group_param_slices_[group]
            ordered = sorted(members, key=lambda item: item[1])
            weights = _restricted_midas_weights(
                [lag for _, lag in ordered],
                weighting=self.weighting,
                params=params[start:end],
                polynomial_order=self.polynomial_order,
                n_steps=self.n_steps,
                step_bounds=self.step_bounds,
            )
            for (column, _), weight in zip(ordered, weights, strict=False):
                coefficients[str(column)] = float(weight)
        return np.asarray([coefficients[name] for name in self.feature_names_in_], dtype=float)


def restricted_midas(
    X: Any,
    y: Any | None = None,
    *,
    weighting: str = "almon",
    polynomial_order: int = 2,
    start_params: Mapping[str, Sequence[float]] | Sequence[float] | None = None,
    n_steps: int = 3,
    step_bounds: tuple[int, ...] | None = None,
    fit_intercept: bool = True,
    maxiter: int = 1000,
    tolerance: float = 1e-8,
) -> ModelFit:
    """Fit a midasr::midas_r-style nonlinear restricted MIDAS regression."""

    weighting_value = _normalize_restricted_weighting(weighting)
    polynomial_order = _validate_polynomial_order(polynomial_order)
    n_steps = _validate_positive_int("n_steps", n_steps)
    step_bounds = _normalize_step_bounds(step_bounds)
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    estimator = _RestrictedMIDASRegressor(
        weighting=weighting_value,
        polynomial_order=polynomial_order,
        start_params=start_params,
        n_steps=n_steps,
        step_bounds=step_bounds,
        fit_intercept=fit_intercept,
        maxiter=maxiter,
        tolerance=tolerance,
    )
    fit = fit_estimator(
        estimator,
        X,
        y,
        model="restricted_midas",
        metadata={
            "weighting": weighting_value,
            "polynomial_order": int(polynomial_order),
            "start_params": _jsonable_start_params(start_params),
            "n_steps": int(n_steps),
            "step_bounds": step_bounds,
            "fit_intercept": bool(fit_intercept),
            "maxiter": int(maxiter),
            "tolerance": float(tolerance),
            "backend": "internal scipy.optimize.least_squares",
            "implementation_note": (
                "midasr::midas_r-aligned nonlinear least-squares objective over "
                "explicit lag columns. Weight maps use nealmon, nbetaMT, or "
                "polystep logic; optimizer differs from R's default optim(BFGS)."
            ),
        },
    )
    fit.metadata["lag_groups"] = {
        group: [column for column, _ in members]
        for group, members in estimator.groups_.items()
    }
    fit.metadata["lag_group_details"] = _lag_group_metadata(estimator.groups_)
    fit.metadata["param_names"] = list(estimator.param_names_)
    fit.metadata["converged"] = bool(estimator.converged_)
    fit.metadata["n_iter"] = int(estimator.n_iter_)
    fit.diagnostics["effective_lag_coefficients"] = estimator.effective_lag_coefficients_
    fit.diagnostics["restricted_parameters"] = pd.Series(
        estimator.params_,
        index=list(estimator.param_names_),
        name="parameter",
    )
    fit.diagnostics["optimizer"] = {
        "converged": bool(estimator.converged_),
        "n_iter": int(estimator.n_iter_),
        "cost": estimator.cost_,
        "message": estimator.message_,
    }
    return fit


def midas_almon(
    X: Any,
    y: Any | None = None,
    *,
    polynomial_order: int = 2,
    theta: tuple[float, ...] | None = None,
    alpha: float = 0.0,
    fit_intercept: bool = True,
) -> ModelFit:
    """Fit linear MIDAS using Almon-polynomial lag weights."""

    polynomial_order = _validate_polynomial_order(polynomial_order)
    theta_value = _normalize_almon_theta(theta, polynomial_order=polynomial_order)
    alpha = _validate_nonnegative_float("alpha", alpha)
    params = {
        "polynomial_order": int(polynomial_order),
        "theta": None if theta is None else theta_value,
        "alpha": alpha,
        "fit_intercept": bool(fit_intercept),
        "implementation_note": (
            "Fixed-shape MIDAS design aligned with the scale-free part of "
            "midasr::nealmon; macroforecast estimates the aggregate coefficient "
            "with a linear/ridge head rather than midasr's joint NLS optimizer."
        ),
    }
    fit = fit_estimator(
        _MIDASRegressor(
            weighting="almon",
            polynomial_order=polynomial_order,
            theta=theta_value,
            alpha=alpha,
            fit_intercept=bool(fit_intercept),
        ),
        X,
        y,
        model="midas_almon",
        metadata=params,
    )
    return _attach_midas_metadata(fit)


def midas_beta(
    X: Any,
    y: Any | None = None,
    *,
    beta_params: tuple[float, float] = (1.0, 1.0),
    alpha: float = 0.0,
    fit_intercept: bool = True,
) -> ModelFit:
    """Fit linear MIDAS using normalized beta lag weights."""

    beta_params = _normalize_beta_params(beta_params)
    alpha = _validate_nonnegative_float("alpha", alpha)
    params = {
        "beta_params": beta_params,
        "alpha": alpha,
        "fit_intercept": bool(fit_intercept),
        "implementation_note": (
            "Fixed-shape MIDAS design aligned with midasr::nbetaMT using "
            "p=(1, a, b, 0); macroforecast estimates the aggregate coefficient "
            "with a linear/ridge head rather than midasr's joint NLS optimizer."
        ),
    }
    fit = fit_estimator(
        _MIDASRegressor(
            weighting="beta",
            beta_params=beta_params,
            alpha=alpha,
            fit_intercept=bool(fit_intercept),
        ),
        X,
        y,
        model="midas_beta",
        metadata=params,
    )
    return _attach_midas_metadata(fit)


def midas_step(
    X: Any,
    y: Any | None = None,
    *,
    n_steps: int = 3,
    step_bounds: tuple[int, ...] | None = None,
    step_weights: tuple[float, ...] | None = None,
    alpha: float = 0.0,
    fit_intercept: bool = True,
) -> ModelFit:
    """Fit linear MIDAS using equal step-function lag buckets."""

    n_steps = _validate_positive_int("n_steps", n_steps)
    step_bounds = _normalize_step_bounds(step_bounds)
    step_weights = _normalize_step_weights(step_weights)
    alpha = _validate_nonnegative_float("alpha", alpha)
    params = {
        "n_steps": n_steps,
        "step_bounds": step_bounds,
        "step_weights": step_weights,
        "alpha": alpha,
        "fit_intercept": bool(fit_intercept),
        "implementation_note": (
            "Fixed-shape MIDAS design aligned with midasr::polystep-style "
            "piecewise-constant lag coefficients. Defaults use equal raw step "
            "heights and normalize the block to a scale-free weight shape."
        ),
    }
    fit = fit_estimator(
        _MIDASRegressor(
            weighting="step",
            n_steps=n_steps,
            step_bounds=step_bounds,
            step_weights=step_weights,
            alpha=alpha,
            fit_intercept=bool(fit_intercept),
        ),
        X,
        y,
        model="midas_step",
        metadata=params,
    )
    return _attach_midas_metadata(fit)


class _UnrestrictedMIDASRegressor:
    """Linear MIDAS regression that keeps every lag coefficient free."""

    def __init__(self, *, alpha: float = 0.0, fit_intercept: bool = True) -> None:
        self.alpha = float(alpha)
        self.fit_intercept = bool(fit_intercept)
        self.feature_names_in_: tuple[str, ...] = ()
        self.groups_: dict[str, list[tuple[str, int]]] = {}
        self._regression: Any = None
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_UnrestrictedMIDASRegressor":
        from sklearn.linear_model import LinearRegression, Ridge

        self.feature_names_in_ = tuple(str(column) for column in X.columns)
        self.groups_ = _midas_groups(pd.Index(self.feature_names_in_))
        frame = X.astype(float)
        if self.alpha > 0.0:
            self._regression = Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept)
        else:
            self._regression = LinearRegression(fit_intercept=self.fit_intercept)
        self._regression.fit(frame, y.astype(float))
        self.coef_ = np.asarray(getattr(self._regression, "coef_", []), dtype=float).reshape(-1)
        self.intercept_ = float(getattr(self._regression, "intercept_", 0.0))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._regression is None:
            return np.zeros(len(X), dtype=float)
        frame = X.reindex(columns=list(self.feature_names_in_), fill_value=np.nan).astype(float)
        return np.asarray(self._regression.predict(frame), dtype=float).reshape(-1)


def unrestricted_midas(
    X: Any,
    y: Any | None = None,
    *,
    alpha: float = 0.0,
    fit_intercept: bool = True,
) -> ModelFit:
    """Fit unrestricted MIDAS over an explicit lag matrix."""

    alpha = _validate_nonnegative_float("alpha", alpha)
    params = {
        "alpha": alpha,
        "fit_intercept": bool(fit_intercept),
        "implementation_note": (
            "Aligned with midasr::midas_u when alpha=0: every supplied lag column "
            "receives its own OLS coefficient. alpha>0 is a macroforecast ridge "
            "extension. Build X with feature_engineering.mixed_frequency_lags()."
        ),
    }
    fit = fit_estimator(
        _UnrestrictedMIDASRegressor(alpha=alpha, fit_intercept=fit_intercept),
        X,
        y,
        model="unrestricted_midas",
        metadata=params,
    )
    fit.metadata["lag_groups"] = {
        group: [column for column, _ in members]
        for group, members in fit.estimator.groups_.items()
    }
    fit.metadata["lag_group_details"] = _lag_group_metadata(fit.estimator.groups_)
    return fit


class _StatsmodelsForecastEstimator:
    def __init__(self, *, method: str, params: dict[str, Any]) -> None:
        self.method = method
        self.params = dict(params)
        self.result_: Any = None
        self.train_index_: pd.Index | None = None
        self.fallback_: float = 0.0
        self.fit_error_: str | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_StatsmodelsForecastEstimator":
        series = pd.Series(y).astype(float).dropna()
        self.train_index_ = series.index
        self.fallback_ = float(series.iloc[-1]) if len(series) else 0.0
        if len(series) < 3:
            return self
        try:
            if self.method == "ets":
                from statsmodels.tsa.exponential_smoothing.ets import ETSModel

                self.result_ = ETSModel(series, **self.params).fit(disp=False)
            elif self.method == "holt_winters":
                from statsmodels.tsa.holtwinters import ExponentialSmoothing

                self.result_ = ExponentialSmoothing(series, **self.params).fit(optimized=True)
            elif self.method == "theta":
                from statsmodels.tsa.forecasting.theta import ThetaModel

                self.result_ = ThetaModel(series, **self.params).fit()
            else:
                raise ValueError(f"unknown method {self.method!r}")
        except Exception as exc:  # noqa: BLE001 - keep callable on short/ill-posed panels
            # Surface the failure instead of silently degrading to a last-value
            # persistence forecast that is indistinguishable from a real fit.
            self.result_ = None
            self.fit_error_ = str(exc)
            warnings.warn(
                f"{self.method} fit failed ({exc}); falling back to last-value "
                "persistence. Inspect estimator.fit_error_.",
                UserWarning,
                stacklevel=2,
            )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        steps = len(X)
        if steps <= 0:
            return np.empty(0, dtype=float)
        if self.result_ is None:
            return np.full(steps, self.fallback_, dtype=float)
        try:
            forecast = self.result_.forecast(steps)
            return np.asarray(forecast, dtype=float).reshape(-1)[:steps]
        except Exception:
            return np.full(steps, self.fallback_, dtype=float)


def ets(
    y: Any,
    *,
    error: str = "add",
    trend: str | None = None,
    seasonal: str | None = None,
    seasonal_periods: int | None = None,
    damped_trend: bool = False,
) -> ModelFit:
    """Fit statsmodels ETSModel for target-only forecasts."""

    params = {
        "error": error,
        "trend": trend,
        "seasonal": seasonal,
        "seasonal_periods": seasonal_periods,
        "damped_trend": bool(damped_trend),
    }
    target = as_series(y)
    dummy = pd.DataFrame({"__origin__": np.arange(len(target), dtype=float)}, index=target.index)
    return fit_estimator(
        _StatsmodelsForecastEstimator(method="ets", params=params),
        dummy,
        target,
        model="ets",
        metadata=params,
    )


def holt_winters(
    y: Any,
    *,
    trend: str | None = "add",
    seasonal: str | None = None,
    seasonal_periods: int | None = None,
    damped_trend: bool = False,
) -> ModelFit:
    """Fit Holt-Winters exponential smoothing for target-only forecasts."""

    params = {
        "trend": trend,
        "seasonal": seasonal,
        "seasonal_periods": seasonal_periods,
        "damped_trend": bool(damped_trend),
    }
    target = as_series(y)
    dummy = pd.DataFrame({"__origin__": np.arange(len(target), dtype=float)}, index=target.index)
    return fit_estimator(
        _StatsmodelsForecastEstimator(method="holt_winters", params=params),
        dummy,
        target,
        model="holt_winters",
        metadata=params,
    )


def theta_method(
    y: Any,
    *,
    period: int | None = None,
    deseasonalize: bool = True,
    use_test: bool = True,
) -> ModelFit:
    """Fit the Theta forecasting method for target-only forecasts."""

    params = {
        "period": period,
        "deseasonalize": bool(deseasonalize),
        "use_test": bool(use_test),
    }
    target = as_series(y)
    dummy = pd.DataFrame({"__origin__": np.arange(len(target), dtype=float)}, index=target.index)
    return fit_estimator(
        _StatsmodelsForecastEstimator(method="theta", params=params),
        dummy,
        target,
        model="theta_method",
        metadata=params,
    )


def _normalize_vars_type(value: str) -> str:
    mapping = {
        "const": "const",
        "constant": "const",
        "c": "const",
        "trend": "trend",
        "t": "trend",
        "both": "both",
        "ct": "both",
        "none": "none",
        "n": "none",
        "no_const": "none",
    }
    key = str(value).lower()
    if key not in mapping:
        allowed = "', '".join(sorted(mapping))
        raise ValueError(f"type must be one of '{allowed}'")
    return mapping[key]


def _vars_rhs(
    values: np.ndarray,
    names: tuple[str, ...],
    n_lag: int,
    *,
    type: str,
    season: int | None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    n_obs, n_vars = values.shape
    yend = values[n_lag:, :]
    rows: list[list[float]] = []
    lag_names: list[str] = []
    for lag in range(1, n_lag + 1):
        lag_names.extend([f"{name}.l{lag}" for name in names])
    for i in range(n_lag, n_obs):
        row: list[float] = []
        for lag in range(1, n_lag + 1):
            row.extend(values[i - lag, :].tolist())
        rows.append(row)
    rhs = np.asarray(rows, dtype=float)
    rhs_names = list(lag_names)
    sample = n_obs - n_lag
    if type in {"const", "both"}:
        rhs = np.column_stack([rhs, np.ones(sample, dtype=float)])
        rhs_names.append("const")
    if type in {"trend", "both"}:
        rhs = np.column_stack([rhs, np.arange(n_lag + 1, n_obs + 1, dtype=float)])
        rhs_names.append("trend")
    if season is not None:
        seasonal = _vars_seasonal_dummies(n_obs, season)[n_lag:, :]
        rhs = np.column_stack([rhs, seasonal])
        rhs_names.extend([f"sd{i}" for i in range(1, season)])
    return yend, rhs, rhs_names


def _vars_forecast(
    coef: np.ndarray,
    values: np.ndarray,
    steps: int,
    n_lag: int,
    *,
    type: str,
    season: int | None,
) -> np.ndarray:
    if steps <= 0:
        return np.empty((0, values.shape[1]), dtype=float)
    n_obs, _ = values.shape
    history = [row.copy() for row in values[-n_lag:]]
    seasonal = _vars_seasonal_dummies(n_obs + steps, season)[n_obs:, :] if season else None
    forecasts: list[np.ndarray] = []
    for h in range(steps):
        row: list[float] = []
        for lag in range(1, n_lag + 1):
            row.extend(history[-lag].tolist())
        if type in {"const", "both"}:
            row.append(1.0)
        if type in {"trend", "both"}:
            row.append(float(n_obs + h + 1))
        if seasonal is not None:
            row.extend(seasonal[h].tolist())
        pred = coef @ np.asarray(row, dtype=float)
        forecasts.append(pred)
        history.append(np.asarray(pred, dtype=float))
    return np.vstack(forecasts)


def _vars_seasonal_dummies(n_obs: int, season: int | None) -> np.ndarray:
    if season is None:
        return np.empty((n_obs, 0), dtype=float)
    base = np.eye(int(season), dtype=float) - 1.0 / float(season)
    base = base[:, : int(season) - 1]
    reps = int(np.ceil(n_obs / float(season)))
    return np.tile(base, (reps, 1))[:n_obs]


def _favar_bvar_draws(
    values: np.ndarray,
    n_lag: int,
    *,
    prior: str,
    b0: float,
    vb0: float,
    nu0: float,
    s0: float | Sequence[Sequence[float]] | None,
    kappa0: float | None,
    kappa1: float | None,
    n_iter: int,
    burnin: int,
    random_state: int,
) -> dict[str, Any]:
    # R FAVAR::BVAR first calls bvartools::gen_var(..., deterministic='none')
    # and then works with y=t(Y), x=t(Z). _var_design(..., intercept=False)
    # creates the same lag-only VAR design.
    from scipy.stats import wishart

    design, response = _var_design(values, n_lag, intercept=False)
    y = response.T
    x = design.T
    k, tnum = y.shape
    n_reg = x.shape[0]
    n_coef = k * n_reg
    if prior == "minnesota":
        if kappa0 is None or kappa1 is None:
            raise ValueError("Minnesota BVAR requires kappa0 and kappa1")
        a_mu_prior, a_v_i_prior = _favar_minnesota_prior(values, n_lag, kappa0, kappa1)
    else:
        a_mu_prior = np.full(n_coef, float(b0), dtype=float)
        a_v_i_prior = np.eye(n_coef, dtype=float) * float(vb0)
    s0_matrix = _prior_scale_matrix(s0, k)
    sigma_df_post = int(tnum + float(nu0))
    if sigma_df_post <= k - 1:
        sigma_df_post = k
    rng = np.random.default_rng(random_state)
    sigma_i = np.eye(k, dtype=float) * 1e-5
    coef_draws: list[np.ndarray] = []
    sigma_draws: list[np.ndarray] = []
    for draw in range(1, n_iter + 1):
        a = _draw_bvar_coefficients(y, x, sigma_i, a_mu_prior, a_v_i_prior, rng)
        a_matrix = a.reshape((k, n_reg), order="F")
        residual = y - a_matrix @ x
        scale_post = s0_matrix + residual @ residual.T
        # R calls rWishart(..., solve(s0 + tcrossprod(u))). SciPy requires the
        # scale argument to be strictly positive definite; nearly collinear
        # macro panels can make the algebraic equivalent only semidefinite.
        # Eigenvalue flooring is a numerical guard, not a different prior.
        precision_scale = _positive_definite(np.linalg.pinv(scale_post))
        sigma_i = np.asarray(
            wishart.rvs(df=sigma_df_post, scale=precision_scale, random_state=rng),
            dtype=float,
        ).reshape(k, k)
        sigma = np.linalg.pinv(sigma_i)
        if draw > burnin:
            coef_draws.append(a_matrix)
            sigma_draws.append(sigma)
    coef_arr = np.stack(coef_draws, axis=0)
    sigma_arr = np.stack(sigma_draws, axis=0)
    posterior_sd = coef_arr.std(axis=0, ddof=1)
    summary = {
        "mean": coef_arr.mean(axis=0),
        # Posterior standard deviation of each coefficient (the measure of
        # coefficient uncertainty). The Monte-Carlo standard error of the
        # posterior mean is exposed separately as 'mcse'.
        "se": posterior_sd,
        "mcse": posterior_sd / np.sqrt(max(1, coef_arr.shape[0])),
        "q025": np.quantile(coef_arr, 0.025, axis=0),
        "q975": np.quantile(coef_arr, 0.975, axis=0),
    }
    return {"coef_draws": coef_arr, "sigma_draws": sigma_arr, "summary": summary}


def _draw_bvar_coefficients(
    y: np.ndarray,
    x: np.ndarray,
    sigma_i: np.ndarray,
    prior_mean: np.ndarray,
    prior_precision: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    k, tnum = y.shape
    n_reg = x.shape[0]
    z = np.zeros((tnum * k, n_reg * k), dtype=float)
    y_vec = np.empty(tnum * k, dtype=float)
    for t in range(tnum):
        z[t * k : (t + 1) * k, :] = np.kron(x[:, t].reshape(1, -1), np.eye(k))
        y_vec[t * k : (t + 1) * k] = y[:, t]
    omega = np.kron(np.eye(tnum), sigma_i)
    precision = z.T @ omega @ z + prior_precision
    rhs = z.T @ omega @ y_vec + prior_precision @ prior_mean
    cov = _positive_definite(_stable_inverse(precision))
    mean = cov @ rhs
    return rng.multivariate_normal(mean, cov, check_valid="ignore")


def _favar_minnesota_prior(
    values: np.ndarray,
    n_lag: int,
    kappa0: float,
    kappa1: float,
) -> tuple[np.ndarray, np.ndarray]:
    _, response = _var_design(values, n_lag, intercept=False)
    k = response.shape[1]
    n_reg = k * n_lag
    sigma = _favar_univariate_ar_sigma(values, n_lag)
    variance = np.empty((k, n_reg), dtype=float)
    for lag in range(1, n_lag + 1):
        for eq in range(k):
            for reg in range(k):
                col = (lag - 1) * k + reg
                if eq == reg:
                    variance[eq, col] = float(kappa0) / float(lag * lag)
                else:
                    variance[eq, col] = (
                        float(kappa0)
                        * float(kappa1)
                        / float(lag * lag)
                        * (sigma[eq] ** 2)
                        / max(sigma[reg] ** 2, 1e-12)
                    )
    prior_mean = np.zeros(k * n_reg, dtype=float)
    prior_precision = np.diag(1.0 / np.maximum(variance.flatten(order="F"), 1e-12))
    return prior_mean, prior_precision


def _favar_univariate_ar_sigma(values: np.ndarray, n_lag: int) -> np.ndarray:
    n_obs, k = values.shape
    if n_obs <= n_lag:
        return np.std(values, axis=0, ddof=1).clip(min=1e-8)
    out = np.empty(k, dtype=float)
    for j in range(k):
        y = values[n_lag:, j]
        rows = []
        for i in range(n_lag, n_obs):
            rows.append([values[i - lag, j] for lag in range(1, n_lag + 1)])
        x = np.asarray(rows, dtype=float)
        coef = np.linalg.lstsq(x, y, rcond=None)[0]
        residual = y - x @ coef
        denom = max(1, len(y) - x.shape[1])
        out[j] = float(np.sqrt(max(float(residual @ residual) / denom, 1e-12)))
    return out


def _prior_scale_matrix(value: float | Sequence[Sequence[float]] | None, k: int) -> np.ndarray:
    if value is None:
        raise ValueError("s0 must not be None for FAVAR::BVAR-compatible fitting")
    if np.isscalar(value):
        return np.full((k, k), float(value), dtype=float)
    arr = np.asarray(value, dtype=float)
    if arr.shape != (k, k):
        raise ValueError(f"s0 must be scalar or a {k}x{k} matrix")
    return arr


def _stable_inverse(matrix: np.ndarray) -> np.ndarray:
    mat = _symmetrize(np.asarray(matrix, dtype=float))
    jitter = 0.0
    for _ in range(6):
        try:
            return np.linalg.inv(mat + jitter * np.eye(mat.shape[0]))
        except np.linalg.LinAlgError:
            jitter = 1e-10 if jitter == 0.0 else jitter * 10.0
    return np.linalg.pinv(mat)


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def _positive_definite(matrix: np.ndarray, *, floor: float = 1e-8) -> np.ndarray:
    mat = _symmetrize(np.asarray(matrix, dtype=float))
    eigvals, eigvecs = np.linalg.eigh(mat)
    scale = max(float(np.nanmax(np.abs(eigvals))) if eigvals.size else 1.0, 1.0)
    clipped = np.maximum(eigvals, float(floor) * scale)
    return _symmetrize((eigvecs * clipped) @ eigvecs.T)


def _jsonable_prior_matrix(value: float | Sequence[Sequence[float]] | None) -> Any:
    if value is None or np.isscalar(value):
        return None if value is None else float(value)
    return np.asarray(value, dtype=float).tolist()


def _bvar_diagnostics(estimator: _BayesianVAR) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    if estimator.summary_:
        for key, value in estimator.summary_.items():
            diagnostics[f"coef_{key}"] = pd.DataFrame(
                np.asarray(value, dtype=float),
                index=list(estimator.names_),
                columns=_var_lag_column_names(estimator.names_, estimator.n_lag),
            )
    if estimator.sigma_draws_ is not None:
        diagnostics["sigma_mean"] = pd.DataFrame(
            estimator.sigma_draws_.mean(axis=0),
            index=list(estimator.names_),
            columns=list(estimator.names_),
        )
    return diagnostics


def _var_lag_column_names(names: tuple[str, ...], n_lag: int) -> list[str]:
    out: list[str] = []
    for lag in range(1, n_lag + 1):
        out.extend([f"{name}.l{lag}" for name in names])
    return out


def _r_scale_matrix(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0)
    scale = np.std(values, axis=0, ddof=1)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    return mean, scale, (values - mean) / scale


def _favar_extr_pc(values: np.ndarray, n_factors: int) -> tuple[np.ndarray, np.ndarray]:
    # FAVAR/R/ExtrPC.R:
    #   xx <- t(X_st) %*% X_st
    #   lam <- sqrt(ncol(X_st)) * eigen(xx)$vectors[, 1:K]
    #   fac <- X_st %*% lam / ncol(X_st)
    xx = values.T @ values
    eigvals, eigvecs = np.linalg.eigh(_symmetrize(xx))
    order = np.argsort(eigvals)[::-1]
    loadings = np.sqrt(values.shape[1]) * eigvecs[:, order[:n_factors]]
    factors = values @ loadings / float(values.shape[1])
    return factors, loadings


def _favar_olssvd(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    # FAVAR/R/facrot.R olssvd(F0, ly):
    #   svd(ly); b = (v * repmat(1/d, ...)) %*% t(u) %*% F0
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    inv_s = np.divide(1.0, s, out=np.zeros_like(s), where=s > 1e-12)
    return vt.T @ np.diag(inv_s) @ u.T @ y


def _favar_facrot(factors: np.ndarray, fast: np.ndarray, slow_factors: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(fast.shape[0], dtype=float), fast, slow_factors])
    b = _favar_olssvd(factors, design)
    return factors - fast @ b[1 : fast.shape[1] + 1, :]


def _favar_bgm(
    values: np.ndarray,
    response: np.ndarray,
    n_factors: int,
    *,
    tolerance: float = 0.001,
    nmax: int = 100,
) -> np.ndarray:
    x_work = np.asarray(values, dtype=float).copy()
    previous: np.ndarray | None = None
    factors = _favar_extr_pc(x_work, n_factors)[0]
    for iteration in range(1, int(nmax) + 1):
        factors = _favar_extr_pc(x_work, n_factors)[0]
        if previous is not None and not bool(np.any(np.abs(factors - previous) > tolerance)):
            break
        previous = factors.copy()
        design = np.column_stack([factors, response])
        beta = np.linalg.pinv(design.T @ design) @ design.T @ x_work
        x_work = x_work - response.reshape(-1, 1) @ beta[-1:, :]
        if iteration >= int(nmax):
            break
    return factors


def _favar_loading_draws(
    x: np.ndarray,
    fy: np.ndarray,
    loading_matrix: np.ndarray,
    n_factors: int,
    factorprior: Mapping[str, Any],
    nburn: int,
    nrep: int,
    random_state: int,
) -> np.ndarray:
    b0 = float(factorprior.get("b0", 0.0))
    b0_vec = np.full(fy.shape[1], b0, dtype=float)
    b0_arg = factorprior.get("B0", factorprior.get("vb0", None))
    if b0_arg is None:
        precision = 4.0 * np.eye(fy.shape[1], dtype=float)
    elif np.isscalar(b0_arg):
        precision = float(b0_arg) * np.eye(fy.shape[1], dtype=float)
    else:
        precision = np.asarray(b0_arg, dtype=float)
    c0 = float(factorprior.get("c0", 0.01))
    d0 = float(factorprior.get("d0", 0.01))
    rng = np.random.default_rng(random_state + 7919)
    draws = np.empty((nrep, fy.shape[1], x.shape[1]), dtype=float)
    xtx = fy.T @ fy
    for j in range(x.shape[1]):
        if j < n_factors:
            draws[:, :, j] = loading_matrix[j, :]
            continue
        y = x[:, j]
        post_precision = xtx + precision
        post_cov = _stable_inverse(post_precision)
        post_mean = post_cov @ (fy.T @ y + precision @ b0_vec)
        residual = y - fy @ post_mean
        shape = 0.5 * (c0 + len(y))
        scale = 0.5 * (d0 + float(residual @ residual))
        for draw in range(nrep):
            sigma2 = 1.0 / rng.gamma(shape=shape, scale=1.0 / max(scale, 1e-12))
            draws[draw, :, j] = rng.multivariate_normal(
                post_mean,
                _positive_definite(post_cov * sigma2),
                check_valid="ignore",
            )
    return draws


def _parse_favar_varprior(prior: Mapping[str, Any]) -> dict[str, Any]:
    mn = prior.get("mn", {}) if prior else {}
    if isinstance(mn, Mapping) and mn.get("kappa0") is not None:
        return {
            "prior": "minnesota",
            "b0": 0.0,
            "vb0": 0.0,
            "nu0": float(prior.get("nu0", 0.0)),
            "s0": prior.get("s0", 0.0),
            "kappa0": float(mn.get("kappa0")),
            "kappa1": float(mn.get("kappa1", 0.5)),
        }
    return {
        "prior": "normal_inverse_wishart",
        "b0": float(prior.get("b0", 0.0)) if prior else 0.0,
        "vb0": float(prior.get("vb0", 0.0)) if prior else 0.0,
        "nu0": float(prior.get("nu0", 0.0)) if prior else 0.0,
        "s0": prior.get("s0", 0.0) if prior else 0.0,
        "kappa0": None,
        "kappa1": None,
    }


def _favar_diagnostics(estimator: _FAVAR) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    if estimator.factorx_ is not None:
        diagnostics["factorx"] = estimator.factorx_
    if estimator.loading_draws_ is not None:
        diagnostics["loading_mean"] = pd.DataFrame(
            estimator.loading_draws_.mean(axis=0),
            index=[f"state_{i + 1}" for i in range(estimator.loading_draws_.shape[1])],
            columns=list(estimator.feature_names_in_),
        )
    if estimator.var_summary_:
        names = tuple([f"factor_{i + 1}" for i in range(estimator.n_factors)] + [estimator.target_name_ or "Y"])
        for key, value in estimator.var_summary_.items():
            diagnostics[f"var_coef_{key}"] = pd.DataFrame(
                np.asarray(value, dtype=float),
                index=list(names),
                columns=_var_lag_column_names(names, estimator.n_lag),
            )
    if estimator.var_sigma_draws_ is not None:
        names = tuple([f"factor_{i + 1}" for i in range(estimator.n_factors)] + [estimator.target_name_ or "Y"])
        diagnostics["var_sigma_mean"] = pd.DataFrame(
            estimator.var_sigma_draws_.mean(axis=0),
            index=list(names),
            columns=list(names),
        )
    return diagnostics


def _var_design(values: np.ndarray, n_lag: int, *, intercept: bool) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    target = []
    for i in range(n_lag, len(values)):
        row: list[float] = [1.0] if intercept else []
        for lag in range(1, n_lag + 1):
            row.extend(values[i - lag].tolist())
        rows.append(row)
        target.append(values[i])
    return np.asarray(rows, dtype=float), np.asarray(target, dtype=float)


def _var_forecast_row(history: list[np.ndarray], n_lag: int, *, intercept: bool) -> np.ndarray:
    row: list[float] = [1.0] if intercept else []
    for lag in range(1, n_lag + 1):
        row.extend(history[-lag].tolist())
    return np.asarray(row, dtype=float)


def _coerce_panel_with_metadata(
    panel: Any,
    *,
    metadata: Mapping[str, Any] | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if hasattr(panel, "panel") and isinstance(getattr(panel, "panel"), pd.DataFrame):
        frame = getattr(panel, "panel").copy()
        base_metadata = dict(getattr(panel, "metadata", {}) or {})
    elif isinstance(panel, tuple) and len(panel) == 2 and isinstance(panel[0], pd.DataFrame):
        frame = panel[0].copy()
        base_metadata = dict(panel[1] or {})
    else:
        frame = as_frame(panel)
        base_metadata = dict(getattr(frame, "attrs", {}).get("macroforecast_metadata", {}) or {})
    if metadata is not None:
        base_metadata.update(dict(metadata))
    return frame, base_metadata


def _prepare_mixed_frequency_panel(
    panel: pd.DataFrame,
    *,
    metadata: Mapping[str, Any],
    monthly_columns: Iterable[str] | None,
    quarterly_columns: Iterable[str] | None,
    unsupported: str,
) -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...]]:
    if not isinstance(panel.index, pd.DatetimeIndex):
        raise TypeError("mixed-frequency DFM panel must have a DatetimeIndex")
    frame = panel.copy()
    frame.index = pd.DatetimeIndex(frame.index).to_period("M").to_timestamp()
    frame = frame.groupby(level=0).last().sort_index()
    frame.index.name = "date"
    explicit_monthly = _column_tuple(monthly_columns)
    explicit_quarterly = _column_tuple(quarterly_columns)
    overlap = sorted(set(explicit_monthly).intersection(explicit_quarterly))
    if overlap:
        raise ValueError(f"columns cannot be both monthly and quarterly: {overlap}")
    missing_explicit = [
        column
        for column in (*explicit_monthly, *explicit_quarterly)
        if column not in frame.columns
    ]
    if missing_explicit:
        raise ValueError(f"frequency override columns are not in the panel: {missing_explicit}")

    frequency_map = _mixed_frequency_map(frame, metadata)
    monthly: list[str] = []
    quarterly: list[str] = []
    unsupported_columns: list[tuple[str, str]] = []
    for column in frame.columns:
        name = str(column)
        if name in explicit_monthly:
            monthly.append(name)
            continue
        if name in explicit_quarterly:
            quarterly.append(name)
            continue
        frequency = frequency_map.get(name, "unknown")
        if frequency == "monthly":
            monthly.append(name)
        elif frequency == "quarterly":
            quarterly.append(name)
        elif str(unsupported).lower() == "drop":
            unsupported_columns.append((name, frequency))
        else:
            raise ValueError(
                "DynamicFactorMQ supports monthly and quarterly columns only; "
                f"column {name!r} has frequency {frequency!r}"
            )
    keep = [*monthly, *quarterly]
    if not keep:
        raise ValueError("mixed-frequency DFM panel has no monthly or quarterly columns")
    selected = frame.loc[:, keep].astype(float)
    empty_columns = [str(column) for column in selected.columns if selected[column].dropna().empty]
    if empty_columns:
        raise ValueError(f"mixed-frequency DFM columns are entirely missing: {empty_columns}")
    if unsupported_columns and str(unsupported).lower() != "drop":
        raise ValueError(f"unsupported mixed-frequency columns: {unsupported_columns}")
    if not monthly:
        raise ValueError("mixed-frequency DFM requires at least one monthly column")
    return selected, tuple(monthly), tuple(quarterly)


def _mixed_frequency_map(panel: pd.DataFrame, metadata: Mapping[str, Any]) -> dict[str, str]:
    raw_map = metadata.get("native_frequency_by_column", {})
    if isinstance(raw_map, Mapping):
        result = {str(column): _normalize_frequency_label(value) for column, value in raw_map.items()}
    else:
        result = {}
    for column in panel.columns:
        name = str(column)
        if name not in result or result[name] in {"unknown", "mixed", "state_monthly"}:
            result[name] = _infer_model_frequency(panel[column])
    return result


def _normalize_frequency_label(value: Any) -> str:
    key = str(value).lower()
    aliases = {
        "m": "monthly",
        "month": "monthly",
        "monthly": "monthly",
        "state_monthly": "monthly",
        "q": "quarterly",
        "quarter": "quarterly",
        "quarterly": "quarterly",
    }
    return aliases.get(key, key)


def _infer_model_frequency(series: pd.Series) -> str:
    observed = series.dropna()
    if observed.shape[0] < 2:
        return "unknown"
    periods = pd.DatetimeIndex(observed.index).to_period("M")
    deltas = [
        int(right.ordinal - left.ordinal)
        for left, right in zip(periods[:-1], periods[1:], strict=False)
        if right.ordinal > left.ordinal
    ]
    if not deltas:
        return "unknown"
    most_common_delta = int(pd.Series(deltas).mode().iloc[0])
    if most_common_delta == 1:
        return "monthly"
    if most_common_delta == 3:
        return "quarterly"
    if most_common_delta == 12:
        return "annual"
    return "irregular"


def _column_tuple(columns: Iterable[str] | None) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(column) for column in columns or ()))


def _mixed_dfm_diagnostics(estimator: _MixedFrequencyDFM, panel: pd.DataFrame) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    if estimator.fitted_target_ is not None and estimator.target in panel.columns:
        observed = panel[estimator.target].astype(float)
        aligned = pd.concat([observed.rename("observed"), estimator.fitted_target_], axis=1).dropna()
        if not aligned.empty:
            residuals = (aligned["observed"] - aligned["fitted"]).rename("residual")
            diagnostics["fitted_values"] = estimator.fitted_target_
            diagnostics["residuals"] = residuals
            diagnostics["metrics"] = _simple_residual_metrics(residuals)
    if estimator.factors_filtered_ is not None:
        diagnostics["factors_filtered"] = estimator.factors_filtered_
    if estimator.params_ is not None:
        diagnostics["params"] = estimator.params_
    if estimator.llf_ is not None:
        diagnostics["llf"] = estimator.llf_
    if estimator.fit_error_ is not None:
        diagnostics["fit_error"] = estimator.fit_error_
    return diagnostics


def _position_target_dates(
    dates: Iterable[Any],
    *,
    frequency: Any,
    anchor_position: str,
) -> pd.DatetimeIndex:
    raw = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    key = str(anchor_position).lower()
    freq = _normalize_frequency_label(frequency)
    if key in {"date", "as_is", "asis"}:
        return pd.DatetimeIndex(raw.to_period("M").to_timestamp(), name="date")
    if key not in {"period_start", "start", "period_end", "end"}:
        raise ValueError("anchor_position must be one of ['date', 'period_start', 'period_end']")
    how = "start" if key in {"period_start", "start"} else "end"
    if freq == "quarterly":
        return pd.DatetimeIndex(raw.to_period("Q").asfreq("M", how=how).to_timestamp(), name="date")
    if freq == "annual":
        return pd.DatetimeIndex(raw.to_period("Y").asfreq("M", how=how).to_timestamp(), name="date")
    return pd.DatetimeIndex(raw.to_period("M").to_timestamp(), name="date")


def _factor_lag_design(
    factors: pd.DataFrame,
    *,
    anchor_index: pd.DatetimeIndex,
    lags: Iterable[int] | int,
) -> pd.DataFrame:
    factor_frame = factors.copy()
    factor_frame.index = pd.DatetimeIndex(factor_frame.index).to_period("M").to_timestamp()
    factor_frame = factor_frame.groupby(level=0).last().sort_index()
    lag_values = _normalize_model_lags(lags)
    result = pd.DataFrame(index=anchor_index)
    for factor_position, column in enumerate(factor_frame.columns, start=1):
        source_name = f"dfm_factor{factor_position}"
        source = factor_frame[column].astype(float)
        for lag in lag_values:
            lookup = pd.DatetimeIndex(anchor_index - pd.DateOffset(months=int(lag))).to_period("M").to_timestamp()
            result[f"{source_name}_lag{lag}"] = source.reindex(lookup).to_numpy(dtype=float)
    result.index.name = "date"
    return result


def _normalize_model_lags(values: Iterable[int] | int) -> tuple[int, ...]:
    if isinstance(values, int):
        if values < 0:
            raise ValueError("lags must be non-negative")
        if values == 0:
            return (0,)
        return tuple(range(1, values + 1))
    normalized = tuple(dict.fromkeys(int(value) for value in values))
    if not normalized:
        raise ValueError("lags must not be empty")
    invalid = [value for value in normalized if value < 0]
    if invalid:
        raise ValueError(f"lags must be non-negative; got {invalid}")
    return normalized


def _simple_residual_metrics(residuals: pd.Series) -> dict[str, float | int]:
    values = residuals.dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return {"n": 0}
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "mae": float(np.mean(np.abs(values))),
        "mse": float(np.mean(values**2)),
        "rmse": float(np.sqrt(np.mean(values**2))),
    }


def _midas_groups(columns: pd.Index) -> dict[str, list[tuple[str, int]]]:
    import re

    groups: dict[str, list[tuple[str, int]]] = {}
    for position, column in enumerate(columns):
        name = str(column)
        match = re.match(r"^(?P<base>.+)_lag(?P<lag>\d+)$", name)
        if match:
            base = match.group("base")
            lag_value = int(match.group("lag"))
        else:
            base = name
            lag_value = position
        groups.setdefault(base, []).append((name, lag_value))
    return groups


def _attach_midas_metadata(fit: ModelFit) -> ModelFit:
    estimator = fit.estimator
    groups = getattr(estimator, "groups_", {})
    weights = getattr(estimator, "weights_", {})
    effective = getattr(estimator, "effective_lag_coefficients_", None)
    fit.metadata["lag_groups"] = {
        group: [column for column, _ in members]
        for group, members in groups.items()
    }
    fit.metadata["lag_group_details"] = _lag_group_metadata(groups)
    fit.metadata["weights"] = {
        str(group): [float(value) for value in group_weights]
        for group, group_weights in weights.items()
    }
    fit.metadata["weighted_columns"] = list(getattr(estimator, "design_columns_", ()))
    if isinstance(effective, pd.Series):
        fit.diagnostics["effective_lag_coefficients"] = effective
    fit.diagnostics["midas_weights"] = {
        str(group): pd.Series(
            [float(value) for value in weights.get(group, [])],
            index=[column for column, _ in members],
            name="weight",
        )
        for group, members in groups.items()
    }
    return fit


def _lag_group_metadata(
    groups: Mapping[str, list[tuple[str, int]]],
) -> dict[str, list[dict[str, int | str]]]:
    return {
        str(group): [
            {"column": str(column), "lag": int(lag)}
            for column, lag in sorted(members, key=lambda item: item[1])
        ]
        for group, members in groups.items()
    }


def _effective_lag_coefficients(
    *,
    groups: Mapping[str, list[tuple[str, int]]],
    weights: Mapping[str, list[float]],
    aggregate_coefficients: np.ndarray | None,
) -> pd.Series | None:
    if aggregate_coefficients is None:
        return None
    values: dict[str, float] = {}
    for group_position, (group, members) in enumerate(groups.items()):
        if group_position >= len(aggregate_coefficients):
            break
        coefficient = float(aggregate_coefficients[group_position])
        group_weights = list(weights.get(group, ()))
        ordered = sorted(members, key=lambda item: item[1])
        if len(group_weights) != len(ordered):
            continue
        for (column, _), weight in zip(ordered, group_weights, strict=False):
            values[str(column)] = coefficient * float(weight)
    if not values:
        return None
    return pd.Series(values, name="effective_lag_coefficient")


def _validate_polynomial_order(value: Any) -> int:
    order = int(value)
    if order < 0:
        raise ValueError("polynomial_order must be non-negative")
    return order


def _normalize_almon_theta(
    theta: tuple[float, ...] | None,
    *,
    polynomial_order: int,
) -> tuple[float, ...]:
    if theta is None:
        return tuple(0.0 for _ in range(polynomial_order))
    values = tuple(float(value) for value in theta)
    expected = polynomial_order
    if len(values) != expected:
        raise ValueError(
            f"theta must contain polynomial_order values; expected {expected}, got {len(values)}"
        )
    return values


def _normalize_beta_params(value: tuple[float, float]) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError("beta_params must contain exactly two positive values")
    params = (float(value[0]), float(value[1]))
    if params[0] <= 0.0 or params[1] <= 0.0:
        raise ValueError("beta_params must be positive")
    return params


def _validate_positive_int(name: str, value: Any) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _normalize_step_bounds(value: tuple[int, ...] | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    bounds = tuple(int(item) for item in value)
    if any(item <= 0 for item in bounds):
        raise ValueError("step_bounds must contain positive interior cut points")
    if tuple(sorted(set(bounds))) != bounds:
        raise ValueError("step_bounds must be strictly increasing")
    return bounds


def _normalize_step_weights(value: tuple[float, ...] | None) -> tuple[float, ...] | None:
    if value is None:
        return None
    weights = tuple(float(item) for item in value)
    if not weights:
        raise ValueError("step_weights must not be empty")
    if not np.isfinite(weights).all() or sum(abs(item) for item in weights) <= 1e-12:
        raise ValueError("step_weights must contain finite, nonzero values")
    return weights


def _normalize_restricted_weighting(value: str) -> str:
    mapping = {
        "almon": "almon",
        "nealmon": "almon",
        "exponential_almon": "almon",
        "beta": "beta",
        "nbeta": "beta",
        "nbetamt": "beta",
        "step": "step",
        "polystep": "step",
    }
    key = str(value).lower()
    if key not in mapping:
        raise ValueError("weighting must be one of 'almon', 'beta', or 'step'")
    return mapping[key]


def _restricted_param_count(
    weighting: str,
    n_lags: int,
    *,
    polynomial_order: int,
    n_steps: int,
    step_bounds: tuple[int, ...] | None,
) -> int:
    if weighting == "almon":
        return 1 + int(polynomial_order)
    if weighting == "beta":
        return 4
    if weighting == "step":
        return len(_restricted_step_buckets(n_lags, n_steps=n_steps, step_bounds=step_bounds))
    raise ValueError("weighting must be one of 'almon', 'beta', or 'step'")


def _restricted_default_start(
    weighting: str,
    n_lags: int,
    *,
    polynomial_order: int,
    n_steps: int,
    step_bounds: tuple[int, ...] | None,
) -> list[float]:
    if weighting == "almon":
        return [1.0, *[0.0 for _ in range(polynomial_order)]]
    if weighting == "beta":
        return [1.0, 1.0, 1.0, 0.0]
    if weighting == "step":
        return [1.0 for _ in _restricted_step_buckets(n_lags, n_steps=n_steps, step_bounds=step_bounds)]
    raise ValueError("weighting must be one of 'almon', 'beta', or 'step'")


def _restricted_midas_weights(
    lags: list[int],
    *,
    weighting: str,
    params: Sequence[float],
    polynomial_order: int,
    n_steps: int,
    step_bounds: tuple[int, ...] | None,
) -> np.ndarray:
    n = len(lags)
    if n == 0:
        return np.empty(0, dtype=float)
    p = np.asarray(params, dtype=float)
    if weighting == "almon":
        if len(p) != 1 + polynomial_order:
            raise ValueError("almon restricted MIDAS parameters must be scale plus polynomial_order shape values")
        positions = np.arange(1, n + 1, dtype=float)
        raw = np.zeros(n, dtype=float)
        for power, value in enumerate(p[1:], start=1):
            raw += float(value) * np.power(positions, power)
        exp_raw = np.exp(raw - raw.max())
        return float(p[0]) * exp_raw / float(exp_raw.sum())
    if weighting == "beta":
        if len(p) != 4:
            raise ValueError("beta restricted MIDAS parameters must match midasr::nbetaMT p=(scale,a,b,offset)")
        if n == 1:
            return np.asarray([float(p[0])], dtype=float)
        eps = np.finfo(float).eps
        z = (np.arange(1, n + 1, dtype=float) - 1.0) / float(n - 1)
        z[0] += eps
        z[-1] -= eps
        nb = np.power(z, float(p[1]) - 1.0) * np.power(1.0 - z, float(p[2]) - 1.0)
        if float(nb.sum()) < eps:
            if abs(float(p[3])) < eps:
                return np.zeros(n, dtype=float)
            return float(p[0]) * np.full(n, 1.0 / n, dtype=float)
        weights = nb / float(nb.sum()) + float(p[3])
        denominator = float(weights.sum())
        if abs(denominator) < eps:
            return np.zeros(n, dtype=float)
        return float(p[0]) * weights / denominator
    if weighting == "step":
        buckets = _restricted_step_buckets(n, n_steps=n_steps, step_bounds=step_bounds)
        if len(p) != len(buckets):
            raise ValueError("step restricted MIDAS parameters must contain one value per step bucket")
        weights = np.zeros(n, dtype=float)
        for bucket, value in zip(buckets, p, strict=False):
            weights[bucket] = float(value)
        return weights
    raise ValueError("weighting must be one of 'almon', 'beta', or 'step'")


def _restricted_step_buckets(
    n_lags: int,
    *,
    n_steps: int,
    step_bounds: tuple[int, ...] | None,
) -> list[np.ndarray]:
    if n_lags <= 0:
        return []
    if step_bounds is None:
        return [bucket for bucket in np.array_split(np.arange(n_lags), max(1, min(n_steps, n_lags))) if len(bucket)]
    if step_bounds[-1] >= n_lags:
        raise ValueError("step_bounds must be interior cut points smaller than the number of lag columns")
    cuts = (0, *step_bounds, n_lags)
    return [np.arange(left, right) for left, right in zip(cuts[:-1], cuts[1:], strict=False)]


def _jsonable_start_params(value: Mapping[str, Sequence[float]] | Sequence[float] | None) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(key): [float(item) for item in sequence] for key, sequence in value.items()}
    return [float(item) for item in value]


def _validate_nonnegative_float(name: str, value: Any) -> float:
    result = float(value)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _midas_weights(
    lags: list[int],
    *,
    weighting: str,
    polynomial_order: int,
    theta: tuple[float, ...] | None,
    beta_params: tuple[float, float],
    n_steps: int,
    step_bounds: tuple[int, ...] | None = None,
    step_weights: tuple[float, ...] | None = None,
) -> np.ndarray:
    n = len(lags)
    if n == 0:
        return np.empty(0, dtype=float)
    if polynomial_order < 0:
        raise ValueError("polynomial_order must be non-negative")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if weighting == "almon":
        theta_values = theta or tuple(0.0 for _ in range(polynomial_order))
        positions = np.arange(1, n + 1, dtype=float)
        raw: np.ndarray = np.zeros(n, dtype=float)
        for power, value in enumerate(theta_values, start=1):
            raw += float(value) * np.power(positions, power)
        weights = np.exp(raw - raw.max())
    elif weighting == "beta":
        a, b = beta_params
        if a <= 0.0 or b <= 0.0:
            raise ValueError("beta_params must be positive")
        if n == 1:
            return np.ones(1, dtype=float)
        eps = np.finfo(float).eps
        z = (np.arange(1, n + 1, dtype=float) - 1.0) / float(n - 1)
        z[0] += eps
        z[-1] -= eps
        weights = np.power(z, a - 1.0) * np.power(1.0 - z, b - 1.0)
    elif weighting == "step":
        if step_bounds is None:
            buckets = np.array_split(np.arange(n), max(1, min(n_steps, n)))
            heights = np.ones(len(buckets), dtype=float)
        else:
            if step_bounds[-1] >= n:
                raise ValueError("step_bounds must be interior cut points smaller than the number of lag columns")
            cuts = (0, *step_bounds, n)
            buckets = [np.arange(left, right) for left, right in zip(cuts[:-1], cuts[1:], strict=False)]
            heights = np.ones(len(buckets), dtype=float)
        if step_weights is not None:
            if len(step_weights) != len(buckets):
                raise ValueError("step_weights must contain one value per step bucket")
            heights = np.asarray(step_weights, dtype=float)
        weights = np.zeros(n, dtype=float)
        for bucket, height in zip(buckets, heights, strict=False):
            weights[bucket] = float(height)
    else:
        raise ValueError("weighting must be 'almon', 'beta', or 'step'")
    total = float(np.sum(np.abs(weights))) if np.any(weights < 0.0) else float(weights.sum())
    if abs(total) <= 1e-12:
        return np.full(n, 1.0 / n, dtype=float)
    return weights / total


__all__ = [
    "ar",
    "bvar_minnesota",
    "bvar_normal_inverse_wishart",
    "dfm_mixed_mariano_murasawa",
    "dfm_unrestricted_midas",
    "ets",
    "far",
    "favar",
    "holt_winters",
    "midas_almon",
    "midas_beta",
    "midas_step",
    "restricted_midas",
    "theta_method",
    "unrestricted_midas",
    "var",
]
