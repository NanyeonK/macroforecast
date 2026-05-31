from __future__ import annotations

from collections.abc import Iterable, Mapping
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
    def __init__(self, *, n_lag: int = 1, target: str | None = None) -> None:
        self.n_lag = max(1, int(n_lag))
        self.target = target
        self._results: Any = None
        self._target_name: str | None = None
        self._last_values: np.ndarray | None = None
        self._fallback: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "_VAR":
        if y is None:
            data = X.dropna()
            target_name = self.target or str(data.columns[0])
        else:
            data = pd.concat([pd.Series(y).rename("__target__"), X], axis=1).dropna()
            target_name = "__target__"
        if data.empty:
            return self
        self._target_name = target_name
        self._fallback = float(pd.to_numeric(data[target_name], errors="coerce").mean())
        if data.shape[0] <= self.n_lag + 1 or data.shape[1] < 2:
            self._last_values = data.to_numpy(dtype=float)
            return self
        from statsmodels.tsa.api import VAR

        try:
            self._results = VAR(data).fit(self.n_lag)
            self._last_values = self._results.endog[-self.n_lag :]
        except Exception:
            self._results = None
            self._last_values = data.to_numpy(dtype=float)[-self.n_lag :]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._results is None or self._target_name is None:
            return np.full(len(X), self._fallback, dtype=float)
        forecast = self._results.forecast(self._results.endog[-self.n_lag :], steps=max(1, len(X)))
        target_index = self._results.names.index(self._target_name)
        return np.asarray(forecast[:, target_index], dtype=float)[: len(X)]


def var(panel: Any, *, target: str | None = None, n_lag: int = 1) -> ModelFit:
    """Fit a vector autoregression on a multivariate panel."""

    frame = as_frame(panel)
    estimator = _VAR(n_lag=n_lag, target=target)
    estimator.fit(frame)
    return ModelFit(
        estimator=estimator,
        model="var",
        feature_names=tuple(str(c) for c in frame.columns),
        target_name=target or str(frame.columns[0]),
        metadata={"n_obs": len(frame.dropna()), "n_lag": int(n_lag)},
    )


class _BayesianVAR:
    """Compact conjugate-prior VAR point-forecast estimator."""

    def __init__(
        self,
        *,
        n_lag: int = 1,
        target: str | None = None,
        prior: str = "minnesota",
        shrinkage: float = 0.2,
        intercept: bool = True,
        random_walk_prior: bool = True,
    ) -> None:
        self.n_lag = max(1, int(n_lag))
        self.target = target
        self.prior = str(prior)
        self.shrinkage = max(float(shrinkage), 1e-8)
        self.intercept = bool(intercept)
        self.random_walk_prior = bool(random_walk_prior)
        self.names_: tuple[str, ...] = ()
        self.target_name_: str | None = None
        self.coef_: np.ndarray | None = None
        self.last_values_: np.ndarray | None = None
        self.fallback_: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "_BayesianVAR":
        if y is None:
            data = X.dropna()
            target_name = self.target or str(data.columns[0])
        else:
            data = pd.concat([pd.Series(y).rename("__target__"), X], axis=1).dropna()
            target_name = "__target__"
        if data.empty:
            return self
        self.names_ = tuple(str(column) for column in data.columns)
        if target_name not in data.columns:
            raise ValueError(f"target {target_name!r} is not in the VAR panel")
        self.target_name_ = target_name
        self.fallback_ = float(data[target_name].mean())
        values = data.to_numpy(dtype=float)
        if len(values) <= self.n_lag:
            self.last_values_ = values[-self.n_lag :]
            return self
        design, response = _var_design(values, self.n_lag, intercept=self.intercept)
        prior_mean = np.zeros((design.shape[1], response.shape[1]), dtype=float)
        if self.prior == "minnesota" and self.random_walk_prior:
            offset = 1 if self.intercept else 0
            for eq in range(response.shape[1]):
                prior_mean[offset + eq, eq] = 1.0
        precision = np.eye(design.shape[1], dtype=float) / (self.shrinkage**2)
        if self.intercept:
            precision[0, 0] = 1e-8
        lhs = design.T @ design + precision
        rhs = design.T @ response + precision @ prior_mean
        self.coef_ = np.linalg.pinv(lhs) @ rhs
        self.last_values_ = values[-self.n_lag :]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.coef_ is None or self.last_values_ is None or self.target_name_ is None:
            return np.full(len(X), self.fallback_, dtype=float)
        history = [row.copy() for row in np.asarray(self.last_values_, dtype=float)]
        target_pos = self.names_.index(self.target_name_)
        preds: list[float] = []
        for _ in range(len(X)):
            row = _var_forecast_row(history, self.n_lag, intercept=self.intercept)
            forecast = row @ self.coef_
            preds.append(float(forecast[target_pos]))
            history.append(np.asarray(forecast, dtype=float))
        return np.asarray(preds, dtype=float)


def bvar_minnesota(
    panel: Any,
    *,
    target: str | None = None,
    n_lag: int = 1,
    shrinkage: float = 0.2,
    intercept: bool = True,
    random_walk_prior: bool = True,
) -> ModelFit:
    """Fit a compact Minnesota-prior Bayesian VAR for point forecasts."""

    frame = as_frame(panel)
    estimator = _BayesianVAR(
        n_lag=n_lag,
        target=target,
        prior="minnesota",
        shrinkage=shrinkage,
        intercept=intercept,
        random_walk_prior=random_walk_prior,
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
            "shrinkage": float(shrinkage),
            "intercept": bool(intercept),
            "random_walk_prior": bool(random_walk_prior),
            "implementation_note": "Conjugate-prior posterior mean VAR point forecast.",
        },
    )


def bvar_normal_inverse_wishart(
    panel: Any,
    *,
    target: str | None = None,
    n_lag: int = 1,
    shrinkage: float = 1.0,
    intercept: bool = True,
) -> ModelFit:
    """Fit a compact normal-inverse-Wishart-style Bayesian VAR."""

    frame = as_frame(panel)
    estimator = _BayesianVAR(
        n_lag=n_lag,
        target=target,
        prior="normal_inverse_wishart",
        shrinkage=shrinkage,
        intercept=intercept,
        random_walk_prior=False,
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
            "shrinkage": float(shrinkage),
            "intercept": bool(intercept),
            "implementation_note": "Normal-prior posterior mean VAR point forecast; variance prior is metadata-only for point forecasts.",
        },
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


def favar(
    X: Any,
    y: Any | None = None,
    *,
    n_factors: int = 3,
    n_lag: int = 1,
    random_state: int = 0,
) -> ModelFit:
    """Fit PCA factors and a VAR on the target plus factors."""

    from sklearn.decomposition import PCA

    frame, target = resolve_xy(X, y)
    n_factors_resolved = min(max(1, int(n_factors)), frame.shape[1], max(1, frame.shape[0] - 1))
    pca = PCA(n_components=n_factors_resolved, random_state=random_state)
    factors = pca.fit_transform(frame.fillna(0.0))
    factor_frame = pd.DataFrame(
        factors,
        index=frame.index,
        columns=[f"factor_{i + 1}" for i in range(n_factors_resolved)],
    )
    estimator = _VAR(n_lag=n_lag, target="__target__")
    estimator.fit(factor_frame, target)
    fit = ModelFit(
        estimator=estimator,
        model="favar",
        feature_names=tuple(factor_frame.columns),
        target_name=str(target.name) if target.name is not None else None,
        metadata={"n_obs": len(factor_frame), "n_factors": n_factors_resolved, "n_lag": int(n_lag)},
    )
    fit.metadata["pca"] = pca
    return fit


class _MixedFrequencyDFM:
    """Statsmodels DynamicFactorMQ wrapper for monthly/quarterly panels."""

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
                "Uses statsmodels DynamicFactorMQ with monthly columns ordered before "
                "quarterly columns and k_endog_monthly set, which applies the "
                "Mariano-Murasawa monthly-to-quarterly aggregation in the state-space model."
            ),
        },
        diagnostics=diagnostics,
    )


class _DFMUnrestrictedMIDAS:
    """Composite DFM-factor plus unrestricted-MIDAS forecast head."""

    def __init__(self, *, midas_estimator: Any, feature_names: tuple[str, ...]) -> None:
        self.midas_estimator = midas_estimator
        self.feature_names_in_ = tuple(feature_names)
        self.coef_ = getattr(midas_estimator, "coef_", None)
        self.intercept_ = getattr(midas_estimator, "intercept_", None)
        self.groups_ = getattr(midas_estimator, "groups_", {})
        self.design_: pd.DataFrame | None = None

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        frame = X.reindex(columns=list(self.feature_names_in_), fill_value=np.nan).astype(float)
        return np.asarray(self.midas_estimator.predict(frame), dtype=float).reshape(-1)


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

    from macroforecast.feature_engineering import mixed_frequency_lags

    frame, meta = _coerce_panel_with_metadata(panel, metadata=metadata)
    if target not in frame.columns:
        raise ValueError(f"target {target!r} is not in the mixed-frequency panel")
    dfm_fit = dfm_mixed_mariano_murasawa(
        (frame, meta),
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
    anchor_index = _position_target_dates(
        frame[target].dropna().index,
        frequency=target_frequency or meta.get("frequency") or "quarterly",
        anchor_position=anchor_position,
    )
    design_parts: list[pd.DataFrame] = []
    factor_design = _factor_lag_design(factors, anchor_index=anchor_index, lags=factor_lags)
    design_parts.append(factor_design)
    lag_column_values = tuple(str(column) for column in lag_columns or ())
    if lag_column_values:
        observed_lags = mixed_frequency_lags(
            (frame, meta),
            target=target,
            columns=lag_column_values,
            lags=lags,
            target_frequency=target_frequency,
            anchor_position=anchor_position,
            drop_missing=False,
        )
        design_parts.append(observed_lags)
    design = pd.concat(design_parts, axis=1)
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
            "lags": list(_normalize_model_lags(lags)),
            "factor_lags": list(_normalize_model_lags(factor_lags)),
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
            "prediction_contract": "predict() expects a prepared feature matrix with the fitted feature columns",
        },
        diagnostics={
            "design": X_fit,
            "dfm": dfm_fit.to_dict(),
            "midas": midas_fit.to_dict(),
        },
    )


class _MIDASRegressor:
    """Linear MIDAS-style regression over lag groups."""

    def __init__(
        self,
        *,
        weighting: str = "almon",
        polynomial_order: int = 2,
        theta: tuple[float, ...] | None = None,
        beta_params: tuple[float, float] = (1.0, 1.0),
        n_steps: int = 3,
        alpha: float = 0.0,
        fit_intercept: bool = True,
    ) -> None:
        self.weighting = str(weighting)
        self.polynomial_order = int(polynomial_order)
        self.theta = None if theta is None else tuple(float(value) for value in theta)
        self.beta_params = (float(beta_params[0]), float(beta_params[1]))
        self.n_steps = max(1, int(n_steps))
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
            )
            self.weights_[group] = weights.tolist()
            block = frame.loc[:, columns].astype(float)
            pieces[f"{group}_midas"] = block.mul(weights, axis=1).sum(
                axis=1,
                min_count=len(columns),
            )
        return pd.DataFrame(pieces, index=frame.index)


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
    alpha: float = 0.0,
    fit_intercept: bool = True,
) -> ModelFit:
    """Fit linear MIDAS using equal step-function lag buckets."""

    n_steps = _validate_positive_int("n_steps", n_steps)
    alpha = _validate_nonnegative_float("alpha", alpha)
    params = {
        "n_steps": n_steps,
        "alpha": alpha,
        "fit_intercept": bool(fit_intercept),
    }
    fit = fit_estimator(
        _MIDASRegressor(
            weighting="step",
            n_steps=n_steps,
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
            "Unrestricted MIDAS: every supplied lag column receives its own "
            "coefficient. Build X with feature_engineering.mixed_frequency_lags()."
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
        except Exception:
            self.result_ = None
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
        return tuple(0.0 for _ in range(polynomial_order + 1))
    values = tuple(float(value) for value in theta)
    expected = polynomial_order + 1
    if len(values) != expected:
        raise ValueError(
            f"theta must contain polynomial_order + 1 values; expected {expected}, got {len(values)}"
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
) -> np.ndarray:
    n = len(lags)
    if n == 0:
        return np.empty(0, dtype=float)
    if polynomial_order < 0:
        raise ValueError("polynomial_order must be non-negative")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    x = np.linspace(0.0, 1.0, n)
    if weighting == "almon":
        theta_values = theta or tuple(0.0 for _ in range(polynomial_order + 1))
        raw: np.ndarray = np.zeros(n, dtype=float)
        for power, value in enumerate(theta_values):
            raw += float(value) * np.power(x, power)
        weights = np.exp(raw - raw.max())
    elif weighting == "beta":
        a, b = beta_params
        if a <= 0.0 or b <= 0.0:
            raise ValueError("beta_params must be positive")
        z = np.linspace(1.0 / (n + 1.0), n / (n + 1.0), n)
        weights = np.power(z, a - 1.0) * np.power(1.0 - z, b - 1.0)
    elif weighting == "step":
        buckets = np.array_split(np.arange(n), max(1, min(n_steps, n)))
        weights = np.zeros(n, dtype=float)
        for bucket in buckets:
            weights[bucket] = 1.0 / (len(buckets) * max(1, len(bucket)))
    else:
        raise ValueError("weighting must be 'almon', 'beta', or 'step'")
    total = float(weights.sum())
    if total <= 1e-12:
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
    "theta_method",
    "unrestricted_midas",
    "var",
]
