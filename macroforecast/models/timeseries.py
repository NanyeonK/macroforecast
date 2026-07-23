from __future__ import annotations

import re
import warnings

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, cast

import numpy as np
import pandas as pd

from macroforecast.models.types import ModelFit
from macroforecast.models.utils import as_frame, as_series, fit_estimator, resolve_xy


_LAG_COL_RE = re.compile(r"_lag(\d+)$")


def _select_lag_columns(X: Any, n_lag: int, target_name: Any = None) -> list[str]:
    """The TARGET's own lag columns ``{base}_lag0..{base}_lag{n_lag-1}``, in lag order.

    Uses the ``n_lag`` MOST RECENT observed one-period values, i.e. lags ``0..n_lag-1``.
    Lag 0 is the value AT the origin (the decision time), which is observed and is NOT
    look-ahead for a future target; it is the most informative predictor, and the
    iterated AR used by the recursive/path policies seeds from exactly these
    ``Y_lag0..Y_lag{n_lag-1}`` values, so including lag 0 makes the direct projection
    match them at horizon 1. When ``target_name`` is given, only the lags whose base
    names the target are kept (an autoregression uses the target's OWN lags, not
    predictor lags such as ``x0_lag1``); the base is matched exactly or as the longest
    prefix of the target name (e.g. base ``Y`` for target ``Y_average_value_h6``).
    Without a ``target_name`` (or no base matches) every ``*_lag`` column in range is
    used, which is correct when the spec carries only target lags (``predictors=[]``).
    """
    # Collect EVERY lag column (any index) so the target base can be matched even
    # when its own lags fall outside [0, n_lag-1]; range restriction happens AFTER
    # base selection so a target whose lags start at 1 is never silently replaced
    # by predictors' *_lag0 columns.
    all_lags: list[tuple[int, str, str]] = []
    for col in pd.DataFrame(X).columns:
        match = _LAG_COL_RE.search(str(col))
        if match is not None:
            base = str(col)[: match.start()]
            all_lags.append((int(match.group(1)), base, str(col)))
    if target_name is not None:
        bases = {base for _, base, _ in all_lags}
        chosen: str | None = None
        if str(target_name) in bases:
            chosen = str(target_name)
        else:
            prefixes = [b for b in bases if b and str(target_name).startswith(b)]
            if prefixes:
                chosen = max(prefixes, key=len)
            elif len(bases) == 1:
                # A single lag base that does not name-match the target is still the
                # target's OWN lags (the spec carries only target lags, predictors=[]);
                # a benign name mismatch, not a predictor block.
                chosen = next(iter(bases))
        if chosen is None:
            # Multiple lag bases and none name the target: the target's own lags are
            # absent. Return empty so the caller falls back to the mean rather than
            # regressing the target on predictors' contemporaneous values.
            return []
        all_lags = [t for t in all_lags if t[1] == chosen]
    in_range = [t for t in all_lags if 0 <= t[0] <= int(n_lag) - 1]
    in_range.sort()
    return [col for _, _, col in in_range]


def _ols_with_intercept(design: np.ndarray, response: np.ndarray) -> np.ndarray:
    """OLS coefficients for ``[1, design] @ beta = response`` (intercept first)."""
    full = np.column_stack([np.ones(len(design)), design])
    return np.linalg.lstsq(full, response, rcond=None)[0]


class _AR:
    def __init__(self, *, n_lag: int = 1, direct: bool = False) -> None:
        self.n_lag = max(1, int(n_lag))
        # direct=True: a one-shot projection of the (h-ahead) target onto the n most
        # recent OBSERVED one-period lags in X (``*_lag0..*_lag{n_lag-1}``), predicted
        # per row. ``*_lag0`` is the value AT the origin (the decision time), which is
        # observed and is NOT look-ahead for a future target; see _select_lag_columns.
        # This is the correct direct multi-step forecast. The legacy roll-forward
        # (direct=False) autoregresses the target's own lags and iterates from the
        # last training value; under a direct policy that value is origin-h stale, so
        # it persists a stale value. direct=False is kept ONLY for the recursive/path
        # policies, where the runner iterates externally on the fresh one-period series.
        self.direct = bool(direct)
        self._coef: np.ndarray | None = None
        self._history: np.ndarray | None = None
        self._fallback: float = 0.0
        self.ssr_: float | None = None
        self.nobs_: int | None = None
        self.n_params_: int | None = None
        self._direct_cols: list[str] | None = None
        self._direct_coef: np.ndarray | None = None

    def _set_mean_ic(self, target: pd.Series) -> None:
        # IC stats for the degenerate mean-only direct model (no usable target lags),
        # so information-criterion order selection can compute BIC/AIC for it.
        resid = pd.Series(target).astype(float).dropna().to_numpy(dtype=float) - self._fallback
        self.ssr_ = float(resid @ resid)
        self.nobs_ = int(resid.shape[0])
        self.n_params_ = 1

    def _warn_if_predictor_lags_present(self, Xdf: pd.DataFrame, target: pd.Series) -> None:
        # A direct AR that finds no usable target lag but DOES carry non-target
        # ``*_lag`` columns is almost always a mis-specified benchmark: the feature
        # spec omitted the target's lag 0 (``feature_spec`` ``target_lags`` is
        # 1-indexed) while including predictors, so the "AR" would have regressed the
        # target on predictors' contemporaneous values. We now fall back to the mean;
        # surface the mistake loudly rather than silently shipping a degenerate
        # benchmark that corrupts every relative metric normalized against it.
        tgt = str(getattr(target, "name", "") or "")
        foreign = [
            c
            for c in map(str, Xdf.columns)
            if (m := _LAG_COL_RE.search(c)) is not None
            and (base := c[: m.start()]) != tgt
            and not (tgt and base and tgt.startswith(base))
        ]
        if foreign:
            warnings.warn(
                f"direct AR for target {tgt!r} found no usable target lag columns "
                f"(n_lag={self.n_lag}) yet the feature matrix carries "
                f"{len(foreign)} predictor lag column(s) (e.g. {foreign[0]!r}); "
                f"falling back to the unconditional mean. The feature spec likely "
                f"omitted the target's lag 0 -- feature_spec target_lags is 1-indexed, "
                f"so pass target_lags=range(0, K) (and predictors=[] for a pure "
                f"autoregression) to include the origin value.",
                UserWarning,
                stacklevel=2,
            )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_AR":
        if self.direct:
            return self._fit_direct(X, y)
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
        # In-sample one-step residuals for information-criterion order selection.
        resid = response - design @ self._coef
        self.ssr_ = float(resid @ resid)
        self.nobs_ = int(response.shape[0])
        self.n_params_ = int(self._coef.shape[0])
        self._history = values[-self.n_lag :]
        return self

    def _fit_direct(self, X: pd.DataFrame, y: pd.Series) -> "_AR":
        Xdf = pd.DataFrame(X)
        target = pd.Series(y).astype(float)
        self._fallback = float(target.dropna().mean()) if not target.dropna().empty else 0.0
        self._direct_cols = _select_lag_columns(Xdf, self.n_lag, target_name=getattr(target, "name", None))
        if not self._direct_cols:
            # No usable target lags (e.g. n_lag restricts to lag0 but the spec's target
            # lags start at 1): fall back to the unconditional mean rather than a
            # stale-persistence forecast. Expose IC stats for the mean-only model so
            # information-criterion order selection can score and skip this order, and
            # warn if predictor lag columns are present (a mis-specified AR benchmark).
            self._warn_if_predictor_lags_present(Xdf, target)
            self._set_mean_ic(target)
            return self
        design_df = Xdf[self._direct_cols].astype(float)
        joined = pd.concat([design_df, target.rename("__target__")], axis=1).dropna()
        if joined.empty:
            self._direct_cols = None
            self._set_mean_ic(target)
            return self
        design = joined[self._direct_cols].to_numpy(dtype=float)
        response = joined["__target__"].to_numpy(dtype=float)
        self._direct_coef = _ols_with_intercept(design, response)
        resid = response - np.column_stack([np.ones(len(design)), design]) @ self._direct_coef
        self.ssr_ = float(resid @ resid)
        self.nobs_ = int(response.shape[0])
        self.n_params_ = int(self._direct_coef.shape[0])  # n_lag columns + intercept
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.direct:
            return self._predict_direct(X)
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

    def _predict_direct(self, X: pd.DataFrame) -> np.ndarray:
        Xdf = pd.DataFrame(X)
        if self._direct_coef is None or not self._direct_cols:
            return np.full(len(Xdf), self._fallback, dtype=float)
        design = Xdf.reindex(columns=self._direct_cols).astype(float).fillna(0.0).to_numpy()
        return np.column_stack([np.ones(len(design)), design]) @ self._direct_coef


def ar(X: Any, y: Any | None = None, *, n_lag: int = 1, direct: bool = False) -> ModelFit:
    """Fit a fixed-order AR(``n_lag``) by OLS.

    By default (``direct=False``) this is a univariate least-squares AR on the
    target series with an explicit intercept (the feature matrix is ignored); it is
    NOT ``stats::ar``. With ``direct=True`` it becomes a DIRECT multi-step
    projection: the (h-ahead) target is regressed on the n most recent observed
    one-period lags ``*_lag0..*_lag{n_lag-1}`` in ``X`` and each row is predicted
    independently. ``*_lag0`` is the value at the origin (the decision time), which
    is observed and is NOT look-ahead for a future target. The runner sets
    ``direct=True`` only for the direct/direct_average policies.
    """

    # The supervised dispatch calls ar(X, y, ...); a bare-series call is ar(series).
    target = as_series(y if y is not None else X)
    if bool(direct):
        # direct projection needs the real feature matrix (the *_lag columns).
        features = as_frame(X)
    else:
        # non-direct autoregresses the target and IGNORES the feature matrix; supply
        # a 1-column dummy so fit_estimator accepts it (univariate behavior, and it
        # works whether X was a real feature matrix, an empty frame, or a bare series).
        features = pd.DataFrame(
            {"__origin__": np.arange(len(target), dtype=float)}, index=target.index
        )
    return fit_estimator(
        _AR(n_lag=n_lag, direct=bool(direct)), features, target,
        model="ar", metadata={"n_lag": int(n_lag), "direct": bool(direct)},
    )


_AR_BIC_CRITERIA = frozenset({"aic", "aicc", "bic"})
_AR_BIC_PARAMETER_COUNTS = frozenset({"standard", "lag_square"})
_AR_BIC_ESTIMATORS = frozenset({"ols", "yule_walker", "burg", "matlab_ar"})
_AR_BIC_FORECAST_MODES = frozenset(
    {"iterated", "direct_lag_projection", "coefficient_power"}
)
_AR_BIC_VARIANCE_FLOOR = float(np.finfo(float).tiny)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    out = int(value)
    if out <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return out


def _bool_param(value: Any, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean")
    return bool(value)


def _ar_lagged_response(values: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    if p <= 0 or len(values) <= p:
        return np.empty(0, dtype=float), np.empty((0, max(0, p)), dtype=float)
    response = values[p:]
    lags = np.column_stack([values[p - j : len(values) - j] for j in range(1, p + 1)])
    valid = np.isfinite(response) & np.isfinite(lags).all(axis=1)
    return response[valid], lags[valid]


def _ar_design(lags: np.ndarray, *, include_constant: bool) -> np.ndarray:
    if include_constant:
        return np.column_stack([np.ones(len(lags)), lags])
    return lags


def _ar_score_parameter_count(
    p: int, *, include_constant: bool, ic_parameter_count: str
) -> int:
    if ic_parameter_count == "lag_square":
        return int(p * p)
    return int(p + 1 if include_constant else p)


def _residual_variance_ic(
    ssr: float, nobs: int, n_params: int, criterion: str
) -> tuple[float, str]:
    if nobs <= 0 or not np.isfinite(ssr):
        return float("inf"), "invalid"
    variance = max(float(ssr) / float(nobs), _AR_BIC_VARIANCE_FLOOR)
    base = float(nobs) * float(np.log(variance))
    if criterion == "aic":
        return base + 2.0 * float(n_params), "ok"
    if criterion == "aicc":
        denom = int(nobs) - int(n_params) - 1
        if denom <= 0:
            return float("inf"), "aicc_degenerate"
        penalty = 2.0 * float(n_params) * float(n_params + 1) / float(denom)
        return base + 2.0 * float(n_params) + penalty, "ok"
    return base + float(n_params) * float(np.log(float(nobs))), "ok"


def _ar_ols_fit(
    values: np.ndarray, p: int, *, include_constant: bool
) -> tuple[float, np.ndarray, float, int, int]:
    response, lags = _ar_lagged_response(values, p)
    if response.size == 0:
        raise ValueError("not enough observations for final AR fit")
    design = _ar_design(lags, include_constant=include_constant)
    beta = np.linalg.lstsq(design, response, rcond=None)[0]
    fitted = design @ beta
    resid = response - fitted
    if include_constant:
        intercept = float(beta[0])
        coef = np.asarray(beta[1:], dtype=float)
    else:
        intercept = 0.0
        coef = np.asarray(beta, dtype=float)
    return intercept, coef, float(resid @ resid), int(len(response)), int(len(beta))


def _ar_common_residuals(
    values: np.ndarray, p: int, intercept: float, coef: np.ndarray
) -> tuple[float, int]:
    response, lags = _ar_lagged_response(values, p)
    if response.size == 0:
        raise ValueError("not enough observations for final AR residuals")
    resid = response - (float(intercept) + lags @ np.asarray(coef, dtype=float))
    return float(resid @ resid), int(len(response))


def _yule_walker_coefficients(u: np.ndarray, p: int) -> tuple[np.ndarray, str]:
    gamma = np.asarray(
        [float(u[k:] @ u[: len(u) - k]) / float(len(u)) for k in range(p + 1)],
        dtype=float,
    )
    toeplitz = gamma[np.abs(np.subtract.outer(np.arange(p), np.arange(p)))]
    rhs = gamma[1 : p + 1]
    status = "ok"
    try:
        condition = float(np.linalg.cond(toeplitz))
    except np.linalg.LinAlgError:
        condition = float("inf")
    if not np.isfinite(condition) or condition > 1.0 / np.sqrt(np.finfo(float).eps):
        coef = np.linalg.pinv(toeplitz) @ rhs
        status = "pinv"
    else:
        try:
            coef = np.linalg.solve(toeplitz, rhs)
        except np.linalg.LinAlgError:
            coef = np.linalg.pinv(toeplitz) @ rhs
            status = "pinv"
    return np.asarray(coef, dtype=float), status


def _burg_coefficients(u: np.ndarray, p: int) -> np.ndarray:
    if len(u) <= p:
        raise ValueError("not enough observations for Burg AR fit")
    forward = np.asarray(u[1:], dtype=float).copy()
    backward = np.asarray(u[:-1], dtype=float).copy()
    coef: np.ndarray = np.empty(0, dtype=float)
    for order in range(p):
        denom = float(forward @ forward + backward @ backward)
        if denom <= _AR_BIC_VARIANCE_FLOOR or not np.isfinite(denom):
            raise ValueError("Burg AR recursion encountered a zero denominator")
        reflection = float(2.0 * (forward @ backward) / denom)
        if coef.size:
            coef = np.concatenate([coef - reflection * coef[::-1], [reflection]])
        else:
            coef = np.asarray([reflection], dtype=float)
        if order < p - 1:
            old_forward = forward
            old_backward = backward
            forward = old_forward[1:] - reflection * old_backward[1:]
            backward = old_backward[:-1] - reflection * old_forward[:-1]
    if not np.isfinite(coef).all():
        raise ValueError("Burg AR fit produced non-finite coefficients")
    return coef


def _matlab_ar_coefficients(u: np.ndarray, p: int) -> np.ndarray:
    if len(u) <= p:
        raise ValueError("not enough observations for MATLAB-compatible AR fit")
    forward_response = u[p:]
    forward_lags = np.column_stack([u[p - j : len(u) - j] for j in range(1, p + 1)])
    origins = np.arange(0, len(u) - p)
    backward_response = u[origins]
    backward_lags = np.column_stack([u[origins + j] for j in range(1, p + 1)])
    response = np.concatenate([forward_response, backward_response])
    design = np.vstack([forward_lags, backward_lags])
    coef = np.linalg.lstsq(design, response, rcond=None)[0]
    if not np.isfinite(coef).all():
        raise ValueError("MATLAB-compatible AR fit produced non-finite coefficients")
    return np.asarray(coef, dtype=float)


class _ARBIC:
    def __init__(
        self,
        *,
        min_lag: int = 1,
        max_lag: int = 12,
        criterion: str = "bic",
        include_constant: bool = True,
        ic_parameter_count: str = "standard",
        estimator: str = "ols",
        forecast_mode: str = "iterated",
        horizon: int = 1,
    ) -> None:
        self.min_lag = _positive_int(min_lag, "min_lag")
        self.max_lag = _positive_int(max_lag, "max_lag")
        if self.min_lag > self.max_lag:
            raise ValueError("min_lag must be less than or equal to max_lag")
        self.criterion = str(criterion).lower()
        if self.criterion not in _AR_BIC_CRITERIA:
            raise ValueError("criterion must be one of: aic, aicc, bic")
        self.include_constant = _bool_param(include_constant, "include_constant")
        self.ic_parameter_count = str(ic_parameter_count).lower()
        if self.ic_parameter_count not in _AR_BIC_PARAMETER_COUNTS:
            raise ValueError("ic_parameter_count must be one of: standard, lag_square")
        self.estimator = str(estimator).lower()
        if self.estimator not in _AR_BIC_ESTIMATORS:
            raise ValueError("estimator must be one of: ols, yule_walker, burg, matlab_ar")
        self.forecast_mode = str(forecast_mode).lower()
        if self.forecast_mode not in _AR_BIC_FORECAST_MODES:
            raise ValueError(
                "forecast_mode must be one of: iterated, direct_lag_projection, "
                "coefficient_power"
            )
        if self.forecast_mode == "direct_lag_projection" and self.estimator != "ols":
            raise ValueError("direct_lag_projection requires estimator='ols'")
        self.horizon = _positive_int(horizon, "horizon")
        self.selected_lag_: int | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.ssr_: float | None = None
        self.nobs_: int | None = None
        self.n_params_: int | None = None
        self.ic_trials_: pd.DataFrame | None = None
        self.selected_ic_: float | None = None
        self.selected_nobs_: int | None = None
        self.selected_n_params_: int | None = None
        self.backend_status_: str = "ok"
        self._values: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_ARBIC":
        del X
        raw = pd.Series(y).astype(float).to_numpy(dtype=float)
        values = raw[np.isfinite(raw)]
        self._values = values
        rows = self._score_lags(values)
        self.ic_trials_ = pd.DataFrame(rows)
        ok = self.ic_trials_[np.isfinite(self.ic_trials_["score"].to_numpy(dtype=float))]
        if ok.empty:
            raise ValueError(
                "No finite AR information-criterion score for lags "
                f"{self.min_lag}..{self.max_lag} using {self.criterion}."
            )
        best = ok.sort_values(["score", "lag"], kind="mergesort").iloc[0]
        self.selected_lag_ = int(best["lag"])
        self.selected_ic_ = float(best["score"])
        self.selected_nobs_ = int(best["nobs"])
        self.selected_n_params_ = int(best["n_params"])
        self._fit_final(values, self.selected_lag_)
        return self

    def _score_lags(self, values: np.ndarray) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for p in range(self.min_lag, self.max_lag + 1):
            response, lags = _ar_lagged_response(values, p)
            nobs = int(len(response))
            n_params = _ar_score_parameter_count(
                p,
                include_constant=self.include_constant,
                ic_parameter_count=self.ic_parameter_count,
            )
            row: dict[str, Any] = {
                "lag": int(p),
                "score": float("inf"),
                "nobs": nobs,
                "n_params": int(n_params),
                "ssr": float("nan"),
                "status": "no_rows" if nobs <= 0 else "ok",
            }
            if nobs > 0:
                try:
                    design = _ar_design(lags, include_constant=self.include_constant)
                    beta = np.linalg.lstsq(design, response, rcond=None)[0]
                    resid = response - design @ beta
                    ssr = float(resid @ resid)
                    score, status = _residual_variance_ic(
                        ssr, nobs, n_params, self.criterion
                    )
                    row.update({"score": float(score), "ssr": ssr, "status": status})
                except np.linalg.LinAlgError as exc:
                    row.update({"status": "error", "error": str(exc)})
            rows.append(row)
        return rows

    def _fit_final(self, values: np.ndarray, p: int) -> None:
        if self.estimator == "ols" or self.forecast_mode == "direct_lag_projection":
            intercept, coef, ssr, nobs, n_params = _ar_ols_fit(
                values, p, include_constant=self.include_constant
            )
            self.intercept_ = intercept
            self.coef_ = coef
            self.ssr_ = ssr
            self.nobs_ = nobs
            self.n_params_ = n_params
            return
        mean = float(np.mean(values)) if self.include_constant else 0.0
        u = values - mean if self.include_constant else values
        if self.estimator == "yule_walker":
            coef, status = _yule_walker_coefficients(u, p)
            self.backend_status_ = status
        elif self.estimator == "burg":
            coef = _burg_coefficients(u, p)
        else:
            coef = _matlab_ar_coefficients(u, p)
        intercept = mean * (1.0 - float(np.sum(coef))) if self.include_constant else 0.0
        ssr, nobs = _ar_common_residuals(values, p, intercept, coef)
        self.intercept_ = float(intercept)
        self.coef_ = np.asarray(coef, dtype=float)
        self.ssr_ = float(ssr)
        self.nobs_ = int(nobs)
        self.n_params_ = int(p + 1 if self.include_constant else p)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        steps = len(X)
        if steps == 0:
            return np.empty(0, dtype=float)
        if self._values is None or self.coef_ is None or self.selected_lag_ is None:
            raise ValueError("ar_bic estimator is not fitted")
        if self.forecast_mode == "direct_lag_projection":
            return self._predict_direct(steps)
        max_step = self.horizon + steps - 1
        if self.forecast_mode == "coefficient_power":
            return self._predict_coefficient_power(max_step)[self.horizon - 1 :]
        return self._predict_iterated(max_step)[self.horizon - 1 :]

    def _latest_state(self) -> np.ndarray:
        if self._values is None or self.selected_lag_ is None:
            raise ValueError("ar_bic estimator is not fitted")
        p = self.selected_lag_
        return self._values[-p:][::-1].astype(float)

    def _predict_iterated(self, max_step: int) -> np.ndarray:
        state = self._latest_state()
        coef = np.asarray(self.coef_, dtype=float)
        out: np.ndarray = np.empty(max_step, dtype=float)
        for i in range(max_step):
            pred = float(self.intercept_ + coef @ state)
            out[i] = pred
            state = np.concatenate([[pred], state[:-1]])
        return out

    def _predict_coefficient_power(self, max_step: int) -> np.ndarray:
        state = self._latest_state()
        coef = np.asarray(self.coef_, dtype=float)
        return np.asarray(
            [float((coef ** step) @ state) for step in range(1, max_step + 1)],
            dtype=float,
        )

    def _predict_direct(self, steps: int) -> np.ndarray:
        if self._values is None or self.selected_lag_ is None:
            raise ValueError("ar_bic estimator is not fitted")
        values = self._values
        p = self.selected_lag_
        state = self._latest_state()
        preds: list[float] = []
        for offset in range(steps):
            step = self.horizon + offset
            origins = np.arange(p - 1, len(values) - step)
            if origins.size == 0:
                raise ValueError(
                    f"not enough observations for direct_lag_projection step {step}"
                )
            response = values[origins + step]
            lags = np.column_stack([values[origins - j] for j in range(p)])
            design = _ar_design(lags, include_constant=self.include_constant)
            beta = np.linalg.lstsq(design, response, rcond=None)[0]
            if self.include_constant:
                pred = float(beta[0] + beta[1:] @ state)
            else:
                pred = float(beta @ state)
            preds.append(pred)
        return np.asarray(preds, dtype=float)


def ar_bic(
    y: Any,
    *,
    min_lag: int = 1,
    max_lag: int = 12,
    criterion: str = "bic",
    include_constant: bool = True,
    ic_parameter_count: str = "standard",
    estimator: str = "ols",
    forecast_mode: str = "iterated",
    horizon: int = 1,
) -> ModelFit:
    """Target-only AR with internal residual-variance IC lag selection.

    The input target is used as supplied: callers own differencing, moving
    averages, scaling, and other leak-free target preparation. Candidate lags
    are scored by AIC/AICc/BIC on target-only OLS lag regressions, then the
    selected lag is refit with the requested AR backend and forecast contract.
    """

    target = as_series(y)
    estimator_obj = _ARBIC(
        min_lag=min_lag,
        max_lag=max_lag,
        criterion=criterion,
        include_constant=include_constant,
        ic_parameter_count=ic_parameter_count,
        estimator=estimator,
        forecast_mode=forecast_mode,
        horizon=horizon,
    )
    dummy = pd.DataFrame(
        {"__origin__": np.arange(len(target), dtype=float)}, index=target.index
    )
    estimator_obj.fit(dummy, target)
    assert estimator_obj.selected_lag_ is not None
    assert estimator_obj.selected_ic_ is not None
    assert estimator_obj.selected_nobs_ is not None
    assert estimator_obj.selected_n_params_ is not None
    metadata = {
        "n_obs": int(len(estimator_obj._values)) if estimator_obj._values is not None else 0,
        "min_lag": estimator_obj.min_lag,
        "max_lag": estimator_obj.max_lag,
        "criterion": estimator_obj.criterion,
        "include_constant": estimator_obj.include_constant,
        "ic_parameter_count": estimator_obj.ic_parameter_count,
        "estimator": estimator_obj.estimator,
        "forecast_mode": estimator_obj.forecast_mode,
        "horizon": estimator_obj.horizon,
        "selected_lag": estimator_obj.selected_lag_,
        "selected_ic": estimator_obj.selected_ic_,
        "selected_nobs": estimator_obj.selected_nobs_,
        "selected_n_params": estimator_obj.selected_n_params_,
    }
    diagnostics = {
        "ic_trials": estimator_obj.ic_trials_,
        "backend_status": estimator_obj.backend_status_,
    }
    return ModelFit(
        estimator=estimator_obj,
        model="ar_bic",
        feature_names=("__origin__",),
        target_name=str(target.name) if target.name is not None else None,
        metadata=metadata,
        diagnostics=diagnostics,
    )


def _infer_seasonal_period(index: Any) -> int | None:
    """Infer a seasonal period from a DatetimeIndex frequency (12 monthly, 4 quarterly)."""
    try:
        freq = getattr(index, "freqstr", None) or pd.infer_freq(index)
    except Exception:
        freq = None
    if not freq:
        return None
    head = str(freq).upper().split("-")[0]
    if head in {"M", "MS", "ME", "BM"}:
        return 12
    if head in {"Q", "QS", "QE", "BQ"}:
        return 4
    if head in {"D", "B"}:
        return 7
    if head in {"H"}:
        return 24
    return None


class _STLForecaster:
    """STL decomposition forecaster (R forecast::stlf).

    Seasonally adjust via STL, forecast the seasonally-adjusted series, then add
    back the last seasonal cycle (seasonal-naive). ``predict(X)`` returns a
    ``len(X)``-step path.
    """

    def __init__(self, *, period: int | None = None, sa_method: str = "ets") -> None:
        self.period = period
        self.sa_method = str(sa_method).lower()
        self._seasonal: np.ndarray | None = None
        self._sa_last: float = 0.0
        self._sa_drift: float = 0.0
        self._sa_forecast: Any = None
        self._fallback: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_STLForecaster":
        series = pd.Series(y).astype(float).dropna()
        self._fallback = float(series.mean()) if not series.empty else 0.0
        if series.empty:
            return self
        period = self.period or _infer_seasonal_period(series.index)
        values = series.to_numpy(dtype=float)
        seasonally_adjusted = values
        self._seasonal = None
        if period and period >= 2 and len(values) >= 2 * period:
            try:
                from statsmodels.tsa.seasonal import STL

                result = STL(series, period=int(period)).fit()
                seasonal = np.asarray(result.seasonal, dtype=float)
                self._seasonal = seasonal[-int(period):]
                seasonally_adjusted = values - seasonal
            except Exception:
                self._seasonal = None
                seasonally_adjusted = values
        # Forecast the seasonally-adjusted component.
        self._sa_last = float(seasonally_adjusted[-1])
        n = len(seasonally_adjusted)
        self._sa_drift = (
            float((seasonally_adjusted[-1] - seasonally_adjusted[0]) / (n - 1))
            if n > 1
            else 0.0
        )
        if self.sa_method == "ets":
            try:
                from statsmodels.tsa.holtwinters import ExponentialSmoothing

                sa_series = pd.Series(seasonally_adjusted, index=series.index)
                self._sa_forecast = ExponentialSmoothing(
                    sa_series, trend="add", seasonal=None
                ).fit()
            except Exception:
                self._sa_forecast = None
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        steps = len(X)
        if self._sa_forecast is not None:
            try:
                sa_path = np.asarray(self._sa_forecast.forecast(steps), dtype=float)
            except Exception:
                sa_path = np.array(
                    [self._sa_last + self._sa_drift * (k + 1) for k in range(steps)], dtype=float
                )
        else:
            sa_path = np.array(
                [self._sa_last + self._sa_drift * (k + 1) for k in range(steps)], dtype=float
            )
        if sa_path.size < steps:
            sa_path = np.resize(sa_path, steps)
        if self._seasonal is not None and len(self._seasonal):
            m = len(self._seasonal)
            seasonal_path = np.array([self._seasonal[k % m] for k in range(steps)], dtype=float)
            return sa_path[:steps] + seasonal_path
        return sa_path[:steps]


def stlf(y: Any, *, period: int | None = None, sa_method: str = "ets", **kwargs: Any) -> ModelFit:
    """STL + forecast: seasonally adjust, forecast, re-seasonalize (R forecast::stlf).

    Decomposes the target with STL, forecasts the seasonally-adjusted series
    (exponential smoothing with additive trend, falling back to a random-walk
    drift), and adds back the last seasonal cycle. ``period`` defaults to the
    index frequency (12 monthly, 4 quarterly); with no detectable seasonality the
    forecast reduces to the seasonally-adjusted path. Target-only.
    """

    target = as_series(y)
    dummy = pd.DataFrame(
        {"__origin__": np.arange(len(target), dtype=float)}, index=target.index
    )
    return fit_estimator(
        _STLForecaster(period=period, sa_method=sa_method),
        dummy,
        target,
        model="stlf",
        metadata={"period": (int(period) if period else None), "sa_method": str(sa_method)},
    )


def _absolute_season_positions(index: Any, length: int) -> np.ndarray:
    """Absolute season positions for a possibly-gapped series index.

    The forecasting harness drops rows with a missing target before a model sees
    them, but preserves the index, so the original season phase is recoverable
    from the index values rather than the positional count. DatetimeIndex maps to
    positions in the complete date range at the inferred frequency; an integer
    index uses its own values; otherwise a contiguous positional fallback is used.
    """
    try:
        if isinstance(index, pd.DatetimeIndex) and len(index) > 1:
            freq = index.freqstr or pd.infer_freq(index)
            if freq:
                full = pd.date_range(index[0], index[-1], freq=freq)
                lookup = {ts: i for i, ts in enumerate(full)}
                return np.asarray([lookup.get(ts, i) for i, ts in enumerate(index)], dtype=int)
        values = np.asarray(index)
        if np.issubdtype(values.dtype, np.integer):
            return values.astype(int)
    except Exception:
        pass
    return np.arange(length, dtype=int)


class _NaiveForecaster:
    """Baseline path forecaster: random-walk, seasonal-naive, or drift.

    ``predict(X)`` returns a ``len(X)``-step path (row ``k`` is the ``k+1``-step
    forecast), matching the package's target-only multi-step contract.
    """

    def __init__(self, *, method: str = "naive", period: int | None = None) -> None:
        self.method = str(method).lower()
        self.period = period
        self._last: float = 0.0
        self._drift: float = 0.0
        self._season: np.ndarray | None = None
        self._season_by_slot: np.ndarray | None = None
        self._period_: int = 1
        self._last_slot: int = 0
        self._fallback: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_NaiveForecaster":
        raw = pd.Series(y).astype(float)
        series = raw.dropna()
        self._fallback = float(series.mean()) if not series.empty else 0.0
        if series.empty:
            self._last = self._fallback
            return self
        values = series.to_numpy(dtype=float)
        self._last = float(values[-1])
        if self.method == "drift":
            n = len(values)
            # forecast::rwf(drift=TRUE): slope = (y_T - y_1) / (T - 1)
            self._drift = float((values[-1] - values[0]) / (n - 1)) if n > 1 else 0.0
        if self.method in {"snaive", "seasonal_naive"}:
            m = max(1, int(self.period)) if self.period else 1
            # forecast::snaive indexes by ABSOLUTE season position: each season slot
            # carries its last OBSERVED value. Positions come from the index (which
            # preserves gaps from dropped/missing observations), so a ragged edge or
            # an interior gap does not shift the seasons.
            positions = _absolute_season_positions(series.index, len(values))
            by_slot: np.ndarray = np.full(m, np.nan, dtype=float)
            for pos, value in zip(positions, values):
                if np.isfinite(value):
                    by_slot[int(pos) % m] = value
            by_slot = np.where(np.isfinite(by_slot), by_slot, self._last)
            self._season_by_slot = by_slot
            self._period_ = m
            self._last_slot = int(positions[-1]) % m
            self._season = by_slot
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        steps = len(X)
        if self.method == "drift":
            return np.asarray(
                [self._last + self._drift * (k + 1) for k in range(steps)], dtype=float
            )
        if (
            self.method in {"snaive", "seasonal_naive"}
            and self._season_by_slot is not None
            and len(self._season_by_slot)
        ):
            m = self._period_
            # step k (1-based) targets absolute position last+k -> season slot
            return np.asarray(
                [self._season_by_slot[(self._last_slot + k) % m] for k in range(1, steps + 1)],
                dtype=float,
            )
        # naive / random walk: last observed value carried forward
        return np.full(steps, self._last, dtype=float)


class _HistoricalMeanForecaster:
    """Constant forecaster equal to the fit-window target mean."""

    def __init__(self, *, window: int | None = None) -> None:
        if window is not None and int(window) <= 0:
            raise ValueError("window must be a positive integer or None")
        self.window = None if window is None else int(window)
        self.mean_: float = 0.0
        self.nobs_: int = 0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_HistoricalMeanForecaster":
        del X
        series = pd.Series(y).astype(float).dropna()
        if self.window is not None:
            series = series.tail(self.window)
        self.nobs_ = int(len(series))
        self.mean_ = float(series.mean()) if not series.empty else 0.0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.mean_, dtype=float)


def hist_mean(y: Any, *, window: int | None = None, **kwargs: Any) -> ModelFit:
    """Historical (prevailing) mean benchmark for the transformed target.

    Fits the canonical expanding historical-mean benchmark used in equity-premium
    and forecast-comparison work: every forecast from an origin equals the mean
    of the target values in that origin's fit window. ``window=None`` uses the
    full expanding fit sample; an integer ``window`` uses the last ``window``
    observed target values. The forecast is horizon-invariant by construction.

    The mean is computed on the target object supplied by the forecasting
    pipeline after target transformation. For example, a level target forecasts
    the historical mean level, while a growth or log-growth target forecasts the
    historical mean growth rate.
    """

    del kwargs
    target = as_series(y)
    dummy = pd.DataFrame(
        {"__origin__": np.arange(len(target), dtype=float)}, index=target.index
    )
    return fit_estimator(
        _HistoricalMeanForecaster(window=window),
        dummy,
        target,
        model="hist_mean",
        metadata={"window": (None if window is None else int(window))},
    )


def _naive_fit(method: str, y: Any, *, period: int | None = None) -> ModelFit:
    target = as_series(y)
    dummy = pd.DataFrame(
        {"__origin__": np.arange(len(target), dtype=float)}, index=target.index
    )
    return fit_estimator(
        _NaiveForecaster(method=method, period=period),
        dummy,
        target,
        model=method if method != "seasonal_naive" else "seasonal_naive",
        metadata={"method": method, "period": (int(period) if period else None)},
    )


def naive(y: Any, **kwargs: Any) -> ModelFit:
    """Random-walk (naive) forecaster: carry the last observed value forward.

    R analogue of ``forecast::naive`` / ``forecast::rwf(drift=FALSE)``. The h-step
    path is constant at ``y_T``. Target-only; the horizon is taken from the number
    of test rows by the forecasting runner.
    """

    return _naive_fit("naive", y)


def seasonal_naive(y: Any, *, period: int | None = None, **kwargs: Any) -> ModelFit:
    """Seasonal-naive forecaster: repeat the last full seasonal cycle.

    R analogue of ``forecast::snaive``. With seasonal period ``m`` the h-step path
    cycles the last ``m`` observed values, so step ``k`` returns ``y_{T-m+1+((k-1) mod m)}``.
    ``period`` defaults to 1 (degenerates to the plain naive forecast).
    """

    return _naive_fit("seasonal_naive", y, period=period)


def random_walk_drift(y: Any, **kwargs: Any) -> ModelFit:
    """Random-walk-with-drift forecaster.

    R analogue of ``forecast::rwf(drift=TRUE)``. The h-step path is
    ``y_T + h * (y_T - y_1) / (T - 1)``: the last value plus the average
    historical change extrapolated linearly.
    """

    return _naive_fit("drift", y)


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
        direct: bool = False,
        direct_horizon: int = 1,
    ) -> None:
        self.n_lag = max(1, int(n_lag))
        self.target = target
        self.type = _normalize_vars_type(type)
        self.season = None if season is None else max(2, int(season))
        self.direct = bool(direct)
        self.direct_horizon = max(1, int(direct_horizon))
        self.coef_: np.ndarray | None = None
        self.direct_coef_: np.ndarray | None = None
        self.names_: tuple[str, ...] = ()
        self.rhs_names_: tuple[str, ...] = ()
        self.direct_rhs_names_: tuple[str, ...] = ()
        self.datamat_: pd.DataFrame | None = None
        self.direct_datamat_: pd.DataFrame | None = None
        self.y_values_: np.ndarray | None = None
        self._target_name: str | None = None
        self._fallback: float = 0.0
        self.ssr_: float | None = None
        self.nobs_: int | None = None
        self.n_params_: int | None = None
        self.residual_variance_: float | None = None

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
        self.y_values_ = data.to_numpy(dtype=float)
        if self.direct:
            return self._fit_direct(data, target_name=target_name)
        if data.shape[0] <= self.n_lag + 1 or data.shape[1] < 2:
            return self
        values = self.y_values_
        yend, rhs, rhs_names = _vars_rhs(
            values,
            self.names_,
            self.n_lag,
            type=self.type,
            season=self.season,
        )
        self.coef_ = np.linalg.lstsq(rhs, yend, rcond=None)[0].T
        resid = yend - rhs @ self.coef_.T
        self.ssr_ = float(np.sum(resid * resid))
        self.nobs_ = int(yend.shape[0])
        self.n_params_ = int(rhs.shape[1])
        denom = max(1, int(yend.shape[0]) - int(rhs.shape[1]))
        self.residual_variance_ = float(np.sum(resid * resid) / denom)
        self.rhs_names_ = tuple(rhs_names)
        self.datamat_ = pd.DataFrame(
            np.column_stack([yend, rhs]),
            index=data.index[self.n_lag :],
            columns=[*self.names_, *self.rhs_names_],
        )
        return self

    def _fit_direct(self, data: pd.DataFrame, *, target_name: str) -> "_VAR":
        values = self.y_values_
        if values is None:
            return self
        target_index = self.names_.index(target_name)
        if values.shape[0] < self.n_lag + self.direct_horizon:
            return self
        response, rhs, rhs_names, origin_positions = _vars_direct_rhs(
            values,
            self.names_,
            self.n_lag,
            self.direct_horizon,
            type=self.type,
            season=self.season,
        )
        if response.size == 0 or rhs.size == 0:
            return self
        target_response = response[:, target_index]
        self.direct_coef_ = np.linalg.lstsq(rhs, target_response, rcond=None)[0]
        resid = target_response - rhs @ self.direct_coef_
        self.ssr_ = float(resid @ resid)
        self.nobs_ = int(target_response.shape[0])
        self.n_params_ = int(self.direct_coef_.shape[0])
        denom = max(1, int(target_response.shape[0]) - int(self.direct_coef_.shape[0]))
        self.residual_variance_ = float((resid @ resid) / denom)
        self.direct_rhs_names_ = tuple(rhs_names)
        self.direct_datamat_ = pd.DataFrame(
            np.column_stack([target_response, rhs]),
            index=data.index[origin_positions + self.direct_horizon],
            columns=[target_name, *self.direct_rhs_names_],
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.direct:
            return self._predict_direct(X)
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

    def _predict_direct(self, X: pd.DataFrame) -> np.ndarray:
        if (
            self.direct_coef_ is None
            or self.y_values_ is None
            or self._target_name is None
        ):
            return np.full(len(X), self._fallback, dtype=float)
        row = _vars_direct_forecast_row(
            self.y_values_,
            self.names_,
            self.n_lag,
            self.direct_horizon,
            type=self.type,
            season=self.season,
        )
        pred = float(row @ self.direct_coef_)
        return np.full(len(X), pred, dtype=float)


def var_restrict(
    panel: Any,
    *,
    n_lag: int = 1,
    type: str = "const",
    season: int | None = None,
    threshold: float = 2.0,
) -> dict[str, Any]:
    """Restricted VAR by sequential elimination of regressors (R vars::restrict).

    Fits a reduced-form VAR and, equation by equation, removes the regressor with
    the smallest absolute ``t`` statistic whenever it falls below ``threshold``,
    re-estimating after each removal (``method="ser"`` in ``vars::restrict``). This
    yields a parsimonious VAR with zero restrictions on the insignificant
    coefficients. Returns, per equation, the retained coefficients (and zeros for
    the eliminated terms), their ``t`` statistics, the names of the eliminated
    regressors, and a ``K x m`` restriction matrix (1 = retained, 0 = restricted).
    """

    frame = as_frame(panel).astype(float)
    if frame.isna().any().any():
        raise ValueError("var_restrict does not allow missing values")
    if frame.shape[1] < 2:
        raise ValueError("VAR panel must contain at least two variables")
    names = tuple(str(c) for c in frame.columns)
    vtype = _normalize_vars_type(type)
    n_lag = max(1, int(n_lag))
    values = frame.to_numpy(dtype=float)
    if values.shape[0] <= n_lag + 1:
        raise ValueError("not enough observations for the requested lag order")

    yend, rhs, rhs_names = _vars_rhs(values, names, n_lag, type=vtype, season=season)
    m = rhs.shape[1]
    n_eq = yend.shape[1]

    def _ols_t(y_col: np.ndarray, design: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xtx_inv = np.linalg.inv(design.T @ design)
        beta = xtx_inv @ (design.T @ y_col)
        resid = y_col - design @ beta
        dof = max(1, design.shape[0] - design.shape[1])
        sigma2 = float(resid @ resid) / dof
        se = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 0.0))
        tvals = np.divide(beta, se, out=np.full_like(beta, np.inf), where=se > 0)
        return beta, tvals

    equations: list[dict[str, Any]] = []
    restriction = np.ones((n_eq, m), dtype=int)
    for j in range(n_eq):
        y_col = yend[:, j]
        active = list(range(m))
        while len(active) > 1:
            design = rhs[:, active]
            beta, tvals = _ols_t(y_col, design)
            abs_t = np.abs(tvals)
            worst = int(np.argmin(abs_t))
            if abs_t[worst] < float(threshold):
                del active[worst]
            else:
                break
        design = rhs[:, active]
        beta, tvals = _ols_t(y_col, design)
        coef_full = np.zeros(m, dtype=float)
        t_full = np.zeros(m, dtype=float)
        for pos, idx in enumerate(active):
            coef_full[idx] = float(beta[pos])
            t_full[idx] = float(tvals[pos])
        eliminated = [rhs_names[i] for i in range(m) if i not in active]
        for i in range(m):
            restriction[j, i] = 1 if i in active else 0
        equations.append({
            "equation": names[j],
            "coefficients": {rhs_names[i]: float(coef_full[i]) for i in range(m)},
            "t_values": {rhs_names[i]: float(t_full[i]) for i in range(m)},
            "retained": [rhs_names[i] for i in active],
            "eliminated": eliminated,
        })

    return {
        "n_vars": int(n_eq),
        "n_lag": int(n_lag),
        "type": vtype,
        "threshold": float(threshold),
        "names": list(names),
        "rhs_names": list(rhs_names),
        "equations": equations,
        "restriction_matrix": restriction.tolist(),
        "n_restricted": int((restriction == 0).sum()),
    }


def var(
    panel: Any,
    *,
    target: str | None = None,
    n_lag: int = 1,
    type: str = "const",
    season: int | None = None,
    direct: bool = False,
    direct_horizon: int = 1,
) -> ModelFit:
    """Fit a vector autoregression on a multivariate panel.

    With ``direct=True``, fit the target equation as an h-step POINT projection
    on the same panel lag block the VAR uses at the forecast origin. This is the
    runner's ``forecast_policy="direct"`` target, not the horizon-average object
    required by ``forecast_policy="direct_average"``.
    """

    frame = as_frame(panel)
    estimator = _VAR(
        n_lag=n_lag,
        target=target,
        type=type,
        season=season,
        direct=bool(direct),
        direct_horizon=direct_horizon,
    )
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
            "direct": bool(direct),
            "direct_horizon": int(estimator.direct_horizon),
            "backend": "internal vars::VAR-aligned OLS",
            "implementation_note": (
                "R vars::VAR-aligned OLS design and predict.varest-style "
                "recursive point forecasts; direct=True fits a target-equation "
                "h-step point projection on the same lag block."
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
    #
    # s0=None (the default) resolves to a data-dependent diagonal prior scale
    # computed in `_favar_bvar_draws` (see `_favar_default_niw_scale`) rather
    # than an exactly-zero improper scale -- WP-V2 finding: the exactly-zero
    # default made the Wishart draw's scale matrix equal the bare (possibly
    # near-singular) sample residual cross-product with no floor, silently
    # diverging the Gibbs sampler on near-singular VAR residual covariance.
    # Passing s0 explicitly (including s0=0.0) is honored exactly as before,
    # now paired with a UserWarning if draws diverge (see
    # `_warn_if_bvar_draws_diverged`).

    def __init__(
        self,
        *,
        n_lag: int = 1,
        target: str | None = None,
        prior: str = "normal_inverse_wishart",
        b0: float = 0.0,
        vb0: float = 0.0,
        nu0: float = 0.0,
        s0: float | Sequence[Sequence[float]] | None = None,
        kappa0: float | None = None,
        kappa1: float | None = None,
        iter: int = 10000,
        burnin: int = 5000,
        random_state: int = 0,
        own_lag_prior_mean: float = 0.0,
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
        self.own_lag_prior_mean = float(own_lag_prior_mean)
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
            own_lag_prior_mean=self.own_lag_prior_mean,
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
    s0: float | Sequence[Sequence[float]] | None = None,
    iter: int = 300,
    burnin: int = 100,
    random_state: int = 0,
    own_lag_prior_mean: float = 0.0,
) -> ModelFit:
    """Fit a FAVAR::BVAR-style Bayesian VAR with Minnesota prior variances.

    ``own_lag_prior_mean`` sets the prior mean of each variable's own first lag
    (default 0.0; pass 1.0 for the classic Litterman random-walk prior). The panel
    is demeaned internally, so 0.0 shrinks toward white noise around the mean.

    ``iter=300``/``burnin=100`` are cheapened defaults (the Gibbs/Wishart draw
    loop cost grows sharply with panel width and ``n_lag``). The deep/
    paper-faithful defaults (``iter=10000``, ``burnin=5000``) remain reachable
    by passing them explicitly.

    ``s0`` (inverse-Wishart prior scale for the residual covariance) defaults
    to ``None``, which resolves to a data-dependent diagonal scale
    ``diag(sigma_1**2, ..., sigma_k**2)`` from each series' own univariate
    AR(``n_lag``)-OLS residual variance -- the standard natural-conjugate
    NIW-BVAR default (Karlsson 2013, Handbook of Economic Forecasting Vol.
    2B Ch. 15; Giannone, Lenza & Primiceri 2015, REStat 97(2)). Passing an
    explicit ``s0`` (including ``0.0``) is honored exactly as before; a
    ``UserWarning`` is now raised if the resulting posterior draws diverge
    (near-singular residual covariance with too-small a scale).
    """

    frame = as_frame(panel)
    estimator = _BayesianVAR(
        n_lag=n_lag,
        target=target,
        prior="minnesota",
        own_lag_prior_mean=own_lag_prior_mean,
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
    s0: float | Sequence[Sequence[float]] | None = None,
    iter: int = 300,
    burnin: int = 100,
    random_state: int = 0,
) -> ModelFit:
    """Fit a FAVAR::BVAR-style Bayesian VAR with normal-Wishart priors.

    ``iter=300``/``burnin=100`` are cheapened defaults (the Gibbs/Wishart draw
    loop cost grows sharply with panel width and ``n_lag``). The deep/
    paper-faithful defaults (``iter=10000``, ``burnin=5000``) remain reachable
    by passing them explicitly.

    ``s0`` (inverse-Wishart prior scale for the residual covariance) defaults
    to ``None``, which resolves to a data-dependent diagonal scale
    ``diag(sigma_1**2, ..., sigma_k**2)`` from each series' own univariate
    AR(``n_lag``)-OLS residual variance -- the standard natural-conjugate
    NIW-BVAR default (Karlsson 2013, Handbook of Economic Forecasting Vol.
    2B Ch. 15; Giannone, Lenza & Primiceri 2015, REStat 97(2)) -- instead of
    the previous exactly-zero ``s0=0.0``, which left the Gibbs sampler's
    Wishart-draw scale matrix equal to the bare (possibly near-singular)
    sample residual cross-product with no floor: on a near-singular VAR
    residual covariance (e.g. a well-explained FAVAR factor block) this
    silently diverged the posterior mean by orders of magnitude with no
    error, warning, or NaN. Passing an explicit ``s0`` (including ``0.0``)
    is honored exactly as before; a ``UserWarning`` is now raised if the
    resulting posterior draws diverge (see `_warn_if_bvar_draws_diverged`).
    """

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
    def __init__(self, *, n_factors: int = 3, n_lag: int = 1, random_state: int = 0,
                 direct: bool = False) -> None:
        self.n_factors = max(1, int(n_factors))
        self.ssr_: float | None = None
        self.nobs_: int | None = None
        self.n_params_: int | None = None
        self.n_lag = max(1, int(n_lag))
        self.random_state = int(random_state)
        # direct=True: factors from the (non-lag) predictor block + the target's own
        # observed ``*_lag0..*_lag{n_lag-1}`` columns (``*_lag0`` = the origin value,
        # observed, not look-ahead), regressed on the h-ahead target and predicted per
        # row. direct=False is the legacy roll-forward (recursive/path).
        self.direct = bool(direct)
        self._pca: Any = None
        self._regression: Any = None
        self._x_mean: pd.Series | None = None
        self._y_history: np.ndarray | None = None
        self._fallback: float = 0.0
        self._direct_pred_cols: list[str] | None = None
        self._direct_lag_cols: list[str] | None = None

    def _direct_design(self, Xdf: pd.DataFrame) -> np.ndarray:
        """Stack PCA factors of the predictor block with the explicit lag columns."""
        parts: list[np.ndarray] = []
        if self._direct_pred_cols and self._pca is not None and self._x_mean is not None:
            block = Xdf.reindex(columns=self._direct_pred_cols).astype(float)
            parts.append(self._pca.transform((block - self._x_mean).fillna(0.0)))
        if self._direct_lag_cols:
            parts.append(Xdf.reindex(columns=self._direct_lag_cols).astype(float).fillna(0.0).to_numpy())
        if not parts:
            return np.empty((len(Xdf), 0), dtype=float)
        return np.column_stack(parts)

    def _fit_direct(self, X: pd.DataFrame, y: pd.Series) -> "_FAR":
        from sklearn.linear_model import LinearRegression

        from macroforecast.feature_engineering.shared import _deterministic_pca

        Xdf = pd.DataFrame(X)
        target = pd.Series(y).astype(float)
        self._fallback = float(target.dropna().mean()) if not target.dropna().empty else 0.0
        self._direct_lag_cols = _select_lag_columns(Xdf, self.n_lag, target_name=getattr(target, "name", None))
        # Factors are built from the PREDICTOR block. Under the direct policy every
        # feature -- target lags AND predictor lags -- is named ``*_lag<k>``, so
        # excluding every ``*_lag*`` column (the previous behaviour) dropped the entire
        # lag-named predictor block and collapsed FAR to plain AR. Exclude only the
        # target's OWN lag columns (the autoregressive regressors, identified by the
        # base name of the selected lags); the predictor lags remain and drive the PCA.
        target_bases = {_LAG_COL_RE.sub("", c) for c in self._direct_lag_cols}
        lag_set = set(self._direct_lag_cols) | {
            c
            for c in map(str, Xdf.columns)
            if (m := _LAG_COL_RE.search(c)) is not None and c[: m.start()] in target_bases
        }
        self._direct_pred_cols = [c for c in map(str, Xdf.columns) if c not in lag_set]
        # Fit PCA on the predictor block over the rows where the target is observed.
        mask = target.reindex(Xdf.index).notna()
        if self._direct_pred_cols:
            block = Xdf.loc[mask, self._direct_pred_cols].astype(float)
            self._x_mean = block.mean(axis=0)
            n_factors = min(self.n_factors, block.shape[1], max(1, block.shape[0] - 1))
            centered = (block - self._x_mean).fillna(0.0)
            self._pca = _deterministic_pca(n_factors, *centered.shape, random_state=self.random_state)
            self._pca.fit(centered)
        design_df = pd.DataFrame(self._direct_design(Xdf), index=Xdf.index)
        joined = pd.concat([design_df, target.rename("__target__")], axis=1).dropna()
        if joined.empty or design_df.shape[1] == 0:
            self._direct_pred_cols = self._direct_lag_cols = None
            return self
        design = joined.drop(columns="__target__").to_numpy(dtype=float)
        response = joined["__target__"].to_numpy(dtype=float)
        self._regression = LinearRegression().fit(design, response)
        resid = response - self._regression.predict(design)
        self.ssr_ = float(resid @ resid)
        self.nobs_ = int(response.shape[0])
        self.n_params_ = int(np.size(self._regression.coef_) + 1)
        return self

    def _predict_direct(self, X: pd.DataFrame) -> np.ndarray:
        Xdf = pd.DataFrame(X)
        if self._regression is None:
            return np.full(len(Xdf), self._fallback, dtype=float)
        design = self._direct_design(Xdf)
        if design.shape[1] == 0:
            return np.full(len(Xdf), self._fallback, dtype=float)
        return np.asarray(self._regression.predict(design), dtype=float)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_FAR":
        if self.direct:
            return self._fit_direct(X, y)
        from sklearn.linear_model import LinearRegression

        joined = pd.concat([X, y.rename("__target__")], axis=1).dropna()
        if joined.empty:
            return self
        X_clean = joined.drop(columns="__target__")
        y_clean = joined["__target__"]
        self._fallback = float(y_clean.mean())
        self._x_mean = X_clean.mean(axis=0)
        n_factors = min(self.n_factors, X_clean.shape[1], max(1, X_clean.shape[0] - 1))
        from macroforecast.feature_engineering.shared import _deterministic_pca
        fit_block = (X_clean - self._x_mean).fillna(0.0)
        self._pca = _deterministic_pca(n_factors, *fit_block.shape, random_state=self.random_state)
        factors = self._pca.fit_transform(fit_block)
        values = y_clean.to_numpy(dtype=float)
        rows = []
        target = []
        for i in range(self.n_lag, len(values)):
            rows.append([*factors[i], *values[i - self.n_lag : i][::-1]])
            target.append(values[i])
        if not rows:
            self._y_history = values[-self.n_lag :]
            return self
        design = np.asarray(rows)
        response = np.asarray(target)
        self._regression = LinearRegression().fit(design, response)
        # In-sample one-step residuals for information-criterion order selection.
        resid = response - self._regression.predict(design)
        self.ssr_ = float(resid @ resid)
        self.nobs_ = int(response.shape[0])
        self.n_params_ = int(np.size(self._regression.coef_) + 1)
        self._y_history = values[-self.n_lag :]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.direct:
            return self._predict_direct(X)
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
    direct: bool = False,
) -> ModelFit:
    """Fit factor-augmented autoregression.

    ``direct=True`` makes it a DIRECT projection: factors of the non-lag predictor
    block plus the target's own observed ``*_lag0..*_lag{n_lag-1}`` columns
    (``*_lag0`` = the origin value, observed, not look-ahead) are regressed on the
    (h-ahead) target and predicted per row (no roll-forward). The runner sets
    ``direct=True`` only for the direct/direct_average policies.
    """

    return fit_estimator(
        _FAR(n_factors=n_factors, n_lag=n_lag, random_state=random_state, direct=bool(direct)),
        X,
        y,
        model="far",
        metadata={"n_factors": int(n_factors), "n_lag": int(n_lag),
                  "random_state": int(random_state), "direct": bool(direct)},
    )


def var_select_order(
    panel: Any,
    *,
    maxlags: int | None = None,
    trend: str = "c",
) -> dict[str, Any]:
    """Select the VAR lag order by information criteria (vars::VARselect).

    Fits VAR(p) for p up to ``maxlags`` and reports the lag order minimising each
    of AIC, BIC (Schwarz), HQ (Hannan-Quinn) and FPE, via statsmodels
    ``VAR.select_order``. ``trend`` is the deterministic term ('n','c','ct','ctt').
    Returns the selected order per criterion plus the criterion values per lag.
    """

    from statsmodels.tsa.api import VAR as _SMVAR

    frame = as_frame(panel)
    data = frame.select_dtypes("number") if hasattr(frame, "select_dtypes") else frame
    matrix = np.asarray(data, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 3:
        raise ValueError("var_select_order needs a 2-D panel with at least 3 rows")
    model = _SMVAR(matrix)
    selected = model.select_order(maxlags=maxlags, trend=trend)
    return {
        "selected_orders": {
            "aic": int(selected.aic),
            "bic": int(selected.bic),
            "hqic": int(selected.hqic),
            "fpe": int(selected.fpe),
        },
        "ics": {key: [float(v) for v in vals] for key, vals in selected.ics.items()},
        "trend": str(trend),
        "n_vars": int(matrix.shape[1]),
    }


def var_roots(fit: ModelFit) -> dict[str, Any]:
    """VAR stability: moduli of the companion-matrix eigenvalues (vars::roots).

    Builds the kp x kp companion matrix from the fitted lag coefficients and
    returns the eigenvalue moduli (descending), the spectral radius, and
    ``is_stable`` (all moduli < 1, i.e. the VAR is covariance-stationary).
    """

    estimator = getattr(fit, "estimator", fit)
    coef = getattr(estimator, "coef_", None)
    if coef is None:
        raise ValueError("var_roots requires a fitted var() model with coefficients")
    coef = np.asarray(coef, dtype=float)
    k = len(getattr(estimator, "names_", ()))
    p = int(getattr(estimator, "n_lag", 1))
    if k == 0 or coef.ndim != 2 or coef.shape[1] < k * p:
        raise ValueError("var_roots could not read the lag-coefficient block")
    lag_block = coef[:, : k * p]
    companion: np.ndarray = np.zeros((k * p, k * p), dtype=float)
    companion[:k, :] = lag_block
    if p > 1:
        companion[k:, : k * (p - 1)] = np.eye(k * (p - 1), dtype=float)
    moduli = sorted((float(m) for m in np.abs(np.linalg.eigvals(companion))), reverse=True)
    return {
        "moduli": moduli,
        "max_modulus": moduli[0] if moduli else float("nan"),
        "is_stable": bool(moduli[0] < 1.0) if moduli else False,
        "n_lag": p,
        "n_vars": k,
    }



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
        fctmethod: str = "BGM",
        slowcode: Sequence[bool] | None = None,
        factorprior: Mapping[str, Any] | None = None,
        varprior: Mapping[str, Any] | None = None,
        nburn: int = 100,
        nrep: int = 200,
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
    fctmethod: str = "BGM",
    slowcode: Sequence[bool] | None = None,
    factorprior: Mapping[str, Any] | None = None,
    varprior: Mapping[str, Any] | None = None,
    nburn: int = 100,
    nrep: int = 200,
    standardize: bool = True,
    random_state: int = 0,
) -> ModelFit:
    """Fit a FAVAR::FAVAR-aligned Bayesian factor-augmented VAR.

    Defaults to ``fctmethod=\"BGM\"``, an iterative factor-purging
    identification (``_favar_bgm``) that needs no ``slowcode``. Passing
    ``fctmethod=\"BBE\"`` selects the slow/fast ``facrot()`` rotation instead,
    which requires an explicit ``slowcode`` mask and raises otherwise.

    ``nburn=100``/``nrep=200`` are cheapened defaults so the model is usable
    out of the box (the Gibbs/Wishart posterior draw loop is expensive per
    iteration for larger ``n_factors``/``n_lag``). The deep/paper-faithful
    ``FAVAR::FAVAR`` defaults (``nburn=5000``, ``nrep=15000``) remain
    reachable by passing them explicitly.
    """

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
    if target_frequency is not None:
        normalized_target_frequency = _normalize_frequency_label(target_frequency)
        if normalized_target_frequency not in {"monthly", "quarterly", "annual"}:
            raise ValueError(
                "target_frequency must be one of ['monthly', 'quarterly', 'annual'] "
                f"(or None); got {target_frequency!r}. A frequency label that does not "
                "match a supported cadence would otherwise silently fall back to a "
                "monthly anchor projection, which is a common source of prediction-time "
                "missing-value failures rather than a clear fit-time error."
            )
    lag_values = _normalize_model_lags(lags)
    factor_lag_values = _normalize_model_lags(factor_lags)
    lag_column_values = tuple(str(column) for column in lag_columns or ())
    dfm_params: dict[str, Any] = {
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
    factors = _extend_dfm_factors(dfm_fit, factors, through_date=anchor_index.max())
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
        values: np.ndarray = np.full(len(frame), float(params[0]) if self.fit_intercept else 0.0, dtype=float)
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
    maxiter: int = 200,
    tolerance: float = 1e-6,
) -> ModelFit:
    """Fit a midasr::midas_r-style nonlinear restricted MIDAS regression.

    ``maxiter=200``/``tolerance=1e-6`` are cheapened defaults for the SciPy
    ``least_squares`` finite-difference-Jacobian solve. The deep/paper-faithful
    defaults (``maxiter=1000``, ``tolerance=1e-8``) remain reachable by passing
    them explicitly.
    """

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


def _ets_auto_select(series: pd.Series, seasonal_periods: int | None) -> dict[str, Any]:
    """Select the (error, trend, seasonal, damped) ETS spec minimising AICc.

    Mirrors ``forecast::ets(y, model="ZZZ")`` default behaviour: search the
    admissible state-space ETS family and keep the minimum-AICc model. Falls back
    to simple exponential smoothing if nothing fits.
    """
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel

    positive = bool(len(series) and (series.to_numpy(dtype=float) > 0).all())
    m = int(seasonal_periods) if seasonal_periods and int(seasonal_periods) > 1 else None
    best: tuple[float, dict[str, Any]] | None = None
    for error in ("add", "mul"):
        if error == "mul" and not positive:
            continue
        for trend in (None, "add"):
            for damped in ((False, True) if trend is not None else (False,)):
                seasonals = (None,) if m is None else (None, "add", "mul")
                for seasonal in seasonals:
                    if seasonal == "mul" and not positive:
                        continue
                    if seasonal is not None and (m is None or len(series) < 2 * m):
                        continue
                    try:
                        res = ETSModel(
                            series,
                            error=error,
                            trend=trend,
                            seasonal=seasonal,
                            seasonal_periods=m if seasonal else None,
                            damped_trend=damped,
                        ).fit(disp=False)
                        aicc = float(res.aicc)
                    except Exception:
                        continue
                    if best is None or aicc < best[0]:
                        best = (aicc, {
                            "error": error, "trend": trend, "seasonal": seasonal,
                            "seasonal_periods": m if seasonal else None,
                            "damped_trend": damped,
                        })
    if best is None:
        return {"error": "add", "trend": None, "seasonal": None,
                "seasonal_periods": None, "damped_trend": False}
    return best[1]


def ets(
    y: Any,
    *,
    error: str = "add",
    trend: str | None = None,
    seasonal: str | None = None,
    seasonal_periods: int | None = None,
    damped_trend: bool = False,
    model: str | None = None,
) -> ModelFit:
    """Fit statsmodels ETSModel for target-only forecasts.

    Pass ``model="auto"`` to select the (error, trend, seasonal, damped) spec by
    minimum AICc over the admissible ETS family (akin to ``forecast::ets``).
    Otherwise the explicit ``error``/``trend``/``seasonal``/``damped_trend`` spec
    is used (default: simple exponential smoothing).
    """

    target = as_series(y)
    if model is not None and str(model).lower() in {"auto", "zzz"}:
        params = _ets_auto_select(target, seasonal_periods)
        params["selection"] = "aicc_auto"
    else:
        params = {
            "error": error,
            "trend": trend,
            "seasonal": seasonal,
            "seasonal_periods": seasonal_periods,
            "damped_trend": bool(damped_trend),
        }
    estimator_params = {k: v for k, v in params.items() if k != "selection"}
    dummy = pd.DataFrame({"__origin__": np.arange(len(target), dtype=float)}, index=target.index)
    return fit_estimator(
        _StatsmodelsForecastEstimator(method="ets", params=estimator_params),
        dummy,
        target,
        model="ets",
        metadata=params,
    )


class _ArimaEstimator:
    """statsmodels ARIMA / SARIMA wrapper with a forecast contract."""

    def __init__(self, *, order: tuple[int, int, int],
                 seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
                 trend: str | None = None) -> None:
        self.order = tuple(int(v) for v in order)
        self.seasonal_order = tuple(int(v) for v in seasonal_order)
        self.trend = trend
        self.result_: Any = None
        self.fallback_: float = 0.0
        self.fit_error_: str | None = None
        self.aic_: float | None = None
        self.aicc_: float | None = None
        self.bic_: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_ArimaEstimator":
        from statsmodels.tsa.arima.model import ARIMA

        series = pd.Series(y).astype(float).dropna()
        self.fallback_ = float(series.iloc[-1]) if len(series) else 0.0
        try:
            res = ARIMA(
                series, order=self.order, seasonal_order=self.seasonal_order,
                trend=self.trend,
            ).fit()
            self.result_ = res
            self.aic_ = float(res.aic)
            self.aicc_ = float(res.aicc)
            self.bic_ = float(res.bic)
        except Exception as exc:  # noqa: BLE001 - keep callable on ill-posed orders
            self.result_ = None
            self.fit_error_ = str(exc)
            warnings.warn(
                f"ARIMA{self.order}x{self.seasonal_order} fit failed ({exc}); "
                "falling back to last-value persistence.",
                UserWarning, stacklevel=2,
            )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        steps = len(X)
        if steps <= 0:
            return np.empty(0, dtype=float)
        if self.result_ is None:
            return np.full(steps, self.fallback_, dtype=float)
        try:
            return np.asarray(self.result_.forecast(steps), dtype=float).reshape(-1)[:steps]
        except Exception:
            return np.full(steps, self.fallback_, dtype=float)


def arima(
    y: Any,
    *,
    order: tuple[int, int, int] = (1, 0, 0),
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
    trend: str | None = None,
) -> ModelFit:
    """Fit a (seasonal) ARIMA model via statsmodels for target-only forecasts.

    ``order`` is ``(p, d, q)`` and ``seasonal_order`` is ``(P, D, Q, m)``. See
    :func:`auto_arima` for automatic order selection (forecast::auto.arima).
    """

    target = as_series(y)
    dummy = pd.DataFrame({"__origin__": np.arange(len(target), dtype=float)}, index=target.index)
    params = {"order": tuple(int(v) for v in order),
              "seasonal_order": tuple(int(v) for v in seasonal_order), "trend": trend}
    return fit_estimator(
        _ArimaEstimator(order=order, seasonal_order=seasonal_order, trend=trend),
        dummy, target, model="arima", metadata=params,
    )


def _ndiffs_kpss(series: pd.Series, *, max_d: int = 2, alpha: float = 0.05) -> int:
    """Number of first differences to reach stationarity (KPSS-based; forecast::ndiffs)."""
    from statsmodels.tsa.stattools import kpss

    current = pd.Series(series).dropna().astype(float)
    d = 0
    while d < max_d and len(current) >= 10:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, pval, *_ = kpss(current.values, regression="c", nlags="auto")
        except Exception:
            break
        if pval >= alpha:  # KPSS null is stationarity -> fail to reject -> stop
            break
        current = current.diff().dropna()
        d += 1
    return d


def auto_arima(
    y: Any,
    *,
    max_p: int = 5,
    max_q: int = 5,
    max_d: int = 2,
    seasonal: bool = False,
    m: int = 1,
    max_P: int = 1,
    max_Q: int = 1,
    seasonal_D: int = 0,
    ic: str = "aicc",
    trend: str | None = None,
) -> ModelFit:
    """Automatic (seasonal) ARIMA order selection (forecast::auto.arima).

    The non-seasonal differencing order ``d`` is chosen by repeated KPSS tests
    (Hyndman-Khandakar), then ``(p, q)`` (and seasonal ``(P, Q)`` when
    ``seasonal`` and ``m>1``) are selected over a grid by minimum ``ic`` ('aicc',
    'aic' or 'bic'). A constant is tried when ``d==0``. Returns the fitted best
    model (an :func:`arima` fit).
    """

    from statsmodels.tsa.arima.model import ARIMA

    if ic not in {"aicc", "aic", "bic"}:
        raise ValueError("ic must be 'aicc', 'aic', or 'bic'")
    target = as_series(y)
    series = pd.Series(target).astype(float).dropna()
    d = _ndiffs_kpss(series, max_d=max_d)
    seas_orders = [(0, 0, 0, 0)]
    if seasonal and m > 1:
        seas_orders = [
            (P, int(seasonal_D), Q, int(m))
            for P in range(max_P + 1) for Q in range(max_Q + 1)
        ]
    trends: list[str | None] = [trend]
    if trend is None and d == 0:
        trends = ["c", None]
    best: tuple[float, tuple[int, int, int], tuple[int, int, int, int], str | None] | None = None
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            for so in seas_orders:
                for tr in trends:
                    if p == 0 and q == 0 and so[0] == 0 and so[2] == 0 and tr is None:
                        continue
                    try:
                        res = ARIMA(series, order=(p, d, q), seasonal_order=so, trend=tr).fit()
                        crit = float(getattr(res, ic))
                    except Exception:
                        continue
                    if not np.isfinite(crit):
                        continue
                    if best is None or crit < best[0]:
                        best = (crit, (p, d, q), so, tr)
    if best is None:
        return arima(target, order=(0, d, 0))
    _, order, so, tr = best
    fit = arima(target, order=order, seasonal_order=so, trend=tr)
    fit.metadata["selection"] = {"ic": ic, "value": float(best[0]), "d_kpss": int(d)}
    return fit


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


def _vars_direct_rhs(
    values: np.ndarray,
    names: tuple[str, ...],
    n_lag: int,
    horizon: int,
    *,
    type: str,
    season: int | None,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Direct VAR projection design.

    Row t uses the origin-dated lag block ``Y_t, ..., Y_{t-p+1}`` and response
    ``Y_{t+h}``. For ``h=1`` this is exactly the target equation row in
    ``_vars_rhs``.
    """

    n_obs, _n_vars = values.shape
    first_origin = int(n_lag) - 1
    last_origin = int(n_obs) - int(horizon) - 1
    if last_origin < first_origin:
        empty_response = np.empty((0, values.shape[1]), dtype=float)
        return empty_response, np.empty((0, 0), dtype=float), [], np.empty(0, dtype=int)
    origin_positions: np.ndarray = np.arange(first_origin, last_origin + 1, dtype=int)
    target_positions: np.ndarray = origin_positions + int(horizon)
    rows: list[list[float]] = []
    lag_names: list[str] = []
    for lag in range(1, n_lag + 1):
        lag_names.extend([f"{name}.l{lag}" for name in names])
    for origin_pos in origin_positions:
        row: list[float] = []
        for lag in range(1, n_lag + 1):
            row.extend(values[origin_pos - lag + 1, :].tolist())
        rows.append(row)
    rhs = np.asarray(rows, dtype=float)
    rhs_names = list(lag_names)
    sample = len(origin_positions)
    if type in {"const", "both"}:
        rhs = np.column_stack([rhs, np.ones(sample, dtype=float)])
        rhs_names.append("const")
    if type in {"trend", "both"}:
        rhs = np.column_stack([rhs, target_positions.astype(float) + 1.0])
        rhs_names.append("trend")
    if season is not None:
        seasonal = _vars_seasonal_dummies(n_obs, season)[target_positions, :]
        rhs = np.column_stack([rhs, seasonal])
        rhs_names.extend([f"sd{i}" for i in range(1, season)])
    return values[target_positions, :], rhs, rhs_names, origin_positions


def _vars_direct_forecast_row(
    values: np.ndarray,
    names: tuple[str, ...],
    n_lag: int,
    horizon: int,
    *,
    type: str,
    season: int | None,
) -> np.ndarray:
    """One direct VAR forecast row at the sample end."""

    n_obs, _n_vars = values.shape
    row: list[float] = []
    for lag in range(1, n_lag + 1):
        row.extend(values[n_obs - lag, :].tolist())
    if type in {"const", "both"}:
        row.append(1.0)
    if type in {"trend", "both"}:
        row.append(float(n_obs + int(horizon)))
    if season is not None:
        seasonal = _vars_seasonal_dummies(n_obs + int(horizon), season)
        row.extend(seasonal[n_obs + int(horizon) - 1].tolist())
    return np.asarray(row, dtype=float)


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
    own_lag_prior_mean: float = 0.0,
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
        a_mu_prior, a_v_i_prior = _favar_minnesota_prior(
            values, n_lag, kappa0, kappa1, own_lag_prior_mean=own_lag_prior_mean
        )
    else:
        a_mu_prior = np.full(n_coef, float(b0), dtype=float)
        a_v_i_prior = np.eye(n_coef, dtype=float) * float(vb0)
    if s0 is None:
        # Data-dependent default (WP-V2 fix) -- see
        # `_favar_default_niw_scale`'s docstring. Replaces the previous
        # exactly-zero s0=0.0 default, which left `scale_post` below equal to
        # the bare (possibly near-singular) sample residual cross-product
        # with no floor.
        s0_matrix = _favar_default_niw_scale(values, n_lag)
    else:
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
    _warn_if_bvar_draws_diverged(coef_arr, design, response)
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
    *,
    own_lag_prior_mean: float = 0.0,
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
    # Own first-lag prior mean (Litterman uses 1.0 for a random-walk prior; the
    # default 0.0 shrinks toward white noise, appropriate for demeaned/stationary
    # or standardized FAVAR inputs). Position of own lag-1 coef of equation eq in
    # the order="F" flattening of the (k, n_reg) coefficient matrix is eq*k + eq.
    if own_lag_prior_mean != 0.0:
        for eq in range(k):
            prior_mean[eq * k + eq] = float(own_lag_prior_mean)
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


def _favar_default_niw_scale(values: np.ndarray, n_lag: int) -> np.ndarray:
    """Data-dependent default inverse-Wishart prior scale S0 (WP-V2 fix).

    Standard natural-conjugate NIW-BVAR convention (Karlsson 2013, Handbook
    of Economic Forecasting Vol. 2B Ch. 15 -- already this repo's own
    reference for the sibling Minnesota-prior sigma^2 scaling fix, see
    CHANGELOG "PR8"; Giannone, Lenza & Primiceri 2015, "Prior Selection for
    Vector Autoregressions", Review of Economics and Statistics 97(2)).

    Under this module's own inverse-Wishart parameterization -- the Gibbs
    loop in `_favar_bvar_draws` draws ``Sigma | data`` with scale
    ``S0 + SSE`` and degrees of freedom ``tnum + nu0``, so
    ``E[Sigma] = scale / (df - k - 1)`` for ``df > k + 1`` -- the minimal
    proper (integer) prior degrees of freedom keeping a no-data prior mean
    finite is ``nu0_min = k + 2`` (``df - k - 1 = 1`` at that value). Setting
    ``S0 = diag(sigma_1**2, ..., sigma_k**2)`` then makes the prior's implied
    residual-covariance mean equal each equation's own univariate
    AR(n_lag)-OLS residual variance -- the standard empirical-Bayes
    Minnesota-style scale -- instead of the previous literal-zero default,
    which left ``S0 + SSE`` equal to the bare (possibly near-singular) sample
    residual cross-product with no floor: on a near-singular VAR residual
    covariance (e.g. a well-explained FAVAR factor block) that silently
    diverged the Gibbs sampler's posterior mean by orders of magnitude.

    ``sigma_i`` reuses `_favar_univariate_ar_sigma`, the same per-equation
    AR-residual-SD helper already used (and independently anchored, see
    ``test_minnesota_prior_variance_matches_documented_formula``) for the
    Minnesota prior's cross-lag variance scaling.
    """
    sigma = _favar_univariate_ar_sigma(values, n_lag)
    return np.diag(sigma**2)


def _prior_scale_matrix(value: float | Sequence[Sequence[float]] | None, k: int) -> np.ndarray:
    if value is None:
        # Callers resolve s0=None to `_favar_default_niw_scale(...)` upstream
        # (see `_favar_bvar_draws`); this function only ever sees an already-
        # resolved scale. Reaching here with None means a caller forgot that
        # resolution step.
        raise ValueError("s0 must not be None for FAVAR::BVAR-compatible fitting")
    if np.isscalar(value):
        # A scalar s0 denotes an isotropic inverse-Wishart scale s0*I. The old
        # np.full((k,k), s0) produced a rank-1 (singular) matrix for k>1.
        # np.isscalar narrowed value to a scalar but mypy cannot follow it.
        return float(cast(Any, value)) * np.eye(k, dtype=float)
    arr = np.asarray(value, dtype=float)
    if arr.shape != (k, k):
        raise ValueError(f"s0 must be scalar or a {k}x{k} matrix")
    return arr


def _warn_if_bvar_draws_diverged(
    coef_arr: np.ndarray,
    design: np.ndarray,
    response: np.ndarray,
    *,
    explosion_multiplier: float = 50.0,
    divergent_fraction_threshold: float = 0.10,
) -> None:
    """Warn (never raise) if the collected Gibbs coefficient draws exploded.

    Divergence guard for the WP-V2 finding: a (near-)zero inverse-Wishart
    prior scale ``s0`` interacting with a near-singular VAR residual
    covariance can make the Gibbs sampler's coefficient draws numerically
    diverge -- silently, with no error or NaN (posterior means observed off
    by 3-4+ orders of magnitude, ~97% of draws divergent). Detected here by
    comparing each draw's coefficient-matrix Frobenius norm against a large
    multiple of the plain OLS estimate's own norm on the same shared VAR
    design -- the same closed-form reference the WP-V2 anchor tests use
    (equation-by-equation OLS is the GLS/Bayesian-posterior-mean limit for a
    flat coefficient prior, for any positive-definite Sigma). If a large
    fraction of post-burn-in draws are many times larger than that reference,
    the chain has not converged to a sane posterior and the caller almost
    certainly needs a larger/explicit ``s0`` or rescaled data.
    """
    ols_coef, *_ = np.linalg.lstsq(design, response, rcond=None)
    ols_scale = float(np.linalg.norm(ols_coef))
    # Floor the reference scale so a (near-)zero OLS fit doesn't make an
    # arbitrarily small absolute coefficient norm look "divergent".
    reference = max(ols_scale, 1.0)
    draw_norms = np.linalg.norm(coef_arr.reshape(coef_arr.shape[0], -1), axis=1)
    if draw_norms.size == 0:
        return
    divergent = draw_norms > explosion_multiplier * reference
    frac_divergent = float(np.mean(divergent))
    if frac_divergent > divergent_fraction_threshold:
        warnings.warn(
            "bvar posterior draws diverged: "
            f"{frac_divergent:.0%} of post-burn-in Gibbs draws have a "
            f"coefficient-matrix norm > {explosion_multiplier:.0f}x the "
            "OLS-based reference scale. Likely cause: near-singular residual "
            "covariance with (near-)zero prior scale s0 -- posterior draws "
            "diverged; set s0>0 or rescale data.",
            UserWarning,
            stacklevel=3,
        )


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
        # np.isscalar narrowed value to a scalar but mypy cannot follow it.
        return None if value is None else float(cast(Any, value))
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
        # np.isscalar narrowed b0_arg to a scalar but mypy cannot follow it.
        precision = float(cast(Any, b0_arg)) * np.eye(fy.shape[1], dtype=float)
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
    # "s0" defaults to None (not 0.0): `_favar_bvar_draws` resolves None to
    # the data-dependent default scale (WP-V2 fix, `_favar_default_niw_scale`)
    # so that favar()'s own default `varprior=None` -- which reaches here as
    # the empty dict `{}` -- transparently inherits the same fix as
    # `bvar_normal_inverse_wishart`'s default, instead of silently forcing
    # the old exactly-zero scale. Passing an explicit `s0` (including `0.0`)
    # in `varprior` is honored exactly as before.
    mn = prior.get("mn", {}) if prior else {}
    if isinstance(mn, Mapping) and mn.get("kappa0") is not None:
        return {
            "prior": "minnesota",
            "b0": 0.0,
            "vb0": 0.0,
            "nu0": float(prior.get("nu0", 0.0)),
            "s0": prior.get("s0"),
            "kappa0": float(cast(Any, mn.get("kappa0"))),
            "kappa1": float(mn.get("kappa1", 0.5)),
        }
    return {
        "prior": "normal_inverse_wishart",
        "b0": float(prior.get("b0", 0.0)) if prior else 0.0,
        "vb0": float(prior.get("vb0", 0.0)) if prior else 0.0,
        "nu0": float(prior.get("nu0", 0.0)) if prior else 0.0,
        "s0": prior.get("s0") if prior else None,
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


def _extend_dfm_factors(
    dfm_fit: Any,
    factors: pd.DataFrame,
    *,
    through_date: Any,
) -> pd.DataFrame:
    """Extend fitted DFM filtered factors forward to cover ``through_date``.

    Prediction-time anchor dates can fall past the last date the DFM factors
    were filtered through (e.g. a "period_end" quarterly anchor projects a
    monthly forecast date onto a later quarter-end month than any observed
    predictor data reaches). Previously this left ``dfm_factor*_lag0`` as NaN
    (a plain ``reindex``), which made ``predict_from_panel`` reject the design
    outright. Instead, extend the already-fitted DynamicFactorMQ state-space
    result with placeholder (all-missing) observations for the gap months and
    read off its own Kalman-filter forecast of the factors -- i.e. propagate
    the fitted factor-VAR state forward rather than manufacturing missing
    values. Falls back to returning ``factors`` unchanged (preserving the
    prior reindex-to-NaN behavior, and therefore the existing missing-value
    error) if the underlying fit is unavailable or the extension itself fails.
    """
    if not isinstance(factors, pd.DataFrame) or factors.empty:
        return factors
    estimator = getattr(dfm_fit, "estimator", None)
    result = getattr(estimator, "result_", None)
    if result is None:
        return factors
    last_date = pd.Timestamp(factors.index.max()).to_period("M").to_timestamp()
    target_date = pd.Timestamp(through_date).to_period("M").to_timestamp()
    if target_date <= last_date:
        return factors
    ext_index = pd.date_range(last_date + pd.DateOffset(months=1), target_date, freq="MS")
    if ext_index.empty:
        return factors
    monthly_columns = list(getattr(estimator, "monthly_columns_", ()))
    quarterly_columns = list(getattr(estimator, "quarterly_columns_", ()))
    if not monthly_columns:
        return factors
    monthly_ext = pd.DataFrame(np.nan, index=ext_index, columns=monthly_columns)
    try:
        if quarterly_columns:
            quarterly_ext = pd.DataFrame(
                np.nan,
                index=pd.period_range(ext_index[0], ext_index[-1], freq="Q"),
                columns=quarterly_columns,
            )
            extended = result.extend(monthly_ext, endog_quarterly=quarterly_ext)
        else:
            extended = result.extend(monthly_ext)
        extended_bunch = extended.factors
        extended_factors = extended_bunch.filtered if extended_bunch is not None else None
    except Exception:  # noqa: BLE001 - fall back to the prior reindex-to-NaN contract
        return factors
    if not isinstance(extended_factors, pd.DataFrame) or extended_factors.empty:
        return factors
    extended_factors = extended_factors.reindex(columns=factors.columns)
    combined = pd.concat([factors, extended_factors])
    return combined[~combined.index.duplicated(keep="last")].sort_index()


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
    params: Sequence[float] | np.ndarray,
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
        positions: np.ndarray = np.arange(1, n + 1, dtype=float)
        raw: np.ndarray = np.zeros(n, dtype=float)
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
        positions: np.ndarray = np.arange(1, n + 1, dtype=float)
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
            heights: np.ndarray = np.ones(len(buckets), dtype=float)
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
    "var_restrict",
    "hist_mean",
    "stlf",
    "random_walk_drift",
    "ar",
    "ar_bic",
    "arima",
    "auto_arima",
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
    "var_select_order",
    "var_roots",
]
