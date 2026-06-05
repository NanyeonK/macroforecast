from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from macroforecast.data import (
    DataBundle,
    DataSpec,
    as_panel,
    attach_metadata,
    panel_info,
    validate_panel,
)


SummaryMetric = Literal["mean", "sd", "min", "max", "skew", "kurtosis", "n_obs", "n_missing"]
CorrelationMethod = Literal["pearson", "spearman", "kendall"]
OutlierMethod = Literal["iqr", "zscore", "multi", "both"]
StationarityTest = Literal["adf", "pp", "kpss", "multi", "none"]
StationarityScope = Literal["all", "target_and_predictors", "target_only", "predictors_only"]

DEFAULT_SUMMARY_METRICS: tuple[SummaryMetric, ...] = (
    "mean",
    "sd",
    "min",
    "max",
    "n_obs",
    "n_missing",
)


@dataclass(frozen=True)
class DataSummaryReport:
    """Container returned by :func:`summarize_data`."""

    overview: dict[str, Any]
    coverage: pd.DataFrame
    univariate: pd.DataFrame
    missing: pd.DataFrame
    correlation: pd.DataFrame | None = None
    outliers: pd.DataFrame | None = None
    stationarity: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "overview": self.overview,
            "coverage": self.coverage.to_dict(orient="index"),
            "univariate": self.univariate.to_dict(orient="index"),
            "missing": self.missing.to_dict(orient="index"),
            "metadata": dict(self.metadata),
        }
        if self.correlation is not None:
            out["correlation"] = self.correlation.to_dict()
        if self.outliers is not None:
            out["outliers"] = self.outliers.to_dict(orient="index")
        if self.stationarity is not None:
            out["stationarity"] = self.stationarity
        return out


def panel_overview(data: Any) -> dict[str, Any]:
    """Return panel-level shape, date range, frequency, and missingness."""

    panel, metadata = _coerce_panel(data)
    info = panel_info(DataBundle(panel, metadata))
    info["metadata_keys"] = sorted(str(key) for key in metadata)
    return info


def panel_snapshot(data: Any) -> dict[str, Any]:
    """Return a compact single-panel snapshot for reports and provenance."""

    panel, metadata = _coerce_panel(data)
    return _compact_panel_info(DataBundle(panel, metadata))


def sample_coverage(data: Any) -> pd.DataFrame:
    """Return per-series sample start, end, observation count, and missingness."""

    panel, _metadata = _coerce_panel(data)
    rows: list[dict[str, Any]] = []
    n_rows = int(panel.shape[0])
    for column in panel.columns:
        series = panel[column]
        observed = series.dropna()
        n_obs = int(observed.shape[0])
        n_missing = int(series.isna().sum())
        rows.append(
            {
                "column": str(column),
                "first_valid": _date_string(observed.index.min()) if n_obs else None,
                "last_valid": _date_string(observed.index.max()) if n_obs else None,
                "n_obs": n_obs,
                "n_missing": n_missing,
                "missing_rate": _safe_ratio(n_missing, n_rows),
            }
        )
    return pd.DataFrame(rows).set_index("column") if rows else pd.DataFrame()


def observation_counts(data: Any) -> pd.Series:
    """Return per-series non-missing observation counts."""

    coverage = sample_coverage(data)
    if coverage.empty:
        return pd.Series(dtype="int64", name="n_obs")
    return coverage["n_obs"].rename("n_obs")


def missing_rates(data: Any) -> pd.Series:
    """Return per-series missing rates."""

    coverage = sample_coverage(data)
    if coverage.empty:
        return pd.Series(dtype="float64", name="missing_rate")
    return coverage["missing_rate"].rename("missing_rate")


def univariate_summary(
    data: Any,
    *,
    metrics: Sequence[SummaryMetric] | None = None,
) -> pd.DataFrame:
    """Return per-series descriptive statistics for numeric panel columns."""

    panel, _metadata = _coerce_panel(data)
    selected = tuple(metrics or DEFAULT_SUMMARY_METRICS)
    _validate_summary_metrics(selected)
    rows: list[dict[str, Any]] = []
    for column in panel.select_dtypes("number").columns:
        series = panel[column].dropna()
        row: dict[str, Any] = {"column": str(column)}
        if "mean" in selected:
            row["mean"] = _float_or_none(series.mean())
        if "sd" in selected:
            row["sd"] = _float_or_none(series.std())
        if "min" in selected:
            row["min"] = _float_or_none(series.min())
        if "max" in selected:
            row["max"] = _float_or_none(series.max())
        if "skew" in selected:
            row["skew"] = _float_or_none(series.skew())
        if "kurtosis" in selected:
            row["kurtosis"] = _float_or_none(series.kurtosis())
        if "n_obs" in selected:
            row["n_obs"] = int(series.shape[0])
        if "n_missing" in selected:
            row["n_missing"] = int(panel[column].isna().sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index("column") if rows else pd.DataFrame()


def missing_summary(data: Any) -> pd.DataFrame:
    """Return per-series missing-count, missing-rate, and longest-gap summary."""

    panel, _metadata = _coerce_panel(data)
    rows: list[dict[str, Any]] = []
    n_rows = int(panel.shape[0])
    for column in panel.columns:
        missing = panel[column].isna()
        n_missing = int(missing.sum())
        rows.append(
            {
                "column": str(column),
                "n_missing": n_missing,
                "missing_rate": _safe_ratio(n_missing, n_rows),
                "longest_missing_run": _longest_true_run(missing),
            }
        )
    return pd.DataFrame(rows).set_index("column") if rows else pd.DataFrame()


def correlation_matrix(
    data: Any,
    *,
    method: CorrelationMethod = "pearson",
    min_periods: int = 1,
) -> pd.DataFrame:
    """Return a numeric correlation matrix for one panel."""

    if method not in {"pearson", "spearman", "kendall"}:
        raise ValueError("method must be one of 'pearson', 'spearman', or 'kendall'")
    if int(min_periods) < 1:
        raise ValueError("min_periods must be a positive integer")
    panel, _metadata = _coerce_panel(data)
    return panel.select_dtypes("number").corr(method=method, min_periods=int(min_periods))


def outlier_summary(
    data: Any,
    *,
    method: OutlierMethod = "iqr",
    iqr_threshold: float = 10.0,
    zscore_threshold: float = 3.0,
) -> pd.DataFrame:
    """Return per-series outlier counts and rates for one panel."""

    if method not in {"iqr", "zscore", "multi", "both"}:
        raise ValueError("method must be one of 'iqr', 'zscore', 'multi', or 'both'")
    _validate_positive(iqr_threshold, "iqr_threshold")
    _validate_positive(zscore_threshold, "zscore_threshold")
    panel, _metadata = _coerce_panel(data)
    numeric = panel.select_dtypes("number")
    rows: list[dict[str, Any]] = []
    include_iqr = method in {"iqr", "multi", "both"}
    include_zscore = method in {"zscore", "multi", "both"}
    if include_iqr:
        median = numeric.median()
        iqr = (numeric.quantile(0.75) - numeric.quantile(0.25)).replace(0, pd.NA)
        iqr_mask = (numeric - median).abs() > float(iqr_threshold) * iqr
    else:
        iqr_mask = pd.DataFrame(False, index=numeric.index, columns=numeric.columns)
    if include_zscore:
        # Match preprocessing.zscore_outlier_clean: population standard
        # deviation with ddof=0. data_analysis should describe the same outlier
        # rule that preprocessing would apply, not a sample-std variant.
        sd = numeric.std(ddof=0).replace(0, pd.NA)
        zscore_mask = ((numeric - numeric.mean()).abs() / sd) > float(zscore_threshold)
    else:
        zscore_mask = pd.DataFrame(False, index=numeric.index, columns=numeric.columns)
    for column in numeric.columns:
        observed_n = int(numeric[column].notna().sum())
        row: dict[str, Any] = {"column": str(column), "n_obs": observed_n}
        if include_iqr:
            count = int(iqr_mask[column].fillna(False).sum())
            row["iqr_outlier_count"] = count
            row["iqr_outlier_rate"] = _safe_ratio(count, observed_n)
        if include_zscore:
            count = int(zscore_mask[column].fillna(False).sum())
            row["zscore_outlier_count"] = count
            row["zscore_outlier_rate"] = _safe_ratio(count, observed_n)
        rows.append(row)
    return pd.DataFrame(rows).set_index("column") if rows else pd.DataFrame()


def stationarity_tests(
    data: Any,
    *,
    test: StationarityTest = "multi",
    scope: StationarityScope = "all",
    target: str | None = None,
    targets: Sequence[str] | None = None,
    alpha: float = 0.05,
    adf_regression: str = "c",
) -> dict[str, Any]:
    """Run ADF, Phillips-Perron, KPSS, or all three on one panel.

    ``adf_regression`` is the ADF deterministic specification passed to
    statsmodels ``adfuller`` ('n', 'c', 'ct', 'ctt'); the default 'c' (constant
    only) follows statsmodels. Note ``tseries::adf.test`` instead defaults to 'ct'
    (constant + linear trend) with a fixed lag, so pass ``adf_regression='ct'`` to
    reproduce that reference.
    """

    if test not in {"adf", "pp", "kpss", "multi", "none"}:
        raise ValueError("test must be one of 'adf', 'pp', 'kpss', 'multi', or 'none'")
    if adf_regression not in {"n", "c", "ct", "ctt"}:
        raise ValueError("adf_regression must be one of 'n', 'c', 'ct', 'ctt'")
    if scope not in {"all", "target_and_predictors", "target_only", "predictors_only"}:
        raise ValueError("scope must be one of 'all', 'target_and_predictors', 'target_only', or 'predictors_only'")
    _validate_alpha(alpha)
    panel, _metadata = _coerce_panel(data)
    columns = _scope_columns(panel, data, scope=scope, target=target, targets=targets)
    if test == "none":
        return {
            "test": "none",
            "scope": scope,
            "alpha": float(alpha),
            "n_series": 0,
            "by_series": {},
        }
    selected_tests = ("adf", "pp", "kpss") if test == "multi" else (test,)
    results: dict[str, dict[str, Any]] = {}
    for column in columns:
        series = pd.to_numeric(panel[column], errors="coerce").dropna()
        name = str(column)
        if series.size < 12 or series.std(ddof=0) == 0:
            results[name] = {"status": "insufficient_data", "n_obs": int(series.size)}
            continue
        column_result: dict[str, Any] = {"n_obs": int(series.size)}
        for selected in selected_tests:
            try:
                column_result[selected] = _run_stationarity(
                    selected, series, float(alpha), adf_regression=adf_regression
                )
            except Exception as exc:  # pragma: no cover - defensive
                column_result[selected] = {"status": "error", "error": str(exc)}
        results[name] = column_result
    return {
        "test": test,
        "scope": scope,
        "alpha": float(alpha),
        "n_series": len(columns),
        "by_series": results,
    }


def adf_test(
    series: Any,
    *,
    regression: str = "c",
    autolag: str | None = "AIC",
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Augmented Dickey-Fuller unit-root test for a single series.

    ``regression`` is the deterministic spec ('n','c','ct','ctt'); default 'c'
    follows statsmodels (``tseries::adf.test`` defaults to 'ct'). Returns a flat
    result dict (the multi-series entry point is ``stationarity_tests``).
    """

    from statsmodels.tsa.stattools import adfuller

    if regression not in {"n", "c", "ct", "ctt"}:
        raise ValueError("regression must be one of 'n', 'c', 'ct', 'ctt'")
    values = pd.Series(series).dropna().astype(float).to_numpy()
    stat, pvalue, used_lag, nobs, *_ = adfuller(values, regression=regression, autolag=autolag)
    return {
        "test": "adf",
        "statistic": float(stat),
        "p_value": float(pvalue),
        "used_lag": int(used_lag),
        "n_obs": int(nobs),
        "regression": regression,
        "reject_unit_root": bool(pvalue < alpha),
    }


def kpss_test(
    series: Any,
    *,
    regression: str = "c",
    nlags: Any = "auto",
    alpha: float = 0.05,
) -> dict[str, Any]:
    """KPSS stationarity test for a single series.

    ``regression='c'`` tests level stationarity (the ``tseries::kpss.test``
    'Level' default); ``'ct'`` tests trend stationarity. Returns a flat result
    dict (the multi-series entry point is ``stationarity_tests``).
    """

    import warnings

    from statsmodels.tsa.stattools import kpss as _kpss

    if regression not in {"c", "ct"}:
        raise ValueError("regression must be 'c' (level) or 'ct' (trend)")
    values = pd.Series(series).dropna().astype(float).to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, pvalue, n_lags, _crit = _kpss(values, regression=regression, nlags=nlags)
    return {
        "test": "kpss",
        "statistic": float(stat),
        "p_value": float(pvalue),
        "n_lags": int(n_lags),
        "regression": regression,
        "reject_stationarity": bool(pvalue < alpha),
    }


def johansen_cointegration(
    panel: Any,
    *,
    det_order: int = 0,
    k_ar_diff: int = 1,
    significance: str = "95",
) -> dict[str, Any]:
    """Johansen cointegration test (R urca::ca.jo / statsmodels coint_johansen).

    Tests the cointegration rank of a multivariate system via the trace and
    maximum-eigenvalue statistics. ``det_order`` is the deterministic term
    (-1 none, 0 constant, 1 linear trend); ``k_ar_diff`` the number of lagged
    differences in the VECM. Returns, for each null rank r, the trace and
    max-eigenvalue statistics with their 90/95/99% critical values, the selected
    cointegration rank under each statistic (sequential test at ``significance``),
    the eigenvalues, and the estimated cointegrating vectors.
    """

    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    frame = pd.DataFrame(panel)
    frame = frame.select_dtypes("number") if hasattr(frame, "select_dtypes") else frame
    names = [str(c) for c in frame.columns]
    k = len(names)
    res = coint_johansen(np.asarray(frame, dtype=float), int(det_order), int(k_ar_diff))
    col = {"90": 0, "95": 1, "99": 2}.get(str(significance), 1)

    def _rank(stats: np.ndarray, crit: np.ndarray) -> int:
        rank = 0
        for r in range(len(stats)):
            if stats[r] > crit[r, col]:
                rank = r + 1
            else:
                break
        return rank

    def _rows(stats: np.ndarray, crit: np.ndarray) -> list[dict[str, Any]]:
        return [
            {"rank_null": int(r), "statistic": float(stats[r]),
             "crit_90": float(crit[r, 0]), "crit_95": float(crit[r, 1]), "crit_99": float(crit[r, 2]),
             "reject": bool(stats[r] > crit[r, col])}
            for r in range(len(stats))
        ]

    rank_trace = _rank(res.lr1, res.cvt)
    return {
        "n_vars": int(k),
        "names": names,
        "det_order": int(det_order),
        "k_ar_diff": int(k_ar_diff),
        "significance": str(significance),
        "trace": _rows(res.lr1, res.cvt),
        "max_eigen": _rows(res.lr2, res.cvm),
        "eigenvalues": [float(v) for v in res.eig],
        "cointegration_rank": {"trace": rank_trace, "max_eigen": _rank(res.lr2, res.cvm)},
        "cointegrating_vectors": np.asarray(res.evec)[:, :max(rank_trace, 1)].tolist(),
    }


def newey_west(
    X: Any,
    y: Any | None = None,
    *,
    lags: int | str = "auto",
    add_intercept: bool = True,
    small_sample: bool = False,
) -> dict[str, Any]:
    """Newey-West HAC covariance for an OLS regression.

    R analogue of ``sandwich::NeweyWest`` combined with ``lmtest::coeftest``:
    fit ``y = X b + e`` by ordinary least squares, then form the
    heteroskedasticity- and autocorrelation-consistent (HAC) covariance of the
    coefficients using a Bartlett (Newey-West) kernel. ``lags`` is the bandwidth
    ``L``; ``"auto"`` uses the Newey-West fixed rule ``floor(4 (T/100)^(2/9))``.
    With ``small_sample=True`` the meat is scaled by ``T / (T - k)`` (the
    finite-sample adjustment used by ``lmtest::coeftest`` defaults).

    Returns the coefficient estimates, HAC standard errors, ``t`` statistics,
    two-sided ``p`` values (Student-``t`` with ``T - k`` degrees of freedom), the
    HAC covariance matrix, the bandwidth, and the regressor names.
    """

    from scipy import stats as _stats

    if y is None:
        frame = pd.DataFrame(X)
        numeric = frame.select_dtypes("number")
        if numeric.shape[1] < 2:
            raise ValueError("newey_west needs a target plus at least one regressor")
        y_series = numeric.iloc[:, 0]
        x_frame = numeric.iloc[:, 1:]
    else:
        x_frame = pd.DataFrame(X)
        x_frame = x_frame.select_dtypes("number") if hasattr(x_frame, "select_dtypes") else x_frame
        y_series = pd.Series(np.asarray(y, dtype=float).ravel())

    names = [str(c) for c in x_frame.columns]
    x_mat = np.asarray(x_frame, dtype=float)
    y_vec = np.asarray(y_series, dtype=float).ravel()

    mask = np.isfinite(y_vec) & np.all(np.isfinite(x_mat), axis=1)
    x_mat = x_mat[mask]
    y_vec = y_vec[mask]
    if add_intercept:
        x_mat = np.column_stack([np.ones(x_mat.shape[0]), x_mat])
        names = ["(intercept)", *names]

    n_obs, k = x_mat.shape
    if n_obs <= k:
        raise ValueError("newey_west needs more observations than coefficients")

    xtx = x_mat.T @ x_mat
    bread = np.linalg.inv(xtx)
    beta = bread @ (x_mat.T @ y_vec)
    resid = y_vec - x_mat @ beta

    if isinstance(lags, str):
        if lags != "auto":
            raise ValueError("lags must be a nonnegative int or 'auto'")
        band = int(np.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0)))
    else:
        band = int(lags)
        if band < 0:
            raise ValueError("lags must be nonnegative")
    band = min(band, n_obs - 1)

    # Score s_t = x_t * e_t (n x k); HAC meat = Gamma_0 + sum_j w_j (Gamma_j + Gamma_j').
    scores = x_mat * resid[:, None]
    meat = scores.T @ scores
    for j in range(1, band + 1):
        weight = 1.0 - j / (band + 1.0)
        gamma = scores[j:].T @ scores[:-j]
        meat += weight * (gamma + gamma.T)
    if small_sample:
        meat *= n_obs / (n_obs - k)

    vcov = bread @ meat @ bread
    se = np.sqrt(np.maximum(np.diag(vcov), 0.0))
    tstat = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    pval = 2.0 * _stats.t.sf(np.abs(tstat), df=n_obs - k)

    coefficients = [
        {"name": names[i], "estimate": float(beta[i]), "std_error": float(se[i]),
         "t_value": float(tstat[i]), "p_value": float(pval[i])}
        for i in range(k)
    ]
    return {
        "n_obs": int(n_obs),
        "n_coef": int(k),
        "names": names,
        "lags": int(band),
        "kernel": "bartlett",
        "coefficients": coefficients,
        "estimate": beta.tolist(),
        "std_error": se.tolist(),
        "t_value": tstat.tolist(),
        "p_value": pval.tolist(),
        "vcov": vcov.tolist(),
    }


def vcov_hc(
    X: Any,
    y: Any | None = None,
    *,
    cov_type: str = "HC1",
    add_intercept: bool = True,
) -> dict[str, Any]:
    """Heteroskedasticity-consistent (White) covariance for an OLS regression.

    R analogue of ``sandwich::vcovHC`` with ``lmtest::coeftest``. Fits
    ``y = X b + e`` by OLS and forms a robust covariance that is consistent
    under heteroskedasticity but assumes no autocorrelation (for serial
    correlation use :func:`newey_west`). ``cov_type`` selects the small-sample
    weighting of the squared residuals:

    - ``"HC0"`` -- White (1980), ``u_i^2``;
    - ``"HC1"`` -- MacKinnon-White degrees-of-freedom scaling ``u_i^2 T/(T-k)``;
    - ``"HC2"`` -- leverage-adjusted ``u_i^2/(1-h_i)``;
    - ``"HC3"`` -- jackknife approximation ``u_i^2/(1-h_i)^2`` (default in R for
      small samples).

    Returns the coefficient table (estimate, robust SE, ``t``, two-sided ``p``
    with ``T - k`` degrees of freedom), the robust covariance matrix, and the
    regressor names.
    """

    from scipy import stats as _stats

    if y is None:
        frame = pd.DataFrame(X)
        numeric = frame.select_dtypes("number")
        if numeric.shape[1] < 2:
            raise ValueError("vcov_hc needs a target plus at least one regressor")
        y_series = numeric.iloc[:, 0]
        x_frame = numeric.iloc[:, 1:]
    else:
        x_frame = pd.DataFrame(X)
        x_frame = x_frame.select_dtypes("number") if hasattr(x_frame, "select_dtypes") else x_frame
        y_series = pd.Series(np.asarray(y, dtype=float).ravel())

    names = [str(c) for c in x_frame.columns]
    x_mat = np.asarray(x_frame, dtype=float)
    y_vec = np.asarray(y_series, dtype=float).ravel()
    mask = np.isfinite(y_vec) & np.all(np.isfinite(x_mat), axis=1)
    x_mat, y_vec = x_mat[mask], y_vec[mask]
    if add_intercept:
        x_mat = np.column_stack([np.ones(x_mat.shape[0]), x_mat])
        names = ["(intercept)", *names]

    n_obs, k = x_mat.shape
    if n_obs <= k:
        raise ValueError("vcov_hc needs more observations than coefficients")

    bread = np.linalg.inv(x_mat.T @ x_mat)
    beta = bread @ (x_mat.T @ y_vec)
    resid = y_vec - x_mat @ beta
    leverage = np.einsum("ij,jk,ik->i", x_mat, bread, x_mat)

    kind = str(cov_type).upper()
    u2 = resid**2
    if kind == "HC0":
        weights = u2
    elif kind == "HC1":
        weights = u2 * (n_obs / (n_obs - k))
    elif kind == "HC2":
        weights = u2 / (1.0 - leverage)
    elif kind == "HC3":
        weights = u2 / (1.0 - leverage) ** 2
    else:
        raise ValueError("cov_type must be one of 'HC0', 'HC1', 'HC2', 'HC3'")

    meat = (x_mat * weights[:, None]).T @ x_mat
    vcov = bread @ meat @ bread
    se = np.sqrt(np.maximum(np.diag(vcov), 0.0))
    tstat = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    pval = 2.0 * _stats.t.sf(np.abs(tstat), df=n_obs - k)

    coefficients = [
        {"name": names[i], "estimate": float(beta[i]), "std_error": float(se[i]),
         "t_value": float(tstat[i]), "p_value": float(pval[i])}
        for i in range(k)
    ]
    return {
        "n_obs": int(n_obs),
        "n_coef": int(k),
        "names": names,
        "cov_type": kind,
        "coefficients": coefficients,
        "estimate": beta.tolist(),
        "std_error": se.tolist(),
        "t_value": tstat.tolist(),
        "p_value": pval.tolist(),
        "vcov": vcov.tolist(),
    }


def breusch_pagan_test(
    X: Any,
    y: Any | None = None,
    *,
    studentize: bool = True,
    add_intercept: bool = True,
) -> dict[str, Any]:
    """Breusch-Pagan test for heteroskedasticity in an OLS regression.

    R analogue of ``lmtest::bptest``. Fits ``y = X b + e`` by OLS and tests the
    null of homoskedasticity by an auxiliary regression of the squared residuals
    on the regressors. With ``studentize=True`` (the ``lmtest`` default) the
    Koenker robust version ``LM = n R^2`` is used, valid without normality; with
    ``studentize=False`` the classic Breusch-Pagan-Godfrey statistic
    ``0.5 * explained-SS`` of the scaled squared residuals is returned. Both are
    chi-squared with ``p`` degrees of freedom (the number of regressors excluding
    the intercept). Returns the statistic, degrees of freedom, ``p`` value, and
    the auxiliary-regression R-squared.
    """

    from scipy import stats as _stats

    if y is None:
        frame = pd.DataFrame(X)
        numeric = frame.select_dtypes("number")
        if numeric.shape[1] < 2:
            raise ValueError("breusch_pagan_test needs a target plus at least one regressor")
        y_series = numeric.iloc[:, 0]
        x_frame = numeric.iloc[:, 1:]
    else:
        x_frame = pd.DataFrame(X)
        x_frame = x_frame.select_dtypes("number") if hasattr(x_frame, "select_dtypes") else x_frame
        y_series = pd.Series(np.asarray(y, dtype=float).ravel())

    x_mat = np.asarray(x_frame, dtype=float)
    y_vec = np.asarray(y_series, dtype=float).ravel()
    mask = np.isfinite(y_vec) & np.all(np.isfinite(x_mat), axis=1)
    x_mat, y_vec = x_mat[mask], y_vec[mask]
    if add_intercept:
        x_mat = np.column_stack([np.ones(x_mat.shape[0]), x_mat])

    n_obs, k = x_mat.shape
    if n_obs <= k:
        raise ValueError("breusch_pagan_test needs more observations than coefficients")

    bread = np.linalg.inv(x_mat.T @ x_mat)
    beta = bread @ (x_mat.T @ y_vec)
    resid = y_vec - x_mat @ beta
    u2 = resid**2

    # Auxiliary regression of squared residuals on the regressors.
    aux_coef = bread @ (x_mat.T @ u2)
    aux_fit = x_mat @ aux_coef
    ss_tot = float(np.sum((u2 - u2.mean()) ** 2))
    ss_res = float(np.sum((u2 - aux_fit) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    ss_explained = ss_tot - ss_res
    df = k - 1 if add_intercept else k

    if studentize:
        statistic = n_obs * r_squared
        version = "koenker"
    else:
        sigma2 = float(np.mean(u2))  # MLE variance, SSR/n
        statistic = 0.5 * ss_explained / (sigma2**2) if sigma2 > 0 else 0.0
        version = "breusch_pagan_godfrey"

    p_value = float(_stats.chi2.sf(statistic, df))
    return {
        "statistic": float(statistic),
        "df": int(df),
        "p_value": p_value,
        "r_squared": float(r_squared),
        "version": version,
        "n_obs": int(n_obs),
    }


def engle_granger(
    y: Any,
    x: Any | None = None,
    *,
    trend: str = "c",
    max_lag: int | None = None,
    autolag: str | None = "aic",
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Engle-Granger two-step residual-based cointegration test.

    R analogue of the two-step Engle-Granger procedure (statsmodels
    ``tsa.stattools.coint``): regress ``y`` on ``x`` by OLS and apply an
    augmented Dickey-Fuller test to the residuals. A rejection indicates the
    residuals are stationary, i.e. ``y`` and ``x`` are cointegrated. ``trend`` is
    the deterministic term in the cointegrating regression ('c', 'ct', or 'n');
    ``max_lag``/``autolag`` control the ADF lag length on the residuals.

    Pass ``y`` and ``x`` separately, or a single panel whose first numeric column
    is the dependent series and the remaining columns the regressors. Returns the
    ADF statistic on the residuals, the MacKinnon ``p`` value, the 1/5/10%
    critical values, the cointegrating-regression coefficients, and the
    cointegration flag at ``alpha``.
    """

    from statsmodels.tsa.stattools import coint

    if x is None:
        frame = pd.DataFrame(y)
        numeric = frame.select_dtypes("number")
        if numeric.shape[1] < 2:
            raise ValueError("engle_granger needs a dependent series plus at least one regressor")
        y0 = numeric.iloc[:, 0]
        x_frame = numeric.iloc[:, 1:]
    else:
        y0 = pd.Series(np.asarray(y, dtype=float).ravel())
        x_frame = pd.DataFrame(x)
        x_frame = x_frame.select_dtypes("number") if hasattr(x_frame, "select_dtypes") else x_frame

    names = [str(c) for c in x_frame.columns]
    y_vec = np.asarray(y0, dtype=float).ravel()
    x_mat = np.asarray(x_frame, dtype=float)
    mask = np.isfinite(y_vec) & np.all(np.isfinite(x_mat), axis=1)
    y_vec, x_mat = y_vec[mask], x_mat[mask]

    if trend not in {"n", "c", "ct", "ctt"}:
        raise ValueError("trend must be one of 'n', 'c', 'ct', 'ctt'")

    stat, pvalue, crit = coint(
        y_vec, x_mat, trend=trend, maxlag=max_lag, autolag=autolag
    )

    # Cointegrating regression for the long-run coefficients (and residuals).
    design = x_mat
    coef_names = list(names)
    if trend in {"c", "ct", "ctt"}:
        design = np.column_stack([np.ones(x_mat.shape[0]), design])
        coef_names = ["(intercept)", *coef_names]
    if trend in {"ct", "ctt"}:
        design = np.column_stack([design, np.arange(x_mat.shape[0], dtype=float)])
        coef_names = [*coef_names, "trend"]
    beta = np.linalg.lstsq(design, y_vec, rcond=None)[0]

    crit_values = {
        "1%": float(crit[0]), "5%": float(crit[1]), "10%": float(crit[2])
    }
    return {
        "test": "engle_granger",
        "statistic": float(stat),
        "p_value": float(pvalue),
        "n_obs": int(y_vec.size),
        "names": names,
        "trend": trend,
        "critical_values": crit_values,
        "cointegrating_coef": {name: float(b) for name, b in zip(coef_names, beta)},
        "cointegrated": bool(pvalue < alpha),
    }


def acf(
    series: Any,
    *,
    nlags: int = 20,
    alpha: float = 0.05,
    adjusted: bool = False,
) -> pd.DataFrame:
    """Sample autocorrelation function (stats::acf / forecast::Acf).

    Returns a tidy table with the autocorrelation at lags 0..nlags and the
    approximate ``1 - alpha`` confidence band (statsmodels acf). ``adjusted``
    selects the n-k (unbiased) divisor instead of the biased 1/n estimator.
    """

    from statsmodels.tsa.stattools import acf as _acf

    x = pd.Series(series).dropna().astype(float).to_numpy()
    values, confint = _acf(x, nlags=int(nlags), alpha=float(alpha), adjusted=bool(adjusted), fft=True)
    rows = [
        {"lag": int(k), "acf": float(values[k]),
         "lower": float(confint[k, 0]), "upper": float(confint[k, 1])}
        for k in range(len(values))
    ]
    table = pd.DataFrame(rows)
    table.attrs["macroforecast_metadata"] = {"kind": "acf", "nlags": int(nlags), "adjusted": bool(adjusted)}
    return table


def pacf(
    series: Any,
    *,
    nlags: int = 20,
    alpha: float = 0.05,
    method: str = "ywadjusted",
) -> pd.DataFrame:
    """Sample partial autocorrelation function (stats::pacf / forecast::Pacf).

    Returns a tidy table with the partial autocorrelation at lags 0..nlags and
    the approximate ``1 - alpha`` confidence band (statsmodels pacf). ``method``
    is the statsmodels PACF estimator ('ywadjusted','ols','ld', ...).
    """

    from statsmodels.tsa.stattools import pacf as _pacf

    x = pd.Series(series).dropna().astype(float).to_numpy()
    values, confint = _pacf(x, nlags=int(nlags), alpha=float(alpha), method=method)
    rows = [
        {"lag": int(k), "pacf": float(values[k]),
         "lower": float(confint[k, 0]), "upper": float(confint[k, 1])}
        for k in range(len(values))
    ]
    table = pd.DataFrame(rows)
    table.attrs["macroforecast_metadata"] = {"kind": "pacf", "nlags": int(nlags), "method": method}
    return table


def ndiffs(
    series: Any,
    *,
    test: str = "kpss",
    max_d: int = 2,
    alpha: float = 0.05,
) -> int:
    """Number of first differences to make a series stationary (forecast::ndiffs).

    Repeatedly applies a unit-root / stationarity test until the differenced
    series is judged stationary (KPSS: fail to reject; ADF/PP: reject the unit
    root), up to ``max_d``.
    """

    key = str(test).lower()
    if key not in {"kpss", "adf", "pp"}:
        raise ValueError("test must be 'kpss', 'adf', or 'pp'")
    current = pd.Series(series).dropna().astype(float)
    d = 0
    while d < int(max_d) and len(current) >= 10:
        try:
            if key == "kpss":
                stationary = not kpss_test(current, alpha=alpha)["reject_stationarity"]
            elif key == "adf":
                stationary = adf_test(current, alpha=alpha)["reject_unit_root"]
            else:
                res = phillips_perron_test(current)
                stationary = bool(res.get("reject_unit_root", False))
        except Exception:
            break
        if stationary:
            break
        current = current.diff().dropna()
        d += 1
    return d


def nsdiffs(
    series: Any,
    *,
    m: int,
    max_D: int = 1,
    threshold: float = 0.64,
) -> int:
    """Number of seasonal differences via seasonal strength (forecast::nsdiffs).

    Uses the Wang-Smyth-Hyndman seasonal strength F_s = max(0, 1 - Var(remainder)
    / Var(seasonal + remainder)) from an STL decomposition; a seasonal difference
    is applied while F_s exceeds ``threshold`` (default 0.64), up to ``max_D``.
    """

    period = int(m)
    if period < 2:
        raise ValueError("m must be >= 2")
    from statsmodels.tsa.seasonal import STL

    current = pd.Series(series).dropna().astype(float)
    D = 0
    while D < int(max_D) and len(current) >= 2 * period:
        try:
            res = STL(current, period=period).fit()
            resid = np.asarray(res.resid, dtype=float)
            seasonal = np.asarray(res.seasonal, dtype=float)
            var_resid = float(np.var(resid))
            var_sr = float(np.var(seasonal + resid))
            strength = max(0.0, 1.0 - var_resid / var_sr) if var_sr > 0 else 0.0
        except Exception:
            break
        if strength < float(threshold):
            break
        current = current.diff(period).dropna()
        D += 1
    return D


def dfgls_test(
    series: Any,
    *,
    trend: str = "c",
    method: str = "aic",
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Elliott-Rothenberg-Stock DF-GLS unit-root test for a single series.

    R analogue of ``urca::ur.ers`` (type ``"DF-GLS"``). The series is locally
    GLS-detrended before an augmented Dickey-Fuller regression, giving higher
    power than the standard ADF test against persistent stationary alternatives.
    ``trend`` is the deterministic specification, ``'c'`` (demeaned) or ``'ct'``
    (de-trended); ``method`` selects the lag length ('aic', 'bic', or 't-stat').
    Returns the statistic, the MacKinnon ``p`` value, the selected lag, the 1/5/10%
    critical values, and the unit-root rejection flag at ``alpha``.
    """

    from arch.unitroot import DFGLS

    if trend not in {"c", "ct"}:
        raise ValueError("trend must be 'c' (demean) or 'ct' (detrend)")
    values = pd.Series(series).dropna().astype(float).to_numpy()
    test = DFGLS(values, trend=trend, method=str(method).lower())
    crit = {str(level): float(value) for level, value in test.critical_values.items()}
    return {
        "test": "dfgls",
        "statistic": float(test.stat),
        "p_value": float(test.pvalue),
        "used_lag": int(test.lags),
        "n_obs": int(values.size),
        "trend": trend,
        "critical_values": crit,
        "reject_unit_root": bool(test.pvalue < alpha),
    }


def zivot_andrews_test(
    series: Any,
    *,
    regression: str = "c",
    trim: float = 0.15,
    maxlag: int | None = None,
    autolag: str | None = "AIC",
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Zivot-Andrews unit-root test allowing one endogenous structural break.

    R analogue of ``urca::ur.za``. Tests the unit-root null against a
    trend-stationary alternative with a single break whose date is chosen
    endogenously to be least favourable to the null. ``regression`` places the
    break in the intercept ('c'), the trend ('t'), or both ('ct'); ``trim`` is
    the fraction of the sample excluded at each end when searching for the break.
    Returns the minimised statistic, the ``p`` value, the 1/5/10% critical
    values, the selected lag, the estimated break position (index and label), and
    the unit-root rejection flag at ``alpha``.
    """

    from statsmodels.tsa.stattools import zivot_andrews

    if regression not in {"c", "t", "ct"}:
        raise ValueError("regression must be 'c', 't', or 'ct'")
    clean = pd.Series(series).dropna().astype(float)
    values = clean.to_numpy()
    stat, pvalue, crit, used_lag, break_idx = zivot_andrews(
        values, trim=trim, maxlag=maxlag, regression=regression, autolag=autolag
    )
    crit_values = {str(level): float(value) for level, value in crit.items()}
    break_idx = int(break_idx)
    break_label = clean.index[break_idx] if 0 <= break_idx < len(clean.index) else None
    return {
        "test": "zivot_andrews",
        "statistic": float(stat),
        "p_value": float(pvalue),
        "used_lag": int(used_lag),
        "n_obs": int(values.size),
        "regression": regression,
        "critical_values": crit_values,
        "break_index": break_idx,
        "break_label": (str(break_label) if break_label is not None else None),
        "reject_unit_root": bool(pvalue < alpha),
    }


def phillips_perron_test(values: Sequence[float] | np.ndarray, *, alpha: float = 0.05) -> dict[str, Any]:
    """Run the native Phillips-Perron Z_tau unit-root test."""

    _validate_alpha(alpha)
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    n = y.size
    if n < 8:
        return {"status": "insufficient_data", "n_obs": int(n)}
    y_t = y[1:]
    y_lag = y[:-1]
    x = np.column_stack([np.ones(n - 1), y_lag])
    xtx = x.T @ x
    try:
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        return {"status": "singular_design", "n_obs": int(n)}
    coef = xtx_inv @ x.T @ y_t
    rho = float(coef[1])
    resid = y_t - x @ coef
    sigma2 = float((resid @ resid) / max(n - 2 - 1, 1))
    bandwidth = max(1, int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))))
    gamma0 = float(np.dot(resid, resid) / n)
    lr_var = gamma0
    for lag in range(1, bandwidth + 1):
        weight = 1.0 - lag / (bandwidth + 1)
        cov = float(np.dot(resid[lag:], resid[:-lag]) / n)
        lr_var += 2.0 * weight * cov
    se_rho = float(np.sqrt(sigma2 * xtx_inv[1, 1]))
    t_rho = (rho - 1.0) / se_rho
    z_tau = float(
        np.sqrt(gamma0 / max(lr_var, 1e-12)) * t_rho
        - 0.5
        * (lr_var - gamma0)
        * np.sqrt(n)
        * np.sqrt(xtx_inv[1, 1])
        / np.sqrt(max(lr_var, 1e-12))
    )
    p_value = mackinnon_pp_pvalue(z_tau, n=n, regression="c")
    return {
        "statistic": z_tau,
        "p_value": p_value,
        "reject_unit_root": bool(p_value < alpha),
        "n_obs": int(n),
        "bandwidth_lags": bandwidth,
    }


_MACKINNON_PP_C_TABLE = {
    0.01: {25: -3.75, 50: -3.58, 100: -3.51, 250: -3.46, 500: -3.44, 1000: -3.43},
    0.05: {25: -3.00, 50: -2.93, 100: -2.89, 250: -2.88, 500: -2.87, 1000: -2.86},
    0.10: {25: -2.63, 50: -2.60, 100: -2.58, 250: -2.57, 500: -2.57, 1000: -2.57},
}


def mackinnon_pp_pvalue(z_tau: float, *, n: int, regression: str = "c") -> float:
    """Approximate MacKinnon p-value for the Phillips-Perron Z_tau statistic."""

    if not np.isfinite(float(z_tau)):
        raise ValueError("z_tau must be finite")
    if int(n) < 1:
        raise ValueError("n must be a positive integer")
    if regression != "c":
        from scipy import stats as _stats

        return float(_stats.norm.cdf(z_tau))
    table = _MACKINNON_PP_C_TABLE
    sizes = sorted(next(iter(table.values())).keys())
    n_clamped = max(sizes[0], min(sizes[-1], int(n)))
    lower_n = max((size for size in sizes if size <= n_clamped), default=sizes[0])
    upper_n = min((size for size in sizes if size >= n_clamped), default=sizes[-1])
    weight = 0.0 if lower_n == upper_n else (n_clamped - lower_n) / (upper_n - lower_n)
    critical_values: dict[float, float] = {}
    for level, by_n in table.items():
        critical_values[level] = by_n[lower_n] * (1 - weight) + by_n[upper_n] * weight
    levels = sorted(critical_values)
    values = [critical_values[level] for level in levels]
    if z_tau <= values[0]:
        return 0.005
    if z_tau >= values[-1]:
        return 0.50
    for idx in range(len(levels) - 1):
        if values[idx] <= z_tau <= values[idx + 1]:
            span = values[idx + 1] - values[idx]
            if span <= 0:
                return float(levels[idx])
            point = (z_tau - values[idx]) / span
            return float(levels[idx] * (1 - point) + levels[idx + 1] * point)
    return 0.50


def summarize_data(
    data: Any,
    *,
    metrics: Sequence[SummaryMetric] | None = None,
    include_correlation: bool = False,
    correlation_method: CorrelationMethod = "pearson",
    include_outliers: bool = False,
    outlier_method: OutlierMethod = "iqr",
    include_stationarity: bool = False,
    stationarity_test: StationarityTest = "multi",
    stationarity_scope: StationarityScope = "all",
) -> DataSummaryReport:
    """Run the standard single-panel summary suite."""

    panel, metadata = _coerce_panel(data)
    bundle = DataBundle(panel, metadata)
    selected_metrics = tuple(metrics or DEFAULT_SUMMARY_METRICS)
    _validate_summary_metrics(selected_metrics)
    report_metadata = attach_metadata(
        metadata,
        "data_analysis",
        {
            "analysis_type": "single_panel",
            "metrics": list(selected_metrics),
            "include_correlation": bool(include_correlation),
            "correlation_method": correlation_method if include_correlation else None,
            "include_outliers": bool(include_outliers),
            "outlier_method": outlier_method if include_outliers else None,
            "include_stationarity": bool(include_stationarity),
            "stationarity_test": stationarity_test if include_stationarity else None,
            "stationarity_scope": stationarity_scope if include_stationarity else None,
            "panel": _compact_panel_info(bundle),
            "input": _metadata_input_summary(metadata),
            "outputs": {
                "coverage": True,
                "univariate": True,
                "missing": True,
                "correlation": bool(include_correlation),
                "outliers": bool(include_outliers),
                "stationarity": bool(include_stationarity),
            },
        },
    )
    coverage = sample_coverage(bundle)
    univariate = univariate_summary(bundle, metrics=selected_metrics)
    missing = missing_summary(bundle)
    correlation = (
        correlation_matrix(bundle, method=correlation_method)
        if include_correlation
        else None
    )
    outliers = (
        outlier_summary(bundle, method=outlier_method)
        if include_outliers
        else None
    )
    stationarity = (
        stationarity_tests(
            data,
            test=stationarity_test,
            scope=stationarity_scope,
        )
        if include_stationarity
        else None
    )
    _attach_metadata(coverage, report_metadata)
    _attach_metadata(univariate, report_metadata)
    _attach_metadata(missing, report_metadata)
    _attach_metadata(correlation, report_metadata)
    _attach_metadata(outliers, report_metadata)
    return DataSummaryReport(
        overview=panel_overview(bundle),
        coverage=coverage,
        univariate=univariate,
        missing=missing,
        correlation=correlation,
        outliers=outliers,
        stationarity=stationarity,
        metadata=report_metadata,
    )


def _coerce_panel(data: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
    if isinstance(data, DataSpec):
        panel = data.panel
        metadata = dict(data.metadata)
    elif isinstance(data, DataBundle):
        panel = data.panel
        metadata = dict(data.metadata)
    elif hasattr(data, "panel") and isinstance(getattr(data, "panel"), pd.DataFrame):
        panel = getattr(data, "panel")
        metadata = dict(getattr(data, "metadata", {}) or {})
    elif isinstance(data, tuple) and len(data) == 2 and isinstance(data[0], pd.DataFrame):
        metadata = dict(data[1])
        panel = as_panel(data[0], metadata=metadata)
    elif isinstance(data, pd.DataFrame):
        metadata = dict(data.attrs.get("macroforecast_metadata", {}))
        panel = as_panel(data, metadata=metadata)
    else:
        raise TypeError("expected DataBundle, DataSpec, PreprocessedData, (panel, metadata), or pandas DataFrame")
    validate_panel(panel)
    return panel, metadata


def _validate_summary_metrics(metrics: Sequence[str]) -> None:
    allowed = set(SummaryMetric.__args__)  # type: ignore[attr-defined]
    unknown = sorted(set(metrics) - allowed)
    if unknown:
        raise ValueError(f"unknown summary metric(s): {unknown}")


def _compact_panel_info(bundle: DataBundle) -> dict[str, Any]:
    info = panel_info(bundle)
    return {
        "n_rows": info["n_rows"],
        "n_columns": info["n_columns"],
        "start": info["start"],
        "end": info["end"],
        "missing_values": info["missing_values"],
        "frequency": info["frequency"],
    }


def _target_names(data: Any, *, target: str | None, targets: Sequence[str] | None) -> set[str]:
    names = set(str(value) for value in (targets or ()))
    if target is not None:
        names.add(str(target))
    if not names:
        data_target = getattr(data, "target", None)
        data_targets = getattr(data, "targets", ())
        if data_target is not None:
            names.add(str(data_target))
        names.update(str(value) for value in data_targets)
    return names


def _scope_columns(
    panel: pd.DataFrame,
    data: Any,
    *,
    scope: StationarityScope,
    target: str | None,
    targets: Sequence[str] | None,
) -> list[str]:
    if scope == "all":
        return list(panel.columns)
    target_names = _target_names(data, target=target, targets=targets)
    if scope in {"target_only", "predictors_only"} and not target_names:
        raise ValueError(f"scope={scope!r} requires target or targets")
    panel_names = {str(column) for column in panel.columns}
    missing_targets = sorted(target_names - panel_names)
    if missing_targets:
        raise ValueError(f"target columns are not in the panel: {missing_targets}")
    if scope == "target_only":
        return [column for column in panel.columns if str(column) in target_names]
    if scope == "predictors_only":
        return [column for column in panel.columns if str(column) not in target_names]
    return list(panel.columns)


def _metadata_input_summary(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset": metadata.get("dataset"),
        "source_family": metadata.get("source_family"),
        "frequency": metadata.get("frequency"),
        "has_panel_report": "panel" in metadata,
        "has_preprocessing": "preprocessing" in metadata,
        "metadata_keys": sorted(str(key) for key in metadata),
    }


def _attach_metadata(frame: pd.DataFrame | None, metadata: Mapping[str, Any]) -> None:
    if frame is not None:
        frame.attrs["macroforecast_metadata"] = dict(metadata)


def _run_stationarity(
    name: str, series: pd.Series, alpha: float, *, adf_regression: str = "c"
) -> dict[str, Any]:
    if name == "adf":
        from statsmodels.tsa.stattools import adfuller

        stat, pvalue, *_ = adfuller(
            series.values, autolag="AIC", regression=adf_regression
        )
        return {
            "statistic": float(stat),
            "p_value": float(pvalue),
            "reject_unit_root": bool(pvalue < alpha),
            "regression": adf_regression,
        }
    if name == "kpss":
        from statsmodels.tsa.stattools import kpss as _kpss
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, pvalue, *_ = _kpss(series.values, regression="c", nlags="auto")
        return {
            "statistic": float(stat),
            "p_value": float(pvalue),
            "reject_stationarity": bool(pvalue < alpha),
        }
    if name == "pp":
        try:
            from arch.unitroot import PhillipsPerron  # type: ignore

            pp = PhillipsPerron(series.values, trend="c")
            return {
                "statistic": float(pp.stat),
                "p_value": float(pp.pvalue),
                "reject_unit_root": bool(pp.pvalue < alpha),
                "implementation": "arch",
            }
        except ImportError:
            pass
        result = phillips_perron_test(series.values, alpha=alpha)
        result["implementation"] = "native"
        return result
    return {"status": "unknown_test", "test": name}


def _date_string(value: Any) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _validate_positive(value: float, name: str) -> None:
    if float(value) <= 0:
        raise ValueError(f"{name} must be > 0")


def _validate_alpha(value: float) -> None:
    alpha = float(value)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")


def _float_or_none(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _longest_true_run(values: pd.Series) -> int:
    longest = 0
    current = 0
    for value in values.astype(bool).tolist():
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


__all__ = [
    "DEFAULT_SUMMARY_METRICS",
    "DataSummaryReport",
    "correlation_matrix",
    "mackinnon_pp_pvalue",
    "missing_summary",
    "missing_rates",
    "observation_counts",
    "outlier_summary",
    "panel_overview",
    "panel_snapshot",
    "adf_test",
    "kpss_test",
    "acf",
    "johansen_cointegration",
    "engle_granger",
    "newey_west",
    "vcov_hc",
    "breusch_pagan_test",
    "pacf",
    "ndiffs",
    "nsdiffs",
    "phillips_perron_test",
    "dfgls_test",
    "zivot_andrews_test",
    "sample_coverage",
    "stationarity_tests",
    "summarize_data",
    "univariate_summary",
]
