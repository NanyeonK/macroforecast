from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

MetricLike = str | Callable[..., float]


def _aligned_frame(*series: Any, names: Sequence[str] | None = None) -> pd.DataFrame:
    labels = list(names or [f"value_{idx}" for idx in range(len(series))])
    if len(labels) != len(series):
        raise ValueError("names length must match series length")
    frames = [
        pd.Series(value).astype(float).rename(name)
        for value, name in zip(series, labels)
    ]
    joined = pd.concat(frames, axis=1).dropna()
    if joined.empty:
        raise ValueError("metric inputs have no aligned non-missing observations")
    return joined


def _aligned_values(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    joined = _aligned_frame(y_true, y_pred, names=("truth", "pred"))
    return joined["truth"].to_numpy(dtype=float), joined["pred"].to_numpy(dtype=float)


def _aligned_relative_frame(
    y_true: Any, y_model: Any, y_benchmark: Any
) -> pd.DataFrame:
    truth = pd.Series(y_true).astype(float).rename("truth")
    model = pd.Series(y_model).astype(float).rename("model")
    benchmark = pd.Series(y_benchmark).astype(float).rename("benchmark")
    candidate = pd.concat([truth, model], axis=1).dropna()
    if candidate.empty:
        raise ValueError(
            "relative metric inputs have no candidate non-missing observations"
        )
    _validate_relative_metric_support(candidate["truth"], benchmark.dropna())
    joined = pd.concat([candidate, benchmark], axis=1).dropna()
    if joined.empty:
        raise ValueError(
            "relative metric inputs have no aligned non-missing observations"
        )
    return joined


def mse(y_true: Any, y_pred: Any) -> float:
    """Mean squared error."""

    truth, pred = _aligned_values(y_true, y_pred)
    return float(np.mean((truth - pred) ** 2))


def rmse(y_true: Any, y_pred: Any) -> float:
    """Root mean squared error."""

    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: Any, y_pred: Any) -> float:
    """Mean absolute error."""

    truth, pred = _aligned_values(y_true, y_pred)
    return float(np.mean(np.abs(truth - pred)))


def bias(y_true: Any, y_pred: Any) -> float:
    """Mean forecast residual, computed as ``actual - prediction``."""

    truth, pred = _aligned_values(y_true, y_pred)
    return float(np.mean(truth - pred))


def medae(y_true: Any, y_pred: Any) -> float:
    """Median absolute error."""

    truth, pred = _aligned_values(y_true, y_pred)
    return float(np.median(np.abs(truth - pred)))


def mad(y_true: Any, y_pred: Any) -> float:
    """Median absolute deviation of forecast errors around their median.

    For forecast errors ``e = actual - prediction``, ``mad`` is
    ``median(|e - median(e)|)`` over the evaluation window. This differs from
    ``medae``, which is the uncentered ``median(|e|)``.
    """

    truth, pred = _aligned_values(y_true, y_pred)
    errors = truth - pred
    return float(np.median(np.abs(errors - np.median(errors))))


def mape(y_true: Any, y_pred: Any, *, eps: float = 1e-10) -> float:
    """Mean absolute percentage error on the 0-100 scale."""

    if eps <= 0:
        raise ValueError("eps must be positive")
    truth, pred = _aligned_values(y_true, y_pred)
    denom = np.where(np.abs(truth) < eps, eps, np.abs(truth))
    return float(np.mean(np.abs(truth - pred) / denom) * 100.0)


def smape(y_true: Any, y_pred: Any, *, eps: float = 1e-10) -> float:
    """Symmetric mean absolute percentage error, M4/Mcomp convention.

    Uses the ``(|A|+|F|)/2`` denominator, so each term is bounded by 200 and the
    statistic ranges on **0-200** (not 0-100); this matches published M4 sMAPE.
    """

    if eps <= 0:
        raise ValueError("eps must be positive")
    truth, pred = _aligned_values(y_true, y_pred)
    denom = np.maximum((np.abs(truth) + np.abs(pred)) / 2.0, eps)
    return float(np.mean(np.abs(truth - pred) / denom) * 100.0)


def theil_u1(y_true: Any, y_pred: Any) -> float:
    """Theil U1 inequality coefficient."""

    truth, pred = _aligned_values(y_true, y_pred)
    numerator = float(np.sqrt(np.mean((truth - pred) ** 2)))
    denominator = float(np.sqrt(np.mean(truth**2)) + np.sqrt(np.mean(pred**2)))
    return numerator / denominator if denominator > 0 else float("nan")


def theil_u2(y_true: Any, y_pred: Any, y_prev: Any) -> float:
    """Proportional-change Theil U relative to a no-change forecast.

    Each squared error is normalised by ``y_prev`` (the Theil/Bliemel
    proportional-change form). This differs from ``forecast::accuracy``'s
    \"Theil's U\", which is the unweighted RMSE ratio (model vs naive).
    """

    joined = _aligned_frame(y_true, y_pred, y_prev, names=("truth", "pred", "prev"))
    if len(joined) < 2:
        return float("nan")
    prev = joined["prev"].to_numpy(dtype=float)
    safe_prev = np.where(np.abs(prev) > 0, prev, np.nan)
    truth = joined["truth"].to_numpy(dtype=float)
    pred = joined["pred"].to_numpy(dtype=float)
    numerator: float = float(np.nansum(((pred - truth) / safe_prev) ** 2))
    denominator: float = float(np.nansum(((truth - prev) / safe_prev) ** 2))
    return float(np.sqrt(numerator / denominator)) if denominator > 0 else float("nan")


def seasonal_naive_mae(y_train: Any, *, m: int = 1) -> float:
    """In-sample (seasonal-)naive mean absolute error: ``mean(|y[t] - y[t-m]|)``.

    This is the scale-free denominator used by MASE (Hyndman & Koehler, 2006).
    ``m=1`` is the random-walk (no-change) scaling; ``m>1`` the seasonal-naive
    scaling. Returns NaN when the training series is shorter than ``m+1``.
    """

    if m < 1:
        raise ValueError("m must be >= 1")
    train = pd.Series(y_train).dropna().astype(float).to_numpy()
    if len(train) <= m:
        return float("nan")
    diffs = np.abs(train[m:] - train[:-m])
    return float(np.mean(diffs)) if diffs.size else float("nan")


def mase(y_true: Any, y_pred: Any, y_train: Any, *, m: int = 1) -> float:
    """Mean Absolute Scaled Error (Hyndman & Koehler, 2006).

    ``MASE = mean(|y_true - y_pred|) / mean(|y_train[t] - y_train[t-m]|)`` -- the
    out-of-sample MAE scaled by the in-sample (seasonal-)naive MAE. ``m=1`` is the
    non-seasonal MASE; ``m>1`` the seasonal MASE. The statistic is scale-free and
    a value below 1 beats the in-sample naive benchmark. Matches forecast::accuracy.
    """

    truth, pred = _aligned_values(y_true, y_pred)
    if truth.size == 0:
        return float("nan")
    scale = seasonal_naive_mae(y_train, m=m)
    if not np.isfinite(scale) or scale <= 0:
        return float("nan")
    return float(np.mean(np.abs(truth - pred)) / scale)


def acf1(values: Any) -> float:
    """Lag-1 autocorrelation of a series (e.g. forecast residuals).

    The ACF1 reported by ``forecast::accuracy``; canonical biased sample
    autocorrelation ``gamma_1 / gamma_0`` with a single global mean.
    """

    arr = pd.Series(values).dropna().astype(float).to_numpy()
    if len(arr) < 2:
        return float("nan")
    centered = arr - arr.mean()
    gamma_0 = float(np.sum(centered * centered))
    if gamma_0 == 0:
        return float("nan")
    gamma_1 = float(np.sum(centered[1:] * centered[:-1]))
    return gamma_1 / gamma_0


def relative_mse(y_true: Any, y_model: Any, y_benchmark: Any) -> float:
    """Candidate model MSE divided by benchmark MSE."""

    joined = _aligned_relative_frame(y_true, y_model, y_benchmark)
    truth = joined["truth"].to_numpy(dtype=float)
    model = joined["model"].to_numpy(dtype=float)
    benchmark = joined["benchmark"].to_numpy(dtype=float)
    numerator = float(np.mean((truth - model) ** 2))
    denominator = float(np.mean((truth - benchmark) ** 2))
    return numerator / denominator if denominator > 0 else float("nan")


def relative_mae(y_true: Any, y_model: Any, y_benchmark: Any) -> float:
    """Candidate model MAE divided by benchmark MAE."""

    joined = _aligned_relative_frame(y_true, y_model, y_benchmark)
    truth = joined["truth"].to_numpy(dtype=float)
    model = joined["model"].to_numpy(dtype=float)
    benchmark = joined["benchmark"].to_numpy(dtype=float)
    numerator = float(np.mean(np.abs(truth - model)))
    denominator = float(np.mean(np.abs(truth - benchmark)))
    return numerator / denominator if denominator > 0 else float("nan")


def mse_reduction(y_true: Any, y_model: Any, y_benchmark: Any) -> float:
    """Benchmark MSE minus candidate model MSE."""

    joined = _aligned_relative_frame(y_true, y_model, y_benchmark)
    truth = joined["truth"].to_numpy(dtype=float)
    model = joined["model"].to_numpy(dtype=float)
    benchmark = joined["benchmark"].to_numpy(dtype=float)
    return float(np.mean((truth - benchmark) ** 2) - np.mean((truth - model) ** 2))


def r2_oos(y_true: Any, y_model: Any, y_benchmark: Any) -> float:
    """Out-of-sample R squared relative to a benchmark forecast."""

    value = relative_mse(y_true, y_model, y_benchmark)
    return float("nan") if np.isnan(value) else float(1.0 - value)


def pinball_loss(y_true: Any, y_quantile: Any, *, quantile: float) -> float:
    """Mean quantile pinball loss."""

    truth, pred = _aligned_values(y_true, y_quantile)
    q = float(quantile)
    if not 0.0 < q < 1.0:
        raise ValueError("quantile must be strictly between 0 and 1")
    error = truth - pred
    return float(np.mean(np.maximum(q * error, (q - 1.0) * error)))


def gaussian_nll(y_true: Any, y_pred: Any, variance: Any) -> float:
    """Gaussian negative log likelihood using supplied predictive variances."""

    joined = _aligned_frame(
        y_true, y_pred, variance, names=("truth", "pred", "variance")
    )
    values = _positive_values(joined["variance"], label="variance")
    errors = joined["truth"].to_numpy(dtype=float) - joined["pred"].to_numpy(
        dtype=float
    )
    return float(np.mean(0.5 * (np.log(2.0 * np.pi * values) + errors**2 / values)))


def log_score(y_true: Any, y_pred: Any, variance: Any) -> float:
    """Alias for Gaussian negative log score; lower is better."""

    return gaussian_nll(y_true, y_pred, variance)


def negative_log_score(y_true: Any, y_pred: Any, variance: Any) -> float:
    """Gaussian negative log score; lower is better."""

    return gaussian_nll(y_true, y_pred, variance)


def crps(y_true: Any, y_pred: Any, variance: Any) -> float:
    """Continuous ranked probability score for Gaussian predictive densities."""

    from scipy import stats as _stats

    joined = _aligned_frame(
        y_true, y_pred, variance, names=("truth", "pred", "variance")
    )
    sigma = np.sqrt(_positive_values(joined["variance"], label="variance"))
    z = (
        joined["truth"].to_numpy(dtype=float) - joined["pred"].to_numpy(dtype=float)
    ) / sigma
    score = sigma * (
        z * (2.0 * _stats.norm.cdf(z) - 1.0)
        + 2.0 * _stats.norm.pdf(z)
        - 1.0 / np.sqrt(np.pi)
    )
    return float(np.mean(score))


def qlike(y_true: Any, variance: Any, *, eps: float = 1e-12) -> float:
    """QLIKE loss for volatility forecasts."""

    if eps <= 0:
        raise ValueError("eps must be positive")
    joined = _aligned_frame(y_true, variance, names=("truth", "variance"))
    realized = _nonnegative_values(joined["truth"], label="realized variance")
    forecast = _positive_values(joined["variance"], label="forecast variance")
    realized = np.maximum(realized, eps)
    return float(np.mean(np.log(forecast) + realized / forecast))


def coverage_rate(y_true: Any, lower: Any, upper: Any) -> float:
    """Share of observations covered by lower/upper forecasts."""

    joined = _aligned_frame(y_true, lower, upper, names=("truth", "lower", "upper"))
    _validate_interval_bounds(joined["lower"], joined["upper"])
    covered = (joined["truth"] >= joined["lower"]) & (
        joined["truth"] <= joined["upper"]
    )
    return float(covered.mean())


def interval_width(lower: Any, upper: Any) -> float:
    """Mean forecast interval width."""

    joined = _aligned_frame(lower, upper, names=("lower", "upper"))
    _validate_interval_bounds(joined["lower"], joined["upper"])
    return float((joined["upper"] - joined["lower"]).mean())


def interval_score(
    y_true: Any, lower: Any, upper: Any, *, alpha: float = 0.05
) -> float:
    """Winkler interval score for a nominal ``1 - alpha`` interval."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    joined = _aligned_frame(y_true, lower, upper, names=("truth", "lower", "upper"))
    _validate_interval_bounds(joined["lower"], joined["upper"])
    truth = joined["truth"].to_numpy(dtype=float)
    lo = joined["lower"].to_numpy(dtype=float)
    hi = joined["upper"].to_numpy(dtype=float)
    width = hi - lo
    under = np.maximum(lo - truth, 0.0)
    over = np.maximum(truth - hi, 0.0)
    return float(np.mean(width + (2.0 / alpha) * under + (2.0 / alpha) * over))


def success_ratio(y_true: Any, y_pred: Any, y_prev: Any) -> float:
    """Directional hit rate relative to a previous actual value."""

    joined = _aligned_frame(y_true, y_pred, y_prev, names=("truth", "pred", "prev"))
    if len(joined) < 2:
        return float("nan")
    truth = joined["truth"].to_numpy(dtype=float)
    pred = joined["pred"].to_numpy(dtype=float)
    prev = joined["prev"].to_numpy(dtype=float)
    return float(np.mean(np.sign(pred - prev) == np.sign(truth - prev)))


def pesaran_timmermann_metric(
    y_true: Any, y_pred: Any, *, threshold: float = 0.0
) -> float:
    """Pesaran-Timmermann directional accuracy statistic."""

    truth, pred = _aligned_values(y_true, y_pred)
    n = len(truth)
    if n < 2:
        return float("nan")
    forecast = (pred > threshold).astype(int)
    actual = (truth > threshold).astype(int)
    success = float((forecast == actual).mean())
    p_y = float(actual.mean())
    p_x = float(forecast.mean())
    p_star = p_y * p_x + (1.0 - p_y) * (1.0 - p_x)
    if p_star <= 0.0 or p_star >= 1.0:
        return float("nan")
    var_p = (p_star * (1.0 - p_star)) / n
    var_p_star = (
        ((2.0 * p_y - 1.0) ** 2 * p_x * (1.0 - p_x)) / n
        + ((2.0 * p_x - 1.0) ** 2 * p_y * (1.0 - p_y)) / n
        + (4.0 * p_y * p_x * (1.0 - p_y) * (1.0 - p_x)) / (n * n)
    )
    denominator = max(var_p - var_p_star, 1e-12)
    return float((success - p_star) / np.sqrt(denominator))


def compute_point_loss(
    y_true: Any,
    y_pred: Any,
    *,
    loss: str = "squared_error",
    variance: Any | None = None,
    quantile: float | None = None,
    eps: float = 1e-12,
) -> pd.Series:
    """Return observation-level forecast loss where lower is better."""

    loss_name = _canonical_point_loss_name(loss, quantile=quantile)
    if loss_name == "squared_error":
        joined = _aligned_frame(y_true, y_pred, names=("truth", "pred"))
        values = (joined["truth"] - joined["pred"]) ** 2
    elif loss_name == "absolute_error":
        joined = _aligned_frame(y_true, y_pred, names=("truth", "pred"))
        values = (joined["truth"] - joined["pred"]).abs()
    elif loss_name.startswith("pinball_loss"):
        if quantile is None:
            raise ValueError("quantile is required when loss='pinball_loss'")
        q = float(quantile)
        if not 0.0 < q < 1.0:
            raise ValueError("quantile must be strictly between 0 and 1")
        joined = _aligned_frame(y_true, y_pred, names=("truth", "pred"))
        error = joined["truth"] - joined["pred"]
        values = np.maximum(q * error, (q - 1.0) * error)
    elif loss_name == "negative_log_score":
        if variance is None:
            raise ValueError("variance is required for negative_log_score")
        joined = _aligned_frame(
            y_true, y_pred, variance, names=("truth", "pred", "variance")
        )
        var = pd.Series(
            _positive_values(joined["variance"], label="variance"), index=joined.index
        )
        errors = joined["truth"] - joined["pred"]
        values = 0.5 * (np.log(2.0 * np.pi * var) + errors**2 / var)
    elif loss_name == "qlike":
        if eps <= 0:
            raise ValueError("eps must be positive")
        joined = _aligned_frame(y_true, y_pred, names=("truth", "variance"))
        realized = pd.Series(
            np.maximum(
                _nonnegative_values(joined["truth"], label="realized variance"), eps
            ),
            index=joined.index,
        )
        forecast = pd.Series(
            _positive_values(joined["variance"], label="forecast variance"),
            index=joined.index,
        )
        values = np.log(forecast) + realized / forecast
    else:  # pragma: no cover - guarded by _canonical_point_loss_name.
        raise ValueError(f"unsupported point loss {loss!r}")
    return pd.Series(values, index=joined.index, name=loss_name).astype(float)


def forecast_returns(
    forecasts: Any,
    *,
    benchmark: str,
    group_cols: Sequence[str] = ("target", "horizon"),
    loss: str = "squared_error",
    model_col: str = "model",
    actual: str = "actual",
    prediction: str = "prediction",
    variance_prediction: str = "variance_prediction",
    support_cols: Sequence[str] | None = None,
    include_benchmark: bool = False,
    quantile: float | None = None,
) -> pd.DataFrame:
    """Construct date-level forecast-return rows relative to a benchmark model."""

    frame = _forecast_frame(forecasts)
    _validate_table_columns(
        frame, (model_col, actual, prediction), label="forecast table"
    )
    groups = [
        column
        for column in (str(value) for value in group_cols)
        if column in frame.columns
    ]
    requested_missing = [
        str(value) for value in group_cols if str(value) not in frame.columns
    ]
    if requested_missing:
        raise ValueError(
            f"group_cols are not present in the forecast table: {requested_missing}"
        )
    supports = _risk_return_support_columns(frame, groups, support_cols=support_cols)
    loss_name = _canonical_point_loss_name(loss, quantile=quantile)
    work = frame.copy()
    work["_macroforecast_loss"] = _point_loss_from_frame(
        work,
        loss=loss,
        actual=actual,
        prediction=prediction,
        variance_prediction=variance_prediction,
        quantile=quantile,
    )
    work = work.dropna(subset=[model_col, "_macroforecast_loss"]).copy()
    benchmark_rows = work.loc[work[model_col].astype(str) == str(benchmark)].copy()
    if benchmark_rows.empty:
        raise ValueError(f"benchmark {benchmark!r} is not present in forecast rows")
    candidate_rows = (
        work.copy()
        if include_benchmark
        else work.loc[work[model_col].astype(str) != str(benchmark)].copy()
    )
    if candidate_rows.empty:
        return _risk_return_frame(
            [], kind="forecast_returns", metadata={"benchmark": benchmark}
        )

    _validate_unique_support(
        benchmark_rows, supports, model_col=None, label="benchmark"
    )
    _validate_unique_support(
        candidate_rows, [*supports, model_col], model_col=None, label="candidate"
    )
    benchmark_indexed = benchmark_rows.set_index(supports)
    rows: list[pd.DataFrame] = []
    candidate_group_keys = [*groups, model_col]
    iterator = (
        candidate_rows.groupby(candidate_group_keys, dropna=False, sort=True)
        if candidate_group_keys
        else [((), candidate_rows)]
    )
    for key, group in iterator:
        candidate = group.set_index(supports)
        group_benchmark = _matching_benchmark_rows(
            benchmark_indexed,
            group,
            supports=supports,
            groups=groups,
        )
        _validate_relative_metric_support(
            candidate["_macroforecast_loss"], group_benchmark["_macroforecast_loss"]
        )
        candidate = candidate.loc[group_benchmark.index].copy()
        if not np.allclose(
            candidate[actual].to_numpy(dtype=float),
            group_benchmark[actual].to_numpy(dtype=float),
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError(
                "benchmark actual values must match candidate actual values"
            )
        out = candidate.reset_index()
        out["model_loss"] = candidate["_macroforecast_loss"].to_numpy(dtype=float)
        out["benchmark_loss"] = group_benchmark["_macroforecast_loss"].to_numpy(
            dtype=float
        )
        out["forecast_return"] = out["benchmark_loss"] - out["model_loss"]
        out["return_sign"] = np.select(
            [out["forecast_return"] > 0.0, out["forecast_return"] < 0.0],
            ["positive", "negative"],
            default="zero",
        )
        out["benchmark_id"] = str(benchmark)
        out["model_id"] = out[model_col].astype(str)
        out["loss_name"] = loss_name
        rows.append(out)
    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    result = _attach_return_path_columns(
        result,
        by=[*groups, model_col, "benchmark_id", "loss_name"],
    )
    columns = [
        *[column for column in supports if column in result.columns],
        model_col,
        "model_id",
        "benchmark_id",
        "loss_name",
        "model_loss",
        "benchmark_loss",
        "forecast_return",
        "return_sign",
        "cumulative_return",
        "drawdown",
    ]
    columns.extend(
        column
        for column in (actual, prediction)
        if column in result.columns and column not in columns
    )
    result = result.loc[:, [column for column in columns if column in result.columns]]
    result.attrs["macroforecast_metadata_schema"] = {
        "kind": "forecast_returns",
        "version": 1,
        "benchmark": str(benchmark),
        "group_cols": groups,
        "support_cols": supports,
        "loss_name": loss_name,
        "model_col": str(model_col),
        "include_benchmark": bool(include_benchmark),
    }
    return result.reset_index(drop=True)


def sharpe_ratio(returns: Any, *, hac_lags: int | str | None = None) -> float:
    """Return mean forecast return divided by naive or HAC return volatility."""

    values = _return_values(returns)
    if values.size == 0:
        return float("nan")
    mean = float(np.mean(values))
    denominator = (
        _hac_long_run_sd(values, hac_lags=hac_lags)
        if hac_lags is not None
        else _sample_sd(values)
    )
    return _safe_ratio(mean, denominator)


def sortino_ratio(returns: Any, *, target_return: float = 0.0) -> float:
    """Return mean excess forecast return divided by downside semideviation."""

    values = _return_values(returns) - float(target_return)
    if values.size == 0:
        return float("nan")
    mean = float(np.mean(values))
    downside = np.minimum(values, 0.0)
    denominator = float(np.sqrt(np.mean(downside**2)))
    return _safe_ratio(mean, denominator)


def omega_ratio(returns: Any, *, threshold: float = 0.0) -> float:
    """Return total upside divided by total downside around a threshold."""

    values = _return_values(returns) - float(threshold)
    if values.size == 0:
        return float("nan")
    upside = float(np.sum(np.maximum(values, 0.0)))
    downside = float(np.sum(np.maximum(-values, 0.0)))
    return _safe_ratio(upside, downside)


def drawdown_series(returns: Any) -> pd.Series:
    """Return cumulative forecast return minus its running peak."""

    series = pd.Series(returns, dtype=float).dropna()
    if series.empty:
        return pd.Series(dtype=float, name="drawdown")
    cumulative = series.cumsum()
    drawdown = cumulative - cumulative.cummax()
    return drawdown.rename("drawdown")


def max_drawdown(returns: Any) -> float:
    """Return the most negative drawdown of a forecast-return path."""

    drawdown = drawdown_series(returns)
    return float(drawdown.min()) if not drawdown.empty else float("nan")


def risk_adjusted_forecast_metrics(
    returns: Any,
    *,
    group_cols: Sequence[str] | None = None,
    return_col: str = "forecast_return",
    hac_lags: int | str | None = "auto",
    target_return: float = 0.0,
    omega_threshold: float = 0.0,
) -> pd.DataFrame:
    """Aggregate forecast-return paths into risk-adjusted performance metrics."""

    frame = pd.DataFrame(returns).copy()
    if return_col not in frame.columns:
        raise ValueError(f"return_col {return_col!r} is not present")
    groups = (
        _default_risk_return_groups(frame)
        if group_cols is None
        else _validate_group_columns(frame, group_cols, label="group_cols")
    )
    sorted_frame = _sort_return_frame(frame)
    iterator = (
        sorted_frame.groupby(groups, dropna=False, sort=True)
        if groups
        else [((), sorted_frame)]
    )
    rows: list[dict[str, Any]] = []
    for key, group in iterator:
        row = _group_row(key, groups)
        values = (
            pd.to_numeric(group[return_col], errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
        )
        row.update(
            _risk_adjusted_row(
                values,
                hac_lags=hac_lags,
                target_return=target_return,
                omega_threshold=omega_threshold,
            )
        )
        rows.append(row)
    out = _risk_return_frame(
        rows,
        kind="risk_adjusted_forecast_metrics",
        metadata={
            "group_cols": groups,
            "return_col": str(return_col),
            "hac_lags": hac_lags,
            "target_return": float(target_return),
            "omega_threshold": float(omega_threshold),
        },
    )
    return out


def edge_ratio(
    forecasts: Any,
    *,
    group_cols: Sequence[str] = ("target", "horizon"),
    loss: str = "squared_error",
    model_col: str = "model",
    actual: str = "actual",
    prediction: str = "prediction",
    variance_prediction: str = "variance_prediction",
    support_cols: Sequence[str] | None = None,
    quantile: float | None = None,
) -> pd.DataFrame:
    """Return frontier-based Edge Ratio by model and group."""

    frame = _forecast_frame(forecasts)
    _validate_table_columns(
        frame, (model_col, actual, prediction), label="forecast table"
    )
    groups = [
        column
        for column in (str(value) for value in group_cols)
        if column in frame.columns
    ]
    requested_missing = [
        str(value) for value in group_cols if str(value) not in frame.columns
    ]
    if requested_missing:
        raise ValueError(
            f"group_cols are not present in the forecast table: {requested_missing}"
        )
    supports = _risk_return_support_columns(frame, groups, support_cols=support_cols)
    work = frame.copy()
    work["_macroforecast_loss"] = _point_loss_from_frame(
        work,
        loss=loss,
        actual=actual,
        prediction=prediction,
        variance_prediction=variance_prediction,
        quantile=quantile,
    )
    work = work.dropna(subset=[model_col, "_macroforecast_loss"]).copy()
    if work.empty:
        return _risk_return_frame([], kind="edge_ratio", metadata={"loss_name": loss})
    _validate_unique_support(
        work, [*supports, model_col], model_col=None, label="edge-ratio"
    )
    rows: list[dict[str, Any]] = []
    support_iterator = work.groupby(supports, dropna=False, sort=True)
    for _support_key, group in support_iterator:
        models = group[model_col].astype(str).to_numpy()
        losses = group["_macroforecast_loss"].to_numpy(dtype=float)
        if len(models) < 2:
            continue
        for idx, row in group.reset_index(drop=True).iterrows():
            other_losses: np.ndarray = np.delete(losses, idx)
            other_min = float(np.min(other_losses))
            edge = other_min - float(row["_macroforecast_loss"])
            out_row = {
                column: row[column]
                for column in [*supports, model_col]
                if column in row.index
            }
            out_row.update(
                {
                    "model_id": str(row[model_col]),
                    "loss_name": _canonical_point_loss_name(loss, quantile=quantile),
                    "model_loss": float(row["_macroforecast_loss"]),
                    "alternative_frontier_loss": other_min,
                    "edge": edge,
                    "edge_sign": (
                        "positive" if edge > 0 else "negative" if edge < 0 else "zero"
                    ),
                    "n_models_at_date": int(len(models)),
                }
            )
            rows.append(out_row)
    edge_path = pd.DataFrame(rows)
    if edge_path.empty:
        return _risk_return_frame([], kind="edge_ratio", metadata={"loss_name": loss})
    agg_groups = [*groups, model_col, "model_id", "loss_name"]
    out_rows: list[dict[str, Any]] = []
    for key, group in edge_path.groupby(agg_groups, dropna=False, sort=True):
        row = _group_row(key, agg_groups)
        values = group["edge"].to_numpy(dtype=float)
        wins = float(np.sum(np.maximum(values, 0.0)))
        regrets = float(np.sum(np.maximum(-values, 0.0)))
        n_models = int(group["n_models_at_date"].max())
        row.update(
            {
                "n_obs": int(len(values)),
                "n_models": n_models,
                "edge_wins": wins,
                "edge_regrets": regrets,
                "edge_ratio": _safe_ratio(
                    wins, regrets, multiplier=max(n_models - 1, 1)
                ),
                "mean_edge": float(np.mean(values)) if len(values) else float("nan"),
                "frontier_win_share": (
                    float(np.mean(values > 0.0)) if len(values) else float("nan")
                ),
            }
        )
        out_rows.append(row)
    result = _risk_return_frame(
        out_rows,
        kind="edge_ratio",
        metadata={
            "group_cols": groups,
            "support_cols": supports,
            "loss_name": _canonical_point_loss_name(loss, quantile=quantile),
            "model_col": str(model_col),
            "sign_convention": "alternative_frontier_loss_minus_model_loss",
        },
    )
    result.attrs["macroforecast_edge_path"] = edge_path
    return result


def evaluate_forecasts(
    forecasts: Any,
    *,
    by: Sequence[str] = ("model", "horizon"),
    metrics: Sequence[str | MetricLike] = ("mse", "rmse", "mae"),
    actual: str = "actual",
    prediction: str = "prediction",
    variance_prediction: str = "variance_prediction",
    volatility_actual: str | None = None,
    quantile_predictions: str = "quantile_predictions",
    previous_actual: str = "previous_actual",
    benchmark_model: str | None = None,
    model_column: str = "model",
) -> pd.DataFrame:
    """Evaluate a forecasting runner output or forecast table."""

    frame = _forecast_frame(forecasts)
    if frame.empty:
        return _evaluation_frame([], metadata={"input_rows": 0})
    _validate_table_columns(frame, (actual, prediction), label="forecast table")
    group_keys = _validate_group_columns(frame, by, label="by")
    resolved_metrics = _resolve_metrics(metrics)
    requested_metric_names = [
        metric_name for _metric_fn, metric_name in resolved_metrics
    ]
    relative_metrics_requested = any(
        metric_name in _RELATIVE_METRIC_NAMES for metric_name in requested_metric_names
    )
    if relative_metrics_requested and benchmark_model is None:
        raise ValueError(
            "benchmark_model is required when relative metrics are requested"
        )
    if relative_metrics_requested and model_column not in group_keys:
        raise ValueError(
            f"by must include model_column {model_column!r} when relative metrics are requested"
        )
    if relative_metrics_requested:
        _validate_relative_table_identity(frame)
    qlike_actual = volatility_actual or actual
    if volatility_actual is not None:
        _validate_table_columns(frame, (volatility_actual,), label="forecast table")
    _validate_requested_metric_columns(
        frame,
        requested_metric_names,
        variance_prediction=variance_prediction,
        quantile_predictions=quantile_predictions,
        previous_actual=previous_actual,
    )
    if group_keys:
        iterator = frame.groupby(group_keys, dropna=False, sort=True)
    else:
        iterator = [((), frame)]
    rows: list[dict[str, Any]] = []
    support_columns = _relative_support_columns(frame)
    benchmark_lookup = _benchmark_lookup(
        frame,
        benchmark_model=benchmark_model,
        model_column=model_column,
        actual=actual,
        prediction=prediction,
        by=group_keys,
        support_columns=support_columns,
    )
    for key, group in iterator:
        row = _group_row(key, group_keys)
        valid = group[[actual, prediction]].dropna()
        row["n"] = int(len(valid))
        if len(valid) > 0:
            for metric_fn, metric_name in resolved_metrics:
                if metric_name in _RELATIVE_METRIC_NAMES:
                    benchmark = benchmark_lookup.get(
                        _benchmark_key(row, group_keys, model_column=model_column)
                    )
                    if benchmark is None:
                        raise ValueError(
                            f"benchmark_model {benchmark_model!r} has no matching "
                            f"benchmark forecast for group {_benchmark_key(row, group_keys, model_column=model_column)}"
                        )
                    truth, pred = _series_for_relative_metric(
                        group,
                        actual=actual,
                        prediction=prediction,
                        support_columns=support_columns,
                    )
                    _validate_relative_metric_support(truth, benchmark[prediction])
                    _validate_benchmark_actuals(
                        truth,
                        benchmark[actual],
                        actual=actual,
                    )
                    row[metric_name] = float(
                        metric_fn(truth, pred, benchmark[prediction])
                    )
                    continue
                if metric_name in _VARIANCE_METRIC_NAMES:
                    if variance_prediction in group.columns:
                        variance_valid = group[
                            [actual, prediction, variance_prediction]
                        ].dropna()
                        if len(variance_valid) > 0:
                            row[metric_name] = float(
                                metric_fn(
                                    variance_valid[actual],
                                    variance_valid[prediction],
                                    variance_valid[variance_prediction],
                                )
                            )
                    continue
                if metric_name in _VOLATILITY_METRIC_NAMES:
                    if variance_prediction in group.columns:
                        variance_valid = group[
                            [qlike_actual, variance_prediction]
                        ].dropna()
                        if len(variance_valid) > 0:
                            row[metric_name] = float(
                                metric_fn(
                                    variance_valid[qlike_actual],
                                    variance_valid[variance_prediction],
                                )
                            )
                    continue
                if metric_name in _DIRECTION_METRIC_NAMES:
                    continue
                if metric_name in _QUANTILE_METRIC_NAMES:
                    continue
                row[metric_name] = float(metric_fn(valid[actual], valid[prediction]))
        if previous_actual in group.columns:
            direction_valid = group[[actual, prediction, previous_actual]].dropna()
            if len(direction_valid) > 0:
                row["theil_u2"] = theil_u2(
                    direction_valid[actual],
                    direction_valid[prediction],
                    direction_valid[previous_actual],
                )
                row["success_ratio"] = success_ratio(
                    direction_valid[actual],
                    direction_valid[prediction],
                    direction_valid[previous_actual],
                )
        if variance_prediction in group.columns:
            variance_valid = group[[actual, prediction, variance_prediction]].dropna()
            if len(variance_valid) > 0:
                row["variance_n"] = int(len(variance_valid))
                row["gaussian_nll"] = gaussian_nll(
                    variance_valid[actual],
                    variance_valid[prediction],
                    variance_valid[variance_prediction],
                )
                row["crps"] = crps(
                    variance_valid[actual],
                    variance_valid[prediction],
                    variance_valid[variance_prediction],
                )
        if quantile_predictions in group.columns:
            row.update(
                _quantile_evaluation(
                    group,
                    actual=actual,
                    quantile_predictions=quantile_predictions,
                )
            )
        rows.append(row)
    return _evaluation_frame(
        rows,
        metadata={
            "by": list(group_keys),
            "requested_metrics": requested_metric_names,
            "benchmark_model": benchmark_model,
            "model_column": model_column,
            "actual": actual,
            "prediction": prediction,
            "variance_prediction": variance_prediction,
            "volatility_actual": volatility_actual,
            "quantile_predictions": quantile_predictions,
            "previous_actual": previous_actual,
            "relative_support_columns": support_columns,
            "input_rows": int(frame.shape[0]),
            "input_columns": [str(column) for column in frame.columns],
            "auto_metric_groups": _auto_metric_groups(
                frame,
                variance_prediction=variance_prediction,
                quantile_predictions=quantile_predictions,
                previous_actual=previous_actual,
            ),
        },
    )


def rank_forecasts(
    evaluation: pd.DataFrame,
    *,
    metric: str = "mse",
    by: Sequence[str] = ("horizon",),
    ascending: bool | None = None,
    rank_column: str = "rank",
) -> pd.DataFrame:
    """Rank evaluated models within horizon/target groups."""

    frame = pd.DataFrame(evaluation).copy()
    if frame.empty:
        frame.attrs["macroforecast_metadata_schema"] = _rank_metadata_schema(
            metric=metric,
            by=[],
            rank_column=rank_column,
        )
        return frame
    if metric not in frame.columns:
        raise ValueError(f"metric column {metric!r} is not present")
    order = _metric_ascending(metric) if ascending is None else bool(ascending)
    group_keys = _validate_group_columns(frame, by, label="by")
    if group_keys:
        frame[rank_column] = frame.groupby(group_keys)[metric].rank(
            method="min", ascending=order
        )
        ranked = frame.sort_values([*group_keys, rank_column]).reset_index(drop=True)
        ranked.attrs["macroforecast_metadata_schema"] = _rank_metadata_schema(
            metric=metric,
            by=group_keys,
            rank_column=rank_column,
            ascending=order,
        )
        return ranked
    frame[rank_column] = frame[metric].rank(method="min", ascending=order)
    ranked = frame.sort_values(rank_column).reset_index(drop=True)
    ranked.attrs["macroforecast_metadata_schema"] = _rank_metadata_schema(
        metric=metric,
        by=group_keys,
        rank_column=rank_column,
        ascending=order,
    )
    return ranked


_METRICS: dict[str, Callable[..., float]] = {
    "mse": mse,
    "msfe": mse,
    "validation_mse": mse,
    "rmse": rmse,
    "validation_rmse": rmse,
    "mae": mae,
    "validation_mae": mae,
    "bias": bias,
    "mean_error": bias,
    "me": bias,
    "mad": mad,
    "medae": medae,
    "median_absolute_error": medae,
    "mape": mape,
    "smape": smape,
    "theil_u1": theil_u1,
    "theil_u2": theil_u2,
    "mase": mase,
    "relative_mse": relative_mse,
    "relative_mae": relative_mae,
    "mse_reduction": mse_reduction,
    "r2_oos": r2_oos,
    "pinball_loss": pinball_loss,
    "gaussian_nll": gaussian_nll,
    "log_score": log_score,
    "negative_log_score": negative_log_score,
    "crps": crps,
    "qlike": qlike,
    "coverage_rate": coverage_rate,
    "interval_width": interval_width,
    "interval_score": interval_score,
    "success_ratio": success_ratio,
    "pesaran_timmermann_metric": pesaran_timmermann_metric,
}

_RELATIVE_METRIC_NAMES = {"relative_mse", "relative_mae", "mse_reduction", "r2_oos"}
_VARIANCE_METRIC_NAMES = {"gaussian_nll", "log_score", "negative_log_score", "crps"}
_VOLATILITY_METRIC_NAMES = {"qlike"}
_DIRECTION_METRIC_NAMES = {"theil_u2", "success_ratio"}
_QUANTILE_METRIC_NAMES = {
    "pinball_loss",
    "coverage_rate",
    "interval_width",
    "interval_score",
    "mase",
    "seasonal_naive_mae",
    "acf1",
}


def get_metric(metric: MetricLike) -> Callable[..., float]:
    """Return a metric callable by name or pass through a callable metric."""

    if callable(metric):
        return metric
    key = metric.lower()
    if key not in _METRICS:
        allowed = ", ".join(sorted(_METRICS))
        raise ValueError(f"Unknown metric {metric!r}. Available metrics: {allowed}.")
    return _METRICS[key]


# The true quantile/interval-distributional subset of ``_QUANTILE_METRIC_NAMES``
# (which also carries ``mase``/``seasonal_naive_mae``/``acf1`` -- point-adjacent
# metrics with a different input shape entirely, deliberately excluded here; see
# ``evaluate_forecasts``'s ``continue``-without-computing handling of them).
_QUANTILE_DENSITY_METRIC_NAMES = frozenset(
    {"pinball_loss", "coverage_rate", "interval_width", "interval_score"}
)

# Every metric name whose table-level evaluation needs a distributional column
# (``variance_prediction`` or ``quantile_predictions``) rather than plain
# ``(y_true, y_pred)`` -- the union ``metric_kind`` classifies as "variance",
# "volatility", or "quantile".
DENSITY_METRIC_NAMES: frozenset[str] = frozenset(
    _VARIANCE_METRIC_NAMES | _VOLATILITY_METRIC_NAMES | _QUANTILE_DENSITY_METRIC_NAMES
)


def metric_kind(metric: MetricLike) -> str:
    """Classify a metric by the forecast-table column(s) its table-level
    evaluation requires (registry plumbing only -- no new metric math).

    Returns one of:

    - ``"variance"``: Gaussian predictive-density metrics needing
      ``(y_true, y_pred, variance_prediction)`` -- ``crps``, ``gaussian_nll``,
      ``log_score``, ``negative_log_score``.
    - ``"volatility"``: realized-vs-forecast variance metrics needing
      ``(realized_variance, variance_prediction)`` -- ``qlike``.
    - ``"quantile"``: interval/quantile metrics needing ``quantile_predictions``
      -- ``pinball_loss``, ``coverage_rate``, ``interval_width``,
      ``interval_score``.
    - ``"relative"``: benchmark-relative metrics needing a benchmark forecast --
      ``relative_mse``, ``relative_mae``, ``mse_reduction``, ``r2_oos``.
    - ``"direction"``: metrics needing a ``previous_actual`` reference --
      ``theil_u2``, ``success_ratio``.
    - ``"point"`` (the default): every ordinary ``(y_true, y_pred)`` metric,
      including any custom callable (classification is name-based, so a
      callable -- which carries no registry name -- is always ``"point"``).
    """

    if not isinstance(metric, str):
        return "point"
    key = metric.lower()
    if key in _VARIANCE_METRIC_NAMES:
        return "variance"
    if key in _VOLATILITY_METRIC_NAMES:
        return "volatility"
    if key in _QUANTILE_DENSITY_METRIC_NAMES:
        return "quantile"
    if key in _RELATIVE_METRIC_NAMES:
        return "relative"
    if key in _DIRECTION_METRIC_NAMES:
        return "direction"
    return "point"


def _resolve_metrics(
    metrics: Sequence[str | MetricLike],
) -> list[tuple[Callable[..., float], str]]:
    resolved = []
    for metric in metrics:
        metric_fn = get_metric(metric)
        metric_name = getattr(metric_fn, "__name__", str(metric))
        resolved.append((metric_fn, metric_name))
    return resolved


def _forecast_frame(forecasts: Any) -> pd.DataFrame:
    if hasattr(forecasts, "to_frame"):
        return forecasts.to_frame()
    return pd.DataFrame(forecasts).copy()


def _evaluation_frame(
    rows: Sequence[Mapping[str, Any]],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame.attrs["macroforecast_metadata_schema"] = {
        "kind": "forecast_metrics",
        "version": 1,
        "row_unit": "by_group",
        **dict(metadata or {}),
    }
    return frame


def _rank_metadata_schema(
    *,
    metric: str,
    by: Sequence[str],
    rank_column: str,
    ascending: bool | None = None,
) -> dict[str, Any]:
    out = {
        "kind": "forecast_metric_ranking",
        "version": 1,
        "metric": str(metric),
        "by": list(by),
        "rank_column": str(rank_column),
    }
    if ascending is not None:
        out["ascending"] = bool(ascending)
        out["direction"] = "lower_is_better" if ascending else "higher_is_better"
    return out


def _validate_table_columns(
    frame: pd.DataFrame, columns: Sequence[str], *, label: str
) -> tuple[str, ...]:
    values = tuple(str(column) for column in columns)
    missing = [column for column in values if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing required column(s): {missing}")
    return values


def _validate_group_columns(
    frame: pd.DataFrame, by: Sequence[str], *, label: str
) -> list[str]:
    values = [str(column) for column in by]
    missing = [column for column in values if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} column(s) are not present in the table: {missing}")
    return values


def _group_row(key: Any, group_keys: Sequence[str]) -> dict[str, Any]:
    if not group_keys:
        return {}
    if len(group_keys) == 1:
        values = (key[0],) if isinstance(key, tuple) else (key,)
    else:
        values = tuple(key)
    return {name: value for name, value in zip(group_keys, values)}


def _benchmark_lookup(
    frame: pd.DataFrame,
    *,
    benchmark_model: str | None,
    model_column: str,
    actual: str,
    prediction: str,
    by: Sequence[str],
    support_columns: Sequence[str],
) -> dict[tuple[Any, ...], pd.DataFrame]:
    if benchmark_model is None:
        return {}
    if model_column not in frame.columns:
        raise ValueError(f"model_column {model_column!r} is not present")
    bench = frame.loc[frame[model_column] == benchmark_model]
    if bench.empty:
        raise ValueError(
            f"benchmark_model {benchmark_model!r} is not present in forecast rows"
        )
    _validate_table_columns(bench, (actual, prediction), label="benchmark rows")
    keys = [key for key in by if key != model_column]
    if not keys:
        return {
            (): _support_indexed_frame(
                bench,
                actual=actual,
                prediction=prediction,
                support_columns=support_columns,
            )
        }
    out: dict[tuple[Any, ...], pd.DataFrame] = {}
    for key, group in bench.groupby(keys, dropna=False, sort=False):
        values = (key,) if len(keys) == 1 and not isinstance(key, tuple) else tuple(key)
        out[values] = _support_indexed_frame(
            group,
            actual=actual,
            prediction=prediction,
            support_columns=support_columns,
        )
    return out


def _benchmark_key(
    row: dict[str, Any], group_keys: Sequence[str], *, model_column: str
) -> tuple[Any, ...]:
    keys = [key for key in group_keys if key != model_column]
    return tuple(row[key] for key in keys if key in row)


def _series_for_relative_metric(
    group: pd.DataFrame,
    *,
    actual: str,
    prediction: str,
    support_columns: Sequence[str],
) -> tuple[pd.Series, pd.Series]:
    columns = [actual, prediction]
    columns.extend(column for column in support_columns if column not in columns)
    valid = group[columns].dropna()
    if support_columns:
        indexed = valid.set_index(list(support_columns))
        return indexed[actual], indexed[prediction]
    return valid[actual].reset_index(drop=True), valid[prediction].reset_index(
        drop=True
    )


def _support_indexed_frame(
    frame: pd.DataFrame,
    *,
    actual: str,
    prediction: str,
    support_columns: Sequence[str],
) -> pd.DataFrame:
    columns = [actual, prediction]
    if support_columns:
        columns = [*support_columns, actual, prediction]
        return (
            frame[columns]
            .dropna()
            .set_index(list(support_columns))[[actual, prediction]]
        )
    return frame[columns].dropna().reset_index(drop=True)


def _relative_support_columns(frame: pd.DataFrame) -> list[str]:
    identity = [
        column for column in ("date", "origin", "origin_pos") if column in frame.columns
    ]
    if not identity:
        return []
    return [
        column
        for column in ("date", "origin", "origin_pos", "target", "horizon")
        if column in frame.columns
    ]


def _validate_relative_table_identity(frame: pd.DataFrame) -> None:
    if not any(column in frame.columns for column in ("date", "origin", "origin_pos")):
        raise ValueError(
            "forecast table must contain at least one support column "
            "('date', 'origin', or 'origin_pos') when relative metrics are requested"
        )


def _validate_relative_metric_support(
    candidate: pd.Series, benchmark: pd.Series
) -> None:
    if not candidate.index.is_unique:
        raise ValueError(
            "candidate forecast support is not unique for relative metric evaluation"
        )
    if not benchmark.index.is_unique:
        raise ValueError(
            "benchmark forecast support is not unique for relative metric evaluation"
        )
    missing = candidate.index.difference(benchmark.index)
    extra = benchmark.index.difference(candidate.index)
    if len(missing) > 0 or len(extra) > 0:
        raise ValueError(
            "benchmark forecast support must match candidate support for relative metrics; "
            f"missing={list(missing[:5])}, extra={list(extra[:5])}"
        )


def _validate_benchmark_actuals(
    candidate: pd.Series, benchmark: pd.Series, *, actual: str
) -> None:
    benchmark_aligned = benchmark.loc[candidate.index]
    candidate_values = pd.Series(candidate).to_numpy(dtype=float)
    benchmark_values = pd.Series(benchmark_aligned).to_numpy(dtype=float)
    if not np.allclose(candidate_values, benchmark_values, rtol=1e-12, atol=1e-12):
        raise ValueError(
            f"benchmark {actual!r} values must match candidate {actual!r} values "
            "for relative metrics"
        )


def _validate_requested_metric_columns(
    frame: pd.DataFrame,
    requested_metrics: Sequence[str],
    *,
    variance_prediction: str,
    quantile_predictions: str,
    previous_actual: str,
) -> None:
    requested = set(requested_metrics)
    if requested & _VARIANCE_METRIC_NAMES and variance_prediction not in frame.columns:
        missing = sorted(requested & _VARIANCE_METRIC_NAMES)
        raise ValueError(
            f"variance_prediction column {variance_prediction!r} is required for requested "
            f"variance/density metric(s): {missing}"
        )
    if (
        requested & _VOLATILITY_METRIC_NAMES
        and variance_prediction not in frame.columns
    ):
        missing = sorted(requested & _VOLATILITY_METRIC_NAMES)
        raise ValueError(
            f"variance_prediction column {variance_prediction!r} is required for requested "
            f"volatility metric(s): {missing}"
        )
    if requested & _DIRECTION_METRIC_NAMES and previous_actual not in frame.columns:
        missing = sorted(requested & _DIRECTION_METRIC_NAMES)
        raise ValueError(
            f"previous_actual column {previous_actual!r} is required for requested "
            f"direction metric(s): {missing}"
        )
    if requested & _QUANTILE_METRIC_NAMES and quantile_predictions not in frame.columns:
        missing = sorted(requested & _QUANTILE_METRIC_NAMES)
        raise ValueError(
            f"quantile_predictions column {quantile_predictions!r} is required for requested "
            f"quantile/interval metric(s): {missing}"
        )


def _auto_metric_groups(
    frame: pd.DataFrame,
    *,
    variance_prediction: str,
    quantile_predictions: str,
    previous_actual: str,
) -> list[str]:
    groups: list[str] = []
    if previous_actual in frame.columns:
        groups.append("direction")
    if variance_prediction in frame.columns:
        groups.append("density")
    if quantile_predictions in frame.columns:
        groups.append("quantile_interval")
    return groups


def _positive_values(values: Any, *, label: str) -> np.ndarray:
    array = pd.Series(values).to_numpy(dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite values")
    if np.any(array <= 0.0):
        raise ValueError(f"{label} must be strictly positive")
    return array


def _nonnegative_values(values: Any, *, label: str) -> np.ndarray:
    array = pd.Series(values).to_numpy(dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite values")
    if np.any(array < 0.0):
        raise ValueError(f"{label} must be nonnegative")
    return array


def _validate_interval_bounds(lower: Any, upper: Any) -> None:
    lo = pd.Series(lower).to_numpy(dtype=float)
    hi = pd.Series(upper).to_numpy(dtype=float)
    if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        raise ValueError("interval bounds must contain only finite values")
    if np.any(hi < lo):
        raise ValueError(
            "interval upper bound must be greater than or equal to lower bound"
        )


def _metric_ascending(metric: str) -> bool:
    key = str(metric)
    higher_is_better = {
        "r2_oos",
        "mse_reduction",
        "success_ratio",
        "pesaran_timmermann_metric",
        "mean_return",
        "sharpe",
        "hac_sharpe",
        "sortino",
        "omega",
        "max_drawdown",
        "final_cumulative_return",
        "win_rate",
        "edge_ratio",
        "mean_edge",
        "frontier_win_share",
    }
    lower_is_better = {
        "mse",
        "msfe",
        "validation_mse",
        "rmse",
        "validation_rmse",
        "mae",
        "validation_mae",
        "mad",
        "medae",
        "median_absolute_error",
        "mape",
        "smape",
        "mase",
        "theil_u1",
        "theil_u2",
        "relative_mse",
        "relative_mae",
        "pinball_loss",
        "gaussian_nll",
        "negative_log_score",
        "log_score",
        "crps",
        "qlike",
        "interval_width",
        "interval_score",
    }
    if key in higher_is_better:
        return False
    if key == "coverage_rate" or key.startswith("coverage_"):
        raise ValueError(
            "ranking direction for coverage metrics is ambiguous; pass ascending explicitly "
            "or rank an interval score metric"
        )
    if (
        key in lower_is_better
        or key.startswith("pinball_loss_")
        or key.startswith("interval_width_")
        or key.startswith("interval_score_")
    ):
        return True
    if key in {"bias", "mean_error", "me"}:
        raise ValueError(
            "ranking direction for signed bias is ambiguous; pass ascending explicitly "
            "or rank an absolute-error metric"
        )
    raise ValueError(
        f"ranking direction for metric {metric!r} is unknown; pass ascending=True or ascending=False"
    )


def _quantile_evaluation(
    group: pd.DataFrame, *, actual: str, quantile_predictions: str
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for row_id, (idx, value) in enumerate(group[quantile_predictions].dropna().items()):
        if not isinstance(value, dict):
            raise ValueError(
                f"{quantile_predictions!r} values must be dictionaries mapping quantile levels "
                "to finite predictions"
            )
        if idx not in group.index:
            continue
        observed = group.at[idx, actual]
        if pd.isna(observed):
            continue
        for level, quantile_value in value.items():
            try:
                q = float(level)
                pred = float(quantile_value)
            except (TypeError, ValueError):
                raise ValueError(
                    f"{quantile_predictions!r} dictionaries must contain numeric quantile "
                    "levels and finite numeric predictions"
                ) from None
            if not 0.0 < q < 1.0:
                raise ValueError(
                    "quantile prediction levels must be strictly between 0 and 1"
                )
            if not np.isfinite(pred):
                raise ValueError("quantile prediction values must be finite")
            rows.append(
                {
                    "row_id": row_id,
                    "actual": float(observed),
                    "level": q,
                    "prediction": pred,
                }
            )
    if not rows:
        return {}
    table = pd.DataFrame(rows)
    out: dict[str, Any] = {"quantile_n": int(len(table))}
    for level, level_group in table.groupby("level", sort=True):
        label = _level_label(float(level))
        out[f"pinball_loss_{label}"] = pinball_loss(
            level_group["actual"], level_group["prediction"], quantile=float(level)
        )
    out.update(_interval_metrics(table))
    return out


def _interval_metrics(table: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    pivot = table.pivot_table(
        index="row_id",
        columns="level",
        values=["actual", "prediction"],
        aggfunc="first",
    )
    levels = sorted(float(level) for level in table["level"].unique())
    for lower in levels:
        upper = 1.0 - lower
        if lower >= upper or upper not in levels:
            continue
        try:
            actual_values = pivot[("actual", lower)]
            lower_values = pivot[("prediction", lower)]
            upper_values = pivot[("prediction", upper)]
        except KeyError:
            continue
        label = f"{_level_label(lower)}_{_level_label(upper)}"
        out[f"coverage_{label}"] = coverage_rate(
            actual_values, lower_values, upper_values
        )
        out[f"interval_width_{label}"] = interval_width(lower_values, upper_values)
        out[f"interval_score_{label}"] = interval_score(
            actual_values,
            lower_values,
            upper_values,
            alpha=max(min(2.0 * lower, 0.999999), 1e-6),
        )
    return out


def _level_label(level: float) -> str:
    return f"q{level:g}".replace(".", "_")


def _canonical_point_loss_name(loss: str, *, quantile: float | None = None) -> str:
    key = str(loss).lower()
    aliases = {
        "se": "squared_error",
        "squared": "squared_error",
        "squared_error": "squared_error",
        "mse": "squared_error",
        "msfe": "squared_error",
        "ae": "absolute_error",
        "absolute": "absolute_error",
        "absolute_error": "absolute_error",
        "mae": "absolute_error",
        "pinball": "pinball_loss",
        "pinball_loss": "pinball_loss",
        "negative_log_score": "negative_log_score",
        "log_score": "negative_log_score",
        "gaussian_nll": "negative_log_score",
        "nll": "negative_log_score",
        "qlike": "qlike",
    }
    if key not in aliases:
        allowed = ", ".join(sorted(aliases))
        raise ValueError(f"Unknown point loss {loss!r}. Available losses: {allowed}.")
    canonical = aliases[key]
    if canonical == "pinball_loss" and quantile is not None:
        return f"pinball_loss_{_level_label(float(quantile))}"
    return canonical


def _point_loss_from_frame(
    frame: pd.DataFrame,
    *,
    loss: str,
    actual: str,
    prediction: str,
    variance_prediction: str,
    quantile: float | None,
) -> pd.Series:
    canonical = _canonical_point_loss_name(loss, quantile=quantile)
    if canonical == "negative_log_score":
        _validate_table_columns(frame, (variance_prediction,), label="forecast table")
        values = compute_point_loss(
            frame[actual],
            frame[prediction],
            loss=loss,
            variance=frame[variance_prediction],
            quantile=quantile,
        )
    elif canonical == "qlike":
        _validate_table_columns(frame, (variance_prediction,), label="forecast table")
        values = compute_point_loss(
            frame[actual],
            frame[variance_prediction],
            loss=loss,
            quantile=quantile,
        )
    else:
        values = compute_point_loss(
            frame[actual],
            frame[prediction],
            loss=loss,
            quantile=quantile,
        )
    return values.reindex(frame.index)


def _risk_return_support_columns(
    frame: pd.DataFrame,
    groups: Sequence[str],
    *,
    support_cols: Sequence[str] | None,
) -> list[str]:
    if support_cols is not None:
        supports = [str(column) for column in support_cols]
        missing = [column for column in supports if column not in frame.columns]
        if missing:
            raise ValueError(
                f"support_cols are not present in the forecast table: {missing}"
            )
        return list(dict.fromkeys([*groups, *supports]))
    identity = [
        column for column in ("date", "origin", "origin_pos") if column in frame.columns
    ]
    if not identity:
        raise ValueError(
            "forecast returns require at least one support column: "
            "'date', 'origin', or 'origin_pos'"
        )
    return list(dict.fromkeys([*groups, *identity]))


def _validate_unique_support(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    model_col: str | None,
    label: str,
) -> None:
    keys = list(columns)
    if model_col is not None and model_col not in keys:
        keys.append(model_col)
    duplicates = frame.duplicated(keys, keep=False)
    if duplicates.any():
        example = frame.loc[duplicates, keys].head().to_dict(orient="records")
        raise ValueError(f"{label} rows are not unique on support columns: {example}")


def _matching_benchmark_rows(
    benchmark_indexed: pd.DataFrame,
    group: pd.DataFrame,
    *,
    supports: Sequence[str],
    groups: Sequence[str],
) -> pd.DataFrame:
    benchmark = benchmark_indexed.reset_index()
    if groups:
        first = group.iloc[0]
        mask = pd.Series(True, index=benchmark.index)
        for column in groups:
            mask &= benchmark[column] == first[column]
        benchmark = benchmark.loc[mask].copy()
    if benchmark.empty:
        key = {column: group.iloc[0][column] for column in groups}
        raise ValueError(f"benchmark has no matching support for group {key}")
    return benchmark.set_index(list(supports))


def _attach_return_path_columns(
    frame: pd.DataFrame,
    *,
    by: Sequence[str],
) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = _sort_return_frame(frame)
    group_keys = [column for column in by if column in out.columns]
    iterator = (
        out.groupby(group_keys, dropna=False, sort=False) if group_keys else [((), out)]
    )
    parts: list[pd.DataFrame] = []
    for _key, group in iterator:
        part = group.copy()
        part["cumulative_return"] = part["forecast_return"].astype(float).cumsum()
        part["drawdown"] = (
            part["cumulative_return"] - part["cumulative_return"].cummax()
        )
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _sort_return_frame(frame: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [
        column
        for column in ("date", "origin", "origin_pos", "horizon", "target", "model")
        if column in frame.columns
    ]
    if not sort_cols:
        return frame.reset_index(drop=True)
    out = frame.copy()
    if "date" in out.columns:
        out["_macroforecast_sort_date"] = pd.to_datetime(out["date"], errors="coerce")
        sort_cols = [
            "_macroforecast_sort_date" if column == "date" else column
            for column in sort_cols
        ]
    out = out.sort_values(sort_cols, kind="stable").drop(
        columns=["_macroforecast_sort_date"], errors="ignore"
    )
    return out.reset_index(drop=True)


def _return_values(returns: Any) -> np.ndarray:
    values = pd.Series(returns, dtype=float).dropna().to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _sample_sd(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1)) if values.size > 1 else float("nan")


def _hac_long_run_sd(values: np.ndarray, *, hac_lags: int | str | None) -> float:
    if values.size <= 1:
        return float("nan")
    lags = _resolve_hac_lags(values.size, hac_lags)
    demeaned = values - float(np.mean(values))
    gamma0 = float(np.mean(demeaned**2))
    long_run_var = gamma0
    for lag in range(1, min(lags, values.size - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        # Normalise the lag-k autocovariance by n (the full sample), not by the
        # overlap length n - k. np.mean over the length-(n-k) product divides by
        # n - k, which overweights higher lags relative to gamma0 (1/n) and the
        # standard Newey-West definition. This matches tests._long_run_variance.
        gamma = float(np.dot(demeaned[lag:], demeaned[:-lag]) / demeaned.size)
        long_run_var += 2.0 * weight * gamma
    return float(np.sqrt(max(long_run_var, 0.0)))


def _resolve_hac_lags(n_obs: int, hac_lags: int | str | None) -> int:
    if hac_lags is None:
        return 0
    if hac_lags == "auto":
        return max(1, int(np.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0))))
    value = int(hac_lags)
    if value < 0:
        raise ValueError("hac_lags must be nonnegative, 'auto', or None")
    return value


def _safe_ratio(
    numerator: float, denominator: float, *, multiplier: float = 1.0
) -> float:
    if np.isnan(denominator):
        return float("nan")
    if abs(denominator) <= 1e-15:
        if abs(numerator) <= 1e-15:
            return float("nan")
        return float(np.inf if numerator > 0 else -np.inf)
    return float((numerator / denominator) * multiplier)


def _risk_adjusted_row(
    values: np.ndarray,
    *,
    hac_lags: int | str | None,
    target_return: float,
    omega_threshold: float,
) -> dict[str, Any]:
    n_obs = int(values.size)
    if n_obs == 0:
        return {
            "n_obs": 0,
            "mean_return": float("nan"),
            "return_sd": float("nan"),
            "hac_return_sd": float("nan"),
            "sharpe": float("nan"),
            "hac_sharpe": float("nan"),
            "sortino": float("nan"),
            "omega": float("nan"),
            "max_drawdown": float("nan"),
            "final_cumulative_return": float("nan"),
            "win_rate": float("nan"),
        }
    mean = float(np.mean(values))
    sd = _sample_sd(values)
    hac_sd = (
        _hac_long_run_sd(values, hac_lags=hac_lags)
        if hac_lags is not None
        else float("nan")
    )
    cumulative = np.cumsum(values)
    return {
        "n_obs": n_obs,
        "mean_return": mean,
        "return_sd": sd,
        "hac_return_sd": hac_sd,
        "sharpe": _safe_ratio(mean, sd),
        "hac_sharpe": _safe_ratio(mean, hac_sd),
        "sortino": sortino_ratio(values, target_return=target_return),
        "omega": omega_ratio(values, threshold=omega_threshold),
        "max_drawdown": max_drawdown(values),
        "final_cumulative_return": float(cumulative[-1]),
        "win_rate": float(np.mean(values > 0.0)),
    }


def _default_risk_return_groups(frame: pd.DataFrame) -> list[str]:
    candidates = (
        "model_id",
        "benchmark_id",
        "target",
        "horizon",
        "sample",
        "sample_id",
        "regime",
        "loss_name",
    )
    groups = [column for column in candidates if column in frame.columns]
    if not groups and "model" in frame.columns:
        groups.append("model")
    return groups


def _risk_return_frame(
    rows: Sequence[Mapping[str, Any]],
    *,
    kind: str,
    metadata: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame.attrs["macroforecast_metadata_schema"] = {
        "kind": str(kind),
        "version": 1,
        **dict(metadata or {}),
    }
    return frame


__all__ = [
    "MetricLike",
    "DENSITY_METRIC_NAMES",
    "bias",
    "compute_point_loss",
    "coverage_rate",
    "crps",
    "drawdown_series",
    "edge_ratio",
    "evaluate_forecasts",
    "forecast_returns",
    "gaussian_nll",
    "get_metric",
    "interval_score",
    "interval_width",
    "log_score",
    "mae",
    "mad",
    "mape",
    "max_drawdown",
    "medae",
    "metric_kind",
    "mse",
    "mse_reduction",
    "negative_log_score",
    "omega_ratio",
    "pesaran_timmermann_metric",
    "pinball_loss",
    "qlike",
    "r2_oos",
    "rank_forecasts",
    "relative_mae",
    "relative_mse",
    "risk_adjusted_forecast_metrics",
    "rmse",
    "sharpe_ratio",
    "smape",
    "sortino_ratio",
    "success_ratio",
    "theil_u1",
    "theil_u2",
]
