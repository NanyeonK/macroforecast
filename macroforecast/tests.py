from __future__ import annotations

import json
import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TestResult:
    """Forecast comparison test result."""

    statistic: float | None
    p_value: float | None
    decision: bool
    alternative: str
    correction_policy: str | None = None
    n_obs: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def stat(self) -> float | None:
        return self.statistic

    @property
    def pvalue(self) -> float | None:
        return self.p_value

    def to_dict(self) -> dict[str, Any]:
        return _json_ready({
            "metadata_schema": {
                "kind": "forecast_test_result",
                "version": 1,
            },
            "statistic": self.statistic,
            "p_value": self.p_value,
            "decision": self.decision,
            "alternative": self.alternative,
            "correction_policy": self.correction_policy,
            "n_obs": self.n_obs,
            "metadata": dict(self.metadata),
        })

    def to_json(
        self,
        path: str | Path | None = None,
        *,
        indent: int | None = 2,
    ) -> str:
        """Return JSON text, and optionally write it to ``path``."""

        text = json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
        if path is not None:
            Path(path).write_text(text + "\n", encoding="utf-8")
        return text

    def summary(self) -> str:
        name = str(self.metadata.get("name", "Forecast comparison test"))
        return (
            f"{name}: statistic={self.statistic}, p_value={self.p_value}, "
            f"decision={self.decision}, alternative={self.alternative}"
        )


def dm_test(
    loss_a: Any,
    loss_b: Any,
    *,
    horizon: int = 1,
    correction: str = "hln",
    kernel: str = "newey_west",
    alpha: float = 0.05,
) -> TestResult:
    """Diebold-Mariano equal predictive ability test."""

    _validate_alpha(alpha)
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    frame = _aligned_frame(loss_a, loss_b, names=("loss_a", "loss_b"))
    diff = frame["loss_a"] - frame["loss_b"]
    hln = _normalize_correction(correction) == "hln"
    stat, p_value = _diebold_mariano_stat(diff, horizon=horizon, hln=hln, kernel=kernel)
    return TestResult(
        statistic=stat,
        p_value=p_value,
        decision=p_value is not None and p_value < alpha,
        alternative="two_sided",
        correction_policy="hln_nw" if hln else "nw",
        n_obs=int(diff.notna().sum()),
        metadata={
            "name": "Diebold-Mariano",
            "horizon": int(horizon),
            "hln_correction": hln,
            "hac_kernel": kernel,
            "mean_loss_difference": _float_or_none(diff.mean()),
        },
    )


def gw_test(
    loss_a: Any,
    loss_b: Any,
    *,
    horizon: int = 1,
    correction: str = "hln",
    kernel: str = "newey_west",
    alpha: float = 0.05,
) -> TestResult:
    """Giacomini-White-style equal predictive ability callable.

    This callable keeps the legacy GW surface. It uses the same HAC loss
    differential statistic as the legacy implementation.
    """

    result = dm_test(
        loss_a,
        loss_b,
        horizon=horizon,
        correction=correction,
        kernel=kernel,
        alpha=alpha,
    )
    return _replace_metadata(result, name="Giacomini-White")


def dmp_test(
    loss_differences: Any,
    *,
    kernel: str = "newey_west",
    alpha: float = 0.05,
) -> TestResult:
    """Diebold-Mariano-Pesaran joint multi-horizon test on stacked losses."""

    _validate_alpha(alpha)
    if isinstance(loss_differences, Sequence) and not isinstance(
        loss_differences, (pd.Series, np.ndarray, str, bytes)
    ):
        arrays = [np.asarray(values, dtype=float).reshape(-1) for values in loss_differences]
        values = np.concatenate(arrays) if arrays else np.asarray([], dtype=float)
    else:
        values = np.asarray(loss_differences, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n < 3:
        return TestResult(None, None, False, "two_sided", "nw", n, {"name": "DMP"})
    mean_diff = float(values.mean())
    lr_var = _long_run_variance(values - mean_diff, kernel=kernel)
    se = float(np.sqrt(max(lr_var / n, 1e-12)))
    stat = mean_diff / se if se > 0 else 0.0
    p_value = _normal_two_sided_p(stat)
    return TestResult(
        statistic=float(stat),
        p_value=p_value,
        decision=p_value is not None and p_value < alpha,
        alternative="two_sided",
        correction_policy="nw",
        n_obs=n,
        metadata={
            "name": "Diebold-Mariano-Pesaran",
            "n_obs_stacked": n,
            "mean_loss_difference": mean_diff,
            "hac_kernel": kernel,
        },
    )


def harvey_newbold_test(
    error_a: Any,
    error_b: Any,
    *,
    horizon: int = 1,
    kernel: str = "newey_west",
    small_sample: bool = True,
    alpha: float = 0.05,
) -> TestResult:
    """Harvey-Leybourne-Newbold forecast encompassing test."""

    _validate_alpha(alpha)
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    frame = _aligned_frame(error_a, error_b, names=("error_a", "error_b"))
    e_a = frame["error_a"].to_numpy(dtype=float)
    e_b = frame["error_b"].to_numpy(dtype=float)
    d = e_a * (e_a - e_b)
    finite = np.isfinite(d)
    n = int(finite.sum())
    if n < 5:
        return TestResult(None, None, False, "one_sided", "hln_nw", n, {"name": "Harvey-Newbold"})
    d_clean = d[finite]
    d_bar = float(np.mean(d_clean))
    lag = max(int(horizon) - 1, 0)
    lr_var = _long_run_variance(d_clean - d_bar, kernel=kernel, lag=lag)
    se = float(np.sqrt(max(lr_var / max(n, 1), 1e-12)))
    stat = d_bar / se if se > 0 else 0.0
    if small_sample:
        factor = (n + 1 - 2 * horizon + horizon * (horizon - 1) / max(n, 1)) / max(n, 1)
        stat *= float(np.sqrt(max(factor, 1e-12)))
    from scipy import stats as _stats

    p_value = float(1.0 - _stats.t.cdf(stat, df=max(n - 1, 1)))
    return TestResult(
        statistic=float(stat),
        p_value=p_value,
        decision=p_value < alpha,
        alternative="one_sided",
        correction_policy="hln_nw",
        n_obs=n,
        metadata={
            "name": "Harvey-Newbold",
            "encompassing": "a_over_b",
            "mean_d": float(np.nanmean(d_clean)),
            "hac_kernel": kernel,
            "small_sample": bool(small_sample),
        },
    )


def hn_test(*args: Any, **kwargs: Any) -> TestResult:
    """Alias for :func:`harvey_newbold_test`."""

    return harvey_newbold_test(*args, **kwargs)


def custom_test(
    name: str,
    func: Callable[..., Any],
    *args: Any,
    alternative: str = "two_sided",
    alpha: float = 0.05,
    correction_policy: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    **params: Any,
) -> TestResult:
    """Run a user-supplied forecast test and coerce it to ``TestResult``."""

    if not name:
        raise ValueError("custom test name must be non-empty")
    if not callable(func):
        raise TypeError("custom test func must be callable")
    _validate_alpha(alpha)
    output = func(*args, **params)
    result = _coerce_custom_test_output(
        output,
        name=str(name),
        alternative=alternative,
        alpha=alpha,
        correction_policy=correction_policy,
    )
    merged_metadata = {
        **dict(result.metadata),
        **dict(metadata or {}),
        "name": str(name),
        "callable": _callable_name(func),
        "params": _json_ready(params),
        "alpha": float(alpha),
        "custom": True,
    }
    return TestResult(
        result.statistic,
        result.p_value,
        result.decision,
        result.alternative,
        result.correction_policy,
        result.n_obs,
        merged_metadata,
    )


def clark_west_test(
    loss_small: Any,
    loss_large: Any,
    forecast_small: Any,
    forecast_large: Any,
    *,
    horizon: int = 1,
    cw_adjustment: bool = True,
    kernel: str = "newey_west",
    alpha: float = 0.05,
) -> TestResult:
    """Clark-West nested forecast comparison test."""

    _validate_alpha(alpha)
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    frame = _aligned_frame(
        loss_small,
        loss_large,
        forecast_small,
        forecast_large,
        names=("loss_small", "loss_large", "forecast_small", "forecast_large"),
    )
    improvement = frame["loss_small"] - frame["loss_large"]
    if cw_adjustment:
        improvement = improvement + (frame["forecast_small"] - frame["forecast_large"]) ** 2
    stat, p_two = _diebold_mariano_stat(improvement, horizon=horizon, hln=False, kernel=kernel)
    p_value = _one_sided_from_two_sided(stat, p_two)
    return TestResult(
        statistic=stat,
        p_value=p_value,
        decision=p_value is not None and p_value < alpha,
        alternative="one_sided",
        correction_policy="nw",
        n_obs=int(improvement.notna().sum()),
        metadata={
            "name": "Clark-West",
            "cw_adjustment": bool(cw_adjustment),
            "mean_adjusted_improvement": _float_or_none(improvement.mean()),
        },
    )


def cw_test(*args: Any, **kwargs: Any) -> TestResult:
    """Alias for :func:`clark_west_test`."""

    return clark_west_test(*args, **kwargs)


def enc_new_test(
    loss_small: Any,
    loss_large: Any,
    *,
    horizon: int = 1,
    kernel: str = "newey_west",
    alpha: float = 0.05,
) -> TestResult:
    """ENC-NEW nested forecast encompassing test."""

    _validate_alpha(alpha)
    return _nested_loss_improvement_test(
        loss_small,
        loss_large,
        horizon=horizon,
        kernel=kernel,
        alpha=alpha,
        name="Enc-New",
    )


def enc_t_test(
    loss_small: Any,
    loss_large: Any,
    *,
    horizon: int = 1,
    kernel: str = "newey_west",
    alpha: float = 0.05,
) -> TestResult:
    """ENC-T nested forecast encompassing test."""

    _validate_alpha(alpha)
    return _nested_loss_improvement_test(
        loss_small,
        loss_large,
        horizon=horizon,
        kernel=kernel,
        alpha=alpha,
        name="Enc-T",
    )


def directional_accuracy_test(
    y_true: Any,
    y_pred: Any,
    *,
    threshold: float = 0.0,
    method: str = "pesaran_timmermann",
    alpha: float = 0.05,
) -> TestResult:
    """Pesaran-Timmermann or Henriksson-Merton directional accuracy test."""

    _validate_alpha(alpha)
    method = _normalize_direction_method(method)
    frame = _aligned_frame(y_true, y_pred, names=("truth", "pred"))
    forecast = (frame["pred"].to_numpy(dtype=float) > threshold).astype(int)
    actual = (frame["truth"].to_numpy(dtype=float) > threshold).astype(int)
    stat, p_value, success = _direction_test_stat(forecast, actual, method=method)
    return TestResult(
        statistic=stat,
        p_value=p_value,
        decision=p_value is not None and p_value < alpha,
        alternative="two_sided",
        correction_policy=None,
        n_obs=int(len(frame)),
        metadata={
            "name": "Directional accuracy",
            "method": method,
            "threshold": float(threshold),
            "success_ratio": success,
        },
    )


def pesaran_timmermann_test(*args: Any, **kwargs: Any) -> TestResult:
    """Pesaran-Timmermann directional accuracy test."""

    kwargs.setdefault("method", "pesaran_timmermann")
    return directional_accuracy_test(*args, **kwargs)


def henriksson_merton_test(*args: Any, **kwargs: Any) -> TestResult:
    """Henriksson-Merton directional accuracy test."""

    kwargs.setdefault("method", "henriksson_merton")
    return directional_accuracy_test(*args, **kwargs)


def density_interval_tests(
    pit: Any,
    *,
    alpha: float = 0.05,
    n_bins: int = 10,
    pit_lag: int = 1,
) -> dict[str, Any]:
    """Berkowitz, KS, Kupiec, Christoffersen, and DQ tests for PIT values."""

    from scipy import stats as _stats

    _validate_alpha(alpha)
    n_bins = _validate_positive_int(n_bins, "n_bins")
    pit_lag = _validate_positive_int(pit_lag, "pit_lag")
    values = np.clip(pd.Series(pit).dropna().to_numpy(dtype=float), 1e-9, 1 - 1e-9)
    if values.size == 0:
        return {
            "metadata_schema": {
                "kind": "density_interval_tests",
                "version": 1,
            },
            "status": "empty",
            "n_obs": 0,
            "n_bins": int(n_bins),
        }
    z = _stats.norm.ppf(values)
    z_mean = z.mean()
    z_std = z.std(ddof=1) if z.size > 1 else 1.0
    berkowitz: dict[str, Any] = {"mean": float(z_mean), "std": float(z_std)}
    if z.size >= 4 and z_std > 0:
        z_lag = z[:-1]
        z_lead = z[1:]
        denom = float(np.dot(z_lag - z_lag.mean(), z_lag - z_lag.mean()))
        rho_hat = (
            float(np.dot(z_lag - z_lag.mean(), z_lead - z_lead.mean()) / denom)
            if denom > 0
            else 0.0
        )
        mu_hat = float(z_lead.mean() - rho_hat * z_lag.mean())
        resid = z_lead - mu_hat - rho_hat * z_lag
        sigma_hat = float(np.sqrt(max(float(np.mean(resid**2)), 1e-10)))
        ll_h0 = float(_stats.norm.logpdf(z, loc=0.0, scale=1.0).sum())
        ll_full = float(_stats.norm.logpdf(z[0], loc=0.0, scale=1.0)) + float(
            _stats.norm.logpdf(z_lead, loc=mu_hat + rho_hat * z_lag, scale=sigma_hat).sum()
        )
        lr = max(-2.0 * (ll_h0 - ll_full), 0.0)
        berkowitz.update(
            {
                "lr_statistic": float(lr),
                "p_value": float(1.0 - _stats.chi2.cdf(lr, df=3)),
                "reject": bool(1.0 - _stats.chi2.cdf(lr, df=3) < alpha),
                "rho": rho_hat,
                "mu": mu_hat,
                "sigma": sigma_hat,
                "df": 3,
            }
        )
    elif z.size >= 2 and z_std > 0:
        ll_h1 = float(_stats.norm.logpdf(z, loc=z_mean, scale=z_std).sum())
        ll_h0 = float(_stats.norm.logpdf(z, loc=0.0, scale=1.0).sum())
        lr = max(-2.0 * (ll_h0 - ll_h1), 0.0)
        berkowitz.update(
            {
                "lr_statistic": float(lr),
                "p_value": float(1.0 - _stats.chi2.cdf(lr, df=2)),
                "reject": bool(1.0 - _stats.chi2.cdf(lr, df=2) < alpha),
                "rho": None,
                "df": 2,
            }
        )
    ks_stat, ks_pvalue = _stats.kstest(values, "uniform")
    hits = (values < alpha).astype(int)
    kupiec = _kupiec_test(hits, alpha)
    christoffersen = _christoffersen_independence_test(hits, alpha)
    dq = _engle_manganelli_dq_test(hits, alpha)
    return _json_ready({
        "metadata_schema": {
            "kind": "density_interval_tests",
            "version": 1,
        },
        "berkowitz": berkowitz,
        "ks": {
            "statistic": float(ks_stat),
            "p_value": float(ks_pvalue),
            "reject": bool(ks_pvalue < alpha),
        },
        "kupiec_pof": kupiec,
        "christoffersen_independence": christoffersen,
        "engle_manganelli_dq": dq,
        "pit_histogram": pit_histogram(values, n_bins=n_bins).to_dict(orient="records"),
        "pit_autocorrelation": pit_autocorrelation_test(values, lag=pit_lag, alpha=alpha).to_dict(),
        "n_obs": int(values.size),
        "n_bins": int(n_bins),
    })


def pit_histogram(pit: Any, *, n_bins: int = 10) -> pd.DataFrame:
    """Return PIT histogram counts against a uniform reference."""

    n_bins = _validate_positive_int(n_bins, "n_bins")
    values = np.clip(pd.Series(pit).dropna().to_numpy(dtype=float), 0.0, 1.0)
    counts, edges = np.histogram(values, bins=n_bins, range=(0.0, 1.0))
    expected = values.size / n_bins if n_bins else 0.0
    rows = [
        {
            "bin": int(idx + 1),
            "lower": float(edges[idx]),
            "upper": float(edges[idx + 1]),
            "count": int(count),
            "expected_count": float(expected),
            "deviation": float(count - expected),
        }
        for idx, count in enumerate(counts)
    ]
    out = pd.DataFrame(rows)
    out.attrs["macroforecast_metadata_schema"] = {
        "kind": "pit_histogram",
        "version": 1,
        "n_bins": int(n_bins),
        "n_obs": int(values.size),
    }
    return out


def pit_autocorrelation_test(
    pit: Any,
    *,
    lag: int = 1,
    alpha: float = 0.05,
) -> TestResult:
    """Normal approximation test for serial dependence in PIT values."""

    _validate_alpha(alpha)
    lag = _validate_positive_int(lag, "lag")
    values = pd.Series(pit).astype(float).dropna().to_numpy(dtype=float)
    n = int(values.size)
    if n <= lag + 2:
        return TestResult(None, None, False, "two_sided", None, n, {"name": "PIT autocorrelation", "lag": lag})
    left = values[lag:]
    right = values[:-lag]
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return TestResult(None, None, False, "two_sided", None, n, {"name": "PIT autocorrelation", "lag": lag})
    rho = float(np.corrcoef(left, right)[0, 1])
    statistic = rho * np.sqrt(n - lag)
    p_value = _normal_two_sided_p(statistic)
    return TestResult(
        statistic=float(statistic),
        p_value=p_value,
        decision=p_value is not None and p_value < alpha,
        alternative="two_sided",
        correction_policy="normal_approximation",
        n_obs=n,
        metadata={"name": "PIT autocorrelation", "lag": lag, "autocorrelation": rho},
    )


def interval_coverage_test(
    y_true: Any,
    lower: Any,
    upper: Any,
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Kupiec and Christoffersen coverage diagnostics for forecast intervals."""

    _validate_alpha(alpha)
    frame = _aligned_frame(y_true, lower, upper, names=("truth", "lower", "upper"))
    truth = frame["truth"].to_numpy(dtype=float)
    lo = frame["lower"].to_numpy(dtype=float)
    hi = frame["upper"].to_numpy(dtype=float)
    misses = ((truth < lo) | (truth > hi)).astype(int)
    coverage = float(1.0 - misses.mean()) if misses.size else None
    kupiec = _kupiec_test(misses, alpha)
    christoffersen = _christoffersen_independence_test(misses, alpha)
    conditional_stat = None
    conditional_p = None
    if kupiec.get("lr_statistic") is not None and christoffersen.get("lr_statistic") is not None:
        from scipy import stats as _stats

        conditional_stat = float(kupiec["lr_statistic"] + christoffersen["lr_statistic"])
        conditional_p = float(1.0 - _stats.chi2.cdf(conditional_stat, df=2))
    return _json_ready(
        {
            "metadata_schema": {
                "kind": "interval_coverage_test",
                "version": 1,
            },
            "coverage_rate": coverage,
            "expected_coverage": float(1.0 - alpha),
            "miss_rate": None if coverage is None else float(1.0 - coverage),
            "n_obs": int(misses.size),
            "kupiec_pof": kupiec,
            "christoffersen_independence": christoffersen,
            "christoffersen_conditional_coverage": {
                "lr_statistic": conditional_stat,
                "p_value": conditional_p,
                "reject": bool(conditional_p is not None and conditional_p < alpha),
            },
        }
    )


def residual_diagnostics(
    residuals: Any,
    *,
    tests: Sequence[str] = (
        "ljung_box_q",
        "arch_lm",
        "jarque_bera_normality",
        "durbin_watson",
    ),
    lag: int = 10,
) -> pd.DataFrame:
    """Run residual diagnostic tests and return one row per test."""

    values = pd.Series(residuals).astype(float).dropna()
    rows = []
    for test_name in tests:
        stat, p_value = _residual_test_statistic(test_name, values, lag)
        rows.append(
            {
                "test": test_name,
                "statistic": stat,
                "p_value": p_value,
                "lag_used": min(lag, max(len(values) - 1, 0)),
                "n_obs": int(len(values)),
            }
        )
    frame = pd.DataFrame(rows)
    frame.attrs["macroforecast_metadata_schema"] = {
        "kind": "residual_diagnostics",
        "version": 1,
        "row_unit": "test",
    }
    return frame


def equal_predictive_tests(
    loss_a: Any,
    loss_b: Any,
    *,
    tests: Sequence[str] = ("dm", "gw", "dmp"),
    error_a: Any | None = None,
    error_b: Any | None = None,
    horizon: int = 1,
    correction: str = "hln",
    kernel: str = "newey_west",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run multiple equal-predictive-ability tests and stack results."""

    rows: list[dict[str, Any]] = []
    for name in tests:
        key = str(name).lower().replace("-", "_")
        if key in {"dm", "dm_test", "diebold_mariano"}:
            result = dm_test(loss_a, loss_b, horizon=horizon, correction=correction, kernel=kernel, alpha=alpha)
        elif key in {"gw", "gw_test", "giacomini_white"}:
            result = gw_test(loss_a, loss_b, horizon=horizon, correction=correction, kernel=kernel, alpha=alpha)
        elif key in {"dmp", "dmp_test", "diebold_mariano_pesaran"}:
            diff = pd.Series(loss_a).sub(pd.Series(loss_b))
            result = dmp_test(diff, kernel=kernel, alpha=alpha)
        elif key in {"hn", "harvey_newbold", "harvey_newbold_test"}:
            if error_a is None or error_b is None:
                raise ValueError("error_a and error_b are required for harvey_newbold in equal_predictive_tests")
            result = harvey_newbold_test(error_a, error_b, horizon=horizon, kernel=kernel, alpha=alpha)
        else:
            raise ValueError(f"unknown equal predictive test {name!r}")
        rows.append(_test_result_row(result, requested_name=str(name)))
    out = pd.DataFrame(rows)
    out.attrs["macroforecast_metadata_schema"] = {
        "kind": "equal_predictive_tests",
        "version": 1,
        "tests": [str(name) for name in tests],
    }
    return out


def nested_tests(
    loss_small: Any,
    loss_large: Any,
    *,
    forecast_small: Any | None = None,
    forecast_large: Any | None = None,
    tests: Sequence[str] = ("clark_west", "enc_new", "enc_t"),
    horizon: int = 1,
    kernel: str = "newey_west",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run multiple nested-model forecast tests and stack results."""

    rows: list[dict[str, Any]] = []
    for name in tests:
        key = str(name).lower().replace("-", "_")
        if key in {"clark_west", "cw", "cw_test"}:
            if forecast_small is None or forecast_large is None:
                raise ValueError("forecast_small and forecast_large are required for clark_west in nested_tests")
            result = clark_west_test(
                loss_small,
                loss_large,
                forecast_small,
                forecast_large,
                horizon=horizon,
                kernel=kernel,
                alpha=alpha,
            )
        elif key in {"enc_new", "enc_new_test"}:
            result = enc_new_test(loss_small, loss_large, horizon=horizon, kernel=kernel, alpha=alpha)
        elif key in {"enc_t", "enc_t_test"}:
            result = enc_t_test(loss_small, loss_large, horizon=horizon, kernel=kernel, alpha=alpha)
        else:
            raise ValueError(f"unknown nested test {name!r}")
        rows.append(_test_result_row(result, requested_name=str(name)))
    out = pd.DataFrame(rows)
    out.attrs["macroforecast_metadata_schema"] = {
        "kind": "nested_tests",
        "version": 1,
        "tests": [str(name) for name in tests],
    }
    return out


def conditional_predictive_ability_test(
    loss_a: Any,
    loss_b: Any,
    *,
    method: str = "giacomini_rossi",
    window_ratio: float = 0.25,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Giacomini-Rossi rolling or Rossi-Sekhposyan recursive fluctuation test."""

    _validate_alpha(alpha)
    if not 0.0 < window_ratio <= 1.0:
        raise ValueError("window_ratio must be in (0, 1]")
    method = _normalize_cpa_method(method)
    frame = _aligned_frame(loss_a, loss_b, names=("loss_a", "loss_b"))
    diff = (frame["loss_a"] - frame["loss_b"]).to_numpy(dtype=float)
    centered = diff - diff.mean()
    n = centered.size
    if n == 0:
        return {
            "metadata_schema": {
                "kind": "conditional_predictive_ability",
                "version": 1,
            },
            "statistic": None,
            "critical_value": None,
            "decision": None,
            "n_obs": 0,
            "method": method,
        }
    m = min(n, max(4, int(round(window_ratio * n))))
    critical = _gr_critical_value(window_ratio, alpha)
    stats = []
    for end in range(m, n + 1):
        window = (
            centered[end - m : end]
            if method == "giacomini_rossi"
            else centered[:end]
        )
        se = _newey_west_se(window, max(1, int(np.ceil(m ** (1 / 3)))))
        if se > 0:
            stats.append(float(window.mean() / se))
    supremum = float(max(abs(value) for value in stats)) if stats else 0.0
    return _json_ready({
        "metadata_schema": {
            "kind": "conditional_predictive_ability",
            "version": 1,
        },
        "statistic": supremum,
        "critical_value": critical,
        "decision": bool(supremum > critical),
        "time_path": stats,
        "window_size": m,
        "n_obs": int(n),
        "method": method,
    })


def model_confidence_set(
    loss_panel: pd.DataFrame,
    *,
    loss: str = "squared_error",
    alpha: float = 0.10,
    n_boot: int = 1000,
    block_length: int | str = "auto",
    bootstrap_method: str = "stationary_bootstrap",
    spa_benchmark_model: str | None = None,
    random_state: int = 0,
    target: str = "target",
    horizon: str = "horizon",
    origin: str = "origin",
    model: str = "model_id",
) -> dict[str, Any]:
    """Single-step MCS approximation plus SPA, Reality Check, and StepM."""

    _validate_alpha(alpha)
    n_boot = _validate_positive_int(n_boot, "n_boot")
    bootstrap_method = _normalize_bootstrap_method(bootstrap_method)
    panel = pd.DataFrame(loss_panel).copy()
    required = {target, horizon, origin, model, loss}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"loss_panel missing required columns: {sorted(missing)}")
    rng = np.random.default_rng(random_state)
    mcs: dict[tuple[Any, ...], set[str]] = {}
    spa: dict[tuple[Any, ...], float] = {}
    reality: dict[tuple[Any, ...], float] = {}
    stepm: dict[tuple[Any, ...], set[str]] = {}
    block_lengths_used: dict[tuple[Any, ...], int] = {}
    for (target_value, horizon_value), group in panel.groupby([target, horizon]):
        wide = group.pivot_table(index=origin, columns=model, values=loss, aggfunc="mean").sort_index()
        wide = wide.dropna(axis=0, how="any")
        if wide.shape[0] < 4 or wide.shape[1] < 2:
            key = (target_value, horizon_value, alpha)
            mcs[key] = {str(column) for column in wide.columns}
            spa[(target_value, horizon_value)] = 1.0
            reality[(target_value, horizon_value)] = 1.0
            stepm[key] = set()
            continue
        matrix = wide.to_numpy(dtype=float)
        n_obs, n_models = matrix.shape
        block = _resolve_block_length(matrix, block_length)
        block_lengths_used[(target_value, horizon_value)] = int(block)
        means = matrix.mean(axis=0)
        centered = matrix - means
        boot_means = _bootstrap_means(
            centered,
            n_boot=n_boot,
            block_length=block,
            method=bootstrap_method,
            rng=rng,
        )
        observed_t, boot_t = _studentized_deviation(means, boot_means)
        critical = float(np.quantile(boot_t.max(axis=1), 1.0 - alpha))
        mcs_set = {
            str(model_name)
            for model_name, t_stat in zip(wide.columns, observed_t)
            if t_stat <= critical
        }
        if not mcs_set:
            mcs_set = {str(wide.columns[int(np.argmin(observed_t))])}
        spa_p = _spa_p_value(
            wide,
            means,
            boot_means,
            benchmark=spa_benchmark_model,
        )
        stepm_set = _stepm_rejected(
            wide.columns,
            means,
            centered,
            alpha=alpha,
            n_boot=n_boot,
            block_length=block,
            rng=rng,
        )
        key = (target_value, horizon_value, alpha)
        mcs[key] = mcs_set
        spa[(target_value, horizon_value)] = spa_p
        reality[(target_value, horizon_value)] = spa_p
        stepm[key] = stepm_set
    return _json_ready({
        "metadata_schema": {
            "kind": "model_confidence_set",
            "version": 1,
        },
        "mcs_inclusion": _mapping_records(
            mcs,
            key_names=("target", "horizon", "alpha"),
            value_name="models",
        ),
        "spa_p_values": _mapping_records(
            spa,
            key_names=("target", "horizon"),
            value_name="p_value",
        ),
        "reality_check_p_values": _mapping_records(
            reality,
            key_names=("target", "horizon"),
            value_name="p_value",
        ),
        "stepm_rejected": _mapping_records(
            stepm,
            key_names=("target", "horizon", "alpha"),
            value_name="models",
        ),
        "bootstrap_n_replications": int(n_boot),
        "block_length": block_length,
        "block_lengths_used": _mapping_records(
            block_lengths_used,
            key_names=("target", "horizon"),
            value_name="block_length",
        ),
        "bootstrap_kind": "stationary_block_bootstrap_per_origin",
    })


def blocked_oob_reality_check(
    loss_panel: pd.DataFrame,
    *,
    benchmark: str,
    loss: str = "squared_error",
    alpha: float = 0.05,
    n_boot: int = 1000,
    block_length: int | str = 4,
    bootstrap_method: str = "fixed_block_bootstrap",
    random_state: int = 0,
    target: str = "target",
    horizon: str = "horizon",
    origin: str = "origin",
    model: str = "model_id",
) -> pd.DataFrame:
    """Block-bootstrap one-sided reality check against a benchmark model.

    The input can be either a long per-origin loss panel with model and loss
    columns, or a wide loss matrix whose columns are model names. Positive
    ``mean_diff`` means the candidate has lower average loss than the
    benchmark.
    """

    _validate_alpha(alpha)
    n_boot = _validate_positive_int(n_boot, "n_boot")
    bootstrap_method = _normalize_bootstrap_method(bootstrap_method)
    panel = pd.DataFrame(loss_panel).copy()
    groups = _reality_check_groups(
        panel,
        benchmark=benchmark,
        loss=loss,
        target=target,
        horizon=horizon,
        origin=origin,
        model=model,
    )
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, Any]] = []
    for target_value, horizon_value, wide in groups:
        if benchmark not in wide.columns:
            raise ValueError(
                f"benchmark {benchmark!r} not in loss columns {list(wide.columns)}"
            )
        numeric = wide.apply(pd.to_numeric, errors="coerce")
        for candidate in numeric.columns:
            if candidate == benchmark:
                continue
            pair = numeric[[benchmark, candidate]].dropna()
            if pair.shape[0] < 2:
                continue
            diff = pair[benchmark].to_numpy(dtype=float) - pair[candidate].to_numpy(dtype=float)
            mean_diff = float(diff.mean())
            resolved_block = _resolve_block_length(diff.reshape(-1, 1), block_length)
            centered = (diff - mean_diff).reshape(-1, 1)
            boot_means = _bootstrap_means(
                centered,
                n_boot=n_boot,
                block_length=resolved_block,
                method=bootstrap_method,
                rng=rng,
            )[:, 0]
            boot_se = float(np.std(boot_means, ddof=1))
            statistic = mean_diff / max(boot_se, 1e-12)
            p_value = float(np.mean(boot_means >= mean_diff))
            rows.append(
                {
                    "target": target_value,
                    "horizon": horizon_value,
                    "model": str(candidate),
                    "benchmark": str(benchmark),
                    "mean_diff": mean_diff,
                    "statistic": float(statistic),
                    "p_value": p_value,
                    "decision": bool(p_value < alpha),
                    "n_obs": int(pair.shape[0]),
                    "alpha": float(alpha),
                    "block_length": int(resolved_block),
                    "n_boot": int(n_boot),
                    "bootstrap_method": bootstrap_method,
                }
            )
    out = pd.DataFrame(rows)
    out.attrs["macroforecast_metadata_schema"] = {
        "kind": "blocked_oob_reality_check",
        "version": 1,
        "benchmark": str(benchmark),
        "loss": str(loss),
        "alpha": float(alpha),
        "n_boot": int(n_boot),
        "block_length": block_length,
        "bootstrap_method": bootstrap_method,
        "positive_mean_diff": "candidate lower loss than benchmark",
    }
    return out


def _nested_loss_improvement_test(
    loss_small: Any,
    loss_large: Any,
    *,
    horizon: int,
    kernel: str,
    alpha: float,
    name: str,
) -> TestResult:
    _validate_alpha(alpha)
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    frame = _aligned_frame(loss_small, loss_large, names=("loss_small", "loss_large"))
    improvement = frame["loss_small"] - frame["loss_large"]
    stat, p_two = _diebold_mariano_stat(improvement, horizon=horizon, hln=False, kernel=kernel)
    p_value = _one_sided_from_two_sided(stat, p_two)
    return TestResult(
        statistic=stat,
        p_value=p_value,
        decision=p_value is not None and p_value < alpha,
        alternative="one_sided",
        correction_policy="nw",
        n_obs=int(improvement.notna().sum()),
        metadata={"name": name, "mean_adjusted_improvement": _float_or_none(improvement.mean())},
    )


def _aligned_frame(*series: Any, names: Sequence[str]) -> pd.DataFrame:
    frames = [pd.Series(value).astype(float).rename(name) for value, name in zip(series, names)]
    joined = pd.concat(frames, axis=1).dropna()
    if joined.empty:
        raise ValueError("inputs must have aligned non-missing observations")
    return joined


def _validate_alpha(alpha: float) -> None:
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be in (0, 1)")


def _validate_positive_int(value: int, name: str) -> int:
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    return value


def _normalize_direction_method(method: str) -> str:
    key = str(method).lower().replace("-", "_")
    if key in {"pesaran_timmermann", "pt"}:
        return "pesaran_timmermann"
    if key in {"henriksson_merton", "hm"}:
        return "henriksson_merton"
    raise ValueError("method must be 'pesaran_timmermann' or 'henriksson_merton'")


def _normalize_cpa_method(method: str) -> str:
    key = str(method).lower().replace("-", "_")
    if key in {"giacomini_rossi", "giacomini_rossi_2010", "gr"}:
        return "giacomini_rossi"
    if key in {"rossi_sekhposyan", "rs"}:
        return "rossi_sekhposyan"
    raise ValueError("method must be 'giacomini_rossi' or 'rossi_sekhposyan'")


def _normalize_bootstrap_method(method: str) -> str:
    key = str(method).lower().replace("-", "_")
    if key in {"stationary", "stationary_bootstrap"}:
        return "stationary_bootstrap"
    if key in {"fixed", "fixed_block", "fixed_block_bootstrap", "moving_block"}:
        return "fixed_block_bootstrap"
    raise ValueError("bootstrap_method must be 'stationary_bootstrap' or 'fixed_block_bootstrap'")


def _reality_check_groups(
    panel: pd.DataFrame,
    *,
    benchmark: str,
    loss: str,
    target: str,
    horizon: str,
    origin: str,
    model: str,
) -> list[tuple[Any, Any, pd.DataFrame]]:
    long_required = {origin, model, loss}
    if long_required <= set(panel.columns):
        working = panel.copy()
        group_columns: list[str] = []
        if target in working.columns:
            group_columns.append(target)
        else:
            working[target] = "all"
            group_columns.append(target)
        if horizon in working.columns:
            group_columns.append(horizon)
        else:
            working[horizon] = "all"
            group_columns.append(horizon)
        groups: list[tuple[Any, Any, pd.DataFrame]] = []
        for key, group in working.groupby(group_columns, dropna=False):
            key_tuple = key if isinstance(key, tuple) else (key,)
            wide = (
                group.pivot_table(
                    index=origin,
                    columns=model,
                    values=loss,
                    aggfunc="mean",
                )
                .sort_index()
                .rename_axis(columns=None)
            )
            groups.append((key_tuple[0], key_tuple[1], wide))
        return groups
    if benchmark not in panel.columns:
        raise ValueError(
            "loss_panel must be either a long panel with "
            f"{sorted(long_required)!r} columns or a wide matrix containing "
            f"benchmark column {benchmark!r}"
        )
    wide = panel.copy()
    if origin in wide.columns:
        wide = wide.set_index(origin)
    return [("all", "all", wide)]


def _normalize_correction(correction: str) -> str:
    key = str(correction).lower().replace("-", "_")
    if key in {"hln", "harvey_leybourne_newbold", "small_sample"}:
        return "hln"
    if key in {"none", "nw", "newey_west"}:
        return "none"
    raise ValueError("correction must be 'hln' or 'none'")


def _replace_metadata(result: TestResult, **metadata: Any) -> TestResult:
    merged = dict(result.metadata)
    merged.update(metadata)
    return TestResult(
        result.statistic,
        result.p_value,
        result.decision,
        result.alternative,
        result.correction_policy,
        result.n_obs,
        merged,
    )


def _test_result_row(result: TestResult, *, requested_name: str) -> dict[str, Any]:
    return {
        "test": requested_name,
        "name": result.metadata.get("name", requested_name),
        "statistic": result.statistic,
        "p_value": result.p_value,
        "decision": result.decision,
        "alternative": result.alternative,
        "correction_policy": result.correction_policy,
        "n_obs": result.n_obs,
        "metadata": dict(result.metadata),
    }


def _coerce_custom_test_output(
    output: Any,
    *,
    name: str,
    alternative: str,
    alpha: float,
    correction_policy: str | None,
) -> TestResult:
    if isinstance(output, TestResult):
        return output
    if isinstance(output, Mapping):
        statistic = output.get("statistic", output.get("stat"))
        p_value = output.get("p_value", output.get("pvalue"))
        decision = output.get("decision")
        if decision is None:
            decision = p_value is not None and float(p_value) < alpha
        return TestResult(
            None if statistic is None else float(statistic),
            None if p_value is None else float(p_value),
            bool(decision),
            str(output.get("alternative", alternative)),
            output.get("correction_policy", correction_policy),
            None if output.get("n_obs") is None else int(output["n_obs"]),
            dict(output.get("metadata", {})),
        )
    if isinstance(output, tuple) and len(output) in {2, 3}:
        statistic, p_value = output[:2]
        n_obs = output[2] if len(output) == 3 else None
        return TestResult(
            None if statistic is None else float(statistic),
            None if p_value is None else float(p_value),
            p_value is not None and float(p_value) < alpha,
            alternative,
            correction_policy,
            None if n_obs is None else int(n_obs),
            {"name": name},
        )
    raise TypeError(
        "custom test callable must return TestResult, a mapping, or "
        "(statistic, p_value[, n_obs])"
    )


def _callable_name(func: Callable[..., Any]) -> str:
    module = getattr(func, "__module__", "")
    qualname = getattr(func, "__qualname__", getattr(func, "__name__", repr(func)))
    return f"{module}.{qualname}" if module else str(qualname)


def _diebold_mariano_stat(
    diff: pd.Series,
    *,
    horizon: int,
    hln: bool = True,
    kernel: str = "newey_west",
) -> tuple[float | None, float | None]:
    clean = diff.dropna()
    n = len(clean)
    if n < 3:
        return None, None
    mean = float(clean.mean())
    lag = max(0, int(horizon) - 1)
    variance = _long_run_variance(clean.to_numpy(dtype=float) - mean, kernel=kernel, lag=lag)
    if variance <= 0:
        return None, None
    statistic = mean / math.sqrt(variance / n)
    if hln:
        adjustment = math.sqrt((n + 1 - 2 * (lag + 1) + (lag + 1) * lag / n) / n)
        statistic *= adjustment if adjustment > 0 else 1.0
    return float(statistic), _normal_two_sided_p(statistic)


def _long_run_variance(
    values: np.ndarray, *, kernel: str = "newey_west", lag: int | None = None
) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    centered = values - values.mean() if abs(values.mean()) > 1e-12 else values
    gamma_0 = float(np.dot(centered, centered) / n)
    if kernel == "andrews":
        if n > 2:
            numerator = float(np.sum(centered[:-1] * centered[1:]))
            denominator = float(np.sum(centered[:-1] ** 2))
            alpha1 = numerator / denominator if denominator > 0 else 0.0
            alpha = (4.0 * alpha1**2) / (max(1.0 - alpha1**2, 1e-12) ** 2)
            lag = max(1, int(np.floor(1.1447 * (alpha * n) ** (1 / 3))))
        else:
            lag = 1
        kernel = "newey_west"
    if lag is None:
        lag = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    bandwidth = max(0, int(lag))
    variance = gamma_0
    if kernel == "newey_west":
        for k in range(1, bandwidth + 1):
            if n > k:
                weight = 1.0 - k / (bandwidth + 1)
                variance += 2.0 * weight * float(np.dot(centered[:-k], centered[k:]) / n)
        return float(variance)
    if kernel == "parzen":
        for k in range(1, bandwidth + 1):
            if n <= k:
                continue
            x = k / bandwidth
            weight = 1.0 - 6.0 * x**2 + 6.0 * x**3 if x <= 0.5 else 2.0 * (1.0 - x) ** 3
            variance += 2.0 * weight * float(np.dot(centered[:-k], centered[k:]) / n)
        return float(variance)
    raise ValueError(f"unknown HAC kernel {kernel!r}")


def _normal_two_sided_p(statistic: float | None) -> float | None:
    if statistic is None:
        return None
    if math.isinf(statistic):
        return 0.0
    return max(0.0, min(1.0, math.erfc(abs(statistic) / math.sqrt(2.0))))


def _one_sided_from_two_sided(stat: float | None, p_two: float | None) -> float | None:
    if stat is None or p_two is None:
        return None
    return p_two / 2.0 if stat > 0 else 1.0 - p_two / 2.0


def _direction_test_stat(
    forecast: np.ndarray, actual: np.ndarray, *, method: str
) -> tuple[float | None, float | None, float | None]:
    n = len(forecast)
    if n < 2:
        return None, None, None
    success = float((forecast == actual).mean())
    p_y = float(actual.mean())
    p_x = float(forecast.mean())
    p_star = p_y * p_x + (1.0 - p_y) * (1.0 - p_x)
    if p_star <= 0.0 or p_star >= 1.0:
        return None, None, success
    var_p = (p_star * (1.0 - p_star)) / n
    var_p_star = (
        ((2.0 * p_y - 1.0) ** 2 * p_x * (1.0 - p_x)) / n
        + ((2.0 * p_x - 1.0) ** 2 * p_y * (1.0 - p_y)) / n
        + (4.0 * p_y * p_x * (1.0 - p_y) * (1.0 - p_x)) / (n * n)
    )
    denominator = max(var_p - var_p_star, 1e-12)
    statistic = (success - p_star) / math.sqrt(denominator)
    if method in {"henriksson_merton", "hm"}:
        up_correct = ((forecast == 1) & (actual == 1)).sum()
        down_correct = ((forecast == 0) & (actual == 0)).sum()
        n_up = max(int((actual == 1).sum()), 1)
        n_down = max(int((actual == 0).sum()), 1)
        joint = (up_correct / n_up) + (down_correct / n_down)
        statistic = (joint - 1.0) * math.sqrt(min(n_up, n_down))
    return float(statistic), _normal_two_sided_p(statistic), success


def _kupiec_test(hits: np.ndarray, alpha: float) -> dict[str, Any]:
    from scipy import stats as _stats

    p_hat = float(hits.mean()) if hits.size else 0.0
    if 0.0 < p_hat < 1.0:
        x_hits = float(hits.sum())
        lr = 2.0 * (
            x_hits * np.log(p_hat / alpha)
            + (hits.size - x_hits) * np.log((1.0 - p_hat) / (1.0 - alpha))
        )
        p_value = float(1.0 - _stats.chi2.cdf(lr, df=1))
    else:
        lr = 0.0
        p_value = 1.0
    return {
        "hits_rate": p_hat,
        "lr_statistic": float(lr),
        "p_value": p_value,
        "reject": bool(p_value < alpha),
    }


def _christoffersen_independence_test(hits: np.ndarray, alpha: float) -> dict[str, Any]:
    from scipy import stats as _stats

    n00 = n01 = n10 = n11 = 0
    for prev, curr in zip(hits[:-1], hits[1:]):
        if prev == 0 and curr == 0:
            n00 += 1
        elif prev == 0 and curr == 1:
            n01 += 1
        elif prev == 1 and curr == 0:
            n10 += 1
        else:
            n11 += 1
    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(hits.size - 1, 1)
    if 0.0 < pi < 1.0 and 0.0 < pi01 < 1.0 and 0.0 < pi11 < 1.0:
        lr = -2.0 * (
            (n01 + n11) * np.log(pi)
            + (n00 + n10) * np.log(1.0 - pi)
            - n01 * np.log(pi01)
            - n00 * np.log(1.0 - pi01)
            - n11 * np.log(pi11)
            - n10 * np.log(1.0 - pi11)
        )
        p_value = float(1.0 - _stats.chi2.cdf(lr, df=1))
    else:
        lr = 0.0
        p_value = 1.0
    return {
        "lr_statistic": float(lr),
        "p_value": p_value,
        "reject": bool(p_value < alpha),
    }


def _engle_manganelli_dq_test(hits: np.ndarray, alpha: float) -> dict[str, Any]:
    from scipy import stats as _stats

    n_lags = 3
    dq_stat = 0.0
    dq_p = 1.0
    if hits.size >= 8:
        try:
            X = np.column_stack(
                [np.ones(hits.size - n_lags)]
                + [hits[n_lags - lag : -lag] for lag in range(1, n_lags + 1)]
            )
            y: np.ndarray = hits[n_lags:].astype(float)
            coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            preds = X @ coef
            ssr = float(np.sum((y - preds) ** 2))
            tss = float(np.sum((y - y.mean()) ** 2))
            r2 = 1.0 - ssr / tss if tss > 0 else 0.0
            dq_stat = float(hits.size * r2)
            dq_p = float(1.0 - _stats.chi2.cdf(dq_stat, df=n_lags))
        except Exception:
            dq_stat = 0.0
            dq_p = 1.0
    return {
        "statistic": dq_stat,
        "p_value": dq_p,
        "reject": bool(dq_p < alpha),
        "n_lags": n_lags,
    }


def _residual_test_statistic(test_name: str, residuals: pd.Series, lag: int) -> tuple[float | None, float | None]:
    values = residuals.astype(float).dropna()
    if len(values) < 3:
        return None, None
    if test_name == "durbin_watson":
        from statsmodels.stats.stattools import durbin_watson

        return float(durbin_watson(values)), None
    if test_name == "jarque_bera_normality":
        from scipy import stats as _stats

        statistic, p_value = _stats.jarque_bera(values)
        return float(statistic), float(p_value)
    max_lag = min(lag, len(values) - 1)
    if max_lag < 1:
        return None, None
    if test_name == "ljung_box_q":
        from statsmodels.stats.diagnostic import acorr_ljungbox

        result = acorr_ljungbox(values, lags=[max_lag], return_df=True)
        return float(result["lb_stat"].iloc[0]), float(result["lb_pvalue"].iloc[0])
    if test_name == "breusch_godfrey_serial_correlation":
        try:
            from statsmodels.regression.linear_model import OLS
            from statsmodels.stats.diagnostic import acorr_breusch_godfrey

            x = np.column_stack([np.ones(len(values)), np.arange(len(values))])
            model = OLS(values.values, x).fit()
            statistic, p_value, _, _ = acorr_breusch_godfrey(model, nlags=max_lag)
            return float(statistic), float(p_value)
        except Exception:
            return None, None
    if test_name == "arch_lm":
        from statsmodels.stats.diagnostic import het_arch

        try:
            statistic, p_value, _, _ = het_arch(values, nlags=max_lag)
            return float(statistic), float(p_value)
        except Exception:
            return None, None
    return None, None


def _gr_critical_value(window_ratio: float, alpha: float) -> float:
    rng = np.random.default_rng(int(round(float(window_ratio) * 100)) + int(alpha * 1000))
    n_sims = 1000
    n_grid = 200
    width = max(1, int(round(window_ratio * n_grid)))
    paths = rng.normal(size=(n_sims, n_grid))
    kernel = np.ones(width)
    rolling = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, paths)
    stats = np.abs((rolling / width) * np.sqrt(width))
    return float(np.quantile(stats.max(axis=1), 1.0 - float(alpha)))


def _newey_west_se(values: np.ndarray, lags: int) -> float:
    n = values.size
    if n == 0:
        return 1.0
    centered = values - values.mean()
    gamma0 = float(np.dot(centered, centered) / n)
    variance = gamma0
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1)
        cov = float(np.dot(centered[lag:], centered[:-lag]) / n)
        variance += 2.0 * weight * cov
    return float(np.sqrt(max(variance / n, 1e-12)))


def _resolve_block_length(matrix: np.ndarray, value: int | str) -> int:
    n = matrix.shape[0]
    if isinstance(value, (int, np.integer)) and value > 0:
        return max(1, min(int(value), n // 2 if n > 1 else 1))
    if isinstance(value, str) and value.isdigit():
        return max(1, min(int(value), n // 2 if n > 1 else 1))
    block = max(1, int(np.floor(2.0 * (4.0 * n / 100.0) ** (1 / 3))))
    return min(block, max(1, n // 2))


def _stationary_bootstrap_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    if block_length <= 1:
        return rng.integers(0, n, size=n)
    p_restart = 1.0 / block_length
    indices: np.ndarray = np.empty(n, dtype=np.int64)
    indices[0] = int(rng.integers(0, n))
    restarts = rng.random(n - 1) < p_restart
    new_starts = rng.integers(0, n, size=n - 1)
    for idx in range(1, n):
        indices[idx] = int(new_starts[idx - 1]) if restarts[idx - 1] else (indices[idx - 1] + 1) % n
    return indices


def _fixed_block_bootstrap_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=n_blocks)
    indices: np.ndarray = np.empty(n_blocks * block_length, dtype=np.int64)
    for block_id, start in enumerate(starts):
        indices[block_id * block_length : (block_id + 1) * block_length] = (
            start + np.arange(block_length)
        ) % n
    return indices[:n]


def _bootstrap_means(
    centered: np.ndarray,
    *,
    n_boot: int,
    block_length: int,
    method: str,
    rng: np.random.Generator,
) -> np.ndarray:
    n_obs, n_models = centered.shape
    boot_means = np.empty((n_boot, n_models))
    for boot_id in range(n_boot):
        indices = (
            _stationary_bootstrap_indices(n_obs, block_length, rng)
            if method == "stationary_bootstrap"
            else _fixed_block_bootstrap_indices(n_obs, block_length, rng)
        )
        boot_means[boot_id] = centered[indices].mean(axis=0)
    return boot_means


def _studentized_deviation(means: np.ndarray, boot_means: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    observed_dev = means - means.mean()
    boot_dev = boot_means - boot_means.mean(axis=1, keepdims=True)
    boot_var = np.where(boot_dev.var(axis=0, ddof=1) <= 0, 1e-12, boot_dev.var(axis=0, ddof=1))
    return observed_dev / np.sqrt(boot_var), boot_dev / np.sqrt(boot_var)


def _spa_p_value(
    wide: pd.DataFrame,
    means: np.ndarray,
    boot_means: np.ndarray,
    *,
    benchmark: str | None,
) -> float:
    if benchmark is None or benchmark not in wide.columns:
        warnings.warn(
            "SPA benchmark model not specified or not present; returning NaN.",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")
    benchmark_idx = list(wide.columns).index(benchmark)
    relative = boot_means - boot_means[:, [benchmark_idx]]
    observed = means - means[benchmark_idx]
    return float((relative.max(axis=1) >= float(observed.max())).mean())


def _stepm_rejected(
    columns: pd.Index,
    means: np.ndarray,
    centered: np.ndarray,
    *,
    alpha: float,
    n_boot: int,
    block_length: int,
    rng: np.random.Generator,
) -> set[str]:
    rejected: set[str] = set()
    active = list(columns)
    active_idx = list(range(len(columns)))
    while len(active_idx) > 1:
        sub_means = means[active_idx]
        sub_centered = centered[:, active_idx]
        sub_boot = _bootstrap_means(
            sub_centered,
            n_boot=n_boot,
            block_length=block_length,
            method="stationary_bootstrap",
            rng=rng,
        )
        sub_t, sub_boot_t = _studentized_deviation(sub_means, sub_boot)
        critical = float(np.quantile(sub_boot_t.max(axis=1), 1.0 - alpha))
        worst_pos = int(np.argmax(sub_t))
        if sub_t[worst_pos] > critical:
            rejected.add(str(active[worst_pos]))
            del active[worst_pos]
            del active_idx[worst_pos]
        else:
            break
    return rejected


def _float_or_none(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _mapping_records(
    mapping: Mapping[Any, Any],
    *,
    key_names: Sequence[str],
    value_name: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for key, value in mapping.items():
        key_values = key if isinstance(key, tuple) else (key,)
        row = {
            str(name): key_values[pos] if pos < len(key_values) else None
            for pos, name in enumerate(key_names)
        }
        if isinstance(value, set):
            row[value_name] = sorted(str(item) for item in value)
        else:
            row[value_name] = value
        records.append(row)
    return records


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, set):
        return sorted(_json_ready(item) for item in value)
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, pd.Series):
        return {
            "name": value.name,
            "index": [_json_ready(item) for item in value.index],
            "data": [_json_ready(item) for item in value.to_list()],
        }
    if isinstance(value, pd.DataFrame):
        return {
            "columns": [str(column) for column in value.columns],
            "index": [_json_ready(item) for item in value.index],
            "data": _json_ready(value.to_dict(orient="list")),
        }
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is pd.NaT or value is pd.NA:
        return None
    return value


__all__ = [
    "TestResult",
    "clark_west_test",
    "conditional_predictive_ability_test",
    "custom_test",
    "cw_test",
    "density_interval_tests",
    "directional_accuracy_test",
    "dm_test",
    "dmp_test",
    "equal_predictive_tests",
    "enc_new_test",
    "enc_t_test",
    "gw_test",
    "harvey_newbold_test",
    "henriksson_merton_test",
    "hn_test",
    "interval_coverage_test",
    "blocked_oob_reality_check",
    "model_confidence_set",
    "nested_tests",
    "pesaran_timmermann_test",
    "pit_autocorrelation_test",
    "pit_histogram",
    "residual_diagnostics",
]
