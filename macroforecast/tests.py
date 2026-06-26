from __future__ import annotations

import json
import math
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
    kernel: str = "acf",
    input_type: str = "loss",
    power: float = 2.0,
    alternative: str = "two_sided",
    alpha: float = 0.05,
) -> TestResult:
    """Diebold-Mariano equal predictive ability test."""

    _validate_alpha(alpha)
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    input_type = _normalize_dm_input_type(input_type)
    alternative = _normalize_alternative(alternative)
    frame = _aligned_frame(loss_a, loss_b, names=("series_a", "series_b"))
    if horizon > len(frame):
        raise ValueError("horizon cannot be longer than the number of aligned observations")
    if input_type == "error":
        diff = frame["series_a"].abs() ** float(power) - frame["series_b"].abs() ** float(power)
    else:
        diff = frame["series_a"] - frame["series_b"]
    hln = _normalize_correction(correction) == "hln"
    stat, p_value = _diebold_mariano_stat(
        diff,
        horizon=horizon,
        hln=hln,
        kernel=kernel,
        alternative=alternative,
    )
    variance_estimator = _normalize_dm_variance_estimator(kernel)
    p_value_status = _p_value_status(stat, p_value)
    return TestResult(
        statistic=stat,
        p_value=p_value,
        decision=p_value is not None and p_value < alpha,
        alternative=alternative,
        correction_policy=f"{'hln_' if hln else ''}{variance_estimator}",
        n_obs=int(diff.notna().sum()),
        metadata={
            "name": "Diebold-Mariano",
            "horizon": int(horizon),
            "hln_correction": hln,
            "variance_estimator": variance_estimator,
            "input_type": input_type,
            "power": float(power),
            "mean_loss_difference": _float_or_none(diff.mean()),
            "statistic_type": "t",
            "null_hypothesis": "equal predictive accuracy",
            "p_value_status": p_value_status,
            "p_value_reference": "Student-t reference with df=n_obs-1",
            "source_reference": "forecast/R/DM2.R::dm.test",
            "external_reference": "forecast::dm.test; Harvey-Leybourne-Newbold modified Diebold-Mariano",
            "r_reference": "forecast/R/DM2.R::dm.test",
            "r_argument_mapping": {
                "loss_a/loss_b": "precomputed loss differential input; no direct R argument",
                "error_a/error_b": "e1/e2 when input_type='error'",
                "horizon": "h",
                "kernel": "varestimator",
                "power": "power",
                "alternative": "alternative",
            },
            "r_alignment": _dm_r_alignment(
                input_type=input_type,
                hln=hln,
                variance_estimator=variance_estimator,
            ),
        },
    )


def gw_test(
    loss_a: Any,
    loss_b: Any,
    *,
    horizon: int = 1,
    correction: str = "hln",
    kernel: str = "acf",
    input_type: str = "loss",
    power: float = 2.0,
    alternative: str = "two_sided",
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
        input_type=input_type,
        power=power,
        alternative=alternative,
        alpha=alpha,
    )
    return _replace_metadata(
        result,
        name="Giacomini-White",
        null_hypothesis="zero mean aligned loss differential",
        source_reference="macroforecast legacy GW-compatible DM-style loss-differential surface",
        external_reference="Giacomini-White conditional predictive ability is not implemented by this callable",
        r_reference=None,
        r_alignment=(
            "No exact R comparator is claimed. This callable preserves the legacy GW surface "
            "by reusing the DM/HLN loss-differential statistic on aligned inputs. Use "
            "conditional_predictive_ability_test for the package's Giacomini-Rossi "
            "time-varying CPA path."
        ),
    )


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
    variance_estimator = _normalize_dm_variance_estimator(kernel)
    base_metadata = {
        "name": "Diebold-Mariano-Pesaran",
        "n_obs_stacked": n,
        "hac_kernel": variance_estimator,
        "statistic_type": "z",
        "null_hypothesis": "zero mean stacked loss differential",
        "p_value_reference": "standard normal two-sided reference",
        "source_reference": "macroforecast stacked Diebold-Mariano-Pesaran-style HAC screen",
        "r_reference": None,
        "r_alignment": (
            "No exact R package comparator was located in the checked R sources. "
            "The callable accepts one or more precomputed loss-difference arrays, stacks "
            "finite values, and tests the stacked mean with a HAC standard error."
        ),
    }
    if n < 3:
        return TestResult(
            None,
            None,
            False,
            "two_sided",
            variance_estimator,
            n,
            {**base_metadata, "p_value_status": "insufficient_observations"},
        )
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
        correction_policy=variance_estimator,
        n_obs=n,
        metadata={
            **base_metadata,
            "mean_loss_difference": mean_diff,
            "p_value_status": _p_value_status(stat, p_value),
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
    """Legacy forecast-encompassing covariance t approximation."""

    _validate_alpha(alpha)
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    frame = _aligned_frame(error_a, error_b, names=("error_a", "error_b"))
    e_a = frame["error_a"].to_numpy(dtype=float)
    e_b = frame["error_b"].to_numpy(dtype=float)
    d = e_a * (e_a - e_b)
    finite = np.isfinite(d)
    n = int(finite.sum())
    variance_estimator = _normalize_dm_variance_estimator(kernel)
    base_metadata = {
        "name": "Harvey-Newbold",
        "encompassing": "a_over_b",
        "hac_kernel": variance_estimator,
        "small_sample": bool(small_sample),
        "statistic_type": "t",
        "null_hypothesis": "forecast a does not encompass forecast b",
        "p_value_reference": "Student-t upper-tail reference with df=n_obs-1",
        "source_reference": "macroforecast legacy forecast-encompassing covariance t approximation",
        "r_reference": None,
        "r_alignment": (
            "No exact R comparator is claimed. forecast::dm.test implements the "
            "Harvey-Leybourne-Newbold equal-accuracy DM test, while this callable "
            "uses the legacy encompassing covariance series e_a * (e_a - e_b)."
        ),
        "note": (
            "This is not forecast::dm.test; it is an encompassing-style covariance "
            "approximation retained as a callable legacy test."
        ),
    }
    if n < 5:
        return TestResult(
            None,
            None,
            False,
            "one_sided",
            f"{'hln_' if small_sample else ''}{variance_estimator}",
            n,
            {**base_metadata, "p_value_status": "insufficient_observations"},
        )
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
        correction_policy=f"{'hln_' if small_sample else ''}{variance_estimator}",
        n_obs=n,
        metadata={
            **base_metadata,
            "mean_d": float(np.nanmean(d_clean)),
            "p_value_status": _p_value_status(stat, p_value),
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
    stat, p_value = _mean_hac_test_statistic(
        improvement,
        horizon=horizon,
        kernel=kernel,
        reference="normal",
        alternative="greater",
    )
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
            "source_reference": "Clark-West adjusted MSPE differential",
            "reference_formula": "e_r^2 - (e_u^2 - (f_r - f_u)^2)",
            "external_reference": "GAUSS cwTest; HypothesisTests.jl ClarkWestTest",
            "r_reference": None,
        },
    )


def cw_test(*args: Any, **kwargs: Any) -> TestResult:
    """Alias for :func:`clark_west_test`."""

    return clark_west_test(*args, **kwargs)


def enc_new_test(
    error_small: Any,
    error_large: Any,
    *,
    critical_value: float | None = None,
    alpha: float = 0.05,
) -> TestResult:
    """ENC-NEW nested forecast encompassing test."""

    _validate_alpha(alpha)
    frame = _aligned_frame(error_small, error_large, names=("error_small", "error_large"))
    e_small = frame["error_small"].to_numpy(dtype=float)
    e_large = frame["error_large"].to_numpy(dtype=float)
    c_values = e_small * (e_small - e_large)
    denominator = float(np.mean(e_large**2))
    statistic = None if denominator <= 0.0 else float(len(c_values) * np.mean(c_values) / denominator)
    decision = bool(statistic is not None and critical_value is not None and statistic > critical_value)
    return TestResult(
        statistic=statistic,
        p_value=None,
        decision=decision,
        alternative="one_sided",
        correction_policy="nonstandard_critical_value",
        n_obs=int(len(c_values)),
        metadata={
            "name": "Enc-New",
            "mean_encompassing_covariance": _float_or_none(np.mean(c_values)),
            "msfe_large": denominator,
            "critical_value": critical_value,
            "source_reference": "Clark-McCracken ENC-New statistic",
            "external_reference": "Stata forecast encompassing example; Clark and McCracken (2001)",
            "r_reference": None,
            "p_value_note": "ENC-New has a nonstandard nested-forecast distribution; pass a design-appropriate critical_value for decision.",
        },
    )


def enc_t_test(
    error_small: Any,
    error_large: Any,
    *,
    horizon: int = 1,
    kernel: str = "newey_west",
    critical_value: float | None = None,
    normal_approximation: bool = False,
    alpha: float = 0.05,
) -> TestResult:
    """ENC-T nested forecast encompassing test."""

    _validate_alpha(alpha)
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    frame = _aligned_frame(error_small, error_large, names=("error_small", "error_large"))
    c_values = frame["error_small"] * (frame["error_small"] - frame["error_large"])
    stat, normal_p = _mean_hac_test_statistic(
        c_values,
        horizon=horizon,
        kernel=kernel,
        reference="normal",
        alternative="greater",
    )
    p_value = normal_p if normal_approximation else None
    decision = (
        bool(stat is not None and stat > critical_value)
        if critical_value is not None
        else bool(p_value is not None and p_value < alpha)
    )
    return TestResult(
        statistic=stat,
        p_value=p_value,
        decision=decision,
        alternative="one_sided",
        correction_policy="normal_approximation" if normal_approximation else "nonstandard_critical_value",
        n_obs=int(c_values.notna().sum()),
        metadata={
            "name": "Enc-T",
            "mean_encompassing_covariance": _float_or_none(c_values.mean()),
            "hac_kernel": kernel,
            "critical_value": critical_value,
            "normal_approximation": bool(normal_approximation),
            "source_reference": "Clark-McCracken ENC-T statistic",
            "external_reference": "Stata forecast encompassing example; Clark and McCracken (2001)",
            "r_reference": None,
            "p_value_note": "Default p_value is None because nested forecast encompassing tests use nonstandard critical values.",
        },
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
    actual_raw = frame["truth"].to_numpy(dtype=float) - float(threshold)
    forecast_raw = frame["pred"].to_numpy(dtype=float) - float(threshold)
    if len(forecast_raw) > 1 and float(np.linalg.norm(np.diff(forecast_raw))) <= 1e-12:
        raise ValueError("forecast is constant; directional accuracy tests cannot be calculated")
    forecast = (forecast_raw > 0.0).astype(int)
    actual = (actual_raw > 0.0).astype(int)
    stat, p_value, success = _direction_test_stat(
        forecast,
        actual,
        method=method,
        forecast_raw=forecast_raw,
        actual_raw=actual_raw,
    )
    reference = _direction_method_reference(method)
    return TestResult(
        statistic=stat,
        p_value=p_value,
        decision=p_value is not None and p_value < alpha,
        alternative="one_sided",
        correction_policy=None,
        n_obs=int(len(frame)),
        metadata={
            "name": "Directional accuracy",
            "method": method,
            "threshold": float(threshold),
            "success_ratio": success,
            "sign_rule": "positive if value > threshold; zero is non-positive",
            "p_value_reference": "one-sided upper-tail normal, 1 - Phi(statistic)",
            **reference,
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


def anatolyev_gerko_test(*args: Any, **kwargs: Any) -> TestResult:
    """Anatolyev-Gerko excess profitability directional accuracy test."""

    kwargs.setdefault("method", "anatolyev_gerko")
    return directional_accuracy_test(*args, **kwargs)


def density_interval_tests(
    pit: Any,
    *,
    alpha: float = 0.05,
    n_bins: int = 10,
    pit_lag: int = 1,
) -> dict[str, Any]:
    """Density and interval diagnostics for PIT values."""

    from scipy import stats as _stats

    _validate_alpha(alpha)
    n_bins = _validate_positive_int(n_bins, "n_bins")
    pit_lag = _validate_positive_int(pit_lag, "pit_lag")
    values = _validate_uniform_values(pit)
    if values.size == 0:
        reference = _density_interval_reference()
        return {
            "metadata_schema": {
                "kind": "density_interval_tests",
                "version": 1,
            },
            "status": "empty",
            "n_obs": 0,
            "n_bins": int(n_bins),
            **reference,
        }
    berkowitz = _berkowitz_density_test(values, lags=pit_lag, alpha=alpha)
    ks_stat, ks_pvalue = _stats.kstest(values, "uniform")
    hits = (values < alpha).astype(int)
    kupiec = _kupiec_test(hits, alpha)
    christoffersen = _christoffersen_independence_test(hits, alpha)
    dq = _engle_manganelli_dq_test(hits, alpha)
    shortfall = shortfall_de_test(values, alpha=alpha, lags=pit_lag)
    reference = _density_interval_reference()
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
        "du_escanciano_shortfall": shortfall,
        "pit_histogram": pit_histogram(values, n_bins=n_bins).to_dict(orient="records"),
        "pit_autocorrelation": pit_autocorrelation_test(values, lag=pit_lag, alpha=alpha).to_dict(),
        "n_obs": int(values.size),
        "n_bins": int(n_bins),
        **reference,
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
    """Kupiec, Christoffersen, and duration diagnostics for forecast intervals."""

    _validate_alpha(alpha)
    frame = _aligned_frame(y_true, lower, upper, names=("truth", "lower", "upper"))
    truth = frame["truth"].to_numpy(dtype=float)
    lo = frame["lower"].to_numpy(dtype=float)
    hi = frame["upper"].to_numpy(dtype=float)
    misses = ((truth < lo) | (truth > hi)).astype(int)
    coverage = float(1.0 - misses.mean()) if misses.size else None
    kupiec = _kupiec_test(misses, alpha)
    christoffersen = _christoffersen_independence_test(misses, alpha)
    duration = _christoffersen_pelletier_duration_test(misses, alpha)
    conditional_stat = None
    conditional_p = None
    if kupiec.get("lr_statistic") is not None and christoffersen.get("lr_statistic") is not None:
        from scipy import stats as _stats

        conditional_stat = float(kupiec["lr_statistic"] + christoffersen["lr_statistic"])
        conditional_p = float(1.0 - _stats.chi2.cdf(conditional_stat, df=2))
    reference = _interval_coverage_reference()
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
                "df": 2,
                "r_reference": "tstests/R/var_cp.R::var_cp_test",
                "rugarch_reference": "rugarch/R/rugarch-tests.R::LR.cc.test",
            },
            "christoffersen_pelletier_duration": duration,
            **reference,
        }
    )


def shortfall_de_test(
    pit: Any,
    *,
    alpha: float = 0.05,
    lags: int = 1,
    boot: bool = False,
    n_boot: int = 2000,
    random_state: int = 0,
) -> dict[str, Any]:
    """Du-Escanciano expected shortfall tests on PIT values."""

    _validate_alpha(alpha)
    lags = _validate_positive_int(lags, "lags")
    values = _validate_uniform_values(pit)
    n = int(values.size)
    if n == 0:
        return {
            "metadata_schema": {
                "kind": "shortfall_de_test",
                "version": 1,
            },
            "status": "empty",
            "n_obs": 0,
            "alpha": float(alpha),
            "lags": int(lags),
            "r_reference": "tstests/R/shortfall_de.R::shortfall_de_test",
            "r_alignment": (
                "Matches Du-Escanciano cumulative tail shortfall formulas; no statistic "
                "is computed for an empty PIT sample."
            ),
        }
    unconditional_stat = _du_escanciano_unconditional_statistic(values, alpha)
    conditional_stat = _du_escanciano_conditional_statistic(values, alpha=alpha, lags=lags)
    if boot:
        rng = np.random.default_rng(random_state)
        unconditional_p = _du_escanciano_bootstrap_pvalue(
            unconditional_stat,
            n=n,
            alpha=alpha,
            lags=None,
            n_boot=n_boot,
            rng=rng,
        )
        conditional_p = _du_escanciano_bootstrap_pvalue(
            conditional_stat,
            n=n,
            alpha=alpha,
            lags=lags,
            n_boot=n_boot,
            rng=rng,
        )
    else:
        unconditional_p = _du_escanciano_unconditional_pvalue(unconditional_stat, n=n, alpha=alpha)
        conditional_p = _du_escanciano_conditional_pvalue(conditional_stat, lags=lags)
    return _json_ready(
        {
            "metadata_schema": {
                "kind": "shortfall_de_test",
                "version": 1,
            },
            "unconditional": {
                "statistic": unconditional_stat,
                "p_value": unconditional_p,
                "reject": bool(unconditional_p is not None and unconditional_p < alpha),
                "distribution": "normal" if not boot else "bootstrap",
            },
            "conditional": {
                "statistic": conditional_stat,
                "p_value": conditional_p,
                "reject": bool(conditional_p is not None and conditional_p < alpha),
                "distribution": "chi_squared" if not boot else "bootstrap",
                "df": int(lags),
            },
            "alpha": float(alpha),
            "lags": int(lags),
            "n_obs": n,
            "boot": bool(boot),
            "n_boot": int(n_boot) if boot else None,
            "r_reference": "tstests/R/shortfall_de.R::shortfall_de_test",
            "r_alignment": (
                "Matches unconditional_de_statistic, unconditional_de_pvalue, "
                "conditional_de_statistic, conditional_de_pvalue, and uniform-bootstrap "
                "simulation logic in tstests/R/shortfall_de.R."
            ),
        }
    )


def dynamic_quantile_test(
    y_true: Any,
    var: Any,
    *,
    alpha: float = 0.05,
    lag: int = 1,
    lag_hit: int = 1,
    lag_var: int = 1,
) -> TestResult:
    """Engle-Manganelli dynamic quantile test for VaR forecasts."""

    _validate_alpha(alpha)
    lag = _validate_positive_int(lag, "lag")
    lag_hit = _validate_positive_int(lag_hit, "lag_hit")
    lag_var = _validate_positive_int(lag_var, "lag_var")
    frame = _aligned_frame(y_true, var, names=("truth", "var"))
    truth = frame["truth"].to_numpy(dtype=float)
    var_values = frame["var"].to_numpy(dtype=float)
    stat, p_value, df, n_used = _dynamic_quantile_statistic(
        truth,
        var_values,
        alpha=alpha,
        lag=lag,
        lag_hit=lag_hit,
        lag_var=lag_var,
    )
    return TestResult(
        statistic=stat,
        p_value=p_value,
        decision=bool(p_value is not None and p_value < alpha),
        alternative="two_sided",
        correction_policy=None,
        n_obs=n_used,
        metadata={
            "name": "Dynamic quantile",
            "alpha": float(alpha),
            "lag": int(lag),
            "lag_hit": int(lag_hit),
            "lag_var": int(lag_var),
            "df": df,
            "r_reference": "segMGarch/R/DQtest.R::DQtest",
            "r_alignment": (
                "Matches segMGarch DQtest with VaR_level mapped to 1-alpha: y is tail-aligned "
                "to VaR length, Hit is 1-alpha for y < VaR and -alpha for y > VaR, X contains "
                "constant, lag-aligned VaR, lagged Hit, and lagged squared y, and the statistic "
                "uses Hit'X(X'X)^-1X'Hit/(alpha*(1-alpha))."
            ),
            "source_url": "https://rdrr.io/cran/segMGarch/src/R/DQtest.R",
        },
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
    alpha: float = 0.05,
    model_df: int = 0,
    exog: Any | None = None,
    demean_arch: bool = False,
) -> pd.DataFrame:
    """Run residual diagnostic tests and return one row per test."""

    _validate_alpha(alpha)
    lag = _validate_positive_int(lag, "lag")
    model_df = max(0, int(model_df))
    values = pd.Series(residuals).astype(float).dropna()
    rows = []
    for test_name in tests:
        rows.append(
            _residual_test_statistic(
                test_name,
                values,
                lag,
                alpha=alpha,
                model_df=model_df,
                exog=exog,
                demean_arch=demean_arch,
            )
        )
    frame = pd.DataFrame(rows)
    frame.attrs["macroforecast_metadata_schema"] = {
        "kind": "residual_diagnostics",
        "version": 1,
        "row_unit": "test",
        "input_contract": "residual_series",
        "alpha": float(alpha),
        "model_df": int(model_df),
        "r_reference": {
            "ljung_box_q": "stats::Box.test(type='Ljung-Box')",
            "arch_lm": "FinTS/R/ArchTest.R::ArchTest",
            "jarque_bera_normality": "tseries/R/test.R::jarque.bera.test",
            "breusch_godfrey_serial_correlation": "lmtest/R/bgtest.R::bgtest",
            "durbin_watson": "lmtest/R/dwtest.R::dwtest",
        },
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
    kernel: str = "acf",
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
        "row_unit": "test",
        "input_contract": "aligned losses; harvey_newbold additionally requires aligned errors",
        "paper_table_ready_columns": [
            "test",
            "name",
            "statistic_type",
            "statistic",
            "p_value",
            "p_value_status",
            "decision",
            "alternative",
            "null_hypothesis",
            "source_reference",
            "r_reference",
            "r_alignment",
        ],
        "r_reference": {
            "dm": "forecast/R/DM2.R::dm.test",
            "gw": None,
            "dmp": None,
            "hn": None,
        },
    }
    return out


def nested_tests(
    loss_small: Any,
    loss_large: Any,
    *,
    forecast_small: Any | None = None,
    forecast_large: Any | None = None,
    error_small: Any | None = None,
    error_large: Any | None = None,
    tests: Sequence[str] = ("clark_west", "enc_new", "enc_t"),
    horizon: int = 1,
    kernel: str = "newey_west",
    enc_critical_value: float | None = None,
    enc_normal_approximation: bool = False,
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
            if error_small is None or error_large is None:
                raise ValueError("error_small and error_large are required for enc_new in nested_tests")
            result = enc_new_test(
                error_small,
                error_large,
                critical_value=enc_critical_value,
                alpha=alpha,
            )
        elif key in {"enc_t", "enc_t_test"}:
            if error_small is None or error_large is None:
                raise ValueError("error_small and error_large are required for enc_t in nested_tests")
            result = enc_t_test(
                error_small,
                error_large,
                horizon=horizon,
                kernel=kernel,
                critical_value=enc_critical_value,
                normal_approximation=enc_normal_approximation,
                alpha=alpha,
            )
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
    window_ratio: float = 0.5,
    dmv_fullsample: bool = True,
    lag_truncate: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Giacomini-Rossi rolling fluctuation test or package recursive extension."""

    _validate_alpha(alpha)
    requested_method = str(method)
    method = _normalize_cpa_method(method)
    if method == "giacomini_rossi":
        window_ratio = _normalize_gr_window_ratio(window_ratio)
        critical = _gr_critical_value(window_ratio, alpha)
    elif not 0.0 < window_ratio <= 1.0:
        raise ValueError("window_ratio must be in (0, 1]")
    else:
        critical = None
    lag_truncate = _validate_gr_lag_truncate(lag_truncate)
    frame = _aligned_frame(loss_a, loss_b, names=("loss_a", "loss_b"))
    diff = (frame["loss_a"] - frame["loss_b"]).to_numpy(dtype=float)
    n = diff.size
    method_reference = _cpa_method_reference(method, requested_method=requested_method)
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
            "requested_method": requested_method,
            **method_reference,
        }
    m = min(n, max(4, int(round(window_ratio * n))))
    stats = (
        _giacomini_rossi_fluctuation_path(
            diff,
            window_size=m,
            dmv_fullsample=dmv_fullsample,
            lag_truncate=lag_truncate,
        )
        if method == "giacomini_rossi"
        else _recursive_loss_fluctuation_path(diff, window_size=m, lag_truncate=lag_truncate)
    )
    supremum = float(max(abs(value) for value in stats)) if stats else 0.0
    decision = bool(supremum > critical) if critical is not None else None
    return _json_ready({
        "metadata_schema": {
            "kind": "conditional_predictive_ability",
            "version": 1,
        },
        "statistic": supremum,
        "critical_value": critical,
        "critical_band": None if critical is None else [-critical, critical],
        "decision": decision,
        "time_path": stats,
        "window_size": m,
        "window_ratio": float(window_ratio),
        "n_obs": int(n),
        "method": method,
        "requested_method": requested_method,
        "variance_scope": "full_sample" if dmv_fullsample else "rolling_window",
        "lag_truncate": int(lag_truncate),
        "loss_difference_orientation": "loss_a - loss_b; positive path values mean loss_a has larger loss than loss_b",
        "statistic_definition": "supremum_absolute_fluctuation_path",
        "available_window_ratio_grid": [round(idx / 10, 1) for idx in range(1, 10)]
        if method == "giacomini_rossi"
        else None,
        "available_alpha": [0.05, 0.10] if method == "giacomini_rossi" else None,
        **method_reference,
    })


def model_confidence_set(
    loss_panel: pd.DataFrame,
    *,
    loss: str = "squared_error",
    alpha: float = 0.10,
    n_boot: int = 1000,
    block_length: int | str = "auto",
    bootstrap_method: str = "mcs_fixed_block",
    statistic: str = "max",
    random_state: int = 0,
    target: str = "target",
    horizon: str = "horizon",
    origin: str = "origin",
    model: str = "model_id",
) -> dict[str, Any]:
    """Exact Hansen-Lunde-Nason model confidence set.

    This is the canonical MCS callable. It follows the R ``MCS`` package's
    ``MCSprocedure`` structure: pairwise loss differences are bootstrapped,
    either ``Tmax`` or ``TR`` is evaluated, one model is removed each step, and
    included/excluded sets are determined from the cumulative MCS p-values.
    """

    return _model_confidence_set_exact(
        loss_panel,
        loss=loss,
        alpha=alpha,
        n_boot=n_boot,
        block_length=block_length,
        bootstrap_method=bootstrap_method,
        statistic=statistic,
        random_state=random_state,
        target=target,
        horizon=horizon,
        origin=origin,
        model=model,
        metadata_kind="model_confidence_set",
    )


def superior_predictive_ability_test(
    loss_panel: pd.DataFrame,
    *,
    benchmark: str,
    loss: str = "squared_error",
    alpha: float = 0.05,
    n_boot: int = 1000,
    block_length: int | str = "auto",
    bootstrap_method: str = "stationary_bootstrap",
    p_value_type: str = "consistent",
    studentize: bool = True,
    nested: bool = False,
    random_state: int = 0,
    target: str = "target",
    horizon: str = "horizon",
    origin: str = "origin",
    model: str = "model_id",
) -> dict[str, Any]:
    """White-Hansen superior predictive ability test via ``arch.bootstrap``."""

    return _arch_benchmark_multiple_comparison(
        loss_panel,
        benchmark=benchmark,
        loss=loss,
        alpha=alpha,
        n_boot=n_boot,
        block_length=block_length,
        bootstrap_method=bootstrap_method,
        p_value_type=p_value_type,
        studentize=studentize,
        nested=nested,
        random_state=random_state,
        target=target,
        horizon=horizon,
        origin=origin,
        model=model,
        test="spa",
    )


def reality_check_test(
    loss_panel: pd.DataFrame,
    *,
    benchmark: str,
    loss: str = "squared_error",
    alpha: float = 0.05,
    n_boot: int = 1000,
    block_length: int | str = "auto",
    bootstrap_method: str = "stationary_bootstrap",
    p_value_type: str = "consistent",
    studentize: bool = True,
    nested: bool = False,
    random_state: int = 0,
    target: str = "target",
    horizon: str = "horizon",
    origin: str = "origin",
    model: str = "model_id",
) -> dict[str, Any]:
    """White reality check against a benchmark via ``arch.bootstrap``."""

    return _arch_benchmark_multiple_comparison(
        loss_panel,
        benchmark=benchmark,
        loss=loss,
        alpha=alpha,
        n_boot=n_boot,
        block_length=block_length,
        bootstrap_method=bootstrap_method,
        p_value_type=p_value_type,
        studentize=studentize,
        nested=nested,
        random_state=random_state,
        target=target,
        horizon=horizon,
        origin=origin,
        model=model,
        test="reality_check",
    )


def stepm_test(
    loss_panel: pd.DataFrame,
    *,
    benchmark: str,
    loss: str = "squared_error",
    alpha: float = 0.05,
    n_boot: int = 1000,
    block_length: int | str = "auto",
    bootstrap_method: str = "stationary_bootstrap",
    studentize: bool = True,
    nested: bool = False,
    random_state: int = 0,
    target: str = "target",
    horizon: str = "horizon",
    origin: str = "origin",
    model: str = "model_id",
) -> dict[str, Any]:
    """Stepwise multiple-comparison test against a benchmark via ``arch.bootstrap``."""

    return _arch_benchmark_multiple_comparison(
        loss_panel,
        benchmark=benchmark,
        loss=loss,
        alpha=alpha,
        n_boot=n_boot,
        block_length=block_length,
        bootstrap_method=bootstrap_method,
        p_value_type="consistent",
        studentize=studentize,
        nested=nested,
        random_state=random_state,
        target=target,
        horizon=horizon,
        origin=origin,
        model=model,
        test="stepm",
    )


def iterative_model_confidence_set(
    loss_panel: pd.DataFrame,
    *,
    loss: str = "squared_error",
    alpha: float = 0.10,
    n_boot: int = 1000,
    block_length: int | str = "auto",
    bootstrap_method: str = "mcs_fixed_block",
    statistic: str = "max",
    random_state: int = 0,
    target: str = "target",
    horizon: str = "horizon",
    origin: str = "origin",
    model: str = "model_id",
) -> dict[str, Any]:
    """Descriptive alias for :func:`model_confidence_set`."""

    return _model_confidence_set_exact(
        loss_panel,
        loss=loss,
        alpha=alpha,
        n_boot=n_boot,
        block_length=block_length,
        bootstrap_method=bootstrap_method,
        statistic=statistic,
        random_state=random_state,
        target=target,
        horizon=horizon,
        origin=origin,
        model=model,
        metadata_kind="iterative_model_confidence_set",
    )


def _model_confidence_set_exact(
    loss_panel: pd.DataFrame,
    *,
    loss: str,
    alpha: float,
    n_boot: int,
    block_length: int | str,
    bootstrap_method: str,
    statistic: str,
    random_state: int,
    target: str,
    horizon: str,
    origin: str,
    model: str,
    metadata_kind: str,
) -> dict[str, Any]:
    """Shared exact MCS engine aligned with R ``MCSprocedure``."""

    _validate_alpha(alpha)
    n_boot = _validate_positive_int(n_boot, "n_boot")
    bootstrap_method = _normalize_bootstrap_method(bootstrap_method)
    statistic = _normalize_mcs_statistic(statistic)
    panel = pd.DataFrame(loss_panel).copy()
    groups = _loss_panel_groups(
        panel,
        loss=loss,
        target=target,
        horizon=horizon,
        origin=origin,
        model=model,
    )
    rng = np.random.default_rng(random_state)
    included: dict[tuple[Any, ...], set[str]] = {}
    rejected: dict[tuple[Any, ...], list[str]] = {}
    p_values: dict[tuple[Any, ...], float] = {}
    block_lengths_used: dict[tuple[Any, ...], int] = {}
    iteration_rows: list[dict[str, Any]] = []
    for target_value, horizon_value, wide in groups:
        key = (target_value, horizon_value, alpha)
        result = _iterative_mcs_wide(
            wide,
            alpha=alpha,
            n_boot=n_boot,
            block_length=block_length,
            bootstrap_method=bootstrap_method,
            statistic=statistic,
            rng=rng,
        )
        included[key] = set(result["included_models"])
        rejected[key] = list(result["rejected_models"])
        p_values[(target_value, horizon_value)] = result["final_p_value"]
        block_lengths_used[(target_value, horizon_value)] = result["block_length"]
        for row in result["iterations"]:
            iteration_rows.append(
                {
                    "target": target_value,
                    "horizon": horizon_value,
                    **row,
                }
            )
    return _json_ready({
        "metadata_schema": {
            "kind": metadata_kind,
            "version": 1,
        },
        "method": "hansen_lunde_nason_mcs",
        "procedure": "sequential_elimination",
        "canonical_function": "model_confidence_set",
        "alias_function": None if metadata_kind == "model_confidence_set" else "iterative_model_confidence_set",
        "mcs_inclusion": _mapping_records(
            included,
            key_names=("target", "horizon", "alpha"),
            value_name="models",
        ),
        "mcs_rejections": _mapping_records(
            rejected,
            key_names=("target", "horizon", "alpha"),
            value_name="models",
        ),
        "p_values": _mapping_records(
            p_values,
            key_names=("target", "horizon"),
            value_name="p_value",
        ),
        "iteration_path": iteration_rows,
        "bootstrap_n_replications": int(n_boot),
        "block_length": block_length,
        "block_lengths_used": _mapping_records(
            block_lengths_used,
            key_names=("target", "horizon"),
            value_name="block_length",
        ),
        "bootstrap_kind": bootstrap_method,
        "statistic": statistic,
        "r_reference": "MCS/R/MCSprocedure.R::MCSprocedure",
        "source_alignment": "GetD loss differences, GetIndices fixed blocks, Tmax/TR sequential elimination",
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
    """Legacy block-bootstrap benchmark-superiority screen.

    The input can be either a long per-origin loss panel with model and loss
    columns, or a wide loss matrix whose columns are model names. Positive
    ``mean_diff`` means the candidate has lower average loss than the
    benchmark. This callable is not the exact White Reality Check; use
    ``reality_check_test``/``superior_predictive_ability_test``/``stepm_test``
    for the arch-backed multiple-comparison procedures.
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
    reference = _legacy_blocked_oob_reference()
    for target_value, horizon_value, wide in groups:
        if benchmark not in wide.columns:
            raise ValueError(
                f"benchmark {benchmark!r} not in loss columns {list(wide.columns)}"
            )
        numeric = wide.apply(pd.to_numeric, errors="coerce")
        familywise = _benchmark_familywise_bootstrap(
            numeric,
            benchmark=benchmark,
            alpha=alpha,
            n_boot=n_boot,
            block_length=block_length,
            bootstrap_method=bootstrap_method,
            rng=rng,
        )
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
                    "familywise_p_value": familywise.get(str(candidate), {}).get("p_value"),
                    "familywise_decision": familywise.get(str(candidate), {}).get("decision"),
                    "familywise_n_obs": familywise.get(str(candidate), {}).get("n_obs"),
                    "n_obs": int(pair.shape[0]),
                    "alpha": float(alpha),
                    "block_length": int(resolved_block),
                    "n_boot": int(n_boot),
                    "bootstrap_method": bootstrap_method,
                    "test_family": "pairwise_benchmark_superiority_bootstrap",
                    **reference,
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
        "familywise_adjustment": "max_centered_block_bootstrap_across_candidates",
        **reference,
    }
    return out


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
    if key in {"anatolyev_gerko", "ag", "excess_profitability"}:
        return "anatolyev_gerko"
    if key in {"henriksson_merton", "hm"}:
        return "henriksson_merton"
    raise ValueError(
        "method must be 'pesaran_timmermann', 'anatolyev_gerko', or 'henriksson_merton'"
    )


def _direction_method_reference(method: str) -> dict[str, Any]:
    if method == "pesaran_timmermann":
        return {
            "source_reference": "Pesaran-Timmermann sign predictability directional accuracy test",
            "external_reference": "tstests dac_test; rugarch DACTest(test='PT')",
            "r_reference": "tstests/R/dac.R::dac_test",
            "rugarch_reference": "rugarch/R/rugarch-tests.R::DACTest(test='PT')",
            "r_alignment": (
                "Matches .pt_test/DACTest(test='PT'): x_t=1{actual>0}, y_t=1{forecast>0}, "
                "z_t=1{forecast*actual>0}, p_star=p_y*p_x+(1-p_y)*(1-p_x), and "
                "p.value=1-pnorm(statistic)."
            ),
        }
    if method == "anatolyev_gerko":
        return {
            "source_reference": "Anatolyev-Gerko excess profitability directional test",
            "external_reference": "tstests dac_test; rugarch DACTest(test='AG')",
            "r_reference": "tstests/R/dac.R::dac_test",
            "rugarch_reference": "rugarch/R/rugarch-tests.R::DACTest(test='AG')",
            "r_alignment": (
                "Matches .ag_test/DACTest(test='AG'): r_t=sign(forecast)*actual, "
                "A_t=mean(r_t), B_t=mean(sign(forecast))*mean(actual), "
                "V_EP=(4/n^2)*p_y*(1-p_y)*sum((actual-mean(actual))^2), and "
                "p.value=1-pnorm(statistic)."
            ),
        }
    return {
        "source_reference": "macroforecast Henriksson-Merton market-timing extension",
        "external_reference": "Henriksson and Merton market-timing diagnostic",
        "r_reference": None,
        "rugarch_reference": None,
        "r_alignment": (
            "No exact comparator in tstests::dac_test or rugarch::DACTest. "
            "This extension reports a normal-screening statistic based on the sum of "
            "up-market and down-market conditional hit rates."
        ),
    }


def _normalize_cpa_method(method: str) -> str:
    key = str(method).lower().replace("-", "_")
    if key in {"giacomini_rossi", "giacomini_rossi_2010", "gr"}:
        return "giacomini_rossi"
    if key in {"recursive", "recursive_fluctuation", "loss_recursive"}:
        return "recursive_fluctuation"
    if key in {"rossi_sekhposyan", "rs"}:
        return "recursive_fluctuation"
    raise ValueError("method must be 'giacomini_rossi' or 'recursive_fluctuation'")


def _cpa_method_reference(method: str, *, requested_method: str) -> dict[str, Any]:
    requested_key = str(requested_method).lower().replace("-", "_")
    if method == "giacomini_rossi":
        return {
            "source_reference": "Giacomini and Rossi (2010) Proposition 1 rolling-window fluctuation test",
            "external_reference": "murphydiagram R package",
            "r_reference": "murphydiagram/R/procs.R::fluctuation_test",
            "r_alignment": (
                "Matches loss1-loss2 orientation, mu grid, Table 1 critical values, "
                "Bartlett HAC denominator, lag_truncate range, and dmv_fullsample branches."
            ),
            "critical_value_source": "Giacomini-Rossi (2010) Table 1 as embedded in murphydiagram",
            "alias_warning": None,
        }
    alias_warning = (
        "method='rossi_sekhposyan' is accepted only as a legacy alias for "
        "recursive_fluctuation; Rossi-Sekhposyan forecast rationality is not implemented here."
        if requested_key in {"rossi_sekhposyan", "rs"}
        else None
    )
    return {
        "source_reference": "macroforecast recursive loss-fluctuation extension",
        "external_reference": None,
        "r_reference": None,
        "r_alignment": (
            "No direct R package comparator; this uses expanding-prefix loss-difference "
            "statistics with the same Bartlett HAC helper used by the Giacomini-Rossi branch."
        ),
        "critical_value_source": None,
        "alias_warning": alias_warning,
    }


def _normalize_bootstrap_method(method: str) -> str:
    key = str(method).lower().replace("-", "_")
    if key in {"mcs", "mcs_block", "mcs_fixed_block", "r_mcs"}:
        return "mcs_fixed_block"
    if key in {"stationary", "stationary_bootstrap"}:
        return "stationary_bootstrap"
    if key in {"fixed", "fixed_block", "fixed_block_bootstrap", "moving_block"}:
        return "fixed_block_bootstrap"
    raise ValueError(
        "bootstrap_method must be 'stationary_bootstrap', "
        "'fixed_block_bootstrap', or 'mcs_fixed_block'"
    )


def _normalize_mcs_statistic(statistic: str) -> str:
    key = str(statistic).lower().replace("-", "_")
    if key in {"range", "tr", "t_range"}:
        return "range"
    if key in {"max", "tmax", "t_max"}:
        return "max"
    raise ValueError("statistic must be 'range' or 'max'")


def _loss_panel_groups(
    panel: pd.DataFrame,
    *,
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
        _validate_unique_long_loss_panel(
            working,
            keys=(*group_columns, origin, model),
            label="loss_panel",
        )
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
    numeric = panel.select_dtypes(include=[np.number]).copy()
    if numeric.shape[1] < 2:
        raise ValueError(
            "loss_panel must be either a long panel with "
            f"{sorted(long_required)!r} columns or a wide numeric loss matrix"
        )
    return [("all", "all", numeric)]


def _iterative_mcs_wide(
    wide: pd.DataFrame,
    *,
    alpha: float,
    n_boot: int,
    block_length: int | str,
    bootstrap_method: str,
    statistic: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    numeric = wide.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
    active = [str(column) for column in numeric.columns]
    if numeric.shape[0] < 4 or len(active) < 2:
        return {
            "included_models": active,
            "rejected_models": [],
            "final_p_value": 1.0,
            "block_length": 1,
            "iterations": [],
        }
    removed: list[str] = []
    rejected: list[str] = []
    included_candidates: set[str] = set()
    iterations: list[dict[str, Any]] = []
    final_p = 1.0
    last_block = 1
    step = 0
    while len(active) >= 2:
        active_frame = numeric.loc[:, active]
        matrix = active_frame.to_numpy(dtype=float)
        block = _resolve_mcs_block_length(matrix, block_length)
        last_block = int(block)
        observed, eliminate_pos, scores, boot_stats = _mcs_statistic(
            matrix,
            n_boot=n_boot,
            block_length=block,
            method=bootstrap_method,
            statistic=statistic,
            rng=rng,
        )
        p_value = (
            1.0
            if observed <= 1e-12 and np.nanmax(np.abs(boot_stats)) <= 1e-12
            else float(np.mean(boot_stats > observed))
        )
        eliminate_model = active[int(eliminate_pos)]
        # Hansen-Lunde-Nason (2011) MCS: membership is decided by the cumulative
        # (running-maximum) MCS p-value, not the raw per-step elimination
        # p-value. Once a step fails to reject (cumulative p > alpha) the set is
        # nested, so every model eliminated thereafter -- even one with a small
        # raw p-value -- stays in the confidence set.
        mcs_p_value = max(final_p if step > 0 else 0.0, p_value)
        if mcs_p_value > alpha:
            included_candidates.add(eliminate_model)
        else:
            rejected.append(eliminate_model)
        removed.append(eliminate_model)
        iterations.append(
            {
                "step": int(step),
                "active_models": list(active),
                "statistic": float(observed),
                "p_value": p_value,
                "mcs_p_value": float(mcs_p_value),
                "eliminated_model": eliminate_model if mcs_p_value <= alpha else None,
                "removed_model": eliminate_model,
                "worst_score": float(scores[int(eliminate_pos)]),
                "mean_losses": {
                    model_name: float(active_frame[model_name].mean())
                    for model_name in active
                },
                "block_length": int(block),
            }
        )
        final_p = mcs_p_value
        active.pop(int(eliminate_pos))
        step += 1
    included_candidates.update(active)
    return {
        "included_models": sorted(included_candidates),
        "rejected_models": rejected,
        "final_p_value": final_p,
        "block_length": last_block,
        "iterations": iterations,
        "removed_models": removed,
    }


def _mcs_statistic(
    matrix: np.ndarray,
    *,
    n_boot: int,
    block_length: int,
    method: str,
    statistic: str,
    rng: np.random.Generator,
) -> tuple[float, int, np.ndarray, np.ndarray]:
    # R alignment: MCS/R/MCSprocedure.R computes GetD(losses), bootstraps
    # resampled GetD values, estimates Var(d_ij) and Var(d_i.), then evaluates
    # either Tmax over t_i or TR over abs(t_ij). This is the same statistic
    # construction, with NumPy arrays and explicit finite-variance guards.
    dbar, dibar = _mcs_loss_differences(matrix)
    boot_dbar, boot_dibar = _mcs_bootstrap_loss_differences(
        matrix,
        n_boot=n_boot,
        block_length=block_length,
        method=method,
        rng=rng,
    )
    dbar_var = np.mean((boot_dbar - dbar[:, :, None]) ** 2, axis=2)
    dibar_var = np.mean((boot_dibar - dibar[:, None]) ** 2, axis=1)
    dbar_se = np.sqrt(np.where(dbar_var <= 0.0, 1e-12, dbar_var))
    dibar_se = np.sqrt(np.where(dibar_var <= 0.0, 1e-12, dibar_var))
    tij = dbar / dbar_se
    np.fill_diagonal(tij, 0.0)
    ti = dibar / dibar_se
    tij_res = (boot_dbar - dbar[:, :, None]) / dbar_se[:, :, None]
    for idx in range(tij_res.shape[0]):
        tij_res[idx, idx, :] = 0.0
    ti_res = (boot_dibar - dibar[:, None]) / dibar_se[:, None]
    if statistic == "range":
        scores = np.nanmax(tij, axis=1)
        return (
            float(np.nanmax(np.abs(tij))),
            int(np.nanargmax(scores)),
            scores,
            np.nanmax(np.abs(tij_res), axis=(0, 1)),
        )
    scores = ti
    return float(np.nanmax(ti)), int(np.nanargmax(scores)), scores, np.nanmax(ti_res, axis=0)


def _mcs_loss_differences(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    diff = matrix[:, :, None] - matrix[:, None, :]
    dbar = diff.mean(axis=0)
    dibar = dbar.sum(axis=1) / max(matrix.shape[1] - 1, 1)
    return dbar, dibar


def _mcs_bootstrap_loss_differences(
    matrix: np.ndarray,
    *,
    n_boot: int,
    block_length: int,
    method: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n_obs, n_models = matrix.shape
    boot_dbar = np.empty((n_models, n_models, n_boot))
    boot_dibar = np.empty((n_models, n_boot))
    for boot_id in range(n_boot):
        indices = _mcs_bootstrap_indices(n_obs, block_length, method=method, rng=rng)
        dbar, dibar = _mcs_loss_differences(matrix[indices])
        boot_dbar[:, :, boot_id] = dbar
        boot_dibar[:, boot_id] = dibar
    return boot_dbar, boot_dibar


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
        _validate_unique_long_loss_panel(
            working,
            keys=(*group_columns, origin, model),
            label="loss_panel",
        )
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


def _arch_benchmark_multiple_comparison(
    loss_panel: pd.DataFrame,
    *,
    benchmark: str,
    loss: str,
    alpha: float,
    n_boot: int,
    block_length: int | str,
    bootstrap_method: str,
    p_value_type: str,
    studentize: bool,
    nested: bool,
    random_state: int,
    target: str,
    horizon: str,
    origin: str,
    model: str,
    test: str,
) -> dict[str, Any]:
    _validate_alpha(alpha)
    n_boot = _validate_positive_int(n_boot, "n_boot")
    bootstrap_method = _normalize_bootstrap_method(bootstrap_method)
    p_value_type = _normalize_arch_p_value_type(p_value_type)
    arch_bootstrap = _arch_bootstrap_method(bootstrap_method)
    try:
        from arch.bootstrap import RealityCheck, SPA, StepM
    except ImportError as exc:  # pragma: no cover - exercised only without optional extra
        raise ImportError(
            "superior_predictive_ability_test, reality_check_test, and "
            "stepm_test require the optional arch backend. Install with "
            "`pip install macroforecast[arch]` or `pip install arch`."
        ) from exc

    groups = _reality_check_groups(
        pd.DataFrame(loss_panel).copy(),
        benchmark=benchmark,
        loss=loss,
        target=target,
        horizon=horizon,
        origin=origin,
        model=model,
    )
    cls = {"spa": SPA, "reality_check": RealityCheck, "stepm": StepM}[test]
    rng = np.random.default_rng(random_state)
    records: list[dict[str, Any]] = []
    for target_value, horizon_value, wide in groups:
        if benchmark not in wide.columns:
            raise ValueError(f"benchmark {benchmark!r} is not present in loss rows")
        numeric = wide.apply(pd.to_numeric, errors="coerce")
        candidate_columns = [column for column in numeric.columns if column != benchmark]
        if not candidate_columns:
            records.append(
                {
                    "target": target_value,
                    "horizon": horizon_value,
                    "benchmark": str(benchmark),
                    "status": "no_candidate_models",
                    "n_obs": 0,
                    "n_models": 0,
                }
            )
            continue
        complete = numeric[[benchmark, *candidate_columns]].dropna(axis=0, how="any")
        if complete.shape[0] < 4:
            records.append(
                {
                    "target": target_value,
                    "horizon": horizon_value,
                    "benchmark": str(benchmark),
                    "status": "insufficient_complete_cases",
                    "n_obs": int(complete.shape[0]),
                    "n_models": int(len(candidate_columns)),
                }
            )
            continue
        resolved_block = _resolve_block_length(complete.to_numpy(dtype=float), block_length)
        seed = int(rng.integers(0, np.iinfo(np.uint32).max))
        benchmark_losses = complete[benchmark]
        model_losses = complete[candidate_columns]
        mean_loss_difference = {
            str(column): float((benchmark_losses - model_losses[column]).mean())
            for column in candidate_columns
        }
        references = _arch_multiple_comparison_references(test)
        if test == "stepm":
            comparison = cls(
                benchmark_losses,
                model_losses,
                size=alpha,
                block_size=resolved_block,
                reps=n_boot,
                bootstrap=arch_bootstrap,
                studentize=bool(studentize),
                nested=bool(nested),
                seed=seed,
            )
            comparison.compute()
            superior = [str(name) for name in comparison.superior_models]
            record = {
                "target": target_value,
                "horizon": horizon_value,
                "benchmark": str(benchmark),
                "superior_models": superior,
                "decision": bool(superior),
                "status": "computed",
            }
        else:
            comparison = cls(
                benchmark_losses,
                model_losses,
                block_size=resolved_block,
                reps=n_boot,
                bootstrap=arch_bootstrap,
                studentize=bool(studentize),
                nested=bool(nested),
                seed=seed,
            )
            comparison.compute()
            p_values = {str(key): float(value) for key, value in comparison.pvalues.items()}
            critical_values = {
                str(key): float(value)
                for key, value in comparison.critical_values(pvalue=alpha).items()
            }
            superior = _arch_superior_model_names(
                comparison.better_models(alpha, pvalue_type=p_value_type),
                candidate_columns,
            )
            record = {
                "target": target_value,
                "horizon": horizon_value,
                "benchmark": str(benchmark),
                "p_value": p_values[p_value_type],
                "p_value_type": p_value_type,
                "p_values": p_values,
                "critical_values": critical_values,
                "superior_models": superior,
                "decision": bool(superior),
                "status": "computed",
            }
        record.update(
            {
                "test": test,
                "alpha": float(alpha),
                "n_obs": int(complete.shape[0]),
                "n_models": int(len(candidate_columns)),
                "block_length": int(resolved_block),
                "n_boot": int(n_boot),
                "bootstrap_method": bootstrap_method,
                "arch_bootstrap_method": arch_bootstrap,
                "studentize": bool(studentize),
                "nested": bool(nested),
                "backend": f"arch.bootstrap.{cls.__name__}",
                "loss_orientation": "positive mean_loss_difference means candidate loss is lower than benchmark loss",
                "mean_loss_difference": mean_loss_difference,
                **references,
            }
        )
        records.append(record)
    kind = {
        "spa": "superior_predictive_ability_test",
        "reality_check": "reality_check_test",
        "stepm": "stepm_test",
    }[test]
    return _json_ready({
        "metadata_schema": {"kind": kind, "version": 1},
        "records": records,
        "backend": f"arch.bootstrap.{cls.__name__}",
        "reference_note": (
            "Delegates White/SPA/StepM bootstrap mechanics to the Python arch "
            "package for the general benchmark-loss versus candidate-loss "
            "matrix contract. R references are recorded per computed record; "
            "ttrTests RC/SPA is trading-rule-parameter-grid specific, while "
            "oosanalysis stepm is a generic Romano-Wolf stepdown reference."
        ),
    })


def _normalize_arch_p_value_type(p_value_type: str) -> str:
    key = str(p_value_type).lower().replace("-", "_")
    if key in {"lower", "consistent", "upper"}:
        return key
    raise ValueError("p_value_type must be 'lower', 'consistent', or 'upper'")


def _arch_bootstrap_method(method: str) -> str:
    if method == "stationary_bootstrap":
        return "stationary"
    if method in {"fixed_block_bootstrap", "mcs_fixed_block"}:
        return "moving block"
    raise ValueError("bootstrap_method must be 'stationary_bootstrap' or 'fixed_block_bootstrap'")


def _arch_superior_model_names(indices: Any, candidate_columns: Sequence[Any]) -> list[str]:
    names: list[str] = []
    for value in list(indices):
        if isinstance(value, (int, np.integer)):
            pos = int(value)
            if 0 <= pos < len(candidate_columns):
                names.append(str(candidate_columns[pos]))
            continue
        names.append(str(value))
    return names


def _arch_multiple_comparison_references(test: str) -> dict[str, Any]:
    # R source audit:
    # - ttrTests/R/dataSnoop.R implements White RC and Hansen SPA by rebuilding
    #   a technical-trading parameter grid on each bootstrapped price sample.
    #   That is conceptually aligned with benchmark-vs-candidate data-snooping
    #   tests, but it is not a reusable forecast-loss-matrix API.
    # - oosanalysis-R-library/R/stepm.R implements the Romano-Wolf stepdown
    #   loop from supplied test statistics and a bootstrap statistic matrix.
    #   arch.bootstrap.StepM performs the same stepdown objective after building
    #   SPA-consistent loss-difference statistics from benchmark/candidate losses.
    if test == "spa":
        return {
            "source_reference": "arch.bootstrap.SPA",
            "external_reference": "Hansen (2005) Superior Predictive Ability",
            "r_reference": "ttrTests/R/dataSnoop.R::dataSnoop(test='SPA')",
            "r_alignment": (
                "conceptual RC/SPA alignment only; ttrTests recomputes a technical-trading "
                "parameter grid on bootstrapped price samples, while macroforecast accepts "
                "precomputed benchmark and candidate loss series."
            ),
        }
    if test == "reality_check":
        return {
            "source_reference": "arch.bootstrap.RealityCheck",
            "external_reference": "White (2000) Reality Check",
            "r_reference": "ttrTests/R/dataSnoop.R::dataSnoop(test='RC')",
            "r_alignment": (
                "conceptual RC/SPA alignment only; ttrTests recomputes a technical-trading "
                "parameter grid on bootstrapped price samples, while macroforecast accepts "
                "precomputed benchmark and candidate loss series."
            ),
        }
    if test == "stepm":
        return {
            "source_reference": "arch.bootstrap.StepM",
            "external_reference": "Romano and Wolf (2005) StepM",
            "r_reference": "oosanalysis-R-library/R/stepm.R::stepm",
            "r_alignment": (
                "same stepdown multiple-testing objective; oosanalysis consumes test statistics "
                "and a bootstrap statistic matrix directly, while macroforecast delegates both "
                "loss-difference statistic construction and stepdown recomputation to arch.bootstrap.StepM."
            ),
        }
    raise ValueError(f"unknown arch multiple-comparison test {test!r}")


def _legacy_blocked_oob_reference() -> dict[str, Any]:
    return {
        "source_reference": "macroforecast legacy blocked_oob_reality_check",
        "external_reference": None,
        "r_reference": None,
        "r_alignment": (
            "No exact R package comparator. This is a legacy one-sided benchmark "
            "superiority screen with pairwise and max-centered family-wise block "
            "bootstrap p-values. It is not ttrTests::dataSnoop White RC/SPA and "
            "not oosanalysis::stepm; exact White/SPA/StepM callables are exposed "
            "through reality_check_test, superior_predictive_ability_test, and stepm_test."
        ),
        "exact_multiple_comparison_callables": [
            "reality_check_test",
            "superior_predictive_ability_test",
            "stepm_test",
        ],
    }


def _benchmark_familywise_bootstrap(
    wide: pd.DataFrame,
    *,
    benchmark: str,
    alpha: float,
    n_boot: int,
    block_length: int | str,
    bootstrap_method: str,
    rng: np.random.Generator,
) -> dict[str, dict[str, Any]]:
    candidate_columns = [column for column in wide.columns if column != benchmark]
    if not candidate_columns:
        return {}
    complete = wide[[benchmark, *candidate_columns]].dropna(axis=0, how="any")
    if complete.shape[0] < 2:
        return {}
    diff = (
        complete[benchmark].to_numpy(dtype=float)[:, None]
        - complete[candidate_columns].to_numpy(dtype=float)
    )
    mean_diff = diff.mean(axis=0)
    resolved_block = _resolve_block_length(diff, block_length)
    centered = diff - mean_diff
    boot_means = _bootstrap_means(
        centered,
        n_boot=n_boot,
        block_length=resolved_block,
        method=bootstrap_method,
        rng=rng,
    )
    # White-style family-wise screening: compare each observed improvement
    # against the max centered bootstrap improvement across candidate models.
    # This is not the arch/R White Reality Check; exact benchmark multiple
    # comparison callables are exposed separately through arch.bootstrap.
    boot_max = boot_means.max(axis=1)
    out: dict[str, dict[str, Any]] = {}
    for column, observed in zip(candidate_columns, mean_diff):
        p_value = float(np.mean(boot_max >= float(observed)))
        out[str(column)] = {
            "p_value": p_value,
            "decision": bool(p_value < alpha),
            "n_obs": int(complete.shape[0]),
        }
    return out


def _normalize_correction(correction: str) -> str:
    key = str(correction).lower().replace("-", "_")
    if key in {"hln", "harvey_leybourne_newbold", "small_sample"}:
        return "hln"
    if key in {"none", "nw", "newey_west"}:
        return "none"
    raise ValueError("correction must be 'hln' or 'none'")


def _normalize_dm_input_type(input_type: str) -> str:
    key = str(input_type).lower().replace("-", "_")
    if key in {"loss", "losses"}:
        return "loss"
    if key in {"error", "errors", "forecast_error", "forecast_errors"}:
        return "error"
    raise ValueError("input_type must be 'loss' or 'error'")


def _normalize_alternative(alternative: str) -> str:
    key = str(alternative).lower().replace("-", "_")
    if key in {"two_sided", "two.sided", "two"}:
        return "two_sided"
    if key in {"less", "left"}:
        return "less"
    if key in {"greater", "right"}:
        return "greater"
    raise ValueError("alternative must be 'two_sided', 'less', or 'greater'")


def _normalize_dm_variance_estimator(kernel: str) -> str:
    key = str(kernel).lower().replace("-", "_")
    if key in {"acf", "autocovariance", "dm_acf"}:
        return "acf"
    if key in {"bartlett", "newey_west", "nw"}:
        return "bartlett"
    if key in {"parzen", "andrews"}:
        return key
    raise ValueError("kernel must be 'acf', 'bartlett', 'newey_west', 'parzen', or 'andrews'")


def _p_value_status(statistic: float | None, p_value: float | None) -> str:
    if p_value is not None:
        return "available"
    if statistic is None:
        return "unavailable_degenerate_or_insufficient_data"
    return "unavailable"


def _dm_r_alignment(*, input_type: str, hln: bool, variance_estimator: str) -> str:
    exact_variance = variance_estimator in {"acf", "bartlett"}
    if input_type == "error" and hln and exact_variance:
        return (
            "Exact formula alignment with forecast::dm.test: d=abs(e1)^power-abs(e2)^power, "
            "long-run variance uses varestimator='acf' or 'bartlett', the HLN factor is "
            "applied, and p-values use Student-t df=n_obs-1."
        )
    notes: list[str] = []
    if input_type == "loss":
        notes.append(
            "uses precomputed losses, so the loss differential is supplied directly instead "
            "of being computed from R forecast::dm.test e1/e2 errors"
        )
    if not hln:
        notes.append("omits the HLN small-sample factor used by forecast::dm.test")
    if not exact_variance:
        notes.append(
            f"uses the macroforecast-only {variance_estimator!r} HAC estimator rather than "
            "forecast::dm.test varestimator='acf' or 'bartlett'"
        )
    return "Partial alignment with forecast::dm.test: " + "; ".join(notes) + "."


def _validate_unique_long_loss_panel(panel: pd.DataFrame, *, keys: Sequence[str], label: str) -> None:
    key_columns = [str(column) for column in keys]
    missing = [column for column in key_columns if column not in panel.columns]
    if missing:
        raise ValueError(f"{label} missing key column(s): {missing}")
    duplicates = panel.duplicated(subset=key_columns, keep=False)
    if duplicates.any():
        examples = (
            panel.loc[duplicates, key_columns]
            .drop_duplicates()
            .head(5)
            .to_dict(orient="records")
        )
        raise ValueError(
            f"{label} contains duplicate loss rows for key columns {key_columns}: {examples}. "
            "Aggregate duplicates explicitly before calling."
        )


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
    metadata = dict(result.metadata)
    return {
        "test": requested_name,
        "name": metadata.get("name", requested_name),
        "statistic": result.statistic,
        "p_value": result.p_value,
        "decision": result.decision,
        "alternative": result.alternative,
        "correction_policy": result.correction_policy,
        "n_obs": result.n_obs,
        "statistic_type": metadata.get("statistic_type"),
        "p_value_status": metadata.get("p_value_status"),
        "p_value_reference": metadata.get("p_value_reference"),
        "null_hypothesis": metadata.get("null_hypothesis"),
        "critical_value": metadata.get("critical_value"),
        "source_reference": metadata.get("source_reference"),
        "external_reference": metadata.get("external_reference"),
        "r_reference": metadata.get("r_reference"),
        "r_alignment": metadata.get("r_alignment"),
        "metadata": metadata,
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
    kernel: str = "acf",
    alternative: str = "two_sided",
) -> tuple[float | None, float | None]:
    clean = diff.dropna()
    n = len(clean)
    if n < 3:
        return None, None
    mean = float(clean.mean())
    lag = max(0, int(horizon) - 1)
    # R alignment: forecast/R/DM2.R::dm.test computes
    # d <- abs(e1)^power - abs(e2)^power;
    # STATISTIC <- mean(d) / sqrt(d.var);
    # STATISTIC <- STATISTIC * k, with k the HLN factor, then p-values from
    # t(df=n-1). Here _long_run_variance returns the numerator LRV and this
    # helper divides by n inside sqrt(... / n), which is algebraically the
    # same scaling as forecast::dm.test's d.var.
    variance = _long_run_variance(clean.to_numpy(dtype=float) - mean, kernel=kernel, lag=lag)
    if variance <= 0:
        if horizon == 1:
            return None, None
        variance = _long_run_variance(clean.to_numpy(dtype=float) - mean, kernel=kernel, lag=0)
        if variance <= 0:
            return None, None
    statistic = mean / math.sqrt(variance / n)
    if hln:
        adjustment = math.sqrt((n + 1 - 2 * int(horizon) + int(horizon) * lag / n) / n)
        statistic *= adjustment if adjustment > 0 else 1.0
    return float(statistic), _t_p_value(statistic, df=n - 1, alternative=alternative)


def _mean_hac_test_statistic(
    series: Any,
    *,
    horizon: int,
    kernel: str,
    reference: str = "normal",
    alternative: str = "greater",
) -> tuple[float | None, float | None]:
    clean = pd.Series(series).astype(float).dropna()
    n = len(clean)
    if n < 3:
        return None, None
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    alternative = _normalize_alternative(alternative)
    mean = float(clean.mean())
    lag = max(0, int(horizon) - 1)
    # Source alignment: Clark-West and ENC-T are mean-of-series HAC tests,
    # not HLN Diebold-Mariano tests. This helper keeps that statistic separate:
    # t = mean(q_t) / sqrt(LRV(q_t) / n), where q_t is the adjusted MSPE
    # differential for Clark-West or e_small * (e_small - e_large) for ENC-T.
    variance = _long_run_variance(clean.to_numpy(dtype=float) - mean, kernel=kernel, lag=lag)
    if variance <= 0.0:
        if lag == 0:
            return None, None
        variance = _long_run_variance(clean.to_numpy(dtype=float) - mean, kernel=kernel, lag=0)
        if variance <= 0.0:
            return None, None
    statistic = float(mean / math.sqrt(variance / n))
    reference = str(reference).lower().replace("-", "_")
    if reference in {"normal", "gaussian", "z"}:
        if alternative == "greater":
            p_value = _normal_one_sided_upper_p(statistic)
        elif alternative == "less":
            _upper = _normal_one_sided_upper_p(statistic)
            p_value = None if _upper is None else 1.0 - _upper
        else:
            p_value = _normal_two_sided_p(statistic)
        return statistic, p_value
    if reference in {"t", "student", "student_t"}:
        return statistic, _t_p_value(statistic, df=n - 1, alternative=alternative)
    raise ValueError("reference must be 'normal' or 't'")


def _long_run_variance(
    values: np.ndarray, *, kernel: str = "acf", lag: int | None = None
) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    centered = values - values.mean() if abs(values.mean()) > 1e-12 else values
    gamma_0 = float(np.dot(centered, centered) / n)
    kernel = _normalize_dm_variance_estimator(kernel)
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
    if kernel == "acf":
        for k in range(1, bandwidth + 1):
            if n > k:
                variance += 2.0 * float(np.dot(centered[:-k], centered[k:]) / n)
        return float(variance)
    if kernel == "bartlett":
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


def _t_p_value(statistic: float | None, *, df: int, alternative: str) -> float | None:
    if statistic is None:
        return None
    if math.isinf(statistic):
        return 0.0
    from scipy import stats as _stats

    if alternative == "two_sided":
        return float(2.0 * _stats.t.sf(abs(statistic), df=max(int(df), 1)))
    if alternative == "less":
        return float(_stats.t.cdf(statistic, df=max(int(df), 1)))
    if alternative == "greater":
        return float(_stats.t.sf(statistic, df=max(int(df), 1)))
    raise ValueError("alternative must be 'two_sided', 'less', or 'greater'")


def _normal_two_sided_p(statistic: float | None) -> float | None:
    if statistic is None:
        return None
    if math.isinf(statistic):
        return 0.0
    return max(0.0, min(1.0, math.erfc(abs(statistic) / math.sqrt(2.0))))


def _normal_one_sided_upper_p(statistic: float | None) -> float | None:
    if statistic is None:
        return None
    if math.isinf(statistic):
        return 0.0 if statistic > 0 else 1.0
    return max(0.0, min(1.0, 0.5 * math.erfc(statistic / math.sqrt(2.0))))


def _one_sided_from_two_sided(stat: float | None, p_two: float | None) -> float | None:
    if stat is None or p_two is None:
        return None
    return p_two / 2.0 if stat > 0 else 1.0 - p_two / 2.0


def _direction_test_stat(
    forecast: np.ndarray,
    actual: np.ndarray,
    *,
    method: str,
    forecast_raw: np.ndarray,
    actual_raw: np.ndarray,
) -> tuple[float | None, float | None, float | None]:
    n = len(forecast)
    if n < 2:
        return None, None, None
    # R alignment: tstests/R/dac.R::.pt_test and
    # rugarch/R/rugarch-tests.R::DACTest compute one-sided Hausman-type
    # directional tests with p.value = 1 - pnorm(stat). PT uses sign
    # predictability; AG uses excess profitability.
    success = float((forecast_raw * actual_raw > 0.0).mean())
    p_y = float(actual.mean())
    p_x = float(forecast.mean())
    p_star = p_y * p_x + (1.0 - p_y) * (1.0 - p_x)
    if method == "anatolyev_gerko":
        forecast_sign = np.sign(forecast_raw)
        trading_return = forecast_sign * actual_raw
        a_t = float(np.mean(trading_return))
        b_t = float(np.mean(forecast_sign) * np.mean(actual_raw))
        p_direction = 0.5 * (1.0 + float(np.mean(forecast_sign)))
        variance = (4.0 / (n * n)) * p_direction * (1.0 - p_direction) * float(
            np.sum((actual_raw - np.mean(actual_raw)) ** 2)
        )
        statistic = (a_t - b_t) / math.sqrt(max(variance, 1e-12))
        ag_success = float((trading_return > 0.0).mean())
        return float(statistic), _normal_one_sided_upper_p(statistic), ag_success
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
    return float(statistic), _normal_one_sided_upper_p(statistic), success


def _validate_uniform_values(pit: Any) -> np.ndarray:
    values = pd.Series(pit).astype(float).dropna().to_numpy(dtype=float)
    if values.size == 0:
        return values
    if not np.all(np.isfinite(values)):
        raise ValueError("pit values must be finite")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("pit values must be between 0 and 1")
    return values


def _berkowitz_density_test(values: np.ndarray, *, lags: int, alpha: float) -> dict[str, Any]:
    from scipy import stats as _stats

    lags = max(0, int(lags))
    z = _stats.norm.ppf(np.clip(values, 1e-9, 1.0 - 1e-9))
    n = int(z.size)
    z_mean = float(np.mean(z)) if n else None
    z_std = float(np.std(z, ddof=1)) if n > 1 else None
    out: dict[str, Any] = {
        "mean": z_mean,
        "std": z_std,
        "lags": int(lags),
        "r_reference": "tstests/R/berkowitz.R::berkowitz_test",
        "r_alignment": (
            "Matches validate_uniform -> qnorm -> ARIMA(lags,0,0) unrestricted likelihood, "
            "Normal(0,1) restricted likelihood, chi-squared df=2+lags, and Jarque-Bera moments."
        ),
    }
    if n < max(3, lags + 3) or z_std is None or z_std <= 0.0:
        out.update(
            {
                "lr_statistic": None,
                "p_value": None,
                "reject": False,
                "df": int(2 + lags),
                "status": "insufficient_or_degenerate",
            }
        )
        return out
    # R alignment: tstests/R/berkowitz.R::berkowitz_test maps PIT values
    # through qnorm, fits ARIMA(lags,0,0), and compares that unrestricted
    # likelihood with Normal(0,1) by a chi-squared LR with df = 2 + lags.
    restricted_likelihood = float(_stats.norm.logpdf(z, loc=0.0, scale=1.0).sum())
    fit_status = "arima"
    try:
        from statsmodels.tsa.arima.model import ARIMA

        model = ARIMA(z, order=(lags, 0, 0), trend="c").fit(method_kwargs={"warn_convergence": False})
        unrestricted_likelihood = float(model.llf)
        ar_params = [float(value) for value in np.asarray(getattr(model, "arparams", []), dtype=float)]
        sigma2 = _float_or_none(getattr(model, "scale", None))
    except Exception:
        fit_status = "normal_mle_fallback"
        sigma = float(np.std(z, ddof=0))
        unrestricted_likelihood = float(_stats.norm.logpdf(z, loc=float(np.mean(z)), scale=max(sigma, 1e-12)).sum())
        ar_params = []
        sigma2 = sigma**2
    lr = float(max(-2.0 * (restricted_likelihood - unrestricted_likelihood), 0.0))
    p_value = float(_stats.chi2.sf(lr, df=2 + lags))
    jb_stat, jb_p = _jarque_bera_from_r_moments(z)
    out.update(
        {
            "lr_statistic": lr,
            "p_value": p_value,
            "reject": bool(p_value < alpha),
            "df": int(2 + lags),
            "fit_status": fit_status,
            "ar_params": ar_params,
            "sigma2": sigma2,
            "jarque_bera": {
                "statistic": jb_stat,
                "p_value": jb_p,
                "reject": bool(jb_p is not None and jb_p < alpha),
                "df": 2,
            },
        }
    )
    return out


def jarque_bera_test(series: Any, *, alpha: float = 0.05) -> TestResult:
    """Jarque-Bera test of normality for a single series.

    Uses population (1/n) skewness and excess-kurtosis moments, JB ~ chi2(2),
    matching ``tseries::jarque.bera.test``. The decision rejects normality when
    p < alpha.
    """

    values = pd.Series(series).dropna().astype(float).to_numpy()
    stat, p_value = _jarque_bera_from_r_moments(values)
    return TestResult(
        statistic=None if stat is None else float(stat),
        p_value=None if p_value is None else float(p_value),
        decision=bool(p_value is not None and p_value < alpha),
        alternative="not_normal",
        correction_policy=None,
        n_obs=int(len(values)),
        metadata={"test": "jarque_bera", "df": 2, "reference": "tseries::jarque.bera.test"},
    )


def _fit_statsmodels_var(panel: Any, n_lag: int, trend: str):
    from statsmodels.tsa.api import VAR

    frame = pd.DataFrame(panel)
    frame = frame.select_dtypes("number") if hasattr(frame, "select_dtypes") else frame
    matrix = np.asarray(frame, dtype=float)
    result = VAR(matrix).fit(maxlags=int(n_lag), trend=trend)
    return result, frame, matrix


def var_serial_test(
    panel: Any,
    *,
    n_lag: int = 1,
    test_lags: int | None = None,
    trend: str = "c",
    adjusted: bool = False,
    alpha: float = 0.05,
) -> TestResult:
    """Multivariate residual serial-correlation test for a VAR (vars::serial.test).

    Lutkepohl Portmanteau / LM test of no autocorrelation in the VAR residual
    vector up to ``test_lags`` lags (statsmodels VARResults.test_whiteness).
    Rejects no-serial-correlation when p < alpha.
    """

    result, frame, _ = _fit_statsmodels_var(panel, n_lag, trend)
    lags = int(test_lags) if test_lags is not None else max(int(n_lag) + 1, 12)
    w = result.test_whiteness(nlags=lags, adjusted=bool(adjusted))
    return TestResult(
        statistic=float(w.test_statistic),
        p_value=float(w.pvalue),
        decision=bool(w.pvalue < alpha),
        alternative="serial_correlation",
        correction_policy="adjusted" if adjusted else None,
        n_obs=int(frame.shape[0]),
        metadata={"test": "var_serial_test", "df": int(w.df), "test_lags": lags, "n_lag": int(n_lag)},
    )


def var_normality_test(
    panel: Any,
    *,
    n_lag: int = 1,
    trend: str = "c",
    alpha: float = 0.05,
) -> TestResult:
    """Multivariate normality test for VAR residuals (vars::normality.test).

    Doornik-Hansen / Lutkepohl joint test of skewness and kurtosis on the
    standardised VAR residuals (statsmodels VARResults.test_normality). Rejects
    multivariate normality when p < alpha.
    """

    result, frame, _ = _fit_statsmodels_var(panel, n_lag, trend)
    nrm = result.test_normality()
    return TestResult(
        statistic=float(nrm.test_statistic),
        p_value=float(nrm.pvalue),
        decision=bool(nrm.pvalue < alpha),
        alternative="not_normal",
        correction_policy=None,
        n_obs=int(frame.shape[0]),
        metadata={"test": "var_normality_test", "df": int(getattr(nrm, "df", 0)), "n_lag": int(n_lag)},
    )


def var_arch_test(
    panel: Any,
    *,
    n_lag: int = 1,
    arch_lags: int = 5,
    trend: str = "c",
    alpha: float = 0.05,
) -> TestResult:
    """Multivariate ARCH-LM test for VAR residuals (vars::arch.test, Lutkepohl).

    Regresses the vech of the residual outer products on ``arch_lags`` of its own
    lags and forms the multivariate ARCH-LM statistic
    ``VARCH_LM = T * N * R2_m`` with ``R2_m = 1 - tr(Omega Omega0^-1)/N`` and
    ``N = K(K+1)/2``, chi-squared with ``arch_lags * N^2`` df. Rejects no
    multivariate ARCH when p < alpha.
    """

    from scipy.stats import chi2

    result, frame, _ = _fit_statsmodels_var(panel, n_lag, trend)
    resid = np.asarray(result.resid, dtype=float)
    t_obs, k = resid.shape
    tril = np.tril_indices(k)
    vech = np.column_stack([resid[:, i] * resid[:, j] for i, j in zip(*tril)])
    n_vech = vech.shape[1]
    q = int(arch_lags)
    if t_obs <= q + n_vech + 1:
        return TestResult(
            None, None, False, "multivariate_arch", None, int(t_obs),
            {"test": "var_arch_test", "p_value_status": "insufficient_observations"},
        )
    y = vech[q:]
    design = [np.ones((y.shape[0], 1))]
    for lag in range(1, q + 1):
        design.append(vech[q - lag : t_obs - lag])
    x = np.hstack(design)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid_reg = y - x @ beta
    n_eff = y.shape[0]
    omega = resid_reg.T @ resid_reg / n_eff
    centered = y - y.mean(axis=0)
    omega0 = centered.T @ centered / n_eff
    r2m = 1.0 - float(np.trace(omega @ np.linalg.pinv(omega0))) / n_vech
    statistic = float(n_eff * n_vech * r2m)
    df = int(q * n_vech * n_vech)
    p_value = float(chi2.sf(max(statistic, 0.0), df=df))
    return TestResult(
        statistic=statistic,
        p_value=p_value,
        decision=bool(p_value < alpha),
        alternative="multivariate_arch",
        correction_policy=None,
        n_obs=int(t_obs),
        metadata={"test": "var_arch_test", "df": df, "arch_lags": q, "n_lag": int(n_lag)},
    )


def giacomini_white_test(
    loss_a: Any,
    loss_b: Any,
    *,
    horizon: int = 1,
    instruments: Any | None = None,
    alpha: float = 0.05,
) -> TestResult:
    """Giacomini-White (2006) conditional predictive ability (Wald) test.

    Tests H0: ``E[h_{t-1} * dL_t] = 0`` where ``dL_t = loss_a - loss_b`` is the
    loss differential and ``h_{t-1}`` is a test-function instrument available at
    the forecast origin (default ``[1, dL_{t-h}]``). The statistic
    ``n * Rbar' Omega^{-1} Rbar ~ chi2(q)`` with ``q`` instruments and ``Omega`` a
    Newey-West HAC of ``R_t = h_{t-1} * dL_t`` (horizon-1 lags). Unlike the
    unconditional Diebold-Mariano test, this tests CONDITIONAL equal predictive
    ability. ``instruments`` may be supplied as an array aligned to ``dL_t``.
    """

    _validate_alpha(alpha)
    h = max(1, int(horizon))
    a = pd.Series(loss_a).astype(float).reset_index(drop=True)
    b = pd.Series(loss_b).astype(float).reset_index(drop=True)
    dl = (a - b).dropna().to_numpy()
    if dl.size <= h + 2:
        return TestResult(
            None, None, False, "conditional_equal_predictive_ability", None,
            int(dl.size), {"test": "giacomini_white", "p_value_status": "insufficient_observations"},
        )
    dl_t = dl[h:]
    if instruments is None:
        inst = np.column_stack([np.ones(dl_t.size), dl[:-h]])
    else:
        arr = np.asarray(instruments, dtype=float)
        arr = arr.reshape(-1, 1) if arr.ndim == 1 else arr
        if arr.shape[0] != dl_t.size:
            raise ValueError("instruments must align with the loss differential after the horizon lag")
        inst = np.column_stack([np.ones(dl_t.size), arr])
    reg = inst * dl_t[:, None]
    n, q = reg.shape
    rbar = reg.mean(axis=0)
    centered = reg - rbar
    omega = centered.T @ centered / n
    for lag in range(1, h):
        weight = 1.0 - lag / float(h)
        gamma = centered[lag:].T @ centered[:-lag] / n
        omega += weight * (gamma + gamma.T)
    omega_inv = np.linalg.pinv(omega)
    statistic = float(n * rbar @ omega_inv @ rbar)
    from scipy.stats import chi2

    p_value = float(chi2.sf(statistic, df=q))
    return TestResult(
        statistic=statistic,
        p_value=p_value,
        decision=bool(p_value < alpha),
        alternative="conditional_equal_predictive_ability",
        correction_policy="newey_west_hac",
        n_obs=int(n),
        metadata={"test": "giacomini_white", "instruments": int(q), "horizon": int(h), "df": int(q)},
    )


def granger_causality(
    panel: Any,
    *,
    caused: str,
    causing: str | Sequence[str],
    n_lag: int = 1,
    kind: str = "f",
    trend: str = "c",
    alpha: float = 0.05,
) -> TestResult:
    """Granger causality test in a VAR (R vars::causality / statsmodels).

    Tests whether ``causing`` Granger-causes ``caused`` in a VAR(``n_lag``) fit on
    ``panel``. ``kind='f'`` uses the F statistic, ``'wald'`` the chi-squared Wald.
    The decision rejects non-causality (i.e. ``causing`` does Granger-cause
    ``caused``) when p < alpha.
    """

    from statsmodels.tsa.api import VAR

    frame = pd.DataFrame(panel)
    frame = frame.select_dtypes("number") if hasattr(frame, "select_dtypes") else frame
    causing_list = [causing] if isinstance(causing, str) else list(causing)
    result = VAR(np.asarray(frame, dtype=float)).fit(maxlags=int(n_lag), trend=trend)
    names = list(frame.columns)
    caused_idx = names.index(caused)
    causing_idx = [names.index(c) for c in causing_list]
    gc = result.test_causality(caused_idx, causing_idx, kind=kind)
    return TestResult(
        statistic=float(gc.test_statistic),
        p_value=float(gc.pvalue),
        decision=bool(gc.pvalue < alpha),
        alternative="granger_causes",
        correction_policy=None,
        n_obs=int(frame.shape[0]),
        metadata={
            "test": "granger_causality",
            "caused": str(caused),
            "causing": [str(c) for c in causing_list],
            "kind": kind,
            "n_lag": int(n_lag),
            "df": [int(v) for v in np.atleast_1d(gc.df)],
        },
    )


def instantaneous_causality(
    panel: Any,
    *,
    caused: str,
    causing: str | Sequence[str] | None = None,
    n_lag: int = 1,
    trend: str = "c",
    alpha: float = 0.05,
) -> TestResult:
    """Instantaneous (contemporaneous) causality test in a VAR (vars::causality).

    Tests for contemporaneous correlation between the residuals of ``caused`` and
    the other variables (or ``causing`` if given). Rejects no-instantaneous-
    causality when p < alpha.
    """

    from statsmodels.tsa.api import VAR

    frame = pd.DataFrame(panel)
    frame = frame.select_dtypes("number") if hasattr(frame, "select_dtypes") else frame
    result = VAR(np.asarray(frame, dtype=float)).fit(maxlags=int(n_lag), trend=trend)
    names = list(frame.columns)
    target = names.index(caused)
    if causing is None:
        ic = result.test_inst_causality(target)
    else:
        causing_list = [causing] if isinstance(causing, str) else list(causing)
        ic = result.test_inst_causality(target, [names.index(c) for c in causing_list])
    return TestResult(
        statistic=float(ic.test_statistic),
        p_value=float(ic.pvalue),
        decision=bool(ic.pvalue < alpha),
        alternative="instantaneous_causality",
        correction_policy=None,
        n_obs=int(frame.shape[0]),
        metadata={"test": "instantaneous_causality", "caused": str(caused), "n_lag": int(n_lag)},
    )


def _jarque_bera_from_r_moments(values: np.ndarray) -> tuple[float | None, float | None]:
    from scipy import stats as _stats

    n = int(values.size)
    if n == 0:
        return None, None
    centered = values - float(np.sum(values) / n)
    m2 = float(np.sum(centered**2) / n)
    if m2 <= 0.0:
        return None, None
    m3 = float(np.sum(centered**3) / n)
    m4 = float(np.sum(centered**4) / n)
    skew_sq = (m3 / (m2 ** 1.5)) ** 2
    kurt = m4 / (m2**2)
    statistic = float(n * skew_sq / 6.0 + n * (kurt - 3.0) ** 2 / 24.0)
    return statistic, float(_stats.chi2.sf(statistic, df=2))


def _du_escanciano_shortfall_series(values: np.ndarray, alpha: float) -> np.ndarray:
    return ((alpha - values) * (values <= alpha)) / alpha


def _du_escanciano_unconditional_statistic(values: np.ndarray, alpha: float) -> float:
    return float(np.mean(_du_escanciano_shortfall_series(values, alpha)))


def _du_escanciano_unconditional_pvalue(statistic: float, *, n: int, alpha: float) -> float | None:
    if n <= 0:
        return None
    mu = alpha / 2.0
    sigma = math.sqrt(alpha * (1.0 / 3.0 - alpha / 4.0))
    if sigma <= 0.0:
        return None
    z_value = abs((math.sqrt(n) * (statistic - mu)) / sigma)
    return _normal_two_sided_p(z_value)


def _du_escanciano_conditional_statistic(values: np.ndarray, *, alpha: float, lags: int) -> float | None:
    n = int(values.size)
    if n <= lags:
        return None
    # R alignment: tstests/R/shortfall_de.R::conditional_de_statistic
    # centers cumulative tail shortfall by alpha/2, then applies a
    # portmanteau statistic to its sample autocorrelations.
    shortfall = _du_escanciano_shortfall_series(values, alpha)
    adjusted = shortfall - alpha / 2.0
    variance = float(np.mean(adjusted**2))
    if variance <= 0.0:
        return None
    autocorr = []
    for lag in range(1, lags + 1):
        cov = float(np.sum(adjusted[lag:] * adjusted[:-lag]) / (n - lag))
        autocorr.append(cov / variance)
    return float(n * np.sum(np.asarray(autocorr, dtype=float) ** 2))


def _du_escanciano_conditional_pvalue(statistic: float | None, *, lags: int) -> float | None:
    if statistic is None:
        return None
    from scipy import stats as _stats

    return float(_stats.chi2.sf(statistic, df=lags))


def _du_escanciano_bootstrap_pvalue(
    statistic: float | None,
    *,
    n: int,
    alpha: float,
    lags: int | None,
    n_boot: int,
    rng: np.random.Generator,
) -> float | None:
    if statistic is None:
        return None
    n_boot = _validate_positive_int(n_boot, "n_boot")
    draws = rng.random((n, n_boot))
    if lags is None:
        simulated = np.asarray(
            [_du_escanciano_unconditional_statistic(draws[:, idx], alpha) for idx in range(n_boot)],
            dtype=float,
        )
        return float(2.0 * min(np.mean(simulated <= statistic), np.mean(simulated >= statistic)))
    simulated = np.asarray(
        [
            _du_escanciano_conditional_statistic(draws[:, idx], alpha=alpha, lags=lags)
            for idx in range(n_boot)
        ],
        dtype=float,
    )
    simulated = simulated[np.isfinite(simulated)]
    if simulated.size == 0:
        return None
    return float(np.mean(simulated >= statistic))


def _dynamic_quantile_statistic(
    truth: np.ndarray,
    var_values: np.ndarray,
    *,
    alpha: float,
    lag: int,
    lag_hit: int,
    lag_var: int,
) -> tuple[float | None, float | None, int | None, int]:
    from scipy import stats as _stats

    n = int(min(truth.size, var_values.size))
    if n <= max(lag, lag_hit, lag_var) + 1:
        return None, None, None, 0
    truth = truth[-n:]
    var_values = var_values[-n:]
    hit = np.where(truth < var_values, 1.0 - alpha, -alpha)
    hit_ahead = hit[lag_hit:]
    var_ahead = var_values[lag_var:]
    hit_lag: np.ndarray = np.zeros((n - lag_hit, lag_hit), dtype=float)
    for col in range(lag_hit):
        hit_lag[:, col] = hit[col : n - (lag_hit - col)]
    y_lag: np.ndarray = truth[lag - 1 : n - 1] ** 2
    min_len = min(len(hit_ahead), len(var_ahead), hit_lag.shape[0], len(y_lag))
    if min_len <= 0:
        return None, None, None, 0
    hit_ahead = hit_ahead[-min_len:]
    var_ahead = var_ahead[-min_len:]
    hit_lag = hit_lag[-min_len:]
    y_lag = y_lag[-min_len:]
    x = np.column_stack([np.ones(min_len), var_ahead, hit_lag, y_lag])
    # R alignment: segMGarch/R/DQtest.R::DQtest builds X from a constant,
    # VaR forecasts, lagged hit values, and lagged squared returns, then uses
    # Hit' X (X'X)^(-1) X' Hit / (alpha * (1 - alpha)).
    projection = x @ np.linalg.pinv(x.T @ x) @ x.T
    statistic = float(hit_ahead.T @ projection @ hit_ahead / (alpha * (1.0 - alpha)))
    df = int(x.shape[1])
    p_value = float(_stats.chi2.sf(statistic, df=df))
    return statistic, p_value, df, int(min_len)


def _christoffersen_pelletier_duration_test(hits: np.ndarray, alpha: float) -> dict[str, Any]:
    from scipy import optimize as _optimize
    from scipy import stats as _stats

    hits = np.asarray(hits, dtype=int)
    n = int(hits.size)
    failure_positions = np.flatnonzero(hits == 1)
    if failure_positions.size <= 1:
        return {
            "weibull_b": None,
            "lr_statistic": None,
            "p_value": None,
            "reject": False,
            "n_failures": int(failure_positions.size),
            "r_reference": "tstests/R/var_cp.R::.duration_test",
            "rugarch_reference": "rugarch/R/rugarch-tests.R::VaRDurTest",
            "status": "insufficient_failures",
        }
    durations = np.diff(failure_positions).astype(float).tolist()
    censor = [0] * len(durations)
    if hits[0] == 0:
        censor = [1] + censor
        durations = [float(failure_positions[0] + 1)] + durations
    if hits[-1] == 0:
        censor = censor + [1]
        durations = durations + [float(n - failure_positions[-1] - 1)]
    d = np.asarray(durations, dtype=float)
    c = np.asarray(censor, dtype=int)
    valid = np.isfinite(d) & (d > 0)
    d = d[valid]
    c = c[valid]
    if d.size <= 1:
        return {
            "weibull_b": None,
            "lr_statistic": None,
            "p_value": None,
            "reject": False,
            "n_failures": int(failure_positions.size),
            "r_reference": "tstests/R/var_cp.R::.duration_test",
            "rugarch_reference": "rugarch/R/rugarch-tests.R::VaRDurTest",
            "status": "insufficient_durations",
        }
    # R alignment: tstests/R/var_cp.R::.duration_test and
    # rugarch/R/rugarch-tests.R::VaRDurTest construct inter-failure durations
    # with left/right censoring, estimate a Weibull duration likelihood, and
    # test the exponential no-memory restriction b=1.  The likelihood formula
    # follows rugarch's .likDurationW density/survival treatment; the current
    # tstests source has the same duration construction and LR shell.
    def objective(b: float) -> float:
        return -_weibull_duration_loglik(float(b), d, c)

    try:
        result = _optimize.minimize_scalar(objective, bounds=(0.001, 10.0), method="bounded")
        b_hat = float(result.x)
        unrestricted = float(-result.fun)
    except Exception:
        b_hat = None
        unrestricted = None
    restricted = _weibull_duration_loglik(1.0, d, c)
    if b_hat is None or unrestricted is None or not np.isfinite(restricted):
        lr = None
        p_value = None
    else:
        lr = float(max(-2.0 * (restricted - unrestricted), 0.0))
        p_value = float(_stats.chi2.sf(lr, df=1))
    return {
        "weibull_b": b_hat,
        "lr_statistic": lr,
        "p_value": p_value,
        "reject": bool(p_value is not None and p_value < alpha),
        "n_failures": int(failure_positions.size),
        "n_durations": int(d.size),
        "r_reference": "tstests/R/var_cp.R::.duration_test",
        "rugarch_reference": "rugarch/R/rugarch-tests.R::VaRDurTest",
        "r_alignment": (
            "Matches Christoffersen-Pelletier duration construction with left/right censoring "
            "and Weibull b=1 exponential restriction; likelihood density/survival treatment "
            "follows rugarch VaRDurTest."
        ),
    }


def _weibull_duration_loglik(shape: float, durations: np.ndarray, censor: np.ndarray) -> float:
    if shape <= 0.0 or durations.size == 0:
        return -math.inf
    numerator = float(durations.size - int(censor[0]) - int(censor[-1]))
    if numerator <= 0.0:
        return -math.inf
    scale = (numerator / float(np.sum(durations**shape))) ** (1.0 / shape)
    if not np.isfinite(scale) or scale <= 0.0:
        return -math.inf
    log_survival = -((scale * durations) ** shape)
    log_density = (
        shape * math.log(scale)
        + math.log(shape)
        + (shape - 1.0) * np.log(durations)
        - (scale * durations) ** shape
    )
    terms = np.where(censor == 1, log_survival, log_density)
    total = float(np.sum(terms))
    return total if np.isfinite(total) else -math.inf


def _kupiec_test(hits: np.ndarray, alpha: float) -> dict[str, Any]:
    from scipy import stats as _stats

    # R alignment: tstests/R/var_cp.R::.lr_unc_coverage and
    # rugarch/R/rugarch-tests.R::.LR.uc compare the restricted Bernoulli
    # likelihood at alpha with the unrestricted likelihood at N/T. Boundary
    # cases such as zero violations are not silently assigned LR = 0.
    hits = np.asarray(hits, dtype=int)
    total = int(hits.size)
    x_hits = int(hits.sum())
    if total == 0:
        lr = None
        p_value = None
        p_hat = None
    else:
        p_hat = x_hits / total
        restricted = _bernoulli_loglik(x_hits, total, float(alpha))
        unrestricted = _bernoulli_loglik(x_hits, total, p_hat)
        lr = float(max(-2.0 * (restricted - unrestricted), 0.0))
        p_value = float(1.0 - _stats.chi2.cdf(lr, df=1))
    return {
        "hits_rate": p_hat,
        "lr_statistic": None if lr is None else float(lr),
        "p_value": p_value,
        "reject": bool(p_value is not None and p_value < alpha),
        "r_reference": "tstests/R/var_cp.R::.lr_unc_coverage",
        "rugarch_reference": "rugarch/R/rugarch-tests.R::.LR.uc",
        "r_alignment": (
            "Restricted Bernoulli likelihood at alpha versus unrestricted likelihood at hit rate; "
            "boundary zero/all-hit cases are evaluated through likelihood terms, not forced to pass."
        ),
    }


def _christoffersen_independence_test(hits: np.ndarray, alpha: float) -> dict[str, Any]:
    from scipy import stats as _stats

    hits = np.asarray(hits, dtype=int)
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
    if hits.size < 2:
        lr = None
        p_value = None
    else:
        pi01 = n01 / max(n00 + n01, 1)
        pi11 = n11 / max(n10 + n11, 1)
        pi = (n01 + n11) / max(hits.size - 1, 1)
        restricted = _transition_loglik(n00, n01, n10, n11, pi, pi)
        unrestricted = _transition_loglik(n00, n01, n10, n11, pi01, pi11)
        lr = float(max(-2.0 * (restricted - unrestricted), 0.0))
        p_value = float(1.0 - _stats.chi2.cdf(lr, df=1))
    return {
        "lr_statistic": None if lr is None else float(lr),
        "p_value": p_value,
        "reject": bool(p_value is not None and p_value < alpha),
        "transition_counts": {
            "n00": int(n00),
            "n01": int(n01),
            "n10": int(n10),
            "n11": int(n11),
        },
        "r_reference": "tstests/R/var_cp.R::.coverage_test",
        "rugarch_reference": "rugarch/R/rugarch-tests.R::.LR.cc",
        "r_alignment": (
            "Markov transition LR comparing common failure probability with separate p01/p11 "
            "transition probabilities; zero-count likelihood terms use the same limiting "
            "convention as the R likelihood products."
        ),
    }


def _density_interval_reference() -> dict[str, Any]:
    return {
        "source_reference": "macroforecast density_interval_tests",
        "external_reference": "tstests density and VaR diagnostic functions",
        "r_reference": {
            "berkowitz": "tstests/R/berkowitz.R::berkowitz_test",
            "shortfall_de": "tstests/R/shortfall_de.R::shortfall_de_test",
            "coverage": "tstests/R/var_cp.R::var_cp_test",
            "rugarch_coverage": "rugarch/R/rugarch-tests.R::VaRTest",
        },
        "r_alignment": (
            "Composite PIT diagnostic wrapper. Berkowitz, Du-Escanciano, Kupiec, and "
            "Christoffersen components are R-aligned individually; engle_manganelli_dq "
            "inside this wrapper is a PIT hit-only proxy, while dynamic_quantile_test is "
            "the full segMGarch-aligned VaR DQ callable."
        ),
    }


def _interval_coverage_reference() -> dict[str, Any]:
    return {
        "source_reference": "macroforecast interval_coverage_test",
        "external_reference": "tstests var_cp_test and rugarch VaRTest/VaRDurTest",
        "r_reference": "tstests/R/var_cp.R::var_cp_test",
        "rugarch_reference": "rugarch/R/rugarch-tests.R::VaRTest",
        "r_alignment": (
            "Uses the same lower-tail/interval miss indicator, Kupiec unconditional coverage LR, "
            "Christoffersen independence LR, combined conditional coverage LR, and "
            "Christoffersen-Pelletier duration LR family."
        ),
    }


def _bernoulli_loglik(successes: int, total: int, probability: float) -> float:
    failures = total - successes
    return _log_probability_term(successes, probability) + _log_probability_term(
        failures,
        1.0 - probability,
    )


def _transition_loglik(
    n00: int,
    n01: int,
    n10: int,
    n11: int,
    p01: float,
    p11: float,
) -> float:
    return (
        _log_probability_term(n00, 1.0 - p01)
        + _log_probability_term(n01, p01)
        + _log_probability_term(n10, 1.0 - p11)
        + _log_probability_term(n11, p11)
    )


def _log_probability_term(count: int, probability: float) -> float:
    if count == 0:
        return 0.0
    if probability <= 0.0:
        return -math.inf
    if probability >= 1.0:
        return 0.0 if count > 0 else 0.0
    return float(count * math.log(probability))


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
        "source_reference": "macroforecast PIT hit-only DQ proxy",
        "r_reference": None,
        "r_alignment": (
            "No direct R comparator. This proxy regresses PIT tail hits on their own lags only; "
            "use dynamic_quantile_test(y_true, var, ...) for the segMGarch-aligned "
            "Engle-Manganelli VaR DQ statistic."
        ),
        "note": "PIT hit-only proxy; use dynamic_quantile_test(y_true, var, ...) for Engle-Manganelli DQ.",
    }


def _residual_test_statistic(
    test_name: str,
    residuals: pd.Series,
    lag: int,
    *,
    alpha: float,
    model_df: int,
    exog: Any | None,
    demean_arch: bool,
) -> dict[str, Any]:
    key = str(test_name).lower().replace("-", "_")
    values = residuals.astype(float).dropna()
    lag_used = min(int(lag), max(len(values) - 1, 0))
    base: dict[str, Any] = {
        "test": key,
        "statistic": None,
        "p_value": None,
        "decision": False,
        "lag_used": int(lag_used),
        "df": None,
        "n_obs": int(len(values)),
        "source_reference": None,
        "r_reference": None,
        "r_alignment": None,
        "status": "computed",
    }
    if len(values) < 3:
        base["status"] = "insufficient_observations"
        return base
    if key == "durbin_watson":
        statistic = _durbin_watson_statistic(values.to_numpy(dtype=float))
        base.update(
            {
                "statistic": statistic,
                "source_reference": "lmtest/R/dwtest.R::dwtest statistic",
                "r_reference": "lmtest/R/dwtest.R::dwtest",
                "r_alignment": (
                    "Statistic matches lmtest dwtest, sum(diff(residuals)^2)/sum(residuals^2). "
                    "P-value is omitted because lmtest computes exact/asymptotic probabilities "
                    "from the original regression design matrix, which is unavailable under the "
                    "residual-series contract."
                ),
                "status": "statistic_only_no_p_value",
            }
        )
        return base
    if key == "jarque_bera_normality":
        statistic, p_value = _jarque_bera_from_r_moments(values.to_numpy(dtype=float))
        base.update(
            {
                "statistic": statistic,
                "p_value": p_value,
                "decision": bool(p_value is not None and p_value < alpha),
                "df": 2,
                "source_reference": "tseries/R/test.R::jarque.bera.test",
                "r_reference": "tseries/R/test.R::jarque.bera.test",
                "r_alignment": (
                    "Matches tseries population-moment convention: n*skewness^2/6 + "
                    "n*(kurtosis-3)^2/24 with chi-squared df=2."
                ),
            }
        )
        return base
    max_lag = lag_used
    if max_lag < 1:
        base["status"] = "insufficient_lag"
        return base
    if key == "ljung_box_q":
        from statsmodels.stats.diagnostic import acorr_ljungbox

        if max_lag <= model_df:
            base["status"] = "lag_not_greater_than_model_df"
            base["source_reference"] = "stats::Box.test(type='Ljung-Box')"
            base["r_reference"] = "stats::Box.test(type='Ljung-Box')"
            base["r_alignment"] = "R returns unavailable p-values when lag is not greater than fitdf."
            return base
        # R alignment: stats::Box.test(..., type="Ljung-Box", fitdf=model_df)
        # and forecast/R/checkresiduals.R compute Q = n(n+2) sum rho_k^2/(n-k)
        # with chi-squared df = lag - fitdf.
        result = acorr_ljungbox(values, lags=[max_lag], model_df=model_df, return_df=True)
        statistic = float(result["lb_stat"].iloc[0])
        p_value = float(result["lb_pvalue"].iloc[0])
        base.update(
            {
                "statistic": statistic,
                "p_value": p_value,
                "decision": bool(p_value < alpha),
                "df": int(max_lag - model_df),
                "source_reference": "stats::Box.test(type='Ljung-Box'); forecast/R/checkresiduals.R",
                "r_reference": "stats::Box.test(type='Ljung-Box')",
                "r_alignment": (
                    "Matches Ljung-Box Q = n(n+2) sum rho_k^2/(n-k), with chi-squared "
                    "df = lag - model_df; model_df maps to R fitdf."
                ),
            }
        )
        return base
    if key == "breusch_godfrey_serial_correlation":
        try:
            statistic, p_value, n_used = _breusch_godfrey_residual_series_statistic(
                values.to_numpy(dtype=float),
                max_lag=max_lag,
                exog=exog,
            )
            base.update(
                {
                    "statistic": statistic,
                    "p_value": p_value,
                    "decision": bool(p_value is not None and p_value < alpha),
                    "df": int(max_lag),
                    "n_obs": int(n_used),
                    "source_reference": "lmtest/R/bgtest.R::bgtest residual-series contract",
                    "r_reference": "lmtest/R/bgtest.R::bgtest",
                    "r_alignment": (
                        "Matches bgtest(type='Chisq') auxiliary LM formula under the residual-series "
                        "contract: fit residuals on exog, add fill=0 lagged residual columns, and use "
                        "n*sum(aux_fitted^2)/sum(original_residuals^2)."
                    ),
                }
            )
            return base
        except Exception:
            base["status"] = "failed"
            base["source_reference"] = "lmtest/R/bgtest.R::bgtest residual-series contract"
            base["r_reference"] = "lmtest/R/bgtest.R::bgtest"
            base["r_alignment"] = "Failed while evaluating residual-series bgtest auxiliary regression."
            return base
    if key == "arch_lm":
        from statsmodels.stats.diagnostic import het_arch

        try:
            arch_values = values.to_numpy(dtype=float)
            if demean_arch:
                arch_values = arch_values - float(np.mean(arch_values))
            # statsmodels documents het_arch as verified against FinTS::ArchTest:
            # regress x_t^2 on a constant and lagged squared residuals, then use
            # n_eff * R^2 with chi-squared df equal to the number of ARCH lags.
            statistic, p_value, _, _ = het_arch(arch_values, nlags=max_lag, ddof=model_df)
            base.update(
                {
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "decision": bool(p_value < alpha),
                    # statsmodels het_arch p-value uses df = nlags - ddof; report the
                    # matching df so the reported df is consistent with the p-value.
                    "df": int(max(1, max_lag - model_df)),
                    "source_reference": "FinTS/R/ArchTest.R::ArchTest",
                    "r_reference": "FinTS/R/ArchTest.R::ArchTest",
                    "r_alignment": (
                        "Matches FinTS ArchTest when model_df=0: embed x^2, regress current squared "
                        "residuals on lagged squared residuals, and use effective sample size times "
                        "auxiliary R^2. model_df uses statsmodels ddof adjustment beyond the R API."
                    ),
                    "demean_arch": bool(demean_arch),
                }
            )
            return base
        except Exception:
            base["status"] = "failed"
            base["source_reference"] = "FinTS/R/ArchTest.R::ArchTest"
            base["r_reference"] = "FinTS/R/ArchTest.R::ArchTest"
            base["r_alignment"] = "Failed while evaluating FinTS-style ARCH LM auxiliary regression."
            return base
    base["status"] = "unknown_test"
    return base


def _durbin_watson_statistic(values: np.ndarray) -> float | None:
    denominator = float(np.sum(values**2))
    if denominator <= 0.0:
        return None
    return float(np.sum(np.diff(values) ** 2) / denominator)


def _breusch_godfrey_residual_series_statistic(
    values: np.ndarray,
    *,
    max_lag: int,
    exog: Any | None,
) -> tuple[float | None, float | None, int]:
    from scipy import stats as _stats

    y = np.asarray(values, dtype=float)
    n = int(y.size)
    x = _coerce_residual_exog(exog, n)
    if n <= max_lag or x.shape[0] != n:
        return None, None, n
    # R alignment: lmtest/R/bgtest.R first fits y on X, builds lagged original
    # residual columns with fill=0, then computes the Chisq LM statistic as
    # n * sum(aux_fitted^2) / sum(original_residuals^2).
    coef, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ coef
    z_columns = [
        np.concatenate([np.zeros(order), resid[: n - order]])
        for order in range(1, max_lag + 1)
    ]
    z = np.column_stack(z_columns)
    aux_x = np.column_stack([x, z])
    aux_coef, _, _, _ = np.linalg.lstsq(aux_x, resid, rcond=None)
    fitted = aux_x @ aux_coef
    denom = float(np.sum(resid**2))
    if denom <= 0.0:
        return None, None, n
    statistic = float(n * np.sum(fitted**2) / denom)
    return statistic, float(_stats.chi2.sf(statistic, df=max_lag)), n


def _coerce_residual_exog(exog: Any | None, n_obs: int) -> np.ndarray:
    if exog is None:
        return np.ones((n_obs, 1), dtype=float)
    frame = pd.DataFrame(exog).astype(float)
    if frame.shape[0] != n_obs:
        raise ValueError("exog must have the same number of rows as residuals")
    values = frame.to_numpy(dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.shape[1] == 0:
        return np.ones((n_obs, 1), dtype=float)
    has_constant = np.any(np.nanstd(values, axis=0) <= 1e-12)
    if not has_constant:
        values = np.column_stack([np.ones(n_obs), values])
    return values


def _gr_critical_value(window_ratio: float, alpha: float) -> float:
    ratio = _normalize_gr_window_ratio(window_ratio)
    if alpha not in {0.05, 0.10}:
        raise ValueError("alpha must be 0.05 or 0.10 for giacomini_rossi critical values")
    # R alignment: murphydiagram/R/procs.R::fluctuation_test uses Table 1 of
    # Giacomini-Rossi (2010), indexed by mu in {0.1, ..., 0.9}.
    cv_5 = {
        0.1: 3.393,
        0.2: 3.179,
        0.3: 3.012,
        0.4: 2.890,
        0.5: 2.779,
        0.6: 2.634,
        0.7: 2.560,
        0.8: 2.433,
        0.9: 2.248,
    }
    cv_10 = {
        0.1: 3.170,
        0.2: 2.948,
        0.3: 2.766,
        0.4: 2.626,
        0.5: 2.500,
        0.6: 2.356,
        0.7: 2.252,
        0.8: 2.130,
        0.9: 1.950,
    }
    return float((cv_5 if alpha == 0.05 else cv_10)[ratio])


def _normalize_gr_window_ratio(window_ratio: float) -> float:
    ratio = round(float(window_ratio), 1)
    if abs(float(window_ratio) - ratio) > 1e-12 or ratio not in {round(idx / 10, 1) for idx in range(1, 10)}:
        raise ValueError("window_ratio must be one of 0.1, 0.2, ..., 0.9 for giacomini_rossi")
    return float(ratio)


def _validate_gr_lag_truncate(lag_truncate: int) -> int:
    lag_truncate = int(lag_truncate)
    if lag_truncate not in range(0, 6):
        raise ValueError("lag_truncate must be in {0, 1, ..., 5}")
    return lag_truncate


def _giacomini_rossi_fluctuation_path(
    loss_difference: np.ndarray,
    *,
    window_size: int,
    dmv_fullsample: bool,
    lag_truncate: int,
) -> list[float]:
    diff = np.asarray(loss_difference, dtype=float)
    n = int(diff.size)
    if n < window_size or window_size < 1:
        return []
    full_hac = _murphydiagram_bartlett_hac(diff, lag_truncate)
    stats: list[float] = []
    for end in range(window_size, n + 1):
        window = diff[end - window_size : end]
        numerator = float(np.mean(window))
        # R alignment: murphydiagram/R/procs.R::fluctuation_test constructs
        # ld <- loss1-loss2, dm1 <- sqrt(m)*mean(ld_window)/sqrt(vHAC(ld)),
        # and dm2 <- mean(ld_window)/sqrt(vHAC(ld_window)/m).  We mirror the
        # two dmv_fullsample branches exactly, using the same Bartlett HAC
        # denominator convention in _murphydiagram_bartlett_hac.
        if dmv_fullsample:
            denominator = math.sqrt(max(full_hac, 1e-12))
            stat = math.sqrt(window_size) * numerator / denominator
        else:
            rolling_hac = _murphydiagram_bartlett_hac(window, lag_truncate)
            stat = numerator / math.sqrt(max(rolling_hac / window_size, 1e-12))
        stats.append(float(stat))
    return stats


def _recursive_loss_fluctuation_path(
    loss_difference: np.ndarray,
    *,
    window_size: int,
    lag_truncate: int,
) -> list[float]:
    diff = np.asarray(loss_difference, dtype=float)
    n = int(diff.size)
    stats: list[float] = []
    for end in range(window_size, n + 1):
        window = diff[:end]
        hac = _murphydiagram_bartlett_hac(window, lag_truncate)
        stats.append(float(math.sqrt(len(window)) * np.mean(window) / math.sqrt(max(hac, 1e-12))))
    return stats


def _murphydiagram_bartlett_hac(values: np.ndarray, lag_truncate: int) -> float:
    x = np.asarray(values, dtype=float).reshape(-1)
    n = int(x.size)
    if n <= 1:
        return 0.0
    # R alignment: murphydiagram/R/procs.R::vHAC with meth="Bartlett" and
    # prewhite=FALSE uses t(u)u/(n-1) plus Bartlett-weighted lag covariances.
    vcv = float(np.dot(x, x) / (n - 1))
    for lag in range(1, min(int(lag_truncate), n - 1) + 1):
        weight = 1.0 - lag / (lag_truncate + 1)
        cov = float(np.dot(x[lag:], x[:-lag]) / (n - 1))
        vcv += 2.0 * weight * cov
    return float(max(vcv, 0.0))


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


def _resolve_mcs_block_length(matrix: np.ndarray, value: int | str, *, min_k: int = 3) -> int:
    n = matrix.shape[0]
    if n <= 2:
        return 1
    if isinstance(value, (int, np.integer)) and int(value) >= 0:
        block = int(value)
    elif isinstance(value, str) and value.isdigit():
        block = int(value)
    else:
        # R alignment: MCS/R/MCSprocedure.R sets k to max(ar(loss_i)$order),
        # then enforces min.k = 3. Here statsmodels' AR order selection is
        # used to reproduce that data-dependent block-length rule in Python.
        block = max(_selected_ar_order(matrix[:, idx]) for idx in range(matrix.shape[1]))
        block = max(block, int(min_k))
    if block < 0:
        raise ValueError("block_length must be nonnegative")
    if block >= n:
        raise ValueError("MCS block_length must be smaller than the number of observations")
    return max(1, block)


def _selected_ar_order(values: np.ndarray) -> int:
    clean = pd.Series(values).astype(float).dropna().to_numpy(dtype=float)
    if clean.size < 8 or np.std(clean) <= 1e-12:
        return 0
    maxlag = max(1, min(10, clean.size // 3))
    try:
        from statsmodels.tsa.ar_model import ar_select_order

        result = ar_select_order(clean, maxlag=maxlag, ic="aic", old_names=False)
        selected = getattr(result, "ar_lags", None)
        if selected is None or len(selected) == 0:
            return 0
        return int(max(selected))
    except Exception:
        return 0


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


def _mcs_bootstrap_indices(
    n: int,
    block_length: int,
    *,
    method: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if method == "stationary_bootstrap":
        return _stationary_bootstrap_indices(n, block_length, rng)
    if method == "fixed_block_bootstrap":
        return _fixed_block_bootstrap_indices(n, block_length, rng)
    if block_length <= 1:
        return rng.integers(0, n, size=n)
    # R alignment: MCS/R/internalFunctions.R::GetIndices samples block starts
    # and concatenates fixed consecutive blocks, truncating the result to T.
    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n - block_length, size=n_blocks)
    indices = np.concatenate([start + np.arange(block_length) for start in starts])
    return indices[:n].astype(np.int64)


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
    "anatolyev_gerko_test",
    "clark_west_test",
    "conditional_predictive_ability_test",
    "custom_test",
    "cw_test",
    "density_interval_tests",
    "directional_accuracy_test",
    "dm_test",
    "jarque_bera_test",
    "giacomini_white_test",
    "var_serial_test",
    "var_normality_test",
    "var_arch_test",
    "granger_causality",
    "instantaneous_causality",
    "dmp_test",
    "dynamic_quantile_test",
    "equal_predictive_tests",
    "enc_new_test",
    "enc_t_test",
    "gw_test",
    "harvey_newbold_test",
    "henriksson_merton_test",
    "hn_test",
    "interval_coverage_test",
    "blocked_oob_reality_check",
    "iterative_model_confidence_set",
    "model_confidence_set",
    "nested_tests",
    "pesaran_timmermann_test",
    "pit_autocorrelation_test",
    "pit_histogram",
    "reality_check_test",
    "residual_diagnostics",
    "shortfall_de_test",
    "stepm_test",
    "superior_predictive_ability_test",
]
