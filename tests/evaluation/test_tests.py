from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf


def test_equal_predictive_accuracy_callables_return_test_results() -> None:
    loss_a = pd.Series([0.2, 0.3, 0.1, 0.4, 0.2, 0.3])
    loss_b = pd.Series([0.4, 0.5, 0.2, 0.5, 0.3, 0.6])
    error_a = np.sqrt(loss_a)
    error_b = np.sqrt(loss_b)

    dm = mf.tests.dm_test(loss_a, loss_b, horizon=1)
    gw = mf.tests.gw_test(loss_a, loss_b, horizon=1)
    dmp = mf.tests.dmp_test([loss_a - loss_b, (loss_a - loss_b) * 0.5])
    stacked = mf.tests.equal_predictive_tests(
        loss_a,
        loss_b,
        tests=("dm", "gw", "dmp", "hn"),
        error_a=error_a,
        error_b=error_b,
    )

    assert isinstance(dm, mf.tests.TestResult)
    assert dm.metadata["name"] == "Diebold-Mariano"
    assert dm.metadata["statistic_type"] == "t"
    assert dm.metadata["p_value_status"] == "available"
    assert dm.metadata["null_hypothesis"] == "equal predictive accuracy"
    assert gw.metadata["name"] == "Giacomini-White"
    assert gw.metadata["r_reference"] is None
    assert "No exact R comparator" in gw.metadata["r_alignment"]
    assert dmp.n_obs == 12
    assert 0.0 <= dmp.p_value <= 1.0
    assert dmp.metadata["statistic_type"] == "z"
    assert dmp.metadata["r_reference"] is None
    assert set(stacked["test"]) == {"dm", "gw", "dmp", "hn"}
    assert {
        "statistic_type",
        "p_value_status",
        "null_hypothesis",
        "source_reference",
        "r_reference",
        "r_alignment",
    } <= set(stacked.columns)
    assert stacked.loc[stacked["test"] == "dm", "r_reference"].iloc[0] == "forecast/R/DM2.R::dm.test"
    assert pd.isna(stacked.loc[stacked["test"] == "gw", "r_reference"].iloc[0])
    assert stacked.attrs["macroforecast_metadata_schema"]["kind"] == "equal_predictive_tests"
    assert "paper_table_ready_columns" in stacked.attrs["macroforecast_metadata_schema"]
    dm_payload = json.loads(dm.to_json())
    assert dm_payload["metadata_schema"]["kind"] == "forecast_test_result"
    assert dm_payload["metadata_schema"]["version"] == 1


def test_dm_test_matches_forecast_package_formula_for_error_inputs() -> None:
    e1 = pd.Series([0.8, -0.5, 0.4, -0.3, 0.7, -0.2, 0.6, -0.1])
    e2 = pd.Series([0.4, -0.3, 0.2, -0.2, 0.5, -0.1, 0.4, -0.2])
    horizon = 2
    d = e1.abs() ** 2 - e2.abs() ** 2
    centered = d - d.mean()
    n = len(d)
    cov0 = float(np.dot(centered, centered) / n)
    cov1 = float(np.dot(centered[:-1], centered[1:]) / n)
    long_run_variance = cov0 + 2.0 * cov1
    hln = math.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    expected_stat = float((d.mean() / math.sqrt(long_run_variance / n)) * hln)
    from scipy import stats as _stats

    expected_two_sided_p = float(2.0 * _stats.t.sf(abs(expected_stat), df=n - 1))
    expected_less_p = float(_stats.t.cdf(expected_stat, df=n - 1))

    result = mf.tests.dm_test(
        e1,
        e2,
        horizon=horizon,
        input_type="error",
        kernel="acf",
        alternative="two_sided",
    )

    assert np.isclose(result.statistic, expected_stat)
    assert np.isclose(result.p_value, expected_two_sided_p)
    assert result.metadata["r_reference"] == "forecast/R/DM2.R::dm.test"
    assert result.metadata["source_reference"] == "forecast/R/DM2.R::dm.test"
    assert result.metadata["variance_estimator"] == "acf"
    assert result.metadata["input_type"] == "error"
    assert result.metadata["p_value_reference"] == "Student-t reference with df=n_obs-1"
    assert result.metadata["p_value_status"] == "available"
    assert "Exact formula alignment" in result.metadata["r_alignment"]

    less_result = mf.tests.dm_test(
        e1,
        e2,
        horizon=horizon,
        input_type="error",
        kernel="acf",
        alternative="less",
    )
    assert np.isclose(less_result.p_value, expected_less_p)


def test_equal_predictive_metadata_distinguishes_dm_extensions() -> None:
    loss_a = pd.Series([0.2, 0.4, 0.3, 0.5, 0.6, 0.4])
    loss_b = pd.Series([0.3, 0.3, 0.2, 0.4, 0.5, 0.3])
    error_a = np.sqrt(loss_a)
    error_b = np.sqrt(loss_b)

    partial_dm = mf.tests.dm_test(
        loss_a,
        loss_b,
        horizon=2,
        input_type="loss",
        correction="none",
        kernel="parzen",
    )
    dmp = mf.tests.dmp_test([loss_a - loss_b])
    hn = mf.tests.harvey_newbold_test(error_a, error_b)
    stacked = mf.tests.equal_predictive_tests(
        loss_a,
        loss_b,
        tests=("dm", "gw", "dmp", "hn"),
        error_a=error_a,
        error_b=error_b,
    )

    assert "Partial alignment" in partial_dm.metadata["r_alignment"]
    assert "precomputed losses" in partial_dm.metadata["r_alignment"]
    assert "parzen" in partial_dm.metadata["r_alignment"]
    assert dmp.metadata["p_value_reference"] == "standard normal two-sided reference"
    assert dmp.metadata["r_reference"] is None
    assert hn.metadata["r_reference"] is None
    assert "not forecast::dm.test" in hn.metadata["note"]
    assert stacked.attrs["macroforecast_metadata_schema"]["r_reference"] == {
        "dm": "forecast/R/DM2.R::dm.test",
        "gw": None,
        "dmp": None,
        "hn": None,
    }
    assert stacked.loc[stacked["test"] == "hn", "p_value_status"].iloc[0] == "available"


def custom_statistic(loss_a, loss_b, *, scale=1.0):
    diff = pd.Series(loss_a).sub(pd.Series(loss_b)).dropna()
    return {
        "statistic": float(diff.mean() * scale),
        "p_value": 0.04,
        "n_obs": int(len(diff)),
        "metadata": {"mean_difference": float(diff.mean())},
    }


def test_custom_test_wraps_user_callable_as_test_result() -> None:
    loss_a = pd.Series([0.2, 0.3, 0.1, 0.4])
    loss_b = pd.Series([0.4, 0.5, 0.2, 0.5])

    result = mf.tests.custom_test(
        "scaled_loss_difference",
        custom_statistic,
        loss_a,
        loss_b,
        scale=2.0,
        alpha=0.05,
    )

    assert isinstance(result, mf.tests.TestResult)
    assert result.decision is True
    assert result.n_obs == 4
    assert result.metadata["name"] == "scaled_loss_difference"
    assert result.metadata["callable"].endswith("custom_statistic")
    assert result.metadata["params"] == {"scale": 2.0}
    assert result.to_dict()["metadata"]["custom"] is True


def test_nested_and_encompassing_callables_return_one_sided_results() -> None:
    y_true = pd.Series([1.3, 0.8, 1.2, 1.6, 1.1, 1.4])
    forecast_small = pd.Series([1.0, 1.1, 1.0, 1.2, 0.9, 1.0])
    forecast_large = pd.Series([1.1, 1.2, 0.9, 1.1, 1.0, 1.1])
    error_small = y_true - forecast_small
    error_large = y_true - forecast_large
    loss_small = error_small**2
    loss_large = error_large**2

    cw = mf.tests.clark_west_test(loss_small, loss_large, forecast_small, forecast_large)
    enc_new = mf.tests.enc_new_test(error_small, error_large)
    enc_t = mf.tests.enc_t_test(error_small, error_large)
    hn = mf.tests.harvey_newbold_test(error_small, error_large)
    stacked = mf.tests.nested_tests(
        loss_small,
        loss_large,
        forecast_small=forecast_small,
        forecast_large=forecast_large,
        error_small=error_small,
        error_large=error_large,
    )

    assert cw.alternative == "one_sided"
    assert cw.metadata["reference_formula"] == "e_r^2 - (e_u^2 - (f_r - f_u)^2)"
    assert enc_new.metadata["name"] == "Enc-New"
    assert enc_new.p_value is None
    assert enc_t.metadata["name"] == "Enc-T"
    assert enc_t.p_value is None
    assert hn.metadata["name"] == "Harvey-Newbold"
    assert hn.metadata["r_reference"] is None
    assert hn.metadata["statistic_type"] == "t"
    assert "forecast::dm.test" in hn.metadata["r_alignment"]
    assert set(stacked["test"]) == {"clark_west", "enc_new", "enc_t"}
    assert stacked.attrs["macroforecast_metadata_schema"]["kind"] == "nested_tests"


def test_nested_encompassing_statistics_match_reference_formulas() -> None:
    y_true = pd.Series([2.0, 2.4, 2.1, 2.7, 2.5, 3.0, 2.8, 3.2])
    forecast_small = pd.Series([1.8, 2.1, 2.0, 2.4, 2.2, 2.7, 2.6, 2.9])
    forecast_large = pd.Series([1.9, 2.3, 1.9, 2.6, 2.4, 2.8, 2.7, 3.0])
    error_small = y_true - forecast_small
    error_large = y_true - forecast_large
    loss_small = error_small**2
    loss_large = error_large**2

    def mean_hac_z(values: pd.Series) -> float:
        clean = values.to_numpy(dtype=float)
        mean = float(clean.mean())
        centered = clean - mean
        variance = float(np.dot(centered, centered) / len(clean))
        return mean / math.sqrt(variance / len(clean))

    adjusted_mspe = loss_small - loss_large + (forecast_small - forecast_large) ** 2
    encompassing_covariance = error_small * (error_small - error_large)
    expected_cw = mean_hac_z(adjusted_mspe)
    expected_enc_t = mean_hac_z(encompassing_covariance)
    expected_enc_new = float(
        len(encompassing_covariance) * encompassing_covariance.mean() / float((error_large**2).mean())
    )
    expected_cw_p = 0.5 * math.erfc(expected_cw / math.sqrt(2.0))

    cw = mf.tests.clark_west_test(loss_small, loss_large, forecast_small, forecast_large)
    enc_new = mf.tests.enc_new_test(error_small, error_large)
    enc_t = mf.tests.enc_t_test(error_small, error_large, normal_approximation=True)

    assert np.isclose(cw.statistic, expected_cw)
    assert np.isclose(cw.p_value, expected_cw_p)
    assert np.isclose(enc_new.statistic, expected_enc_new)
    assert np.isclose(enc_t.statistic, expected_enc_t)
    assert enc_t.metadata["p_value_note"].startswith("Default p_value is None")
    assert enc_new.metadata["external_reference"].startswith("Stata forecast encompassing example")


def test_multi_horizon_spa_test_returns_uspa_and_aspa_results() -> None:
    base = pd.DataFrame(
        {
            "h1": [1.20, 1.10, 1.30, 1.15, 1.25, 1.18, 1.22, 1.28, 1.12],
            "h2": [1.05, 1.00, 1.10, 1.02, 1.08, 1.04, 1.09, 1.06, 1.03],
        }
    )
    improved = base - pd.DataFrame(
        {
            "h1": [0.20, 0.15, 0.18, 0.16, 0.19, 0.17, 0.20, 0.18, 0.16],
            "h2": [0.10, 0.11, 0.09, 0.12, 0.10, 0.11, 0.09, 0.10, 0.12],
        }
    )

    uspa = mf.tests.multi_horizon_spa_test(
        base,
        improved,
        statistic="uspa",
        n_boot=99,
        block_length=3,
        random_state=11,
    )
    aspa = mf.tests.multi_horizon_spa_test(
        base,
        improved,
        statistic="aspa",
        weights=[0.25, 0.75],
        n_boot=99,
        block_length=3,
        random_state=11,
    )

    assert isinstance(uspa, mf.tests.TestResult)
    assert isinstance(aspa, mf.tests.TestResult)
    assert uspa.statistic > 0
    assert aspa.statistic > 0
    assert 0.0 <= uspa.p_value <= 1.0
    assert 0.0 <= aspa.p_value <= 1.0
    assert uspa.metadata["statistic"] == "uspa"
    assert aspa.metadata["statistic"] == "aspa"
    assert uspa.metadata["block_length"] == 3
    assert aspa.metadata["weights"] == [0.25, 0.75]


def test_direction_density_and_residual_diagnostics() -> None:
    y_true = pd.Series([1.0, -1.0, 2.0, -0.5, 0.8, -0.2])
    y_pred = pd.Series([0.7, -0.8, 1.5, 0.1, 0.9, -0.3])
    pit = np.linspace(0.05, 0.95, 20)
    residuals = pd.Series(np.sin(np.arange(30)))

    pt = mf.tests.pesaran_timmermann_test(y_true, y_pred)
    hm = mf.tests.henriksson_merton_test(y_true, y_pred)
    ag = mf.tests.anatolyev_gerko_test(y_true, y_pred)
    density = mf.tests.density_interval_tests(pit, alpha=0.1, n_bins=5, pit_lag=1)
    histogram = mf.tests.pit_histogram(pit, n_bins=5)
    pit_acf = mf.tests.pit_autocorrelation_test(pit, lag=1)
    interval = mf.tests.interval_coverage_test(
        y_true=[1.0, 2.0, 3.0, 4.0],
        lower=[0.5, 1.5, 2.5, 3.5],
        upper=[1.5, 2.5, 3.5, 4.5],
        alpha=0.1,
    )
    diagnostics = mf.tests.residual_diagnostics(residuals)

    assert pt.metadata["method"] == "pesaran_timmermann"
    assert hm.metadata["method"] == "henriksson_merton"
    assert ag.metadata["method"] == "anatolyev_gerko"
    assert pt.metadata["rugarch_reference"] == "rugarch/R/rugarch-tests.R::DACTest(test='PT')"
    assert hm.metadata["r_reference"] is None
    assert "No exact comparator" in hm.metadata["r_alignment"]
    assert {"berkowitz", "ks", "kupiec_pof", "christoffersen_independence"}.issubset(
        density
    )
    assert density["metadata_schema"]["kind"] == "density_interval_tests"
    assert density["r_reference"]["berkowitz"] == "tstests/R/berkowitz.R::berkowitz_test"
    assert density["berkowitz"]["r_reference"] == "tstests/R/berkowitz.R::berkowitz_test"
    assert density["engle_manganelli_dq"]["r_reference"] is None
    assert density["du_escanciano_shortfall"]["metadata_schema"]["kind"] == "shortfall_de_test"
    assert len(density["pit_histogram"]) == 5
    assert histogram.attrs["macroforecast_metadata_schema"]["kind"] == "pit_histogram"
    assert pit_acf.metadata["name"] == "PIT autocorrelation"
    assert interval["metadata_schema"]["kind"] == "interval_coverage_test"
    assert interval["coverage_rate"] == 1.0
    assert interval["r_reference"] == "tstests/R/var_cp.R::var_cp_test"
    assert "christoffersen_pelletier_duration" in interval
    assert set(diagnostics["test"]) == {
        "ljung_box_q",
        "arch_lm",
        "jarque_bera_normality",
        "durbin_watson",
    }
    assert diagnostics.attrs["macroforecast_metadata_schema"]["kind"] == "residual_diagnostics"


def test_pesaran_timmermann_matches_tstests_and_rugarch_formula() -> None:
    actual = np.asarray([1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 2.0, -1.0], dtype=float)
    forecast = np.asarray([0.5, -0.3, -0.2, -0.1, 0.2, 0.1, 0.4, -0.8], dtype=float)
    n = len(actual)
    x_t = (actual > 0.0).astype(float)
    y_t = (forecast > 0.0).astype(float)
    z_t = ((forecast * actual) > 0.0).astype(float)
    p_y = float(np.mean(y_t))
    p_x = float(np.mean(x_t))
    p_hat = float(np.mean(z_t))
    p_star = p_y * p_x + (1.0 - p_y) * (1.0 - p_x)
    p_hat_var = (p_star * (1.0 - p_star)) / n
    p_star_var = (
        ((2.0 * p_y - 1.0) ** 2 * (p_x * (1.0 - p_x))) / n
        + ((2.0 * p_x - 1.0) ** 2 * (p_y * (1.0 - p_y))) / n
        + (4.0 * p_x * p_y * (1.0 - p_x) * (1.0 - p_y)) / (n * n)
    )
    expected_stat = (p_hat - p_star) / math.sqrt(p_hat_var - p_star_var)

    result = mf.tests.pesaran_timmermann_test(actual, forecast)

    assert np.isclose(result.statistic, expected_stat)
    assert np.isclose(result.metadata["success_ratio"], p_hat)
    assert result.metadata["r_reference"] == "tstests/R/dac.R::dac_test"
    assert result.metadata["rugarch_reference"] == "rugarch/R/rugarch-tests.R::DACTest(test='PT')"
    assert ".pt_test" in result.metadata["r_alignment"]


def test_anatolyev_gerko_matches_tstests_formula() -> None:
    y_true = pd.Series([1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 2.0, -1.0])
    y_pred = pd.Series([0.5, -0.3, -0.2, -0.1, 0.2, 0.1, 0.4, -0.8])
    forecast_sign = np.sign(y_pred.to_numpy(dtype=float))
    actual = y_true.to_numpy(dtype=float)
    trading_return = forecast_sign * actual
    n = len(actual)
    a_t = float(np.mean(trading_return))
    b_t = float(np.mean(forecast_sign) * np.mean(actual))
    p_direction = 0.5 * (1.0 + float(np.mean(forecast_sign)))
    variance = (4.0 / (n * n)) * p_direction * (1.0 - p_direction) * float(
        np.sum((actual - np.mean(actual)) ** 2)
    )
    expected_stat = (a_t - b_t) / math.sqrt(variance)

    result = mf.tests.anatolyev_gerko_test(y_true, y_pred)

    assert np.isclose(result.statistic, expected_stat)
    assert result.alternative == "one_sided"
    assert result.metadata["r_reference"] == "tstests/R/dac.R::dac_test"
    assert result.metadata["rugarch_reference"] == "rugarch/R/rugarch-tests.R::DACTest(test='AG')"
    assert ".ag_test" in result.metadata["r_alignment"]


def test_henriksson_merton_is_macroforecast_extension_without_dac_r_comparator() -> None:
    actual = np.asarray([1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 2.0, -1.0], dtype=float)
    forecast = np.asarray([0.5, -0.3, -0.2, -0.1, 0.2, 0.1, 0.4, -0.8], dtype=float)
    actual_sign = (actual > 0.0).astype(int)
    forecast_sign = (forecast > 0.0).astype(int)
    up_correct = int(((forecast_sign == 1) & (actual_sign == 1)).sum())
    down_correct = int(((forecast_sign == 0) & (actual_sign == 0)).sum())
    n_up = int((actual_sign == 1).sum())
    n_down = int((actual_sign == 0).sum())
    joint = (up_correct / n_up) + (down_correct / n_down)
    expected_stat = (joint - 1.0) * math.sqrt(min(n_up, n_down))

    result = mf.tests.henriksson_merton_test(actual, forecast)

    assert np.isclose(result.statistic, expected_stat)
    assert result.metadata["r_reference"] is None
    assert result.metadata["rugarch_reference"] is None
    assert result.metadata["source_reference"] == "macroforecast Henriksson-Merton market-timing extension"


def test_shortfall_de_matches_tstests_formula_without_bootstrap() -> None:
    pit = np.asarray([0.01, 0.03, 0.12, 0.40, 0.02, 0.70, 0.04, 0.90], dtype=float)
    alpha = 0.05
    lags = 2
    shortfall = ((alpha - pit) * (pit <= alpha)) / alpha
    unconditional_stat = float(np.mean(shortfall))
    mu = alpha / 2.0
    sigma = math.sqrt(alpha * (1.0 / 3.0 - alpha / 4.0))
    z_value = abs((math.sqrt(len(pit)) * (unconditional_stat - mu)) / sigma)
    expected_unconditional_p = math.erfc(z_value / math.sqrt(2.0))
    adjusted = shortfall - alpha / 2.0
    variance = float(np.mean(adjusted**2))
    autocorr = [
        float(np.sum(adjusted[lag:] * adjusted[:-lag]) / (len(pit) - lag)) / variance
        for lag in range(1, lags + 1)
    ]
    expected_conditional_stat = float(len(pit) * np.sum(np.asarray(autocorr) ** 2))

    result = mf.tests.shortfall_de_test(pit, alpha=alpha, lags=lags)

    assert np.isclose(result["unconditional"]["statistic"], unconditional_stat)
    assert np.isclose(result["unconditional"]["p_value"], expected_unconditional_p)
    assert np.isclose(result["conditional"]["statistic"], expected_conditional_stat)
    assert result["r_reference"] == "tstests/R/shortfall_de.R::shortfall_de_test"
    assert result["r_alignment"].startswith("Matches unconditional_de_statistic")


def test_dynamic_quantile_matches_segmgarch_formula() -> None:
    y_true = np.asarray([-0.5, 0.1, -1.2, 0.3, -0.8, 0.2, -1.5, 0.4], dtype=float)
    var = np.asarray([-1.0, -0.7, -0.9, -0.8, -0.7, -0.9, -1.0, -0.8], dtype=float)
    alpha = 0.05
    hit = np.where(y_true < var, 1.0 - alpha, -alpha)
    hit_ahead = hit[1:]
    var_ahead = var[1:]
    hit_lag = hit[:-1].reshape(-1, 1)
    y_lag = y_true[:-1] ** 2
    min_len = min(len(hit_ahead), len(var_ahead), hit_lag.shape[0], len(y_lag))
    x = np.column_stack(
        [
            np.ones(min_len),
            var_ahead[-min_len:],
            hit_lag[-min_len:],
            y_lag[-min_len:],
        ]
    )
    h = hit_ahead[-min_len:]
    expected_stat = float(h.T @ x @ np.linalg.pinv(x.T @ x) @ x.T @ h / (alpha * (1.0 - alpha)))

    result = mf.tests.dynamic_quantile_test(y_true, var, alpha=alpha)

    assert np.isclose(result.statistic, expected_stat)
    assert result.metadata["df"] == 4
    assert result.metadata["r_reference"] == "segMGarch/R/DQtest.R::DQtest"
    assert result.metadata["source_url"] == "https://rdrr.io/cran/segMGarch/src/R/DQtest.R"
    assert "VaR_level mapped to 1-alpha" in result.metadata["r_alignment"]


def test_interval_coverage_matches_tstests_and_rugarch_likelihoods() -> None:
    alpha = 0.25
    y_true = [-2.0, 0.0, -2.0, 0.0, 0.0, -2.0, 0.0, -2.0]
    result = mf.tests.interval_coverage_test(
        y_true=y_true,
        lower=[-1.0] * len(y_true),
        upper=[1.0] * len(y_true),
        alpha=alpha,
    )
    hits = np.asarray([1, 0, 1, 0, 0, 1, 0, 1], dtype=int)
    total = len(hits)
    n_hits = int(hits.sum())
    p_hat = n_hits / total
    restricted_uc = n_hits * math.log(alpha) + (total - n_hits) * math.log(1.0 - alpha)
    unrestricted_uc = n_hits * math.log(p_hat) + (total - n_hits) * math.log(1.0 - p_hat)
    expected_uc = -2.0 * (restricted_uc - unrestricted_uc)

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
    pi = (n01 + n11) / (total - 1)
    p01 = n01 / (n00 + n01)
    p11 = n11 / (n10 + n11)

    def log_term(count: int, probability: float) -> float:
        if count == 0:
            return 0.0
        if probability <= 0.0:
            return -math.inf
        if probability >= 1.0:
            return 0.0
        return count * math.log(probability)

    restricted_ind = (
        log_term(n00, 1.0 - pi)
        + log_term(n01, pi)
        + log_term(n10, 1.0 - pi)
        + log_term(n11, pi)
    )
    unrestricted_ind = (
        log_term(n00, 1.0 - p01)
        + log_term(n01, p01)
        + log_term(n10, 1.0 - p11)
        + log_term(n11, p11)
    )
    expected_ind = -2.0 * (restricted_ind - unrestricted_ind)

    assert result["kupiec_pof"]["lr_statistic"] == pytest.approx(expected_uc)
    assert result["kupiec_pof"]["r_reference"] == "tstests/R/var_cp.R::.lr_unc_coverage"
    assert result["christoffersen_independence"]["lr_statistic"] == pytest.approx(expected_ind)
    assert result["christoffersen_independence"]["transition_counts"] == {
        "n00": 1,
        "n01": 3,
        "n10": 3,
        "n11": 0,
    }
    assert result["christoffersen_conditional_coverage"]["lr_statistic"] == pytest.approx(
        expected_uc + expected_ind
    )
    assert result["rugarch_reference"] == "rugarch/R/rugarch-tests.R::VaRTest"


def test_interval_duration_reports_christoffersen_pelletier_statistic() -> None:
    result = mf.tests.interval_coverage_test(
        y_true=[-2.0, 0.0, 0.0, -1.5, 0.0, 0.0, -1.7, 0.0, 0.0, 0.0],
        lower=[-1.0] * 10,
        upper=[1.0] * 10,
        alpha=0.1,
    )
    duration = result["christoffersen_pelletier_duration"]

    assert duration["r_reference"] == "tstests/R/var_cp.R::.duration_test"
    assert duration["n_failures"] == 3
    assert duration["lr_statistic"] is not None
    assert 0.0 <= duration["p_value"] <= 1.0


def test_interval_coverage_kupiec_zero_hit_boundary_is_not_silently_passed() -> None:
    result = mf.tests.interval_coverage_test(
        y_true=[1.0] * 100,
        lower=[0.0] * 100,
        upper=[2.0] * 100,
        alpha=0.1,
    )

    expected_lr = -2.0 * (100 * math.log(1.0 - 0.1))

    assert np.isclose(result["kupiec_pof"]["lr_statistic"], expected_lr)
    assert result["kupiec_pof"]["p_value"] < 0.01
    assert result["kupiec_pof"]["reject"] is True


def test_residual_diagnostics_match_r_reference_formulas() -> None:
    values = np.asarray(
        [0.2, -0.1, 0.4, -0.3, 0.5, -0.2, 0.1, -0.4, 0.3, -0.05, 0.15, -0.25],
        dtype=float,
    )
    lag = 4
    result = mf.tests.residual_diagnostics(
        values,
        tests=(
            "ljung_box_q",
            "arch_lm",
            "jarque_bera_normality",
            "durbin_watson",
            "breusch_godfrey_serial_correlation",
        ),
        lag=lag,
    ).set_index("test")

    centered = values - values.mean()
    denom = float(np.sum(centered**2))
    acf = [
        float(np.sum(centered[:-order] * centered[order:]) / denom)
        for order in range(1, lag + 1)
    ]
    expected_ljung_box = float(
        len(values) * (len(values) + 2) * sum(rho**2 / (len(values) - order) for order, rho in enumerate(acf, start=1))
    )

    squared = values**2
    arch_y = squared[lag:]
    arch_x = np.column_stack(
        [np.ones(len(arch_y))]
        + [squared[lag - order : len(squared) - order] for order in range(1, lag + 1)]
    )
    arch_coef, _, _, _ = np.linalg.lstsq(arch_x, arch_y, rcond=None)
    arch_fit = arch_x @ arch_coef
    arch_r2 = 1.0 - float(np.sum((arch_y - arch_fit) ** 2)) / float(np.sum((arch_y - arch_y.mean()) ** 2))
    expected_arch = float(len(arch_y) * arch_r2)

    jb_stat, jb_p = mf.tests._jarque_bera_from_r_moments(values)

    resid = values - values.mean()
    z = np.column_stack(
        [np.concatenate([np.zeros(order), resid[: len(resid) - order]]) for order in range(1, lag + 1)]
    )
    aux_x = np.column_stack([np.ones(len(resid)), z])
    aux_coef, _, _, _ = np.linalg.lstsq(aux_x, resid, rcond=None)
    expected_bg = float(len(resid) * np.sum((aux_x @ aux_coef) ** 2) / np.sum(resid**2))

    expected_dw = float(np.sum(np.diff(values) ** 2) / np.sum(values**2))

    assert np.isclose(result.loc["ljung_box_q", "statistic"], expected_ljung_box)
    assert result.loc["ljung_box_q", "df"] == lag
    assert result.loc["ljung_box_q", "source_reference"].startswith("stats::Box.test")
    assert result.loc["ljung_box_q", "r_reference"] == "stats::Box.test(type='Ljung-Box')"
    assert "lag - model_df" in result.loc["ljung_box_q", "r_alignment"]
    assert np.isclose(result.loc["arch_lm", "statistic"], expected_arch)
    assert result.loc["arch_lm", "source_reference"] == "FinTS/R/ArchTest.R::ArchTest"
    assert result.loc["arch_lm", "r_reference"] == "FinTS/R/ArchTest.R::ArchTest"
    assert "model_df=0" in result.loc["arch_lm", "r_alignment"]
    assert np.isclose(result.loc["jarque_bera_normality", "statistic"], jb_stat)
    assert np.isclose(result.loc["jarque_bera_normality", "p_value"], jb_p)
    assert result.loc["jarque_bera_normality", "source_reference"] == "tseries/R/test.R::jarque.bera.test"
    assert result.loc["jarque_bera_normality", "r_reference"] == "tseries/R/test.R::jarque.bera.test"
    assert np.isclose(result.loc["breusch_godfrey_serial_correlation", "statistic"], expected_bg)
    assert result.loc["breusch_godfrey_serial_correlation", "source_reference"] == "lmtest/R/bgtest.R::bgtest residual-series contract"
    assert result.loc["breusch_godfrey_serial_correlation", "r_reference"] == "lmtest/R/bgtest.R::bgtest"
    assert "residual-series contract" in result.loc["breusch_godfrey_serial_correlation", "r_alignment"]
    assert np.isclose(result.loc["durbin_watson", "statistic"], expected_dw)
    assert result.loc["durbin_watson", "r_reference"] == "lmtest/R/dwtest.R::dwtest"
    assert result.loc["durbin_watson", "status"] == "statistic_only_no_p_value"
    assert "P-value is omitted" in result.loc["durbin_watson", "r_alignment"]


def test_arch_lm_demean_matches_fints_archtest_demean_option() -> None:
    values = np.asarray([0.4, -0.2, 0.7, -0.5, 0.6, -0.3, 0.2, -0.1], dtype=float)
    lag = 2
    demeaned = values - values.mean()
    squared = demeaned**2
    arch_y = squared[lag:]
    arch_x = np.column_stack(
        [np.ones(len(arch_y))]
        + [squared[lag - order : len(squared) - order] for order in range(1, lag + 1)]
    )
    arch_coef, _, _, _ = np.linalg.lstsq(arch_x, arch_y, rcond=None)
    arch_fit = arch_x @ arch_coef
    arch_r2 = 1.0 - float(np.sum((arch_y - arch_fit) ** 2)) / float(np.sum((arch_y - arch_y.mean()) ** 2))
    expected_arch = float(len(arch_y) * arch_r2)

    result = mf.tests.residual_diagnostics(
        values,
        tests=("arch_lm",),
        lag=lag,
        demean_arch=True,
    ).iloc[0]

    assert np.isclose(result["statistic"], expected_arch)
    assert bool(result["demean_arch"]) is True
    assert result["r_reference"] == "FinTS/R/ArchTest.R::ArchTest"


def test_conditional_predictive_ability_and_model_confidence_set() -> None:
    rng = np.random.default_rng(123)
    loss_a = pd.Series(rng.normal(0.8, 0.1, 24))
    loss_b = pd.Series(rng.normal(1.0, 0.1, 24))
    cpa = mf.tests.conditional_predictive_ability_test(
        loss_a,
        loss_b,
        method="giacomini_rossi",
        window_ratio=0.5,
    )

    loss_panel = pd.DataFrame(
        {
            "target": ["y"] * 24,
            "horizon": [1] * 24,
            "origin": list(range(12)) * 2,
            "model_id": ["a"] * 12 + ["b"] * 12,
            "squared_error": [0.3 + idx * 0.01 for idx in range(12)]
            + [0.5 + idx * 0.01 for idx in range(12)],
        }
    )
    mcs = mf.tests.model_confidence_set(
        loss_panel,
        n_boot=20,
        block_length=2,
        random_state=123,
    )
    reality = mf.tests.blocked_oob_reality_check(
        loss_panel,
        benchmark="a",
        n_boot=20,
        block_length=2,
        random_state=123,
    )
    wide_reality = mf.tests.blocked_oob_reality_check(
        loss_panel.pivot_table(
            index="origin",
            columns="model_id",
            values="squared_error",
        ),
        benchmark="a",
        n_boot=20,
        block_length=2,
        random_state=123,
    )

    assert cpa["method"] == "giacomini_rossi"
    assert cpa["metadata_schema"]["kind"] == "conditional_predictive_ability"
    assert cpa["n_obs"] == 24
    assert cpa["critical_value"] == 2.779
    assert cpa["r_reference"] == "murphydiagram/R/procs.R::fluctuation_test"
    assert cpa["external_reference"] == "murphydiagram R package"
    assert cpa["critical_value_source"].startswith("Giacomini-Rossi")
    assert cpa["loss_difference_orientation"].startswith("loss_a - loss_b")
    assert cpa["available_window_ratio_grid"] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    assert mcs["metadata_schema"]["kind"] == "model_confidence_set"
    assert mcs["method"] == "hansen_lunde_nason_mcs"
    assert mcs["procedure"] == "sequential_elimination"
    assert mcs["canonical_function"] == "model_confidence_set"
    assert mcs["r_reference"] == "MCS/R/MCSprocedure.R::MCSprocedure"
    mcs_record = next(item for item in mcs["mcs_inclusion"] if item["target"] == "y")
    assert mcs_record["horizon"] == 1
    assert mcs_record["alpha"] == 0.1
    assert set(mcs_record["models"]).issubset({"a", "b"})
    assert mcs["iteration_path"][0]["removed_model"] in {"a", "b"}
    assert mcs["bootstrap_n_replications"] == 20
    assert reality.attrs["macroforecast_metadata_schema"]["kind"] == "blocked_oob_reality_check"
    assert reality.loc[0, "model"] == "b"
    assert reality.loc[0, "benchmark"] == "a"
    assert reality.loc[0, "mean_diff"] < 0.0
    assert 0.0 <= reality.loc[0, "p_value"] <= 1.0
    assert 0.0 <= reality.loc[0, "familywise_p_value"] <= 1.0
    assert reality.loc[0, "test_family"] == "pairwise_benchmark_superiority_bootstrap"
    assert reality.loc[0, "source_reference"] == "macroforecast legacy blocked_oob_reality_check"
    assert reality.loc[0, "r_reference"] is None
    assert "No exact R package comparator" in reality.loc[0, "r_alignment"]
    assert reality.attrs["macroforecast_metadata_schema"]["source_reference"] == (
        "macroforecast legacy blocked_oob_reality_check"
    )
    assert "reality_check_test" in reality.attrs["macroforecast_metadata_schema"]["exact_multiple_comparison_callables"]
    assert wide_reality.loc[0, "target"] == "all"
    assert wide_reality.loc[0, "horizon"] == "all"
    json.dumps(mcs)


def test_giacomini_rossi_fluctuation_matches_murphydiagram_formula() -> None:
    loss_a = np.asarray([1.2, 0.9, 1.1, 1.0, 0.8, 1.3, 1.1, 0.7, 0.9, 1.0])
    loss_b = np.asarray([1.0, 1.1, 0.9, 1.2, 1.1, 1.0, 1.3, 1.0, 1.2, 1.1])
    diff = loss_a - loss_b
    mu = 0.5
    m = round(mu * len(diff))

    def bartlett_hac(values: np.ndarray, lag_truncate: int) -> float:
        values = np.asarray(values, dtype=float)
        n = len(values)
        hac = float(np.dot(values, values) / (n - 1))
        for lag in range(1, lag_truncate + 1):
            weight = 1.0 - lag / (lag_truncate + 1)
            cov = float(np.dot(values[lag:], values[:-lag]) / (n - 1))
            hac += 2.0 * weight * cov
        return hac

    full_hac = bartlett_hac(diff, 1)
    expected_path = [
        float(math.sqrt(m) * np.mean(diff[end - m : end]) / math.sqrt(full_hac))
        for end in range(m, len(diff) + 1)
    ]

    result = mf.tests.conditional_predictive_ability_test(
        loss_a,
        loss_b,
        method="giacomini_rossi",
        window_ratio=mu,
        dmv_fullsample=True,
        lag_truncate=1,
        alpha=0.05,
    )

    assert np.allclose(result["time_path"], expected_path)
    assert result["statistic"] == pytest.approx(max(abs(value) for value in expected_path))
    assert result["critical_band"] == [-2.779, 2.779]


def test_giacomini_rossi_rolling_hac_branch_matches_murphydiagram_formula() -> None:
    loss_a = np.asarray([1.2, 0.9, 1.1, 1.0, 0.8, 1.3, 1.1, 0.7, 0.9, 1.0])
    loss_b = np.asarray([1.0, 1.1, 0.9, 1.2, 1.1, 1.0, 1.3, 1.0, 1.2, 1.1])
    diff = loss_a - loss_b
    mu = 0.5
    m = round(mu * len(diff))

    def bartlett_hac(values: np.ndarray, lag_truncate: int) -> float:
        values = np.asarray(values, dtype=float)
        n = len(values)
        hac = float(np.dot(values, values) / (n - 1))
        for lag in range(1, lag_truncate + 1):
            weight = 1.0 - lag / (lag_truncate + 1)
            cov = float(np.dot(values[lag:], values[:-lag]) / (n - 1))
            hac += 2.0 * weight * cov
        return hac

    expected_path = []
    for end in range(m, len(diff) + 1):
        window = diff[end - m : end]
        rolling_hac = bartlett_hac(window, 1)
        expected_path.append(float(np.mean(window) / math.sqrt(rolling_hac / m)))

    result = mf.tests.conditional_predictive_ability_test(
        loss_a,
        loss_b,
        method="giacomini_rossi",
        window_ratio=mu,
        dmv_fullsample=False,
        lag_truncate=1,
        alpha=0.05,
    )

    assert np.allclose(result["time_path"], expected_path)
    assert result["variance_scope"] == "rolling_window"
    assert result["r_alignment"].startswith("Matches loss1-loss2 orientation")


def test_conditional_predictive_ability_recursive_alias_is_explicitly_metadata_only() -> None:
    loss_a = np.asarray([1.2, 0.9, 1.1, 1.0, 0.8, 1.3, 1.1, 0.7, 0.9, 1.0])
    loss_b = np.asarray([1.0, 1.1, 0.9, 1.2, 1.1, 1.0, 1.3, 1.0, 1.2, 1.1])

    result = mf.tests.conditional_predictive_ability_test(
        loss_a,
        loss_b,
        method="rossi_sekhposyan",
        window_ratio=0.5,
        lag_truncate=1,
    )

    assert result["method"] == "recursive_fluctuation"
    assert result["requested_method"] == "rossi_sekhposyan"
    assert result["r_reference"] is None
    assert result["critical_value"] is None
    assert result["alias_warning"].startswith("method='rossi_sekhposyan'")


def test_iterative_model_confidence_set_sequentially_eliminates_models() -> None:
    rng = np.random.default_rng(321)
    rows = []
    for origin in range(36):
        common = 0.01 * np.sin(origin / 3.0)
        for model, base in (("a", 0.8), ("b", 0.35), ("c", 0.38)):
            rows.append(
                {
                    "target": "y",
                    "horizon": 1,
                    "origin": origin,
                    "model_id": model,
                    "squared_error": base + common + rng.normal(0.0, 0.015),
                }
            )
    loss_panel = pd.DataFrame(rows)

    result = mf.tests.iterative_model_confidence_set(
        loss_panel,
        alpha=0.1,
        n_boot=30,
        block_length=3,
        random_state=123,
    )
    included = result["mcs_inclusion"][0]["models"]
    rejected = result["mcs_rejections"][0]["models"]

    assert result["metadata_schema"]["kind"] == "iterative_model_confidence_set"
    assert result["canonical_function"] == "model_confidence_set"
    assert result["alias_function"] == "iterative_model_confidence_set"
    assert "b" in included
    assert "a" in rejected
    assert result["iteration_path"][0]["eliminated_model"] == "a"
    assert result["block_lengths_used"][0]["block_length"] == 3
    json.dumps(result)


def test_iterative_model_confidence_set_accepts_wide_loss_matrix() -> None:
    wide = pd.DataFrame(
        {
            "a": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
            "b": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
        }
    )

    result = mf.tests.iterative_model_confidence_set(
        wide,
        n_boot=20,
        random_state=123,
    )

    assert result["mcs_inclusion"][0]["target"] == "all"
    assert result["mcs_inclusion"][0]["horizon"] == "all"
    assert result["mcs_inclusion"][0]["models"] == ["a", "b"]
    assert result["iteration_path"][0]["p_value"] == 1.0


def test_model_confidence_set_is_exact_mcs_canonical_callable() -> None:
    rng = np.random.default_rng(321)
    rows = []
    for origin in range(36):
        common = 0.01 * np.sin(origin / 3.0)
        for model, base in (("a", 0.8), ("b", 0.35), ("c", 0.38)):
            rows.append(
                {
                    "target": "y",
                    "horizon": 1,
                    "origin": origin,
                    "model_id": model,
                    "squared_error": base + common + rng.normal(0.0, 0.015),
                }
            )
    loss_panel = pd.DataFrame(rows)

    result = mf.tests.model_confidence_set(
        loss_panel,
        alpha=0.1,
        n_boot=30,
        block_length=3,
        random_state=123,
    )

    included = result["mcs_inclusion"][0]["models"]
    rejected = result["mcs_rejections"][0]["models"]

    assert result["metadata_schema"]["kind"] == "model_confidence_set"
    assert result["method"] == "hansen_lunde_nason_mcs"
    assert result["statistic"] == "max"
    assert result["bootstrap_kind"] == "mcs_fixed_block"
    assert result["source_alignment"].startswith("GetD loss differences")
    assert "b" in included
    assert "a" in rejected
    assert result["iteration_path"][0]["eliminated_model"] == "a"


def test_arch_multiple_comparison_callables_return_backend_results() -> None:
    pytest.importorskip("arch.bootstrap")
    loss_panel = pd.DataFrame(
        {
            "target": ["y"] * 36,
            "horizon": [1] * 36,
            "origin": list(range(12)) * 3,
            "model_id": ["benchmark"] * 12 + ["good"] * 12 + ["weak"] * 12,
            "squared_error": [1.0 + idx * 0.01 for idx in range(12)]
            + [0.5 + idx * 0.01 for idx in range(12)]
            + [1.1 + idx * 0.01 for idx in range(12)],
        }
    )

    spa = mf.tests.superior_predictive_ability_test(
        loss_panel,
        benchmark="benchmark",
        n_boot=30,
        block_length=2,
        random_state=123,
    )
    rc = mf.tests.reality_check_test(
        loss_panel,
        benchmark="benchmark",
        n_boot=30,
        block_length=2,
        random_state=123,
    )
    stepm = mf.tests.stepm_test(
        loss_panel,
        benchmark="benchmark",
        n_boot=30,
        block_length=2,
        random_state=123,
    )

    spa_record = spa["records"][0]
    rc_record = rc["records"][0]
    stepm_record = stepm["records"][0]

    assert spa["metadata_schema"]["kind"] == "superior_predictive_ability_test"
    assert rc["metadata_schema"]["kind"] == "reality_check_test"
    assert stepm["metadata_schema"]["kind"] == "stepm_test"
    assert spa_record["backend"] == "arch.bootstrap.SPA"
    assert rc_record["backend"] == "arch.bootstrap.RealityCheck"
    assert stepm_record["backend"] == "arch.bootstrap.StepM"
    assert spa_record["r_reference"] == "ttrTests/R/dataSnoop.R::dataSnoop(test='SPA')"
    assert rc_record["r_reference"] == "ttrTests/R/dataSnoop.R::dataSnoop(test='RC')"
    assert stepm_record["r_reference"] == "oosanalysis-R-library/R/stepm.R::stepm"
    assert spa_record["mean_loss_difference"]["good"] > 0.0
    assert rc_record["mean_loss_difference"]["weak"] < 0.0
    assert stepm_record["loss_orientation"].startswith("positive mean_loss_difference")
    assert 0.0 <= spa_record["p_value"] <= 1.0
    assert 0.0 <= rc_record["p_value"] <= 1.0
    assert "good" in spa_record["superior_models"]
    assert "good" in rc_record["superior_models"]
    assert "good" in stepm_record["superior_models"]
    assert "technical-trading" in spa_record["r_alignment"]
    assert "stepdown" in stepm_record["r_alignment"]


def test_statistical_test_callables_reject_invalid_options() -> None:
    with pytest.raises(ValueError, match="method"):
        mf.tests.directional_accuracy_test([1, -1], [1, -1], method="unknown")
    with pytest.raises(ValueError, match="window_ratio"):
        mf.tests.conditional_predictive_ability_test([1, 2, 3], [1, 2, 4], window_ratio=0)
    with pytest.raises(ValueError, match="bootstrap_method"):
        mf.tests.model_confidence_set(
            pd.DataFrame(
                {
                    "target": ["y", "y"],
                    "horizon": [1, 1],
                    "origin": [0, 0],
                    "model_id": ["a", "b"],
                    "squared_error": [1.0, 2.0],
                }
            ),
            bootstrap_method="unknown",
        )
    with pytest.raises(ValueError, match="statistic"):
        mf.tests.iterative_model_confidence_set(
            pd.DataFrame({"a": [1, 2, 3, 4], "b": [1, 2, 3, 4]}),
            statistic="unknown",
        )
    small = mf.tests.density_interval_tests([0.25, 0.75])
    assert small["berkowitz"]["df"] == 3
    with pytest.raises(ValueError, match="pit values"):
        mf.tests.density_interval_tests([0.1, 1.1])


def test_multiple_model_tests_reject_duplicate_long_loss_keys() -> None:
    duplicated = pd.DataFrame(
        {
            "target": ["y", "y", "y"],
            "horizon": [1, 1, 1],
            "origin": [0, 0, 0],
            "model_id": ["a", "a", "b"],
            "squared_error": [1.0, 1.2, 0.8],
        }
    )

    with pytest.raises(ValueError, match="duplicate loss rows"):
        mf.tests.model_confidence_set(duplicated, n_boot=10)

    with pytest.raises(ValueError, match="duplicate loss rows"):
        mf.tests.iterative_model_confidence_set(duplicated, n_boot=10)

    with pytest.raises(ValueError, match="duplicate loss rows"):
        mf.tests.blocked_oob_reality_check(duplicated, benchmark="a", n_boot=10)
