from __future__ import annotations

from itertools import combinations
import time

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
import macroforecast.models.bayesian as bayesian_models


def _supervised_frame(n: int = 24, p: int = 4) -> tuple[pd.DataFrame, pd.Series]:
    idx = pd.date_range("2000-01-31", periods=n, freq="ME")
    base = np.linspace(-1.0, 1.0, n)
    data = {
        f"x{j}": np.sin(base * (j + 1)) + (j + 1) * base**2
        for j in range(p)
    }
    X = pd.DataFrame(data, index=idx)
    values = 0.5 + 0.05 * np.arange(n, dtype=float)
    if p >= 1:
        values = values + 0.8 * X["x0"].to_numpy(dtype=float)
    if p >= 2:
        values = values - 0.3 * X["x1"].to_numpy(dtype=float)
    if p >= 3:
        values = values + 0.1 * X["x2"].to_numpy(dtype=float) ** 2
    y = pd.Series(values, index=idx, name="y")
    return X, y


def _ols_subset_prediction(
    X: pd.DataFrame, y: pd.Series, X_new: pd.DataFrame, subset: tuple[int, ...]
) -> np.ndarray:
    x_values = X.to_numpy(dtype=float)[:, subset]
    design = np.column_stack([np.ones(len(X), dtype=float), x_values])
    params = np.linalg.lstsq(design, y.to_numpy(dtype=float), rcond=None)[0]
    new_design = np.column_stack(
        [np.ones(len(X_new), dtype=float), X_new.to_numpy(dtype=float)[:, subset]]
    )
    return new_design @ params


def _literal_loo_predictions(
    X: pd.DataFrame, y: pd.Series, subset: tuple[int, ...]
) -> np.ndarray:
    x_values = X.to_numpy(dtype=float)
    y_values = y.to_numpy(dtype=float)
    out = np.zeros(len(X), dtype=float)
    for pos in range(len(X)):
        mask = np.ones(len(X), dtype=bool)
        mask[pos] = False
        design = np.column_stack(
            [np.ones(mask.sum(), dtype=float), x_values[mask][:, subset]]
        )
        params = np.linalg.lstsq(design, y_values[mask], rcond=None)[0]
        row = np.r_[1.0, x_values[pos, list(subset)]]
        out[pos] = float(row @ params)
    return out


def test_csr_exact_small_case_averages_all_subsets() -> None:
    X, y = _supervised_frame(n=18, p=4)
    X_new = X.iloc[-3:]

    fit = mf.models.csr(X, y, k=2, max_subsets=100, random_state=123)
    pred = fit.predict(X_new)

    expected = np.mean(
        [
            _ols_subset_prediction(X, y, X_new, subset)
            for subset in combinations(range(4), 2)
        ],
        axis=0,
    )
    np.testing.assert_allclose(pred.to_numpy(dtype=float), expected, rtol=1e-12, atol=1e-12)
    assert len(fit.estimator.subsets_) == 6
    assert fit.metadata["k"] == 2


def test_csr_subset_cap_is_seed_deterministic() -> None:
    X, y = _supervised_frame(n=30, p=8)

    first = mf.models.csr(X, y, k=3, max_subsets=10, random_state=7)
    second = mf.models.csr(X, y, k=3, max_subsets=10, random_state=7)

    assert first.estimator.subsets_ == second.estimator.subsets_
    np.testing.assert_allclose(first.predict(X).to_numpy(), second.predict(X).to_numpy())


def test_csr_rejects_subset_size_larger_than_predictor_count() -> None:
    X, y = _supervised_frame(n=12, p=3)

    with pytest.raises(ValueError, match="at least k=4 predictors; got p=3"):
        mf.models.csr(X, y, k=4)


def test_csr_model_spec_exposes_random_state_for_pipeline_seed_derivation() -> None:
    spec = mf.models.get_model("csr")

    assert spec.input_kind == "supervised"
    assert spec.default_params["random_state"] == 1071


def test_jma_weights_match_brute_force_grid_on_tiny_case() -> None:
    X, y = _supervised_frame(n=22, p=2)

    fit = mf.models.jma(X, y)
    loo = fit.estimator.loo_predictions_
    assert loo is not None
    y_values = y.to_numpy(dtype=float)
    grid = np.linspace(0.0, 1.0, 10001)
    losses = np.asarray(
        [
            np.sum((y_values - (weight * loo[:, 0] + (1.0 - weight) * loo[:, 1])) ** 2)
            for weight in grid
        ],
        dtype=float,
    )
    best = float(grid[int(np.argmin(losses))])

    np.testing.assert_allclose(fit.estimator.weights_, [best, 1.0 - best], atol=1e-3)


def test_jma_single_candidate_gets_unit_weight() -> None:
    X, y = _supervised_frame(n=14, p=1)

    fit = mf.models.jma(X, y)

    np.testing.assert_allclose(fit.estimator.weights_, [1.0], atol=0.0)
    assert np.isfinite(fit.predict(X.iloc[:2]).to_numpy(dtype=float)).all()


def test_jma_is_deterministic() -> None:
    X, y = _supervised_frame(n=26, p=4)

    first = mf.models.jma(X, y)
    second = mf.models.jma(X, y)

    np.testing.assert_allclose(first.estimator.weights_, second.estimator.weights_)
    np.testing.assert_allclose(first.predict(X).to_numpy(), second.predict(X).to_numpy())


def test_jma_hat_matrix_loo_matches_literal_refits() -> None:
    X, y = _supervised_frame(n=18, p=3)

    fit = mf.models.jma(X, y)
    loo = fit.estimator.loo_predictions_
    assert loo is not None
    for candidate_pos, subset in enumerate(fit.estimator.candidate_indices_):
        expected = _literal_loo_predictions(X, y, subset)
        np.testing.assert_allclose(loo[:, candidate_pos], expected, rtol=1e-10, atol=1e-10)


def test_jma_model_spec_uses_nested_candidates() -> None:
    spec = mf.models.get_model("jma")

    assert spec.input_kind == "supervised"
    assert spec.default_params["candidates"] == "nested"


def test_ucsv_ksc_constants_match_log_chi_square_moments() -> None:
    mean = float(np.sum(bayesian_models._KSC_PROB * bayesian_models._KSC_MEAN))
    variance = float(
        np.sum(
            bayesian_models._KSC_PROB
            * (
                bayesian_models._KSC_VAR
                + (bayesian_models._KSC_MEAN - mean) ** 2
            )
        )
    )

    assert mean == pytest.approx(-1.2704, abs=5e-5)
    assert variance == pytest.approx(4.9348, abs=5e-4)


def _local_level_sv(seed: int = 0, n: int = 120) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-31", periods=n, freq="ME")
    phase = np.linspace(0.0, 3.0 * np.pi, n)
    obs_log_vol = -1.7 + 0.4 * np.sin(phase)
    level_log_vol = -5.0 + 0.3 * np.cos(phase)
    tau = np.zeros(n, dtype=float)
    tau[0] = 2.0
    for pos in range(1, n):
        tau[pos] = tau[pos - 1] + rng.normal(scale=float(np.sqrt(np.exp(level_log_vol[pos - 1]))))
    y = tau + rng.normal(scale=np.sqrt(np.exp(obs_log_vol)), size=n)
    return pd.Series(y, index=idx, name="y"), pd.Series(tau, index=idx, name="tau")


def test_ucsv_tracks_synthetic_local_level_sv_trend() -> None:
    y, tau = _local_level_sv(seed=11, n=100)

    fit = mf.models.ucsv(y, n_draws=220, burn=80, gamma=0.05, random_state=123)
    trend = fit.diagnostics["trend"]

    noisy_rmse = float(np.sqrt(np.mean((y.to_numpy(dtype=float) - tau.to_numpy(dtype=float)) ** 2)))
    trend_rmse = float(
        np.sqrt(np.mean((trend.to_numpy(dtype=float) - tau.to_numpy(dtype=float)) ** 2))
    )
    assert trend_rmse < noisy_rmse
    assert fit.metadata["forecast_is_final_trend"] is True


def test_ucsv_recovers_constant_observation_variance() -> None:
    rng = np.random.default_rng(202405)
    truth = 4.0
    idx = pd.date_range("1980-01-31", periods=400, freq="ME")
    y = pd.Series(
        rng.normal(scale=np.sqrt(truth), size=len(idx)),
        index=idx,
        name="y",
    )

    fit = mf.models.ucsv(y, n_draws=260, burn=100, gamma=0.01, random_state=918)
    recovered = float(
        np.mean(np.exp(fit.diagnostics["obs_log_vol"].to_numpy(dtype=float)))
    )

    assert recovered == pytest.approx(truth, rel=0.25)


def test_ucsv_is_deterministic_given_random_state() -> None:
    y, _tau = _local_level_sv(seed=13, n=80)

    first = mf.models.ucsv(y, n_draws=120, burn=40, gamma=0.05, random_state=7)
    second = mf.models.ucsv(y, n_draws=120, burn=40, gamma=0.05, random_state=7)

    np.testing.assert_allclose(
        first.diagnostics["trend"].to_numpy(dtype=float),
        second.diagnostics["trend"].to_numpy(dtype=float),
    )
    np.testing.assert_allclose(
        first.predict(pd.DataFrame(index=range(3))).to_numpy(dtype=float),
        second.predict(pd.DataFrame(index=range(3))).to_numpy(dtype=float),
    )


def test_ucsv_forecast_is_horizon_invariant_final_trend() -> None:
    y, _tau = _local_level_sv(seed=17, n=70)

    fit = mf.models.ucsv(y, n_draws=100, burn=30, gamma=0.05, random_state=5)
    pred = fit.predict(pd.DataFrame({"h": range(6)}))

    assert pred.nunique() == 1
    assert pred.iloc[0] == pytest.approx(float(fit.diagnostics["trend"].iloc[-1]))


@pytest.mark.slow
def test_ucsv_default_draws_runtime_guard_on_500_obs() -> None:
    y, _tau = _local_level_sv(seed=19, n=500)

    start = time.perf_counter()
    fit = mf.models.ucsv(y, random_state=0)
    elapsed = time.perf_counter() - start

    assert np.isfinite(fit.predict(pd.DataFrame(index=range(1))).iloc[0])
    assert elapsed < 60.0


def test_ucsv_model_spec_is_target_kind_and_exposes_random_state() -> None:
    spec = mf.models.get_model("ucsv")

    assert spec.input_kind == "target"
    assert spec.default_params["random_state"] == 1071
