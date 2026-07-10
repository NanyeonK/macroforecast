from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf


_ADD2_BIC_LAG_SQUARE_SELECTED_LAG = 4
_ADD2_BIC_LAG_SQUARE_COEFFICIENT_POWER = 2.6204213101651392
_ADD2_BIC_ORACLE_LEVEL_SHIFT = 4.9319909761602885


def _controlled_target(n: int = 80) -> pd.Series:
    t = np.arange(n, dtype=float)
    y = (
        0.35 * np.sin(t / 2.3)
        + 0.20 * np.cos(t / 5.0)
        + 0.015 * t
        + 0.003 * (t % 7)
    )
    return pd.Series(y, index=pd.date_range("2000-01-31", periods=n, freq="ME"), name="y")


def _lag_square_bic_oracle_target(n: int = 80) -> pd.Series:
    rng = np.random.default_rng(1)
    y = np.zeros(n, dtype=float)
    y[:4] = rng.normal(scale=0.2, size=4)
    innovations = rng.normal(scale=0.08, size=n)
    ar_weights = np.asarray([0.15, 0.05, -0.20, 0.80])
    for idx in range(4, n):
        y[idx] = (
            float(ar_weights @ y[idx - 4 : idx][::-1])
            + innovations[idx]
            + 0.01 * np.sin(idx / 3.0)
        )
    y = y + np.linspace(-0.2, 0.3, n) + _ADD2_BIC_ORACLE_LEVEL_SHIFT
    return pd.Series(
        y, index=pd.date_range("2000-01-31", periods=n, freq="ME"), name="y"
    )


def _reference_lag_bic_lag_square(
    y: np.ndarray, *, min_lag: int, max_lag: int
) -> tuple[int, float, int, int]:
    best: tuple[float, int, int, int] | None = None
    for lag in range(min_lag, max_lag + 1):
        response = y[lag:]
        design = np.column_stack([y[lag - j : len(y) - j] for j in range(1, lag + 1)])
        full = np.column_stack([np.ones(len(design)), design])
        beta = np.linalg.lstsq(full, response, rcond=None)[0]
        resid = response - full @ beta
        ssr = float(resid @ resid)
        nobs = int(len(response))
        n_params = int(lag * lag)
        variance = max(ssr / nobs, float(np.finfo(float).tiny))
        score = float(nobs * np.log(variance) + n_params * np.log(nobs))
        if best is None or score < best[0]:
            best = (score, lag, nobs, n_params)
    assert best is not None
    score, lag, nobs, n_params = best
    return lag, score, nobs, n_params


def _reference_matlab_ar_phi(y: np.ndarray, lag: int) -> np.ndarray:
    centered = y - float(np.mean(y))
    forward_response = centered[lag:]
    forward_lags = np.column_stack(
        [centered[lag - j : len(centered) - j] for j in range(1, lag + 1)]
    )
    origins = np.arange(0, len(centered) - lag)
    backward_response = centered[origins]
    backward_lags = np.column_stack([centered[origins + j] for j in range(1, lag + 1)])
    return np.linalg.lstsq(
        np.vstack([forward_lags, backward_lags]),
        np.concatenate([forward_response, backward_response]),
        rcond=None,
    )[0]


def test_ar_bic_lag_square_coefficient_power_matches_reference_path() -> None:
    y = _lag_square_bic_oracle_target()
    values = y.to_numpy(dtype=float)
    lag, score, nobs, n_params = _reference_lag_bic_lag_square(
        values, min_lag=1, max_lag=6
    )
    phi = _reference_matlab_ar_phi(values, lag)
    horizon = 4
    state = values[-lag:][::-1]
    expected = float((phi**horizon) @ state)

    # The expected lag is the independently hand-derived BIC optimum under the
    # lag_square parameter count for this controlled series; it is not read from
    # ar_bic output.
    assert lag == _ADD2_BIC_LAG_SQUARE_SELECTED_LAG
    np.testing.assert_allclose(
        np.asarray([expected]),
        np.asarray([_ADD2_BIC_LAG_SQUARE_COEFFICIENT_POWER]),
        rtol=0,
        atol=1e-12,
    )

    fit = mf.models.ar_bic(
        y,
        min_lag=1,
        max_lag=6,
        criterion="bic",
        ic_parameter_count="lag_square",
        estimator="matlab_ar",
        forecast_mode="coefficient_power",
        horizon=horizon,
    )
    pred = fit.predict(pd.DataFrame(index=[y.index[-1] + pd.offsets.MonthEnd()]))

    assert fit.model == "ar_bic"
    assert fit.metadata["selected_lag"] == lag
    assert fit.metadata["selected_nobs"] == nobs
    assert fit.metadata["selected_n_params"] == n_params
    np.testing.assert_allclose(fit.metadata["selected_ic"], score, rtol=0, atol=1e-10)
    np.testing.assert_allclose(fit.coef_, phi, rtol=0, atol=1e-12)
    np.testing.assert_allclose(pred.to_numpy(), np.asarray([expected]), rtol=0, atol=1e-12)


def test_ar_bic_registry_is_target_only_and_exposes_search_space() -> None:
    y = _controlled_target()
    X = pd.DataFrame({"junk": np.linspace(100.0, -50.0, len(y))}, index=y.index)
    direct = mf.models.ar_bic(y, max_lag=4)
    via_registry = mf.models.get_model("ar_bic").fit(X, y, max_lag=4)

    spec = mf.models.get_model("ar_bic")
    assert mf.ar_bic is mf.models.ar_bic
    assert spec.input_kind == "target"
    assert spec.backend == "internal"
    assert spec.default_params == {
        "min_lag": 1,
        "max_lag": 12,
        "criterion": "bic",
        "include_constant": True,
        "ic_parameter_count": "standard",
        "estimator": "ols",
        "forecast_mode": "iterated",
        "horizon": 1,
    }
    assert mf.models.model_search_space("ar_bic", preset="small") == {
        "max_lag": (4, 8, 12)
    }
    assert direct.feature_names == ("__origin__",)
    np.testing.assert_allclose(
        direct.predict(pd.DataFrame(index=range(3))).to_numpy(),
        via_registry.predict(pd.DataFrame(index=range(3))).to_numpy(),
    )


@pytest.mark.parametrize("estimator", ["ols", "yule_walker", "burg", "matlab_ar"])
def test_ar_bic_backends_produce_finite_paths(estimator: str) -> None:
    y = _controlled_target(90)

    fit = mf.models.ar_bic(y, max_lag=4, estimator=estimator)
    pred = fit.predict(pd.DataFrame(index=range(5))).to_numpy()

    assert fit.selected_lag_ in {1, 2, 3, 4}
    assert np.isfinite(fit.coef_).all()
    assert np.isfinite(fit.intercept_)
    assert np.isfinite(pred).all()
    assert len(fit.ic_trials_) == 4


def test_ar_bic_direct_lag_projection_is_ols_only_and_finite() -> None:
    y = _controlled_target(70)

    fit = mf.models.ar_bic(
        y,
        max_lag=3,
        estimator="ols",
        forecast_mode="direct_lag_projection",
        horizon=2,
    )
    pred = fit.predict(pd.DataFrame(index=range(3))).to_numpy()

    assert np.isfinite(pred).all()
    with pytest.raises(ValueError, match="direct_lag_projection requires estimator='ols'"):
        mf.models.ar_bic(
            y,
            max_lag=3,
            estimator="yule_walker",
            forecast_mode="direct_lag_projection",
        )


def test_ar_bic_validation_errors() -> None:
    y = _controlled_target(20)

    with pytest.raises(ValueError, match="min_lag"):
        mf.models.ar_bic(y, min_lag=0)
    with pytest.raises(ValueError, match="min_lag must be less"):
        mf.models.ar_bic(y, min_lag=5, max_lag=2)
    with pytest.raises(ValueError, match="criterion"):
        mf.models.ar_bic(y, criterion="hq")
    with pytest.raises(ValueError, match="include_constant"):
        mf.models.ar_bic(y, include_constant="yes")
    with pytest.raises(ValueError, match="ic_parameter_count"):
        mf.models.ar_bic(y, ic_parameter_count="constant")
    with pytest.raises(ValueError, match="estimator"):
        mf.models.ar_bic(y, estimator="matlab")
    with pytest.raises(ValueError, match="forecast_mode"):
        mf.models.ar_bic(y, forecast_mode="oracle")
    with pytest.raises(ValueError, match="No finite AR information-criterion score"):
        mf.models.ar_bic(pd.Series([1.0, np.nan, np.inf]), max_lag=1)


def test_default_ar_signature_metadata_and_forecast_unchanged() -> None:
    signature = inspect.signature(mf.models.ar)
    assert list(signature.parameters) == ["X", "y", "n_lag", "direct"]
    assert signature.parameters["y"].default is None
    assert signature.parameters["n_lag"].default == 1
    assert signature.parameters["direct"].default is False

    y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], name="y")
    fit = mf.models.ar(y)
    pred = fit.predict(pd.DataFrame(index=range(3))).to_numpy()

    assert fit.model == "ar"
    assert fit.metadata == {"n_obs": 5, "n_lag": 1, "direct": False}
    assert fit.feature_names == ("__origin__",)
    assert fit.estimator.__class__.__name__ == "_AR"
    np.testing.assert_allclose(pred, np.asarray([6.0, 7.0, 8.0]), rtol=0, atol=5e-15)
