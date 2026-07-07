import numpy as np
import pandas as pd
import pytest

import macroforecast as mf


PHI = np.asarray(
    [
        [0.62, 0.18, -0.08],
        [0.05, 0.48, 0.10],
        [-0.04, 0.08, 0.38],
    ],
    dtype=float,
)


def _stationary_var_panel(n: int = 2200, *, seed: int = 442) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    values = np.zeros((n, 3), dtype=float)
    shocks = rng.normal(scale=0.08, size=(n, 3))
    for t in range(1, n):
        values[t] = PHI @ values[t - 1] + shocks[t]
    idx = pd.date_range("1980-01-01", periods=n, freq="MS", name="date")
    return pd.DataFrame(values, index=idx, columns=["y0", "y1", "y2"])


def _origin_with_nonpersistent_target(panel: pd.DataFrame, horizon: int) -> int:
    phi_h = np.linalg.matrix_power(PHI, horizon)
    for pos in range(len(panel) - horizon - 1, 500, -1):
        state = panel.iloc[pos].to_numpy(dtype=float)
        population = float((phi_h @ state)[0])
        persistence = float(state[0])
        if abs(persistence - population) > 0.08:
            return pos
    raise AssertionError("synthetic VAR did not produce a usable origin")


@pytest.mark.parametrize("horizon", [1, 6, 12])
def test_var_direct_projection_matches_population_oracle(horizon: int) -> None:
    panel = _stationary_var_panel()
    origin_pos = _origin_with_nonpersistent_target(panel, horizon)
    train = panel.iloc[: origin_pos + 1]
    state = train.iloc[-1].to_numpy(dtype=float)
    population_coef = np.linalg.matrix_power(PHI, horizon)[0]
    population_mean = float(population_coef @ state)

    direct_fit = mf.models.var(
        train,
        target="y0",
        n_lag=1,
        type="const",
        direct=True,
        direct_horizon=horizon,
    )
    direct_estimator = direct_fit.estimator
    direct_pred = float(direct_fit.predict(pd.DataFrame(index=range(horizon))).iloc[-1])

    assert direct_estimator.direct_coef_ is not None
    np.testing.assert_allclose(
        direct_estimator.direct_coef_[:3],
        population_coef,
        atol=0.10,
        rtol=0.0,
    )
    assert abs(float(direct_estimator.direct_coef_[-1])) < 0.04

    if horizon == 1:
        iterated_fit = mf.models.var(train, target="y0", n_lag=1, type="const")
        iterated_pred = float(iterated_fit.predict(pd.DataFrame(index=range(1))).iloc[0])
        assert direct_pred == pytest.approx(iterated_pred, abs=1e-6)

    if horizon == 12:
        persistence = float(state[0])
        assert abs(direct_pred - population_mean) < abs(persistence - population_mean)


def test_var_direct_runner_does_not_use_future_target_values() -> None:
    horizon = 6
    origin_pos = 460
    panel = _stationary_var_panel(n=520)
    mutated = panel.copy()
    mutated.loc[mutated.index[origin_pos + 1 :], "y0"] += 10_000.0

    win = mf.window.from_cutoffs(
        test_start=panel.index[origin_pos],
        test_end=panel.index[origin_pos],
        mode="expanding",
        val_method="last_block",
        retrain_every=999,
        horizon=horizon,
    )

    def _forecast(frame: pd.DataFrame) -> pd.DataFrame:
        bundle = mf.data.custom_dataset(
            frame,
            transform_codes={column: 1 for column in frame.columns},
        )
        return mf.forecasting.run(
            bundle,
            "var",
            window=win,
            features=None,
            target="y0",
            horizons=[horizon],
            forecast_policy="direct",
            save_models=False,
        ).to_frame()

    base = _forecast(panel)
    changed = _forecast(mutated)

    assert len(base) == 1
    assert len(changed) == 1
    assert base.iloc[0]["forecast_policy"] == "direct"
    assert base.iloc[0]["horizon"] == horizon
    assert base.iloc[0]["params"]["direct"] is True
    assert base.iloc[0]["params"]["direct_horizon"] == horizon
    assert float(base.iloc[0]["prediction"]) == pytest.approx(
        float(changed.iloc[0]["prediction"]),
        abs=1e-12,
    )
