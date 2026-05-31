from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf


def test_equal_predictive_accuracy_callables_return_test_results() -> None:
    loss_a = pd.Series([0.2, 0.3, 0.1, 0.4, 0.2, 0.3])
    loss_b = pd.Series([0.4, 0.5, 0.2, 0.5, 0.3, 0.6])

    dm = mf.tests.dm_test(loss_a, loss_b, horizon=1)
    gw = mf.tests.gw_test(loss_a, loss_b, horizon=1)
    dmp = mf.tests.dmp_test([loss_a - loss_b, (loss_a - loss_b) * 0.5])

    assert isinstance(dm, mf.tests.TestResult)
    assert dm.metadata["name"] == "Diebold-Mariano"
    assert gw.metadata["name"] == "Giacomini-White"
    assert dmp.n_obs == 12
    assert 0.0 <= dmp.p_value <= 1.0
    dm_payload = json.loads(dm.to_json())
    assert dm_payload["metadata_schema"]["kind"] == "forecast_test_result"
    assert dm_payload["metadata_schema"]["version"] == 1


def test_nested_and_encompassing_callables_return_one_sided_results() -> None:
    loss_small = pd.Series([0.5, 0.6, 0.4, 0.7, 0.5, 0.6])
    loss_large = pd.Series([0.3, 0.4, 0.2, 0.5, 0.4, 0.3])
    forecast_small = pd.Series([1.0, 1.1, 1.0, 1.2, 0.9, 1.0])
    forecast_large = pd.Series([1.1, 1.2, 0.9, 1.1, 1.0, 1.1])

    cw = mf.tests.clark_west_test(loss_small, loss_large, forecast_small, forecast_large)
    enc_new = mf.tests.enc_new_test(loss_small, loss_large)
    enc_t = mf.tests.enc_t_test(loss_small, loss_large)
    hn = mf.tests.harvey_newbold_test(loss_small**0.5, loss_large**0.5)

    assert cw.alternative == "one_sided"
    assert enc_new.metadata["name"] == "Enc-New"
    assert enc_t.metadata["name"] == "Enc-T"
    assert hn.metadata["name"] == "Harvey-Newbold"


def test_direction_density_and_residual_diagnostics() -> None:
    y_true = pd.Series([1.0, -1.0, 2.0, -0.5, 0.8, -0.2])
    y_pred = pd.Series([0.7, -0.8, 1.5, 0.1, 0.9, -0.3])
    pit = np.linspace(0.05, 0.95, 20)
    residuals = pd.Series(np.sin(np.arange(30)))

    pt = mf.tests.pesaran_timmermann_test(y_true, y_pred)
    hm = mf.tests.henriksson_merton_test(y_true, y_pred)
    density = mf.tests.density_interval_tests(pit, alpha=0.1)
    diagnostics = mf.tests.residual_diagnostics(residuals)

    assert pt.metadata["method"] == "pesaran_timmermann"
    assert hm.metadata["method"] == "henriksson_merton"
    assert {"berkowitz", "ks", "kupiec_pof", "christoffersen_independence"}.issubset(
        density
    )
    assert density["metadata_schema"]["kind"] == "density_interval_tests"
    assert set(diagnostics["test"]) == {
        "ljung_box_q",
        "arch_lm",
        "jarque_bera_normality",
        "durbin_watson",
    }
    assert diagnostics.attrs["macroforecast_metadata_schema"]["kind"] == "residual_diagnostics"


def test_conditional_predictive_ability_and_model_confidence_set() -> None:
    rng = np.random.default_rng(123)
    loss_a = pd.Series(rng.normal(0.8, 0.1, 24))
    loss_b = pd.Series(rng.normal(1.0, 0.1, 24))
    cpa = mf.tests.conditional_predictive_ability_test(
        loss_a,
        loss_b,
        method="giacomini_rossi",
        window_ratio=0.25,
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
        spa_benchmark_model="a",
        random_state=123,
    )

    assert cpa["method"] == "giacomini_rossi"
    assert cpa["metadata_schema"]["kind"] == "conditional_predictive_ability"
    assert cpa["n_obs"] == 24
    assert mcs["metadata_schema"]["kind"] == "model_confidence_set"
    mcs_record = next(item for item in mcs["mcs_inclusion"] if item["target"] == "y")
    assert mcs_record["horizon"] == 1
    assert mcs_record["alpha"] == 0.1
    assert set(mcs_record["models"]).issubset({"a", "b"})
    assert any(
        item["target"] == "y" and item["horizon"] == 1
        for item in mcs["spa_p_values"]
    )
    assert mcs["bootstrap_n_replications"] == 20
    json.dumps(mcs)


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
    small = mf.tests.density_interval_tests([0.25, 0.75])
    assert small["berkowitz"]["df"] == 2
