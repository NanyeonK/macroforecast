import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.pipeline import Arm, EvalSpec, pipeline_spec, run_pipeline


def _bundle(n=72):
    idx = pd.date_range("2000-01-31", periods=n, freq="ME", name="date")
    x = np.linspace(0.0, 1.0, n)
    frame = pd.DataFrame({"y": 1.0 + 2.0 * x, "x1": x}, index=idx)
    return mf.data.custom_dataset(frame, transform_codes={"y": 1, "x1": 1})


def _features():
    return mf.feature_engineering.feature_spec(
        target="y",
        predictors=["x1"],
        lags=1,
        target_lags=(0, 1),
    )


def _window():
    return mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=36),
        val=mf.window.val_last_block(size=12),
        test=mf.window.test_origins(horizon=1, step=8),
    )


def _spec(*, n_jobs=1):
    feats = _features()
    return pipeline_spec(
        data=_bundle(),
        targets=["y"],
        horizons=[1, 3],
        window=_window(),
        arms=[
            Arm("linear", model="ols", features=feats, tags={"NL": 0, "family": "linear"}),
            Arm(
                "ridge",
                model="ridge",
                features=feats,
                tags={"NL": 1, "family": "regularized", "selected": True},
            ),
        ],
        evaluation=EvalSpec(benchmark="linear", metrics=("rmse",)),
        save_models=False,
        n_jobs=n_jobs,
    )


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_arm_tags_propagate_to_master_serial_and_parallel(n_jobs):
    report = run_pipeline(_spec(n_jobs=n_jobs))
    forecasts = report.forecasts

    assert {"tag_NL", "tag_family", "tag_selected"} <= set(forecasts.columns)
    by_arm = forecasts.groupby("arm", sort=True)
    assert set(by_arm.get_group("linear")["tag_NL"]) == {0}
    assert set(by_arm.get_group("linear")["tag_family"]) == {"linear"}
    assert by_arm.get_group("linear")["tag_selected"].isna().all()
    assert set(by_arm.get_group("ridge")["tag_NL"]) == {1}
    assert set(by_arm.get_group("ridge")["tag_family"]) == {"regularized"}
    assert set(by_arm.get_group("ridge")["tag_selected"]) == {True}


def test_arm_tags_echo_in_provenance():
    report = run_pipeline(_spec())

    arms = {
        arm["name"]: arm
        for arm in report.provenance["spec_echo"]["arms"]
    }
    assert arms["linear"]["tags"] == {"NL": 0, "family": "linear"}
    assert arms["ridge"]["tags"] == {
        "NL": 1,
        "family": "regularized",
        "selected": True,
    }


def test_arm_tag_validation_rejects_non_identifier_keys_and_non_scalar_values():
    with pytest.raises(ValueError, match="valid identifiers"):
        Arm("bad-key", model="ols", tags={"not-a-column": 1})

    with pytest.raises(TypeError, match="scalar str, int, float, or bool"):
        Arm("bad-value", model="ols", tags={"axis": ["NL"]})

    with pytest.raises(ValueError, match="finite"):
        Arm("bad-float", model="ols", tags={"axis": float("nan")})
