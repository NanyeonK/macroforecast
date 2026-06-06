"""Stage 6: the pipeline can implement a GCLS-2021-style transformation comparison.

GCLS compare DATA TRANSFORMATIONS holding the model fixed, reporting relative RMSE
vs an AR benchmark with Clark-West significance, across horizons. Here each arm is a
transformation (different feature representation of the same predictors), so the
contender x horizon accuracy matrix IS the transformation comparison.
"""
import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.pipeline import (
    Arm, CombinationContender, EvalSpec, pipeline_spec, run_pipeline,
)


def _bundle(n=160):
    idx = pd.date_range("1990-01-31", periods=n, freq="ME", name="date")
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n).cumsum()
    y1 = 0.6 * np.r_[0, x[:-1]] + rng.standard_normal(n) * 0.5      # INDPRO-like
    y2 = 0.3 * np.r_[0, x[:-1]] + rng.standard_normal(n) * 0.5      # second target
    frame = pd.DataFrame({"INDPRO": y1, "PRICE": y2, "x1": x}, index=idx)
    return mf.data.custom_dataset(frame, transform_codes={"INDPRO": 1, "PRICE": 1, "x1": 1})


def _window():
    return mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=72),
        val=mf.window.val_last_block(size=24),
        test=mf.window.test_origins(horizon=1, step=6),
    )


def _gcls_spec():
    # AR benchmark + two "transformations" (different feature representations), ridge model
    ar = mf.feature_engineering.feature_spec(target="INDPRO", predictors=[], lags=None, target_lags=(0, 1, 2))
    t_lags = mf.feature_engineering.feature_spec(target="INDPRO", predictors=["x1"], lags=(0, 1), target_lags=(0, 1))
    t_roll = mf.feature_engineering.feature_spec(target="INDPRO", predictors=["x1"], lags=(0, 1),
                                                 target_lags=(0, 1), rolling_windows=3)
    return pipeline_spec(
        data=_bundle(),
        targets=[mf.pipeline.TargetSpec("INDPRO", transform="level"),
                 mf.pipeline.TargetSpec("PRICE", transform="level")],
        horizons=[1, 3],
        window=_window(),
        arms=[Arm("AR", model="ar", features=ar),
              Arm("LAGS", model="ridge", features=t_lags, nested_in_benchmark=True),
              Arm("ROLL", model="ridge", features=t_roll, nested_in_benchmark=True)],
        evaluation=EvalSpec(benchmark="AR", tests=["dm", "cw", "mcs"], primary_axis="transform"),
        combinations=[CombinationContender("POOL", method="constrained_ls", params={"min_periods": 10})],
        save_models=False,
    )


def test_gcls_transformation_matrix_and_significance():
    report = run_pipeline(_gcls_spec())
    acc = report.accuracy
    # transformation x target x horizon relative-RMSE matrix
    assert set(acc["contender"]) >= {"AR", "LAGS", "ROLL", "POOL"}
    assert set(acc["target"]) == {"INDPRO", "PRICE"}
    assert set(acc["horizon"]) == {1, 3}
    # benchmark normalised to 1
    bench = acc[acc["is_benchmark"]]
    assert np.allclose(bench["relative_mse"].to_numpy(), 1.0)
    # Clark-West significance present for the non-benchmark transformations
    sig = report.significance
    assert {"cw_p", "dm_p"}.issubset(sig.columns)
    assert set(sig["contender"]) <= {"LAGS", "ROLL", "POOL"}
    # provenance records the GCLS framing
    assert report.provenance["benchmark"] == "AR"
    assert "POOL" in report.provenance["combinations"]
    # common-sample column present (audit fix) so the matrix is comparable
    assert "n_common" in acc.columns


def test_primary_axis_recorded():
    spec = _gcls_spec()
    assert spec.evaluation.primary_axis == "transform"
