"""A one-member combination must not vanish silently.

A pool of one series is that series, so a simple pooling rule should produce it. An
estimated-weight rule genuinely needs two or more contenders to fit weights across, and
should say so rather than disappearing -- the old behaviour dropped the contender without
a word, so every metric derived from it read NaN with nothing to explain it.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.pipeline import (
    Arm,
    CombinationContender,
    EvalSpec,
    TargetSpec,
    pipeline_spec,
    run_pipeline,
)


def _report(members, method="mean"):
    idx = pd.date_range("2000-01-31", periods=120, freq="ME", name="date")
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "y": 1.0 + 2.0 * np.linspace(0.0, 1.0, 120) + rng.normal(0.0, 0.2, 120),
            "x1": np.linspace(0.0, 1.0, 120),
            "x2": rng.normal(size=120),
        },
        index=idx,
    )
    bundle = mf.data.custom_dataset(frame, transform_codes={c: 1 for c in frame.columns})
    window = mf.window.from_cutoffs(
        test_start=idx[80], horizon=1, embargo=0, val_method="expanding", val_min_train_size=24
    )
    arms = [
        Arm(
            "HA",
            model="hist_mean",
            features=mf.feature_engineering.feature_spec(target="y", target_lags=(1,)),
            is_benchmark=True,
        )
    ]
    for member in members:
        arms.append(
            Arm(
                member,
                model="ols",
                features=mf.feature_engineering.feature_spec(
                    target="y", predictors=[member], lags=0, target_lags=None
                ),
            )
        )
    spec = pipeline_spec(
        data=bundle,
        targets=[TargetSpec("y", transform="level")],
        horizons=[1],
        window=window,
        arms=arms,
        evaluation=EvalSpec(benchmark="HA", metrics=("r2_oos",)),
        combinations=[CombinationContender(name="POOL", method=method, over=tuple(members))],
        save_models=False,
    )
    return run_pipeline(spec)


def test_single_member_mean_pool_is_produced() -> None:
    report = _report(["x1"])
    forecasts = report.forecasts
    pool = forecasts[forecasts["contender"] == "POOL"]
    member = forecasts[forecasts["contender"] == "x1"]

    assert len(pool) == len(member) > 0
    # the pool of one series is that series
    np.testing.assert_allclose(
        pool.sort_values("origin")["prediction"].to_numpy(float),
        member.sort_values("origin")["prediction"].to_numpy(float),
    )
    assert (report.accuracy["contender"] == "POOL").any()


def test_two_member_pool_still_works() -> None:
    report = _report(["x1", "x2"])
    forecasts = report.forecasts
    assert len(forecasts[forecasts["contender"] == "POOL"]) > 0
    assert (report.accuracy["contender"] == "POOL").any()


def test_single_member_estimated_weight_pool_warns_instead_of_vanishing() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = _report(["x1"], method="bates_granger")

    # weights cannot be fitted across a single contender, so it is still skipped ...
    assert len(report.forecasts[report.forecasts["contender"] == "POOL"]) == 0
    # ... but the caller is told why
    assert any(
        "POOL" in str(w.message) and "at least two" in str(w.message) for w in caught
    )
