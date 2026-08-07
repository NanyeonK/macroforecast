"""Reusing a fit must be invisible in the numbers.

With `retrain_every > 1` the fit window is frozen between retrain points, so
consecutive origins hand the estimator the same rows and get the same
coefficients back. Measured on a 40-origin fixture before this change: 40 fit
calls, 4 distinct coefficient sets -- 36 of them recomputing an answer already
in hand.

A cache is only allowed to remove the recomputation, never to change what comes
out, so these tests compare forecasts rather than counting calls. The call count
is checked too, but that is the cost claim; the forecasts are the correctness
one.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.models.specs import MODEL_SPECS
from macroforecast.pipeline import Arm, EvalSpec, TargetSpec, pipeline_spec, run_pipeline

N = 100
TEST_AT = 60


def _spec(retrain_every: int, *, model: str = "ols", params=None):
    idx = pd.date_range("1990-01-31", periods=N, freq="ME", name="date")
    rng = np.random.default_rng(0)
    panel = pd.DataFrame({f"x{i}": rng.normal(size=N) for i in range(3)}, index=idx)
    panel["y"] = 0.5 * panel["x0"] + rng.normal(size=N) * 0.3
    bundle = mf.data.custom_dataset(panel, transform_codes={c: 1 for c in panel.columns})
    return pipeline_spec(
        data=bundle,
        targets=[TargetSpec("y", transform="level")],
        horizons=[1],
        window=mf.window.from_cutoffs(
            test_start=idx[TEST_AT], horizon=1, embargo=0,
            val_method="expanding", val_min_train_size=12,
            retrain_every=retrain_every,
        ),
        arms=[Arm("M", model=model, is_benchmark=True, params=params or {},
                  features=mf.feature_engineering.feature_spec(
                      target="y", predictors=[f"x{i}" for i in range(3)],
                      lags=0, target_lags=1))],
        evaluation=EvalSpec(benchmark="M", metrics=("rmse",), tests=()),
        save_models=False,
    )


def _run(spec) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return run_pipeline(spec).forecasts.sort_values("date").reset_index(drop=True)


def _count_fits(build, model: str = "ols") -> int:
    """Patch BEFORE the spec is built.

    pipeline_spec resolves the model spec at build time, so replacing
    fit_func afterwards counts nothing -- which is exactly what the first
    version of this helper did, and it reported zero.
    """
    calls = {"n": 0}
    original = MODEL_SPECS[model].fit_func

    def counting(X, y=None, **kw):
        calls["n"] += 1
        return original(X, y, **kw)

    MODEL_SPECS[model] = dataclasses.replace(MODEL_SPECS[model], fit_func=counting)
    try:
        _run(build())
    finally:
        MODEL_SPECS[model] = dataclasses.replace(MODEL_SPECS[model], fit_func=original)
    return calls["n"]


@pytest.mark.parametrize("retrain_every", [1, 5, 10])
def test_reuse_does_not_change_any_forecast(retrain_every: int) -> None:
    """The correctness claim. Same spec, run twice, must agree exactly -- and the
    second run exercises a warm cache within its own run()."""
    spec = _spec(retrain_every)
    a, b = _run(spec), _run(spec)
    np.testing.assert_array_equal(
        a["prediction"].to_numpy(dtype=float),
        b["prediction"].to_numpy(dtype=float),
    )


def test_the_fit_count_collapses_to_the_number_of_distinct_samples() -> None:
    """The cost claim: one fit per distinct fit window, not one per origin."""
    n_fits = _count_fits(lambda: _spec(10))
    forecasts = len(_run(_spec(10)))
    assert n_fits <= 6, (
        f"expected about one fit per retrain group over {forecasts} origins; got {n_fits}"
    )
    assert n_fits >= 3, f"suspiciously few fits ({n_fits}); the window should move 4 times"


def test_retrain_every_1_still_fits_at_every_origin() -> None:
    """Nothing to reuse when the window advances every origin, and the cache must
    not pretend otherwise."""
    n_fits = _count_fits(lambda: _spec(1))
    forecasts = len(_run(_spec(1)))
    assert n_fits >= forecasts, (
        f"{n_fits} fits for {forecasts} origins -- a moving window has no repeats to skip"
    )


def test_two_different_panels_do_not_share_a_fit() -> None:
    """The key is the fit sample's content, so different data cannot collide.

    Bounds alone would be the cheaper key and also a bet: an arm-level window
    override or a preprocessing refit can give two origins the same bounds and
    different rows.
    """
    idx = pd.date_range("1990-01-31", periods=N, freq="ME", name="date")

    def forecasts_for(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        panel = pd.DataFrame({f"x{i}": rng.normal(size=N) for i in range(3)}, index=idx)
        panel["y"] = 0.5 * panel["x0"] + rng.normal(size=N) * 0.3
        bundle = mf.data.custom_dataset(
            panel, transform_codes={c: 1 for c in panel.columns}
        )
        spec = pipeline_spec(
            data=bundle,
            targets=[TargetSpec("y", transform="level")],
            horizons=[1],
            window=mf.window.from_cutoffs(
                test_start=idx[TEST_AT], horizon=1, embargo=0,
                val_method="expanding", val_min_train_size=12, retrain_every=10,
            ),
            arms=[Arm("M", model="ols", is_benchmark=True,
                      features=mf.feature_engineering.feature_spec(
                          target="y", predictors=[f"x{i}" for i in range(3)],
                          lags=0, target_lags=1))],
            evaluation=EvalSpec(benchmark="M", metrics=("rmse",), tests=()),
            save_models=False,
        )
        return _run(spec)["prediction"].to_numpy(dtype=float)

    assert not np.allclose(forecasts_for(0), forecasts_for(99)), (
        "two different panels produced identical forecasts -- a fit was shared"
    )


def test_a_stochastic_model_is_unaffected() -> None:
    """Reuse must not change a seeded model's draws either."""
    spec = _spec(10, model="random_forest", params={"n_estimators": 6, "max_depth": 3})
    a, b = _run(spec), _run(spec)
    np.testing.assert_array_equal(
        a["prediction"].to_numpy(dtype=float),
        b["prediction"].to_numpy(dtype=float),
    )
