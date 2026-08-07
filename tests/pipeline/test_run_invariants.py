"""Invariants a forecasting run must satisfy regardless of how it was assembled.

Each of these is a property the ANSWER must have, not a property of the code that
produced it -- so they are written against the public pipeline and would survive a
rewrite of anything underneath.

Three come from an external audit of `main`; the benchmark one extends what
`test_target_only_fit_sample.py` already pins. That file checks the benchmark's
first forecast does not move when a predictor it never reads has a gap. The
audit's point is that a benchmark can also drift in ways a single prediction
hides -- a different scored sample, a different loss, a different `R2_OS`
denominator -- so the comparison here covers the whole scored record.

Serial/parallel and cache equivalence are deliberately NOT re-tested; they are
already covered by `test_mrf_parallel_is_bit_identical_to_serial_when_seeded`,
`test_pipeline_seed_drives_model_random_state_serial_and_parallel`,
`test_pipeline_cache_dir_preserves_numbers` and others.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.pipeline import Arm, EvalSpec, TargetSpec, pipeline_spec, run_pipeline

N = 90
TEST_AT = 70


def _panel(seed: int = 0) -> pd.DataFrame:
    idx = pd.date_range("1990-01-31", periods=N, freq="ME", name="date")
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({f"x{i}": rng.normal(size=N) for i in range(4)}, index=idx)
    frame["y"] = 0.6 * frame["x0"] + rng.normal(size=N) * 0.4
    return frame


def _bundle(frame: pd.DataFrame):
    return mf.data.custom_dataset(frame, transform_codes={c: 1 for c in frame.columns})


def _window(frame: pd.DataFrame, horizon: int = 1):
    return mf.window.from_cutoffs(
        test_start=frame.index[TEST_AT],
        horizon=horizon,
        embargo=0,
        val_method="expanding",
        val_min_train_size=12,
    )


def _features(frame: pd.DataFrame):
    return mf.feature_engineering.feature_spec(
        target="y",
        predictors=[c for c in frame.columns if c != "y"],
        lags=0,
        target_lags=1,
    )


def _run(frame: pd.DataFrame, arms, *, horizon: int = 1, seed: int = 42):
    spec = pipeline_spec(
        data=_bundle(frame),
        targets=[TargetSpec("y", transform="level")],
        horizons=[horizon],
        window=_window(frame, horizon),
        arms=arms,
        evaluation=EvalSpec(
            benchmark="AR",
            metrics=("rmse", "relative_mse", "r2_oos"),
            tests=("dm",),
        ),
        save_models=False,
        seed=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return run_pipeline(spec)


def _canonical(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort so a comparison is about content, not emission order."""
    keys = [c for c in ("target", "horizon", "contender", "date", "origin") if c in frame]
    return frame.sort_values(keys).reset_index(drop=True)


def _assert_frames_identical(a: pd.DataFrame, b: pd.DataFrame, what: str) -> None:
    assert not a.empty, f"{what}: nothing to compare"
    assert list(a.columns) == list(b.columns), f"{what}: columns differ"
    assert len(a) == len(b), f"{what}: row counts differ ({len(a)} vs {len(b)})"
    for col in a.columns:
        if pd.api.types.is_numeric_dtype(a[col]):
            np.testing.assert_allclose(
                a[col].to_numpy(dtype=float),
                b[col].to_numpy(dtype=float),
                rtol=0,
                atol=1e-12,
                err_msg=f"{what}.{col} moved",
            )
        else:
            assert a[col].tolist() == b[col].tolist(), f"{what}.{col} moved"


# --------------------------------------------------------------------------- #
# A. the benchmark is a property of itself, not of its company
# --------------------------------------------------------------------------- #


def test_the_benchmark_record_is_identical_whoever_else_is_in_the_run() -> None:
    """Every scored quantity, not only the first prediction.

    A benchmark that changes when an unrelated contender joins makes every ratio
    in the report a function of the arm list, which is the one thing a benchmark
    may not be. #491 closed one route to that; this pins the whole scored record
    so a second route cannot open quietly.
    """
    frame = _panel()
    bench = Arm("AR", model="ar", is_benchmark=True)
    ols = Arm("OLS", model="ols", features=_features(frame))
    # the shape of arm that used to truncate the shared fit sample
    long_lag = Arm(
        "LONGLAG",
        model="ols",
        features=mf.feature_engineering.feature_spec(
            target="y", predictors=["x0", "x1"], lags=range(1, 25), target_lags=1
        ),
    )

    alone = _run(frame, [bench, ols])
    crowded = _run(frame, [bench, ols, long_lag])

    for name in ("forecasts", "accuracy"):
        a = _canonical(getattr(alone, name).query("contender == 'AR'"))
        b = _canonical(getattr(crowded, name).query("contender == 'AR'"))
        _assert_frames_identical(a, b, f"benchmark {name}")


# --------------------------------------------------------------------------- #
# B. the order arms are listed in is not information
# --------------------------------------------------------------------------- #


def test_results_do_not_depend_on_the_order_arms_are_listed_in() -> None:
    """Seeds included: a stochastic arm must draw from its identity, not its slot.

    If a per-arm seed came from position, reordering the list would silently
    reassign randomness -- results moving for a reason that is nowhere in the
    specification.
    """
    frame = _panel()

    def arms(order):
        built = {
            "AR": Arm("AR", model="ar", is_benchmark=True),
            "OLS": Arm("OLS", model="ols", features=_features(frame)),
            # A stochastic arm is the point of this test -- a deterministic pair
            # would pass whatever the seeding does. Kept deliberately tiny: what
            # matters is that the draw is reproducible, not that the forest is good.
            "RF": Arm(
                "RF",
                model="random_forest",
                features=_features(frame),
                params={"n_estimators": 4, "max_depth": 3},
            ),
        }
        return [built[name] for name in order]

    a = _run(frame, arms(["AR", "OLS", "RF"]))
    b = _run(frame, arms(["RF", "AR", "OLS"]))

    for name in ("forecasts", "accuracy"):
        _assert_frames_identical(
            _canonical(getattr(a, name)),
            _canonical(getattr(b, name)),
            f"{name} (arm order)",
        )


# --------------------------------------------------------------------------- #
# C. rows after the origin cannot reach back
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("horizon", [1, 3])
def test_corrupting_the_future_leaves_the_first_forecast_unchanged(horizon: int) -> None:
    """Mutation, not appending -- and only the forecast that can be compared.

    Appending rows past the end only shows the window stops where it should.
    Overwriting the PREDICTORS with a value no estimator could ignore is
    stronger: it shows no fitted state -- scaling, PCA, selection,
    hyperparameters -- ever saw them.

    Only the FIRST test origin's forecast is comparable, and the corruption has
    to start STRICTLY after it. A forecast made at origin `t` legitimately reads
    `X_t`; it is the origin's own observation, not the future. Later origins
    legitimately read rows this test has poisoned, so their forecasts are
    expected to differ and say nothing about leakage.

    `actual` is excluded throughout: those are the future, and they are supposed
    to change.
    """
    frame = _panel()
    arms = [
        Arm("AR", model="ar", is_benchmark=True),
        Arm("OLS", model="ols", features=_features(frame)),
    ]
    clean = _run(frame, arms, horizon=horizon)

    poisoned = frame.copy()
    predictors = [c for c in poisoned.columns if c != "y"]
    poisoned.loc[poisoned.index[TEST_AT + 1 :], predictors] = 1e12
    dirty = _run(poisoned, arms, horizon=horizon)

    a = _canonical(clean.forecasts).groupby("contender", as_index=False).first()
    b = _canonical(dirty.forecasts).groupby("contender", as_index=False).first()
    merged = a.merge(b, on=["contender", "date"], suffixes=("_clean", "_dirty"))
    assert len(merged) == len(a), "the two runs did not produce the same first cells"
    np.testing.assert_allclose(
        merged["prediction_clean"].to_numpy(dtype=float),
        merged["prediction_dirty"].to_numpy(dtype=float),
        rtol=0,
        atol=1e-10,
        err_msg=(
            "the first test forecast changed when predictor values STRICTLY "
            "after its origin were replaced with 1e12 -- something was fitted "
            "on rows that origin could not see"
        ),
    )
