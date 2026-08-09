"""A0 — characterization tests for the contracts a structural refactor must preserve.

These pin four properties that nothing currently guarantees, because the same research
design is resolved independently in several places: `pipeline/spec.py` resolves the
target, `pipeline/run.py` retargets the arm's `FeatureSpec`, `forecasting/policy_config.py`
re-derives transform/horizon/target-mode, and `pipeline/result_store.py` retargets the
features *again* for cache identity.

Nothing here proposes a new structure. Each test states a property that must survive
`ResolvedForecastTask`, canonical `WindowSpec`, `OriginContext` and the compiled stage
plans, so that a refactor which breaks one is caught by a failing test rather than by a
wrong number six months later.

Where current behaviour already violates a property, the test says so explicitly rather
than encoding the violation as expected.
"""
from __future__ import annotations

import dataclasses as dc

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.pipeline import Arm, EvalSpec, TargetSpec, pipeline_spec, run_pipeline

N = 90
TEST_START = 70


@pytest.fixture(scope="module")
def panel():
    idx = pd.date_range("1990-01-31", periods=N, freq="ME", name="date")
    rng = np.random.default_rng(4)
    frame = pd.DataFrame({f"x{i}": rng.normal(size=N) for i in range(4)}, index=idx)
    frame["y"] = 0.5 * frame["x0"] + rng.normal(size=N) * 0.3
    bundle = mf.data.custom_dataset(frame, transform_codes={c: 1 for c in frame.columns})
    return idx, bundle


def _features():
    return mf.feature_engineering.feature_spec(
        target="y", predictors=["x0", "x1", "x2", "x3"], lags=0, target_lags=2
    )


def _window(idx, horizon=1):
    return mf.window.from_cutoffs(
        test_start=idx[TEST_START],
        horizon=horizon,
        embargo=0,
        val_method="expanding",
        val_min_train_size=20,
    )


def _spec(idx, bundle, *, arms=None, horizons=(1,), **kw):
    return pipeline_spec(
        data=bundle,
        targets=[TargetSpec("y", transform="level")],
        horizons=list(horizons),
        window=_window(idx, max(horizons)),
        arms=(arms := arms or [Arm("OLS", model="ols", features=_features(), is_benchmark=True)]),
        evaluation=EvalSpec(benchmark=arms[0].name, metrics=("rmse",), tests=()),
        save_models=False,
        n_jobs=1,
        seed=11,
        **kw,
    )


# --------------------------------------------------------------------------- #
# 1. Task-resolution parity
# --------------------------------------------------------------------------- #

def test_one_cell_pipeline_matches_a_direct_forecasting_run(panel):
    """The same task, resolved by two different code paths, must forecast the same.

    `pipeline.run_pipeline` resolves target/transform/policy/features/window through
    `pipeline_spec` + `run.py`; `forecasting.run` resolves them through
    `policy_config.py`. Today those are separate resolvers. If a refactor unifies them
    behind one `ResolvedForecastTask`, this must keep passing; if a refactor breaks the
    agreement, this is where it shows.
    """
    idx, bundle = panel
    spec = _spec(idx, bundle)
    piped = run_pipeline(spec).forecasts

    direct = mf.forecasting.run(
        bundle,
        target="y",
        horizon=1,
        model="ols",
        features=_features(),
        window=_window(idx, 1),
        target_transform="level",
    )
    direct_frame = direct.forecasts

    p = piped.sort_values("date").reset_index(drop=True)
    d = direct_frame.sort_values("date").reset_index(drop=True)
    assert len(p) == len(d), (
        f"the two paths produced different numbers of forecasts: {len(p)} vs {len(d)}"
    )
    np.testing.assert_allclose(
        p["prediction"].to_numpy(float),
        d["prediction"].to_numpy(float),
        rtol=0,
        atol=1e-12,
        err_msg="a one-cell pipeline and a direct forecasting.run disagree",
    )


def test_the_pipeline_resolves_the_feature_target_exactly_once(panel):
    """`FeatureSpec` retargeting happens in at least three places today.

    `pipeline/run.py` replaces the arm's spec, `forecasting` re-derives it per policy,
    and `result_store._retargeted_features` does it a third time for cache identity.
    They must agree on the target; a refactor should make that structural rather than
    coincidental.
    """
    from macroforecast.pipeline.result_store import _retargeted_features

    features = mf.feature_engineering.feature_spec(
        target="SOMETHING_ELSE", predictors=["x0"], lags=0, target_lags=1
    )
    retargeted = _retargeted_features(features, "y")
    assert getattr(retargeted, "target", None) == "y", (
        "the result-store retargeting no longer produces the pipeline's target; "
        "cache identity and execution would then disagree about which series was "
        "being forecast"
    )


# --------------------------------------------------------------------------- #
# 2. Spec immutability
# --------------------------------------------------------------------------- #

@pytest.mark.xfail(
    strict=True,
    reason="FINDING (2026-08-09): pipeline_spec stores Arm.params BY REFERENCE. "
    " is True, so a later mutation changes the "
    "run, the digest and the provenance echo. frozen=True freezes the reference, "
    "not the mapping. Fix is deep canonicalization at the builder boundary; "
    "strict=True so this flips to a failure the moment it is fixed.",
)
def test_mutating_the_caller_dict_after_building_cannot_change_the_spec(panel):
    """A frozen spec must be frozen through its containers, not only at the top level.

    `Arm.params` is a plain mapping on a `frozen=True` dataclass, so the freeze stops at
    the reference. If the caller's dict is stored by reference, a later mutation silently
    changes the run, the result-store digest, and the provenance echo — after the spec was
    supposedly fixed.
    """
    idx, bundle = panel
    params = {"alpha": 1.0}
    arm = Arm("RIDGE", model="ridge", features=_features(), params=params, is_benchmark=True)
    spec = _spec(idx, bundle, arms=[arm])

    params["alpha"] = 999.0

    got = spec.arms[0].params["alpha"]
    assert got == 1.0, (
        f"mutating the caller's dict after pipeline_spec() changed the spec "
        f"(alpha is now {got}). The spec is frozen at the top level only, so the run, "
        f"the result-store digest and the provenance echo can all move after the spec "
        f"was supposedly fixed."
    )


@pytest.mark.xfail(
    strict=True,
    reason="FINDING (2026-08-09): the same by-reference storage moves the result-store "
    "digest. A cached cell can be served for a configuration that no longer matches, "
    "or recomputed for one that does.",
)
def test_the_result_store_digest_does_not_move_when_the_caller_dict_moves(panel):
    """The digest is what makes a cached cell reusable; it must be as frozen as the spec."""
    from macroforecast.pipeline.result_store import result_cell_identity as cell_identity

    idx, bundle = panel
    params = {"alpha": 1.0}
    arm = Arm("RIDGE", model="ridge", features=_features(), params=params, is_benchmark=True)
    spec = _spec(idx, bundle, arms=[arm])

    before = cell_identity(spec, spec.arms[0], spec.targets[0], horizon=1, data_identity={"fingerprint": "fp"})
    params["alpha"] = 999.0
    after = cell_identity(spec, spec.arms[0], spec.targets[0], horizon=1, data_identity={"fingerprint": "fp"})

    assert before.digest == after.digest, (
        "the result-store digest changed because the caller mutated a dict it passed in. "
        "A cached cell could then be served for a different configuration, or recomputed "
        "for an identical one."
    )


# --------------------------------------------------------------------------- #
# 3. Window conflict
# --------------------------------------------------------------------------- #

def test_a_window_cannot_carry_two_disagreeing_validation_sizes():
    """`WindowSpec` holds both nested and legacy flat fields for the same quantity.

    `val=ValWindow(size=24)` and `validation_size=36` describe one thing twice. Silently
    preferring one is the failure mode worth preventing: the caller believes the value
    they passed is in force. Either construction refuses, or the flat field is a
    documented alias that compiles into the nested one — but it must not be a coin flip.
    """
    from macroforecast.window.core import TestWindow, ValWindow, WindowSpec

    try:
        spec = WindowSpec(
            estimation=None,
            val=ValWindow(method="last_block", size=24),
            test=TestWindow(start=0, end=10, horizon=1),
            validation_size=36,
        )
    except (ValueError, TypeError) as exc:
        assert str(exc), "the refusal must say what conflicts"
        return

    effective = getattr(spec.val, "size", None)
    flat = getattr(spec, "validation_size", None)
    assert effective == flat, (
        f"WindowSpec accepted two disagreeing validation sizes and silently kept "
        f"val.size={effective} while validation_size={flat}. A caller reading either "
        f"field gets a different answer about the same window."
    )


def test_horizon_is_not_stored_in_two_places_that_can_disagree(panel):
    """`WindowSpec.horizon` and `WindowSpec.test.horizon` are the same quantity."""
    idx, _ = panel
    window = _window(idx, horizon=3)
    flat = getattr(window, "horizon", None)
    nested = getattr(getattr(window, "test", None), "horizon", None)
    if flat is None or nested is None:
        pytest.skip("one of the two horizon fields no longer exists — the duplication is gone")
    assert int(flat) == int(nested), (
        f"the window reports horizon={flat} at the top level and {nested} under .test; "
        f"downstream code reads both"
    )


# --------------------------------------------------------------------------- #
# 4. Identity parity
# --------------------------------------------------------------------------- #

def test_the_cache_echo_describes_the_task_that_actually_ran(panel):
    """`cell_echo` is what a cached result claims about itself.

    It is built by `result_store` from the spec, while the run is executed by
    `forecasting`. If they ever disagree about target, transform, policy or horizon,
    a cache hit returns a result for a different question than the one asked.
    """
    from macroforecast.pipeline.result_store import result_cell_identity as cell_identity

    idx, bundle = panel
    spec = _spec(idx, bundle)
    identity = cell_identity(spec, spec.arms[0], spec.targets[0], horizon=1, data_identity={"fingerprint": "fp"})
    echo = identity.cell_echo
    assert echo is not None, f"no echo produced: {identity.reason}"

    assert echo["target"]["name"] == "y"
    assert echo["target"]["transform"] == "level"
    assert int(echo["horizon"]) == 1

    result = mf.forecasting.run(
        bundle,
        target="y",
        horizon=1,
        model="ols",
        features=_features(),
        window=_window(idx, 1),
        target_transform="level",
    )
    ran_on = set(result.forecasts["target"].unique()) if "target" in result.forecasts else set()
    if ran_on:
        assert echo["target"]["name"] in {str(t).split("__")[-1] for t in ran_on} | ran_on, (
            f"the cache echo says target={echo['target']['name']!r} but the run produced "
            f"{ran_on}"
        )
