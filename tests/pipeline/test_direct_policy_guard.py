"""WP5: direct-policy guard for iterated/state-space models.

``ar``/``far`` had a CRITICAL stale-persistence defect under the
``direct``/``direct_average`` forecast policy (see CHANGELOG [Unreleased], GCLS
replication Bug 3), fixed by giving them a true direct-projection mode. The other
iterated/state-space models -- target-kind statsmodels forecasters, panel VAR/DFM
models, and ``favar`` -- still forecast a horizon by ITERATING their own dynamics,
so the same defect remains latent for them under direct-like policies. Rather than
reject the combination (deliberate use, e.g. an intentionally weak benchmark,
stays possible), ``pipeline_spec`` emits a ``UserWarning`` at spec-build time.

These tests pin: (1) the warning fires for a representative target-kind and
panel-kind guarded model under ``direct``/``direct_average``; (2) it does NOT fire
for ``ar``/``far`` (excluded -- they have the real fix) or for a genuine
supervised model, nor for ``recursive``/``path_average`` policies; (3) the guarded
model set is derived from ``macroforecast.list_model_specs()`` so it cannot
silently rot as the models lane adds or removes models; (4) the warning has ZERO
effect on the computed forecasts.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.pipeline import Arm, EvalSpec, TargetSpec, pipeline_spec, run_pipeline
from macroforecast.pipeline.spec import DIRECT_POLICY_GUARD_MODELS


def _toy_inputs():
    idx = pd.date_range("1990-01-01", periods=80, freq="MS")
    rng = np.random.default_rng(3)
    panel = pd.DataFrame(
        {"Y": np.cumsum(rng.normal(size=80)), "X1": rng.normal(size=80)},
        index=idx,
    )
    panel.index.name = "date"
    bundle = mf.data.custom_dataset(panel, transform_codes={"Y": 1, "X1": 1})
    win = mf.window.from_cutoffs(
        test_start="1996-01-01",
        test_end="1996-06-01",
        mode="expanding",
        val_method="last_block",
        retrain_every=1,
    )
    return bundle, win


def _spec(model_name, policy, *, arm_name=None):
    bundle, win = _toy_inputs()
    return pipeline_spec(
        data=bundle,
        targets=[TargetSpec(name="Y", policy=policy, transform="level")],
        horizons=[1],
        window=win,
        arms=[Arm(name=arm_name or model_name.upper(), model=model_name, is_benchmark=True)],
        evaluation=EvalSpec(benchmark=arm_name or model_name.upper(), metrics=("rmse",)),
    )


def test_guard_set_matches_model_specs():
    """The hardcoded guard set must equal (target-kind | panel-kind) + favar,
    exactly the formula the WP5 spec calls for -- so it cannot silently rot as
    the models lane adds/removes models without updating this set too.
    """
    df = mf.list_model_specs()
    expected = set(df.loc[df["input_kind"].isin(["target", "panel"]), "name"]) | {"favar"}
    assert DIRECT_POLICY_GUARD_MODELS == expected
    # ar/far deliberately excluded even though they share favar's input_kind.
    assert "ar" not in DIRECT_POLICY_GUARD_MODELS
    assert "far" not in DIRECT_POLICY_GUARD_MODELS


@pytest.mark.parametrize("model_name", ["arima", "ets", "theta_method"])
def test_warns_for_target_kind_model_under_direct(model_name):
    with pytest.warns(UserWarning, match=model_name):
        _spec(model_name, "direct")


@pytest.mark.parametrize("model_name", ["var", "bvar_minnesota", "dfm_unrestricted_midas"])
def test_warns_for_panel_kind_model_under_direct_average(model_name):
    with pytest.warns(UserWarning, match=model_name):
        _spec(model_name, "direct_average")


def test_warns_for_favar_under_direct():
    with pytest.warns(UserWarning, match="favar"):
        _spec("favar", "direct")


@pytest.mark.parametrize("model_name", ["ar", "far"])
@pytest.mark.parametrize("policy", ["direct", "direct_average"])
def test_no_warning_for_ar_far_direct_projection_models(model_name, policy):
    """ar/far are EXCLUDED: they now have a validated direct-projection mode."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _spec(model_name, policy)
    guard_hits = [w for w in caught if "iterat" in str(w.message)]
    assert not guard_hits, [str(w.message) for w in guard_hits]


def test_no_warning_for_supervised_model_under_direct():
    """A genuine direct-projection (feature-matrix) model must never trigger this
    guard -- it is not in the iterated-dynamics set at all.
    """
    bundle, win = _toy_inputs()
    feats = mf.feature_engineering.feature_spec(target="Y", predictors=["X1"], lags=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline_spec(
            data=bundle,
            targets=[TargetSpec(name="Y", policy="direct", transform="level")],
            horizons=[1],
            window=win,
            arms=[Arm(name="RIDGE", model="ridge", features=feats, is_benchmark=True)],
            evaluation=EvalSpec(benchmark="RIDGE", metrics=("rmse",)),
        )
    guard_hits = [w for w in caught if "iterat" in str(w.message)]
    assert not guard_hits


@pytest.mark.parametrize("policy", ["recursive", "path_average"])
def test_no_warning_for_non_direct_like_policies(policy):
    """The guard is specific to direct/direct_average; recursive and path_average
    already iterate correctly and are the RECOMMENDED alternative this warning
    points users toward, so they must never trigger it themselves.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _spec("arima", policy)
    guard_hits = [w for w in caught if "iterat" in str(w.message)]
    assert not guard_hits, [str(w.message) for w in guard_hits]


def test_guard_does_not_change_forecasts():
    """The guard is a pure ``warnings.warn`` at spec-build time: it must not move
    a single forecast number. Pin by running the identical spec twice -- once
    with warnings surfaced normally, once with them filtered out -- and requiring
    byte-identical output.
    """
    spec_warn = _spec("arima", "direct")
    with pytest.warns(UserWarning):
        with_warning = run_pipeline(spec_warn).forecasts

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec_quiet = _spec("arima", "direct")
    quiet = run_pipeline(spec_quiet).forecasts

    assert not with_warning.empty
    assert not quiet.empty
    cols = ["horizon", "date", "prediction"]
    pd.testing.assert_frame_equal(
        with_warning[cols].reset_index(drop=True),
        quiet[cols].reset_index(drop=True),
        atol=1e-12,
    )
