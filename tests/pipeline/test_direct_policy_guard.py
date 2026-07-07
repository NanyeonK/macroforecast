"""WP5: direct-policy guard for iterated/state-space models.

``ar``/``far``/``var`` had a CRITICAL stale-persistence defect under direct-like
forecast policies (see CHANGELOG [Unreleased], GCLS replication Bug 3). ``ar``
and ``far`` support true direct/direct_average projections; ``var`` supports a
true direct POINT projection only. The other iterated/state-space models --
target-kind statsmodels forecasters, panel BVAR/DFM models, and ``favar`` --
still forecast a horizon by ITERATING their own dynamics, so the same defect
remains latent for them under direct-like policies.
``pipeline_spec`` now rejects unsupported combinations by default, with explicit
``warn`` and ``reroute`` opt-outs.

These tests pin: (1) default ``error`` rejects; (2) ``warn`` preserves the old
warning-only behavior; (3) ``reroute`` emits rows labeled ``recursive``; (4) the
guard does NOT fire for ``ar``/``far``/``var`` (excluded -- they have the real
fix) or for a genuine supervised model, nor for ``recursive``/``path_average``
policies; (5) the guarded model set is derived from
``macroforecast.list_model_specs()`` so it cannot silently rot as the models
lane adds or removes models.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.pipeline import Arm, EvalSpec, TargetSpec, pipeline_spec, run_pipeline
from macroforecast.pipeline.spec import (
    DIRECT_AVERAGE_GUARD_MODELS,
    DIRECT_POLICY_GUARD_MODELS,
)


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


def _spec(model_name, policy, *, arm_name=None, on_unsupported_direct="error"):
    bundle, win = _toy_inputs()
    return pipeline_spec(
        data=bundle,
        targets=[TargetSpec(name="Y", policy=policy, transform="level")],
        horizons=[1],
        window=win,
        arms=[Arm(name=arm_name or model_name.upper(), model=model_name, is_benchmark=True)],
        evaluation=EvalSpec(benchmark=arm_name or model_name.upper(), metrics=("rmse",)),
        on_unsupported_direct=on_unsupported_direct,
    )


def test_guard_set_matches_model_specs():
    """The hardcoded guard set must equal target/panel models without direct
    projection support, plus favar, so it cannot silently rot as models move.
    """
    df = mf.list_model_specs()
    expected = {
        str(row["name"])
        for _, row in df.iterrows()
        if row["input_kind"] in {"target", "panel"}
        and "direct" not in mf.get_model(str(row["name"])).default_params
    } | {"favar"}
    assert DIRECT_POLICY_GUARD_MODELS == expected
    # ar/far deliberately excluded even though they share favar's input_kind;
    # var deliberately excluded from the plain-direct panel bucket after issue #442.
    assert "ar" not in DIRECT_POLICY_GUARD_MODELS
    assert "far" not in DIRECT_POLICY_GUARD_MODELS
    assert "var" not in DIRECT_POLICY_GUARD_MODELS
    assert DIRECT_AVERAGE_GUARD_MODELS == frozenset({"var"})


def test_default_errors_for_guarded_model_under_direct():
    with pytest.raises(ValueError, match="on_unsupported_direct='warn'"):
        _spec("arima", "direct")


def test_default_errors_for_var_under_direct_average():
    with pytest.raises(ValueError, match="horizon-average target"):
        _spec("var", "direct_average")


@pytest.mark.parametrize("model_name", ["arima", "ets", "theta_method"])
def test_warn_mode_warns_for_target_kind_model_under_direct(model_name):
    with pytest.warns(UserWarning, match=model_name):
        _spec(model_name, "direct", on_unsupported_direct="warn")


@pytest.mark.parametrize("model_name", ["bvar_minnesota", "dfm_unrestricted_midas"])
def test_warn_mode_warns_for_panel_kind_model_under_direct_average(model_name):
    with pytest.warns(UserWarning, match=model_name):
        _spec(model_name, "direct_average", on_unsupported_direct="warn")


def test_warns_for_favar_under_direct():
    with pytest.warns(UserWarning, match="favar"):
        _spec("favar", "direct", on_unsupported_direct="warn")


def test_warns_for_var_under_direct_average():
    with pytest.warns(UserWarning, match="horizon-average target"):
        spec = _spec("var", "direct_average", on_unsupported_direct="warn")
    assert spec.policy_overrides == {}


@pytest.mark.parametrize("model_name", ["ar", "far"])
@pytest.mark.parametrize("policy", ["direct", "direct_average"])
def test_no_warning_for_direct_projection_models(model_name, policy):
    """ar/far are EXCLUDED: they have validated direct-projection modes."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _spec(model_name, policy)
    guard_hits = [w for w in caught if "iterat" in str(w.message)]
    assert not guard_hits, [str(w.message) for w in guard_hits]


def test_no_warning_for_var_under_direct():
    """var is supported under point-direct forecasts, but not direct_average."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _spec("var", "direct")
    guard_hits = [w for w in caught if "direct_average" in str(w.message)]
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
    """``warn`` mode preserves the old warning-only behavior and does not move a
    single forecast number.
    """
    with pytest.warns(UserWarning):
        spec_warn = _spec("arima", "direct", on_unsupported_direct="warn")
        with_warning = run_pipeline(spec_warn).forecasts

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec_quiet = _spec("arima", "direct", on_unsupported_direct="warn")
    quiet = run_pipeline(spec_quiet).forecasts

    assert not with_warning.empty
    assert not quiet.empty
    cols = ["horizon", "date", "prediction"]
    pd.testing.assert_frame_equal(
        with_warning[cols].reset_index(drop=True),
        quiet[cols].reset_index(drop=True),
        atol=1e-12,
    )


def test_reroute_mode_labels_rows_recursive():
    with pytest.warns(UserWarning, match="Rerouting"):
        spec = _spec("naive", "direct", on_unsupported_direct="reroute")
    out = run_pipeline(spec).forecasts
    assert not out.empty
    assert set(out["forecast_policy"]) == {"recursive"}
    assert spec.policy_overrides == {("NAIVE", "Y"): "recursive"}


def test_var_direct_average_reroute_labels_rows_recursive():
    with pytest.warns(UserWarning, match="Rerouting"):
        spec = _spec("var", "direct_average", on_unsupported_direct="reroute")
    out = run_pipeline(spec).forecasts
    assert not out.empty
    assert set(out["forecast_policy"]) == {"recursive"}
    assert spec.policy_overrides == {("VAR", "Y"): "recursive"}
