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
lane adds or removes models. ``hist_mean`` is the narrow target-kind exception:
it is a constant mean projection of the already transformed target, not an
iterated dynamic forecast.
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

_DIRECT_SAFE_TARGET_EXCEPTIONS = frozenset({"hist_mean"})


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


def _spec(
    model_name,
    policy,
    *,
    arm_name=None,
    on_unsupported_direct="error",
    model_selection=None,
):
    """A one-arm spec for the guard tests.

    ``model_selection`` defaults to ``None``, which is exactly what ``Arm`` already
    defaults to, so every existing caller builds the identical spec it did before. It is
    threaded through for the panel-input arms, which must opt out of the default search
    space explicitly (see ``test_panel_arm_needs_an_explicit_search_opt_out``) before
    they can run at all -- and that is a selection question, not the policy question
    these tests are about.
    """
    bundle, win = _toy_inputs()
    return pipeline_spec(
        data=bundle,
        targets=[TargetSpec(name="Y", policy=policy, transform="level")],
        horizons=[1],
        window=win,
        arms=[
            Arm(
                name=arm_name or model_name.upper(),
                model=model_name,
                is_benchmark=True,
                model_selection=model_selection,
            )
        ],
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
        and str(row["name"]) not in _DIRECT_SAFE_TARGET_EXCEPTIONS
    } | {"favar"}
    assert DIRECT_POLICY_GUARD_MODELS == expected
    # ar/far deliberately excluded even though they share favar's input_kind;
    # var deliberately excluded from the plain-direct panel bucket after issue #442.
    assert "ar" not in DIRECT_POLICY_GUARD_MODELS
    assert "far" not in DIRECT_POLICY_GUARD_MODELS
    assert "var" not in DIRECT_POLICY_GUARD_MODELS
    assert "hist_mean" not in DIRECT_POLICY_GUARD_MODELS
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
    """What the reroute does to the ROWS, with the selection question taken off the table.

    ``var`` reads a panel, and panel-input forecasting refuses an unpinned default search
    space (#571), so the arm must opt out of selection before it produces any rows at
    all. That opt-out is passed here so the assertions below are about the policy
    reroute and nothing else; the selection contract itself is pinned separately by
    ``test_panel_arm_needs_an_explicit_search_opt_out``.
    """
    with pytest.warns(UserWarning, match="Rerouting"):
        spec = _spec(
            "var",
            "direct_average",
            on_unsupported_direct="reroute",
            model_selection={"var": None},
        )
    out = run_pipeline(spec).forecasts
    assert not out.empty
    assert set(out["forecast_policy"]) == {"recursive"}
    assert spec.policy_overrides == {("VAR", "Y"): "recursive"}


def test_panel_arm_needs_an_explicit_search_opt_out():
    """The accepted #571 contract, at the level a user meets it.

    A panel-input arm left on ``model_selection=None`` does not quietly run with some
    default search: the cell fails, and ``run_pipeline`` keeps its fail-open-cell
    behaviour, so the run completes and reports the failure with the exact call that
    fixes it rather than raising out of the pipeline. Passing that opt-out then runs.

    This is what the reroute test above stopped covering once it started opting out, and
    it is pinned by ``run_pipeline`` rather than by the validator directly so that a
    change which turned the message into a silent empty cell would be caught here.
    """
    with pytest.warns(UserWarning, match="Rerouting"):
        implicit = _spec("var", "direct_average", on_unsupported_direct="reroute")
    with pytest.warns(RuntimeWarning, match="does not tune model parameters yet"):
        report = run_pipeline(implicit)

    assert report.forecasts.empty, "the cell failed, so it contributes no rows"
    assert len(report.failed_cells) == 1
    failure = report.failed_cells[0]
    assert failure["target"] == "Y"
    assert failure["arm"] == "VAR"
    assert "does not tune model parameters yet" in failure["error"]
    assert "model_selection={'var': None}" in failure["error"], (
        "the reported error must name the call that fixes it"
    )

    with pytest.warns(UserWarning, match="Rerouting"):
        explicit = _spec(
            "var",
            "direct_average",
            on_unsupported_direct="reroute",
            model_selection={"var": None},
        )
    out = run_pipeline(explicit).forecasts
    assert not out.empty, "the explicit opt-out runs the same arm"
