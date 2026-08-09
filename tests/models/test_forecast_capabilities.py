"""A2 — forecast capability lives on the model, and the guards read it.

Before this, a model's compatibility with a forecast policy was written out in three
literal sets across two layers: `DIRECT_POLICY_GUARD_MODELS` and
`DIRECT_AVERAGE_GUARD_MODELS` in `pipeline/spec.py`, and
`_TARGET_LAGS_BY_DESIGN_MODELS` in `forecasting/policy_config.py`. Adding a model meant
editing three files, and a drift test was what kept them agreeing.

These tests pin the property that replaced that arrangement: capability is a fact about
the model, and every consumer derives from it rather than repeating it.
"""
from __future__ import annotations

import pytest

import macroforecast.forecasting.policy_config as policy_config
import macroforecast.pipeline.spec as pipeline_spec_mod
from macroforecast.models import MODEL_SPECS
from macroforecast.models.specs import ForecastCapabilities, forecast_capabilities


def test_every_registered_model_has_a_capability():
    """No model may be silently absent from the contract."""
    missing = [name for name, spec in MODEL_SPECS.items() if forecast_capabilities(spec) is None]
    assert not missing, missing


def test_the_default_follows_input_kind():
    """A model that iterates its own dynamics cannot honour a direct h-step projection."""
    assert ForecastCapabilities.for_input_kind("target").direct is False
    assert ForecastCapabilities.for_input_kind("panel").direct is False
    assert ForecastCapabilities.for_input_kind("supervised").direct is True
    assert ForecastCapabilities.for_input_kind("volatility").direct is True


@pytest.mark.parametrize(
    "name, attr, expected, why",
    [
        ("hist_mean", "direct", True,
         "a constant-mean benchmark has no dynamics to iterate, so a direct h-step "
         "projection is exactly what it does -- input_kind alone would guard it"),
        ("var", "direct", True, "valid as a direct point forecast"),
        ("var", "direct_average", False, "not valid as a horizon-average target"),
        ("favar", "direct", False,
         "input_kind supervised, but genuinely iterated internally unlike ar/far"),
        ("ar", "builds_own_target_lags", True, "documented usage supplies target_lags"),
        ("far", "builds_own_target_lags", True, "documented usage supplies target_lags"),
    ],
)
def test_the_three_exceptions_are_explicit(name, attr, expected, why):
    """The overrides are the whole point: each has a reason, not a convention.

    If `input_kind` alone were enough there would be no override table, and if the
    overrides were silent a reader could not tell a deliberate carve-out from an
    oversight.
    """
    if name not in MODEL_SPECS:
        pytest.skip(f"{name} is not registered in this build")
    assert getattr(forecast_capabilities(MODEL_SPECS[name]), attr) is expected, why


def test_the_guards_are_derived_not_repeated():
    """Every consumer reads the capability; none keeps its own list.

    This is the regression that matters. If someone re-adds a literal set, it will
    drift from the capability the moment a model changes, and this catches it while the
    two still agree rather than after they stop.
    """
    derived_direct = {
        name for name, spec in MODEL_SPECS.items() if not forecast_capabilities(spec).direct
    }
    assert pipeline_spec_mod.DIRECT_POLICY_GUARD_MODELS == derived_direct

    derived_avg = {
        name
        for name, spec in MODEL_SPECS.items()
        if forecast_capabilities(spec).direct and not forecast_capabilities(spec).direct_average
    }
    assert pipeline_spec_mod.DIRECT_AVERAGE_GUARD_MODELS == derived_avg

    derived_lags = {
        name
        for name, spec in MODEL_SPECS.items()
        if forecast_capabilities(spec).builds_own_target_lags
    }
    assert policy_config._target_lags_by_design_models() == derived_lags


def test_a_new_model_needs_no_edit_outside_its_own_registration():
    """The property A2 exists to create.

    A supervised model registered today is direct-capable and not guarded, without any
    edit to `pipeline/spec.py` or `forecasting/policy_config.py`. Before A2 the guard
    sets were literals, so this could only be true by someone remembering.
    """
    import macroforecast as mf

    def _fit(X, y=None, **kwargs):  # pragma: no cover - never called
        raise AssertionError("capability resolution must not fit the model")

    spec = mf.models.custom_model("a2_probe_supervised", _fit, input_kind="supervised")
    cap = forecast_capabilities(spec)
    assert cap.direct is True
    assert cap.direct_average is True
    assert cap.builds_own_target_lags is False, (
        "a fresh supervised model must not be treated as supplying its own target "
        "lags; that carve-out belongs to ar/far and is opt-in"
    )
