"""Construction-time validation for custom ModelSpec objects."""

from __future__ import annotations

import pytest

import macroforecast as mf


def _fit(X, y):
    return None


def test_custom_model_validates_input_kind():
    with pytest.raises(ValueError, match="input_kind"):
        mf.models.custom_model("bad_kind", _fit, input_kind="matrix")  # type: ignore[arg-type]


def test_custom_model_requires_callable_and_usable_signature():
    with pytest.raises(TypeError, match="callable"):
        mf.models.custom_model("not_callable", object())  # type: ignore[arg-type]

    def no_args():
        return None

    with pytest.raises(TypeError, match="at least 2"):
        mf.models.custom_model("no_args", no_args)


def test_custom_model_validates_default_preset_against_search_spaces():
    with pytest.raises(ValueError, match="default_preset 'large' is not in search_spaces"):
        mf.models.custom_model(
            "bad_preset",
            _fit,
            default_preset="large",
            search_spaces={"small": {"alpha": (0.1, 1.0)}},
        )

    with pytest.raises(ValueError, match="default_preset=None is ambiguous"):
        mf.models.custom_model(
            "none_preset",
            _fit,
            default_preset=None,
            search_spaces={"small": {"alpha": (0.1, 1.0)}},
        )


def test_custom_model_default_preset_none_without_search_spaces_uses_standard():
    spec = mf.models.custom_model("plain", _fit, default_preset=None)
    assert spec.default_preset == "standard"
    assert spec.preset == "standard"


def test_custom_model_volatility_signature_allows_keyword_exog():
    def fit_volatility(y, *, X=None):
        return object()

    spec = mf.models.custom_model(
        "vol_custom",
        fit_volatility,
        input_kind="volatility",
    )

    assert spec.input_kind == "volatility"


def test_custom_model_mf_digest_stamps_fit_func():
    if hasattr(_fit, "__mf_digest__"):
        delattr(_fit, "__mf_digest__")

    spec = mf.models.custom_model("digestible", _fit, mf_digest="digest-v1")

    assert spec.fit_func is _fit
    assert getattr(_fit, "__mf_digest__") == "digest-v1"
