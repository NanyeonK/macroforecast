from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf


def _xy() -> tuple[pd.DataFrame, pd.Series]:
    idx = pd.date_range("2020-01-31", periods=40, freq="ME")
    x1 = np.linspace(0.0, 1.0, len(idx))
    x2 = np.sin(np.arange(len(idx)) / 3.0)
    X = pd.DataFrame({"x1": x1, "x2": x2}, index=idx)
    y = pd.Series(1.0 + 2.0 * x1 - 0.5 * x2, index=idx, name="y")
    return X, y


def test_linear_coefficients_and_tree_importance() -> None:
    X, y = _xy()
    linear = mf.models.ridge(X, y, alpha=0.1)
    tree = mf.models.random_forest(X, y, n_estimators=10, random_state=0)

    coef = mf.interpretation.linear_coefficients(linear)
    importance = mf.interpretation.tree_importance(tree)

    assert set(coef["feature"]) == {"x1", "x2"}
    assert "coefficient" in coef.columns
    assert coef.attrs["macroforecast_metadata_schema"]["kind"] == "linear_coefficients"
    assert set(importance["feature"]) == {"x1", "x2"}
    assert importance["importance"].sum() > 0.0
    assert importance.attrs["macroforecast_metadata_schema"]["kind"] == "tree_importance"


def test_model_agnostic_interpretation_helpers() -> None:
    X, y = _xy()
    fit = mf.models.ridge(X, y, alpha=0.1)

    perm = mf.interpretation.permutation_importance(
        fit,
        X,
        y,
        n_repeats=2,
        random_state=0,
    )
    pdp = mf.interpretation.partial_dependence(fit, X, features="x1", grid_size=5)
    ale = mf.interpretation.accumulated_local_effect(fit, X, feature="x1", bins=5)

    assert set(perm["feature"]) == {"x1", "x2"}
    assert perm.attrs["macroforecast_metadata_schema"]["kind"] == "permutation_importance"
    assert len(pdp) == 5
    assert set(pdp.columns) == {"feature", "value", "prediction"}
    assert pdp.attrs["macroforecast_metadata_schema"]["kind"] == "partial_dependence"
    assert len(ale) == 5
    assert set(ale.columns) == {"feature", "bin", "center", "ale", "local_effect"}
    assert ale.attrs["macroforecast_metadata_schema"]["kind"] == "accumulated_local_effect"


def test_custom_interpretation_wraps_user_callable() -> None:
    X, y = _xy()
    fit = mf.models.ridge(X, y, alpha=0.1)

    def mean_prediction(model, X, *, y=None, metadata=None, scale=1.0):
        return {
            "mean_prediction": float(model.predict(X).mean() * scale),
            "target_supplied": y is not None,
            "metadata_keys": len(metadata or {}),
        }

    out = mf.custom_interpretation(
        fit,
        X.iloc[:5],
        mean_prediction,
        y=y.iloc[:5],
        name="mean_prediction_check",
        metadata={"sample": "test"},
        scale=2.0,
    )

    schema = out.attrs["macroforecast_metadata_schema"]
    assert schema["kind"] == "custom_interpretation"
    assert schema["method"] == "mean_prediction_check"
    assert schema["metadata"]["params"] == {"scale": 2.0}
    assert bool(out.loc[0, "target_supplied"]) is True
    assert out.loc[0, "metadata_keys"] == 1


def test_shap_values_uses_optional_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    X, y = _xy()
    fit = mf.models.ridge(X, y, alpha=0.1)

    class FakeExplainer:
        def __init__(self, predict_fn, background):
            self.predict_fn = predict_fn
            self.background = background

        def __call__(self, frame, **kwargs):
            values = np.full(frame.shape, 0.25)
            base_values = np.full(len(frame), 1.0)
            return types.SimpleNamespace(values=values, base_values=base_values)

    fake_shap = types.SimpleNamespace(Explainer=FakeExplainer)
    monkeypatch.setitem(sys.modules, "shap", fake_shap)

    out = mf.interpretation.shap_values(fit, X.iloc[:3], background=X.iloc[:5])

    assert len(out) == 3 * X.shape[1]
    assert set(out.columns) == {
        "row",
        "index",
        "feature",
        "feature_value",
        "shap_value",
        "base_value",
    }
    assert set(out["shap_value"]) == {0.25}
    assert out.attrs["macroforecast_metadata_schema"]["kind"] == "shap_values"
    assert out.attrs["macroforecast_metadata_schema"]["metadata"]["background_n_obs"] == 5


def test_shap_values_reports_optional_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    X, y = _xy()
    fit = mf.models.ridge(X, y, alpha=0.1)

    def missing_import(name: str):
        if name == "shap":
            raise ImportError("missing shap")
        return __import__(name)

    monkeypatch.setattr("macroforecast.interpretation.core.import_module", missing_import)

    with pytest.raises(ImportError, match="macroforecast\\[interpretation\\]"):
        mf.interpretation.shap_values(fit, X.iloc[:3])
