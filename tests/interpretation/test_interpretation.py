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
    assert coef.attrs["macroforecast_metadata_schema"]["reference"]["alignment"] == "direct_attribute_read"
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
    ice = mf.interpretation.individual_conditional_expectation(
        fit,
        X.iloc[:4],
        features="x1",
        grid_size=5,
        center=True,
    )
    ice_alias = mf.interpretation.ice_curves(fit, X.iloc[:4], features="x1", grid_size=5)
    ale = mf.interpretation.accumulated_local_effect(fit, X, feature="x1", bins=5)

    assert set(perm["feature"]) == {"x1", "x2"}
    assert perm.attrs["macroforecast_metadata_schema"]["kind"] == "permutation_importance"
    assert len(pdp) == 5
    assert set(pdp.columns) == {"feature", "value", "prediction"}
    assert pdp.attrs["macroforecast_metadata_schema"]["kind"] == "partial_dependence"
    assert len(ice) == 20
    assert set(ice.columns) == {
        "feature",
        "row",
        "index",
        "value",
        "prediction",
        "centered_prediction",
    }
    assert ice.attrs["macroforecast_metadata_schema"]["kind"] == "individual_conditional_expectation"
    assert ice.attrs["macroforecast_metadata_schema"]["reference"]["reference"].endswith("ice=TRUE")
    assert ice.loc[ice["value"] == ice["value"].min(), "centered_prediction"].abs().max() == 0.0
    assert ice_alias.attrs["macroforecast_metadata_schema"]["method"] == "ice_curves_alias"
    assert len(ale) == 5
    assert set(ale.columns) == {"feature", "bin", "center", "ale", "local_effect"}
    assert ale.attrs["macroforecast_metadata_schema"]["kind"] == "accumulated_local_effect"


def test_interpretation_helpers_are_namespace_scoped() -> None:
    assert hasattr(mf.interpretation, "ice_curves")
    assert hasattr(mf.interpretation, "partial_dependence")
    assert not hasattr(mf, "ice_curves")
    assert not hasattr(mf, "partial_dependence")


def test_legacy_importance_gap_callables() -> None:
    X, y = _xy()
    fit = mf.models.ridge(X, y, alpha=0.1)

    conditional = mf.interpretation.permutation_importance_strobl(
        fit,
        X,
        y,
        n_repeats=2,
        n_bins=3,
        random_state=0,
    )
    lofo = mf.interpretation.lofo_importance(fit, X, y)
    hstat = mf.interpretation.friedman_h_interaction(fit, X, grid_size=4)
    decomposition = mf.interpretation.forecast_decomposition(fit, X)
    cumulative = mf.interpretation.cumulative_r2_contribution(fit, X, y)
    inclusion = mf.interpretation.lasso_inclusion_frequency(fit)

    assert conditional.attrs["macroforecast_metadata_schema"]["kind"] == "permutation_importance_strobl"
    assert conditional.attrs["macroforecast_metadata_schema"]["reference"]["class"] == "approximation"
    assert set(conditional["feature"]) == {"x1", "x2"}
    assert lofo.attrs["macroforecast_metadata_schema"]["kind"] == "lofo_importance"
    assert set(hstat.columns) >= {"feature_1", "feature_2", "h_statistic"}
    assert decomposition.attrs["macroforecast_metadata_schema"]["kind"] == "forecast_decomposition"
    assert "__intercept__" in set(decomposition["feature"])
    assert cumulative.attrs["macroforecast_metadata_schema"]["kind"] == "cumulative_r2_contribution"
    assert inclusion.attrs["macroforecast_metadata_schema"]["kind"] == "lasso_inclusion_frequency"


def test_attribution_aggregation_and_attention_helpers() -> None:
    importance = pd.DataFrame(
        {
            "feature": ["real_lag1", "real_lag2", "price_lag1"],
            "importance": [1.0, 0.5, 0.25],
        }
    )
    grouped = mf.interpretation.group_aggregate(
        importance,
        groups={"real_activity": ["real_lag1", "real_lag2"], "prices": ["price_lag1"]},
    )
    lineage = mf.interpretation.lineage_attribution(
        importance,
        {
            "real_lag1": {"pipeline_name": "lags"},
            "real_lag2": {"pipeline_name": "lags"},
            "price_lag1": {"pipeline_name": "prices"},
        },
    )
    evaluation = pd.DataFrame(
        {
            "model": ["raw", "processed", "raw", "processed"],
            "target": ["y", "y", "y", "y"],
            "horizon": [1, 1, 2, 2],
            "mse": [2.0, 1.0, 3.0, 2.0],
        }
    )
    transform = mf.interpretation.transformation_attribution(evaluation)
    X, y = _xy()
    weights = mf.interpretation.attention_weights(X.iloc[:8], X.iloc[8:10])
    dual = mf.interpretation.dual_decomposition(X.iloc[:8], y.iloc[:8], X.iloc[8:10])

    assert grouped.attrs["macroforecast_metadata_schema"]["kind"] == "group_aggregate"
    assert grouped.loc[grouped["group"] == "real_activity", "importance"].iloc[0] == 1.5
    assert lineage.attrs["macroforecast_metadata_schema"]["kind"] == "lineage_attribution"
    assert "pipeline_name" in lineage.columns
    assert transform.attrs["macroforecast_metadata_schema"]["kind"] == "transformation_attribution"
    assert set(transform["pipeline"]) == {"raw", "processed"}
    assert weights.attrs["macroforecast_metadata_schema"]["kind"] == "attention_weights"
    assert len(weights) == 16
    assert dual.attrs["macroforecast_metadata_schema"]["kind"] == "dual_decomposition"
    assert "contribution" in dual.columns


def test_transformation_attribution_uses_loss_improvement_scale() -> None:
    evaluation = pd.DataFrame(
        {
            "model": ["worst", "middle", "best"],
            "mse": [3.0, 2.0, 1.0],
        }
    )

    out = mf.interpretation.transformation_attribution(evaluation)

    schema = out.attrs["macroforecast_metadata_schema"]
    assert schema["metadata"]["scale"] == "utility_improvement_from_baseline"
    assert schema["metadata"]["component_decomposition"] is False
    assert out.loc[out["pipeline"] == "best", "contribution"].iloc[0] > 0.0
    assert out.loc[out["pipeline"] == "worst", "contribution"].iloc[0] < 0.0
    assert np.isclose(out["contribution"].sum(), 1.0)


def test_tree_forecast_decomposition_fallback_is_marked_non_additive() -> None:
    X, y = _xy()
    tree = mf.models.random_forest(X, y, n_estimators=10, random_state=0)

    out = mf.interpretation.forecast_decomposition(tree, X)

    schema = out.attrs["macroforecast_metadata_schema"]
    assert schema["kind"] == "forecast_decomposition"
    assert schema["method"] == "tree_importance_fallback_not_additive"
    assert schema["metadata"]["prediction_additivity"] is False
    assert set(out["status"]) == {"tree_importance_fallback_not_additive"}
    assert out["contribution"].isna().all()


def test_attention_weights_do_not_penalize_intercept() -> None:
    X_train = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    X_test = pd.DataFrame({"x": [10.0]})

    weights = mf.interpretation.attention_weights(
        X_train,
        X_test,
        add_intercept=True,
        ridge=10.0,
    )

    schema = weights.attrs["macroforecast_metadata_schema"]
    assert schema["metadata"]["intercept_penalized"] is False
    assert np.isclose(weights["weight"].sum(), 1.0)


def test_temporal_and_mrf_interpretation_helpers() -> None:
    X, y = _xy()
    fit = mf.models.ridge(X, y, alpha=0.1)

    rolling = mf.interpretation.rolling_recompute(
        fit,
        X,
        y,
        window=10,
        step=10,
        n_repeats=1,
        random_state=0,
    )
    boot = mf.interpretation.bootstrap_jackknife(
        fit,
        X,
        y,
        n_replications=3,
        random_state=0,
    )
    estimator = types.SimpleNamespace(
        output_={
            "betas": np.array([[1.0, 0.2, -0.1], [1.1, 0.3, -0.2]]),
            "YandX": pd.DataFrame(columns=["y", "x1", "x2"]),
        },
        x_columns=("x1", "x2"),
    )
    mrf_fit = mf.models.ModelFit(
        estimator=estimator,
        model="macro_random_forest",
        feature_names=("x1", "x2"),
    )
    gtvp = mf.interpretation.mrf_gtvp(mrf_fit, X.iloc[:2])

    assert rolling.attrs["macroforecast_metadata_schema"]["kind"] == "rolling_recompute"
    assert set(rolling["window_id"]) == {0, 1, 2, 3}
    assert boot.attrs["macroforecast_metadata_schema"]["kind"] == "bootstrap_jackknife"
    assert set(boot["feature"]) == {"x1", "x2"}
    assert gtvp.attrs["macroforecast_metadata_schema"]["kind"] == "mrf_gtvp"
    assert set(gtvp["feature"]) == {"__intercept__", "x1", "x2"}
    assert "summary" in gtvp.attrs


def test_gradient_helpers_report_missing_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    X, y = _xy()
    fit = mf.models.ridge(X, y, alpha=0.1)

    def missing_import(name: str):
        if name == "torch":
            raise ImportError("missing torch")
        return __import__(name)

    monkeypatch.setattr("macroforecast.interpretation.core.import_module", missing_import)

    with pytest.raises(ImportError, match="macroforecast\\[deep\\]"):
        mf.interpretation.saliency_map(fit, X.iloc[:3])


def test_var_interpretation_callables() -> None:
    idx = pd.date_range("2020-01-31", periods=80, freq="ME")
    base = np.sin(np.arange(len(idx)) / 5.0)
    panel = pd.DataFrame(
        {
            "y": base + np.linspace(0.0, 0.2, len(idx)),
            "x": np.roll(base, 1),
            "z": np.cos(np.arange(len(idx)) / 7.0),
        },
        index=idx,
    ).iloc[2:]
    fit = mf.models.var(panel, target="y", n_lag=1)

    girf = mf.interpretation.generalized_irf(fit, n_periods=4, target="y")
    orth = mf.interpretation.orthogonalised_irf(fit, n_periods=4, target="y")
    variance = mf.interpretation.fevd(fit, n_periods=4, target="y")
    history = mf.interpretation.historical_decomposition(fit, max_lag=4, target="y")

    assert girf.attrs["macroforecast_metadata_schema"]["kind"] == "generalized_irf"
    assert orth.attrs["macroforecast_metadata_schema"]["kind"] == "orthogonalised_irf"
    assert variance.attrs["macroforecast_metadata_schema"]["kind"] == "fevd"
    assert variance.attrs["macroforecast_metadata_schema"]["reference"]["reference"] == "statsmodels VARResults.fevd"
    assert history.attrs["macroforecast_metadata_schema"]["kind"] == "historical_decomposition"
    assert set(girf["feature"]) == {"y", "x", "z"}
    assert np.isfinite(girf["importance"]).all()


def test_custom_interpretation_wraps_user_callable() -> None:
    X, y = _xy()
    fit = mf.models.ridge(X, y, alpha=0.1)

    def mean_prediction(model, X, *, y=None, metadata=None, scale=1.0):
        return {
            "mean_prediction": float(model.predict(X).mean() * scale),
            "target_supplied": y is not None,
            "metadata_keys": len(metadata or {}),
        }

    out = mf.interpretation.custom_interpretation(
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
    importance = mf.interpretation.shap_importance(fit, X.iloc[:3], background=X.iloc[:5])

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
    assert importance.attrs["macroforecast_metadata_schema"]["kind"] == "shap_importance"
    assert set(importance["importance"]) == {0.25}


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
