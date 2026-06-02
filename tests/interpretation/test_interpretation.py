from __future__ import annotations

import importlib.util
import importlib
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


def _has_anatomy() -> bool:
    return importlib.util.find_spec("anatomy") is not None


def test_linear_coefficients_and_tree_importance() -> None:
    X, y = _xy()
    linear = mf.models.ridge(X, y, alpha=0.1)
    tree = mf.models.random_forest(X, y, n_estimators=10, random_state=0)

    coef = mf.interpretation.linear_coefficients(linear)
    importance = mf.interpretation.tree_importance(tree)

    assert set(coef["feature"]) == {"x1", "x2"}
    assert "coefficient" in coef.columns
    assert coef.attrs["macroforecast_metadata_schema"]["kind"] == "linear_coefficients"
    assert (
        coef.attrs["macroforecast_metadata_schema"]["reference"]["alignment"]
        == "direct_attribute_read"
    )
    assert set(importance["feature"]) == {"x1", "x2"}
    assert importance["importance"].sum() > 0.0
    assert (
        importance.attrs["macroforecast_metadata_schema"]["kind"] == "tree_importance"
    )


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
    ice_alias = mf.interpretation.ice_curves(
        fit, X.iloc[:4], features="x1", grid_size=5
    )
    ale = mf.interpretation.accumulated_local_effect(fit, X, feature="x1", bins=5)

    assert set(perm["feature"]) == {"x1", "x2"}
    assert (
        perm.attrs["macroforecast_metadata_schema"]["kind"] == "permutation_importance"
    )
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
    assert (
        ice.attrs["macroforecast_metadata_schema"]["kind"]
        == "individual_conditional_expectation"
    )
    assert ice.attrs["macroforecast_metadata_schema"]["reference"][
        "reference"
    ].endswith("ice=TRUE")
    assert (
        ice.loc[ice["value"] == ice["value"].min(), "centered_prediction"].abs().max()
        == 0.0
    )
    assert (
        ice_alias.attrs["macroforecast_metadata_schema"]["method"] == "ice_curves_alias"
    )
    assert len(ale) == 5
    assert set(ale.columns) == {"feature", "bin", "center", "ale", "local_effect"}
    assert (
        ale.attrs["macroforecast_metadata_schema"]["kind"] == "accumulated_local_effect"
    )


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

    assert (
        conditional.attrs["macroforecast_metadata_schema"]["kind"]
        == "permutation_importance_strobl"
    )
    assert (
        conditional.attrs["macroforecast_metadata_schema"]["reference"]["class"]
        == "approximation"
    )
    assert set(conditional["feature"]) == {"x1", "x2"}
    assert lofo.attrs["macroforecast_metadata_schema"]["kind"] == "lofo_importance"
    assert set(hstat.columns) >= {"feature_1", "feature_2", "h_statistic"}
    assert (
        decomposition.attrs["macroforecast_metadata_schema"]["kind"]
        == "forecast_decomposition"
    )
    assert "__intercept__" in set(decomposition["feature"])
    assert (
        cumulative.attrs["macroforecast_metadata_schema"]["kind"]
        == "cumulative_r2_contribution"
    )
    assert (
        inclusion.attrs["macroforecast_metadata_schema"]["kind"]
        == "lasso_inclusion_frequency"
    )


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
    assert (
        lineage.attrs["macroforecast_metadata_schema"]["kind"] == "lineage_attribution"
    )
    assert "pipeline_name" in lineage.columns
    assert (
        transform.attrs["macroforecast_metadata_schema"]["kind"]
        == "transformation_attribution"
    )
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


def test_dual_observation_weights_reconstruct_ridge_and_krr_predictions() -> None:
    X_train = pd.DataFrame(
        {"x1": [1.0, 0.0, 1.0, 2.0], "x2": [0.0, 1.0, 1.0, 1.0]},
        index=pd.date_range("2020-01-31", periods=4, freq="ME"),
    )
    X_test = pd.DataFrame(
        {"x1": [0.5, 1.5], "x2": [1.0, 0.25]},
        index=pd.date_range("2020-05-31", periods=2, freq="ME"),
    )
    y = pd.Series([1.0, -0.5, 0.5, 1.5], index=X_train.index)

    ridge_weights = mf.interpretation.observation_weights(
        None,
        X_train,
        X_test,
        method="ridge",
        lambda_=0.2,
        ridge_penalty_scale="none",
    )
    krr_weights = mf.interpretation.observation_weights(
        None,
        X_train,
        X_test,
        method="krr",
        kernel="linear",
        lambda_=0.2,
    )
    ridge_matrix = ridge_weights.attrs["weight_matrix"]
    krr_matrix = krr_weights.attrs["weight_matrix"]
    beta = np.linalg.pinv(X_train.to_numpy().T @ X_train.to_numpy() + 0.2 * np.eye(2))
    beta = beta @ X_train.to_numpy().T @ y.to_numpy()

    assert np.allclose(ridge_matrix, krr_matrix)
    assert np.allclose(ridge_matrix @ y.to_numpy(), X_test.to_numpy() @ beta)
    assert (
        ridge_weights.attrs["macroforecast_metadata_schema"]["kind"]
        == "observation_weights"
    )
    assert (
        ridge_weights.attrs["macroforecast_metadata_schema"]["reference"]["reference"]
        == "Dual Interpretation of Machine Learning Forecasts, Goulet Coulombe, Goebel, and Klieber (2024)"
    )


def test_dual_random_forest_weights_reconstruct_prediction() -> None:
    from sklearn.ensemble import RandomForestRegressor

    X_train = pd.DataFrame({"x": [0.0, 0.1, 1.0, 1.1]})
    y = pd.Series([0.0, 0.0, 1.0, 1.0])
    X_test = pd.DataFrame({"x": [0.05, 1.05]})
    forest = RandomForestRegressor(
        n_estimators=1,
        max_depth=1,
        bootstrap=False,
        random_state=0,
    ).fit(X_train, y)

    weights = mf.interpretation.observation_weights(
        forest,
        X_train,
        X_test,
        method="random_forest",
    )
    matrix = weights.attrs["weight_matrix"]

    assert np.allclose(matrix.sum(axis=1), 1.0)
    assert np.allclose(matrix @ y.to_numpy(), forest.predict(X_test))
    assert set(weights["channel"]) == {"random_forest"}


def test_dual_data_portfolio_contributions_and_diagnostics() -> None:
    weights = pd.DataFrame(
        {
            "test_row": [0, 0, 0, 1, 1, 1],
            "test_index": ["f1", "f1", "f1", "f2", "f2", "f2"],
            "train_row": [0, 1, 2, 0, 1, 2],
            "train_index": ["t1", "t2", "t3", "t1", "t2", "t3"],
            "weight": [0.5, -0.25, 0.75, 0.25, 0.0, 0.75],
        }
    )
    y = pd.Series([1.0, 2.0, 4.0], index=["t1", "t2", "t3"])

    contrib = mf.interpretation.outcome_contributions(weights, y)
    diag = mf.interpretation.data_portfolio_diagnostics(weights, top_q=1 / 3)
    top = mf.interpretation.top_episodes(contrib, n=1, sort_by="abs_weight")
    grouped = mf.interpretation.episode_group_weights(
        contrib,
        {"early": ["t1", "t2"], "late": ["t3"]},
    )

    assert contrib.loc[contrib["test_index"] == "f1", "prediction"].iloc[0] == 3.0
    first = diag.loc[diag["test_index"] == "f1"].iloc[0]
    second = diag.loc[diag["test_index"] == "f2"].iloc[0]
    assert np.isclose(first["concentration"], 0.5)
    assert np.isclose(first["short_position"], -0.25)
    assert np.isclose(first["leverage"], 1.0)
    assert np.isclose(first["gross_leverage"], 1.5)
    assert np.isnan(first["turnover"])
    assert np.isclose(second["turnover"], 0.5)
    assert set(top["train_index"]) == {"t3"}
    late = grouped[
        (grouped["test_index"] == "f1") & (grouped["episode_group"] == "late")
    ].iloc[0]
    assert np.isclose(late["weight"], 0.75)
    assert np.isclose(late["contribution"], 3.0)


def test_dual_namespace_builds_result_tables() -> None:
    X_train = pd.DataFrame(
        {"x1": [1.0, 0.0, 1.0, 2.0], "x2": [0.0, 1.0, 1.0, 1.0]},
        index=pd.date_range("2020-01-31", periods=4, freq="ME"),
    )
    X_test = pd.DataFrame(
        {"x1": [0.5], "x2": [1.0]},
        index=pd.date_range("2020-05-31", periods=1, freq="ME"),
    )
    y = pd.Series([1.0, -0.5, 0.5, 1.5], index=X_train.index)

    result = mf.interpretation.dual.dual_interpretation(
        None,
        X_train,
        y,
        X_test,
        method="ridge",
        lambda_=0.2,
        ridge_penalty_scale="none",
        groups={"early": [X_train.index[0], X_train.index[1]]},
        top_n=2,
    )
    tables = result.to_tables(prefix="demo")

    assert isinstance(result, mf.interpretation.DualInterpretationResult)
    assert result.metadata_schema["kind"] == "dual_interpretation_result"
    assert {
        "demo_observation_weights",
        "demo_observation_contributions",
        "demo_forecast_diagnostics",
        "demo_top_observations",
        "demo_group_observation_weights",
        "demo_metadata",
    } <= set(tables)
    assert (
        tables["demo_forecast_diagnostics"].attrs["macroforecast_metadata_schema"]["kind"]
        == "dual_forecast_diagnostics_table"
    )
    assert mf.interpretation.forecast_diagnostics(result.weights).equals(
        mf.interpretation.dual.forecast_diagnostics(result.weights)
    )


def test_dual_from_forecast_result_attaches_sidecar() -> None:
    X_train = pd.DataFrame(
        {"x1": [1.0, 0.0, 1.0, 2.0], "x2": [0.0, 1.0, 1.0, 1.0]},
        index=pd.date_range("2020-01-31", periods=4, freq="ME"),
    )
    X_test = pd.DataFrame(
        {"x1": [0.5], "x2": [1.0]},
        index=pd.date_range("2020-05-31", periods=1, freq="ME"),
    )
    y = pd.Series([1.0, -0.5, 0.5, 1.5], index=X_train.index)
    forecast_result = mf.forecasting.ForecastResult(
        pd.DataFrame(
            {
                "date": [X_test.index[0]],
                "origin": [X_train.index[-1]],
                "model": ["ridge"],
                "prediction": [0.0],
                "actual": [0.0],
            }
        ),
        metadata={"metadata_schema": {"kind": "forecast_result", "version": 1}},
    )

    attached = mf.interpretation.dual_from_forecast_result(
        forecast_result,
        None,
        X_train,
        y,
        X_test,
        method="ridge",
        lambda_=0.2,
        ridge_penalty_scale="none",
        sidecar_name="dual_view",
    )
    detached = mf.interpretation.dual_from_forecast_result(
        forecast_result,
        None,
        X_train,
        y,
        X_test,
        method="ridge",
        lambda_=0.2,
        ridge_penalty_scale="none",
        attach=False,
    )
    method_attached = forecast_result.with_dual(
        None,
        X_train,
        y,
        X_test,
        method="ridge",
        lambda_=0.2,
        ridge_penalty_scale="none",
    )

    assert attached is not forecast_result
    assert isinstance(detached, mf.interpretation.DualInterpretationResult)
    assert attached.get_sidecar("dual_view").metadata["forecast_result"]["forecast_rows"] == 1
    assert attached.metadata["sidecars"]["dual_view"]["metadata_schema"]["kind"] == (
        "dual_interpretation_result"
    )
    assert method_attached.sidecar_names() == ("dual",)
    assert isinstance(method_attached.get_sidecar("dual"), mf.interpretation.DualInterpretationResult)


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

    monkeypatch.setattr(
        "macroforecast.interpretation.core.import_module", missing_import
    )

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
    assert (
        variance.attrs["macroforecast_metadata_schema"]["reference"]["reference"]
        == "statsmodels VARResults.fevd"
    )
    assert (
        history.attrs["macroforecast_metadata_schema"]["kind"]
        == "historical_decomposition"
    )
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
    importance = mf.interpretation.shap_importance(
        fit, X.iloc[:3], background=X.iloc[:5]
    )

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
    assert (
        out.attrs["macroforecast_metadata_schema"]["metadata"]["background_n_obs"] == 5
    )
    assert (
        importance.attrs["macroforecast_metadata_schema"]["kind"] == "shap_importance"
    )
    assert set(importance["importance"]) == {0.25}


def test_shap_values_reports_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X, y = _xy()
    fit = mf.models.ridge(X, y, alpha=0.1)

    def missing_import(name: str):
        if name == "shap":
            raise ImportError("missing shap")
        return __import__(name)

    monkeypatch.setattr(
        "macroforecast.interpretation.core.import_module", missing_import
    )

    with pytest.raises(ImportError, match="macroforecast\\[interpretation\\]"):
        mf.interpretation.shap_values(fit, X.iloc[:3])


def test_performance_shapley_value_decomposes_point_loss() -> None:
    contributions = pd.DataFrame(
        {
            "row": [0, 0, 1],
            "feature": ["x1", "x2", "x1"],
            "forecast_contribution": [2.0, 1.0, 1.0],
            "base_value": [0.0, 0.0, 0.0],
        }
    )
    y = pd.Series([2.5, 0.0])

    local = mf.interpretation.performance_shapley_value(
        contributions,
        y,
        contribution_col="forecast_contribution",
        return_local=True,
    )
    global_table = mf.interpretation.performance_shapley_value(
        contributions,
        y,
        contribution_col="forecast_contribution",
    )

    schema = local.attrs["macroforecast_metadata_schema"]
    assert schema["kind"] == "performance_shapley_value"
    assert schema["reference"]["class"] == "paper_formula_adapter"
    assert schema["metadata"]["max_efficiency_error"] < 1e-12
    assert (
        local.groupby("row")["pbsv"]
        .sum()
        .equals(local.groupby("row")["full_loss"].first())
    )
    assert (
        local.loc[(local["row"] == 0) & (local["feature"] == "x1"), "pbsv"].iloc[0]
        < 0.0
    )
    assert (
        local.loc[(local["row"] == 1) & (local["feature"] == "x1"), "pbsv"].iloc[0]
        > 0.0
    )
    assert "__base__" in set(global_table["feature"])

    single_row = mf.interpretation.performance_shapley_value(
        pd.DataFrame(
            {
                "feature": ["__intercept__", "x1"],
                "contribution": [1.0, 2.0],
            }
        ),
        [4.0],
    )
    assert (
        single_row.attrs["macroforecast_metadata_schema"]["metadata"]["row_column"]
        == "row"
    )
    assert single_row["n_rows"].iloc[0] == 1


def test_anatomy_explain_wraps_optional_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class FakeModelCombination:
        def __init__(self, groups):
            self.groups = groups

    class FakeOutputTransformer:
        def __init__(self, transform):
            self.transform = transform

    class FakeAnatomyObject:
        def explain(
            self, *, model_sets=None, transformer=None, explanation_subset=None
        ):
            calls["model_sets"] = model_sets
            calls["transformer"] = transformer
            calls["explanation_subset"] = explanation_subset
            index = pd.MultiIndex.from_tuples(
                [("combo", pd.Timestamp("2020-01-31"))],
                names=["model_set", "date"],
            )
            return pd.DataFrame(
                {"base_contribution": [1.0], "x1": [0.25], "x2": [-0.1]},
                index=index,
            )

    class FakeAnatomy:
        @staticmethod
        def load(path):
            calls["path"] = path
            return FakeAnatomyObject()

    class FakeMAS:
        class LossType:
            LOWER_IS_BETTER = "lower"
            LARGER_IS_BETTER = "larger"

        class MASType:
            IMPORTANCE_WEIGHTED = "importance_weighted"
            EQUAL_WEIGHTED = "equal_weighted"

        def __init__(self, is_vi, oos_pbsv, pbsv_loss_type):
            calls["mas_is_vi"] = is_vi
            calls["mas_oos_pbsv"] = oos_pbsv
            calls["mas_loss_type"] = pbsv_loss_type

        def compute(
            self,
            *,
            mas_type=None,
            hypothesis_test=True,
            h0_alpha=0.5,
            n_samples=1000000,
        ):
            calls["mas_type"] = mas_type
            calls["hypothesis_test"] = hypothesis_test
            calls["h0_alpha"] = h0_alpha
            calls["n_samples"] = n_samples
            return {"mas": 0.75, "mas_p_value": 0.125}

    fake_module = types.SimpleNamespace(
        Anatomy=FakeAnatomy,
        AnatomyModelCombination=FakeModelCombination,
        AnatomyModelOutputTransformer=FakeOutputTransformer,
        MAS=FakeMAS,
    )
    monkeypatch.setitem(sys.modules, "anatomy", fake_module)

    out = mf.interpretation.anatomy_explain(
        "precomputed-anatomy.pkl",
        model_groups={"combo": ["model_a", "model_b"]},
        metric="squared_error",
        explanation_subset=[pd.Timestamp("2020-01-31")],
    )

    schema = out.attrs["macroforecast_metadata_schema"]
    assert calls["path"] == "precomputed-anatomy.pkl"
    assert calls["model_sets"].groups == {"combo": ["model_a", "model_b"]}
    assert isinstance(calls["transformer"], FakeOutputTransformer)
    assert schema["kind"] == "anatomy_explain"
    assert schema["metadata"]["backend"] == "anatomy"
    assert set(out["feature"]) == {"base_contribution", "x1", "x2"}
    assert (
        bool(out.loc[out["feature"] == "base_contribution", "is_base"].iloc[0]) is True
    )

    vi = mf.interpretation.oshapley_vi("precomputed-anatomy.pkl")
    loss = mf.interpretation.pbsv("precomputed-anatomy.pkl", loss="rmse")
    generic_vi = mf.interpretation.shapley_variable_importance(out)
    i_vi = mf.interpretation.ishapley_vi(out)
    mas = mf.interpretation.model_accordance_score(
        vi,
        loss,
        hypothesis_test=True,
        n_samples=25,
        random_state=123,
    )

    assert vi.attrs["macroforecast_metadata_schema"]["kind"] == "oshapley_vi"
    assert set(vi["feature"]) == {"x1", "x2"}
    assert loss.attrs["macroforecast_metadata_schema"]["kind"] == "pbsv"
    assert loss.attrs["macroforecast_metadata_schema"]["metadata"]["loss"] == "rmse"
    assert (
        generic_vi.attrs["macroforecast_metadata_schema"]["kind"]
        == "shapley_variable_importance"
    )
    assert i_vi.attrs["macroforecast_metadata_schema"]["kind"] == "ishapley_vi"
    assert (
        i_vi.attrs["macroforecast_metadata_schema"]["metadata"]["vi_scope"]
        == "in_sample"
    )
    assert set(i_vi["feature"]) == {"x1", "x2"}
    assert (
        mas.attrs["macroforecast_metadata_schema"]["kind"] == "model_accordance_score"
    )
    assert mas.loc[0, "mas"] == 0.75
    assert calls["mas_loss_type"] == "lower"
    assert calls["mas_type"] == "importance_weighted"
    assert calls["n_samples"] == 25


def test_anatomy_from_forecast_result_attaches_pipeline_sidecar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anatomy_module = importlib.import_module("macroforecast.interpretation.anatomy")
    X, y = _xy()
    window = mf.window.from_cutoffs(
        test_start=X.index[20],
        test_end=X.index[22],
        estimation_min_size=12,
        horizon=1,
        step=1,
    )
    forecast_result = mf.forecasting.ForecastResult(
        pd.DataFrame(
            {
                "date": [X.index[20]],
                "origin": [X.index[19]],
                "model": ["ols"],
                "prediction": [0.0],
                "actual": [0.0],
            }
        ),
        metadata={"metadata_schema": {"kind": "forecast_result", "version": 1}},
    )
    fake_pipeline = mf.interpretation.AnatomyPipelineResult(
        anatomy=None,
        explanations={"forecast": pd.DataFrame({"feature": ["x1"], "contribution": [1.0]})},
        variable_importance=pd.DataFrame({"feature": ["x1"], "importance": [1.0]}),
        performance_values={"rmse": pd.DataFrame({"feature": ["x1"], "contribution": [-0.1]})},
        metadata={"kind": "anatomy_pipeline"},
    )
    calls: dict[str, object] = {}

    def fake_anatomy_pipeline(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return fake_pipeline

    monkeypatch.setattr(anatomy_module, "anatomy_pipeline", fake_anatomy_pipeline)

    attached = mf.interpretation.oshapley_from_forecast_result(
        forecast_result,
        X,
        y,
        {"ols": "ols"},
        window=window,
        sidecar_name="accuracy_anatomy",
        losses=("rmse",),
    )
    detached = mf.interpretation.oshapley_from_forecast_result(
        forecast_result,
        X,
        y,
        {"ols": "ols"},
        window=window,
        attach=False,
    )

    assert attached is not forecast_result
    assert attached.get_sidecar("accuracy_anatomy") is fake_pipeline
    assert attached.metadata["sidecars"]["accuracy_anatomy"]["metadata_schema"]["kind"] == (
        "anatomy_pipeline_result"
    )
    assert detached is fake_pipeline
    assert calls["kwargs"]["window"] is window
    assert fake_pipeline.metadata["forecast_result"]["forecast_rows"] == 1
    assert mf.interpretation.oshapley_pipeline(X, y, {"ols": "ols"}, window=window) is fake_pipeline
    assert (
        mf.interpretation.oshapley_from_forecast_result(
            forecast_result,
            X,
            y,
            {"ols": "ols"},
            window=window,
            attach=False,
        )
        is fake_pipeline
    )

    with pytest.raises(ValueError, match="window is required"):
        mf.interpretation.oshapley_from_forecast_result(
            forecast_result,
            X,
            y,
            {"ols": "ols"},
            window=None,
        )


def test_forecast_shapley_output_selects_result_sidecar_tables() -> None:
    forecast_result = mf.forecasting.ForecastResult(
        pd.DataFrame({"model": ["ridge"], "prediction": [1.0], "actual": [1.5]}),
        metadata={"run_id": "demo"},
    )
    forecast = pd.DataFrame({"feature": ["x1"], "contribution": [0.4]})
    vi = pd.DataFrame({"feature": ["x1"], "importance": [0.4]})
    pbsv = pd.DataFrame({"feature": ["x1"], "contribution": [-0.2]})
    result = mf.interpretation.ForecastShapleyResult(
        anatomy=None,
        explanations={"forecast": forecast},
        variable_importance=vi,
        performance_values={"rmse": pbsv},
        metadata={"window": {"method": "expanding"}, "models": ["ridge"]},
    )
    attached = forecast_result.with_sidecar("oshapley", result)

    pd.testing.assert_frame_equal(
        mf.interpretation.forecast_shapley_output(attached, output="forecast"),
        forecast,
    )
    pd.testing.assert_frame_equal(
        mf.interpretation.oshapley_output(attached, output="oshapley"),
        vi,
    )
    pd.testing.assert_frame_equal(
        mf.interpretation.oshapley_output(attached, output="vi"),
        vi,
    )
    pd.testing.assert_frame_equal(
        mf.interpretation.oshapley_output(attached, output="pbsv", loss="rmse"),
        pbsv,
    )
    pd.testing.assert_frame_equal(
        mf.interpretation.forecast_shapley_output(result, output="loss"),
        pbsv,
    )

    metadata = mf.interpretation.oshapley_output(attached, output="metadata")
    assert metadata.attrs["macroforecast_metadata_schema"]["kind"] == (
        "forecast_shapley_metadata_table"
    )
    assert {"window.method", "models[0]"} <= set(metadata["path"])

    summary = mf.interpretation.oshapley_output(attached, output="summary")
    assert summary["metadata_schema"]["kind"] == "anatomy_pipeline_result"

    tables = mf.interpretation.oshapley_output(attached, output="tables")
    assert set(tables) == {
        "oshapley_explanation_forecast",
        "oshapley_metadata",
        "oshapley_performance_rmse",
        "oshapley_variable_importance",
    }
    selected = mf.interpretation.oshapley_output(
        attached,
        output="tables",
        table="oshapley_variable_importance",
    )
    assert selected.equals(vi)

    with pytest.raises(KeyError, match="PBSV loss"):
        mf.interpretation.oshapley_output(attached, output="pbsv", loss="mae")
    with pytest.raises(KeyError, match="table"):
        mf.interpretation.oshapley_output(attached, output="tables", table="missing")
    with pytest.raises(ValueError, match="no forecast-Shapley sidecar"):
        mf.interpretation.oshapley_output(forecast_result)


def test_forecast_shapley_output_requires_sidecar_name_when_ambiguous() -> None:
    forecast_result = mf.forecasting.ForecastResult(
        pd.DataFrame({"model": ["ridge"], "prediction": [1.0], "actual": [1.5]})
    )
    one = mf.interpretation.ForecastShapleyResult(
        anatomy=None,
        variable_importance=pd.DataFrame({"feature": ["x1"], "importance": [1.0]}),
    )
    two = mf.interpretation.ForecastShapleyResult(
        anatomy=None,
        variable_importance=pd.DataFrame({"feature": ["x2"], "importance": [2.0]}),
    )
    attached = forecast_result.with_sidecar("first", one).with_sidecar("second", two)

    with pytest.raises(ValueError, match="multiple ForecastResult sidecars"):
        mf.interpretation.oshapley_output(attached)

    out = mf.interpretation.oshapley_output(attached, sidecar_name="second")

    pd.testing.assert_frame_equal(out, two.variable_importance)
    with pytest.raises(KeyError, match="was not found"):
        mf.interpretation.oshapley_output(attached, sidecar_name="missing")


@pytest.mark.skipif(not _has_anatomy(), reason="requires optional anatomy backend")
def test_anatomy_pipeline_builds_provider_from_macroforecast_window() -> None:
    X, y = _xy()
    window = mf.window.from_cutoffs(
        test_start=X.index[20],
        test_end=X.index[22],
        estimation_min_size=12,
        val_method="last_block",
        val_size=3,
        horizon=1,
        step=1,
    )

    subsets = mf.interpretation.window_to_anatomy_subsets(window, X.index)
    provider = mf.interpretation.anatomy_provider(
        X,
        y,
        {"ols": "ols"},
        window=window,
    )
    result = mf.interpretation.anatomy_pipeline(
        X,
        y,
        {"ols": "ols"},
        window=window,
        losses=("squared_error", "rmse"),
        n_iterations=4,
        n_jobs=1,
    )
    forecast_result = mf.forecasting.ForecastResult(
        pd.DataFrame({"model": ["ols"], "prediction": [0.0], "actual": [0.0]}),
        metadata={"run_id": "demo"},
    )
    loss = forecast_result.anatomy_pbsv(result.anatomy, loss="squared_error")
    loss_alias = forecast_result.pbsv(result.anatomy, loss="squared_error")
    vi_alias = forecast_result.oshapley_vi(result.anatomy)

    assert subsets.n_periods == 3
    assert provider.n_periods == 3
    assert provider.model_names == ["ols"]
    assert result.variable_importance is not None
    assert (
        result.variable_importance.attrs["macroforecast_metadata_schema"]["kind"]
        == "oshapley_vi"
    )
    assert set(result.performance_values) == {"squared_error", "rmse"}
    assert (
        result.explanations["forecast"].attrs["macroforecast_metadata_schema"]["kind"]
        == "anatomy_explain"
    )
    assert loss.attrs["macroforecast_forecast_result"]["run_id"] == "demo"
    assert loss_alias.attrs["macroforecast_forecast_result"]["run_id"] == "demo"
    assert vi_alias.attrs["macroforecast_forecast_result"]["run_id"] == "demo"
