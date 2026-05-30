from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf


def _xy(n: int = 60) -> tuple[pd.DataFrame, pd.Series]:
    idx = pd.date_range("2000-01-31", periods=n, freq="ME")
    x1 = np.linspace(0.0, 1.0, n)
    x2 = np.sin(np.arange(n) / 3.0)
    X = pd.DataFrame({"x1": x1, "x2": x2}, index=idx)
    y = pd.Series(1.0 + 2.0 * x1 - 0.5 * x2, index=idx, name="y")
    return X, y


def test_linear_models_return_model_fit_with_series_predictions() -> None:
    X, y = _xy()
    fit = mf.models.ridge(X, y, alpha=0.5)

    pred = fit.predict(X.iloc[:5])

    assert isinstance(fit, mf.models.ModelFit)
    assert list(pred.index) == list(X.index[:5])
    assert pred.name == "prediction"
    assert fit.metadata["alpha"] == 0.5


def test_lasso_path_is_not_public_model_family() -> None:
    assert not hasattr(mf.models, "lasso_path")
    assert "lasso_path" not in mf.models.__all__


def test_model_specs_own_default_params_and_search_spaces() -> None:
    spec = mf.models.get_model("lasso", preset="small")

    assert isinstance(spec, mf.models.ModelSpec)
    assert spec.default_params["alpha"] == 1.0
    assert spec.search_space() == {"alpha": (0.01, 0.1, 1.0)}

    table = mf.models.describe_model(spec)
    assert set(table["parameter"]) == {"alpha", "max_iter"}
    assert table.loc[table["parameter"] == "alpha", "small_space"].iloc[0] == (0.01, 0.1, 1.0)


def test_model_spec_parameter_metadata_is_consistent() -> None:
    for name, spec in mf.models.MODEL_SPECS.items():
        parameter_names = {parameter.name for parameter in spec.parameters}
        tunable_names = {parameter.name for parameter in spec.parameters if parameter.tunable}
        search_names = set().union(*(space.keys() for space in spec.search_spaces.values())) if spec.search_spaces else set()

        assert set(spec.default_params).issubset(parameter_names), name
        assert search_names.issubset(parameter_names), name
        if spec.search_spaces:
            assert tunable_names.issubset(search_names), name


def test_model_spec_can_fit_like_model_callable() -> None:
    X, y = _xy()
    spec = mf.models.get_model("ridge", params={"alpha": 0.5})

    fit = spec(X, y)

    assert isinstance(fit, mf.models.ModelFit)
    assert fit.metadata["alpha"] == 0.5


def test_model_fit_and_spec_export_metadata() -> None:
    X, y = _xy()
    spec = mf.models.get_model("ridge", preset="small", params={"alpha": 0.5})
    fit = spec(X, y)

    fit_dict = fit.to_dict()
    spec_dict = spec.to_dict()
    spec_metadata = spec.to_metadata()

    assert fit_dict["model"] == "ridge"
    assert fit_dict["feature_names"] == ["x1", "x2"]
    assert fit.to_metadata()["fit"]["n_features"] == 2
    assert spec_dict["parameters"][0]["name"] == "alpha"
    assert spec_metadata["model"] == "ridge"
    assert spec_metadata["search_space"]["alpha"] == [0.01, 0.1, 1.0]
    json.dumps({"fit": fit_dict, "spec": spec_dict, "metadata": spec_metadata})


def test_pls_and_far_fit() -> None:
    X, y = _xy()

    pls_fit = mf.models.pls(X, y, n_components=1)
    far_fit = mf.models.far(X, y, n_factors=1, n_lag=1)

    assert len(pls_fit.predict(X.iloc[-3:])) == 3
    assert pls_fit.metadata["n_components"] == 1
    assert len(far_fit.predict(X.iloc[-3:])) == 3


def test_supervised_pca_fit_records_selected_features() -> None:
    X, y = _xy()

    fit = mf.models.supervised_pca(
        X,
        y,
        n_components=1,
        n_selected=1,
        min_abs_corr=0.0,
        alpha=0.1,
    )
    pred = fit.predict(X.iloc[-3:])

    assert len(pred) == 3
    assert fit.metadata["n_components"] == 1
    assert fit.estimator.selected_features_
    assert set(fit.estimator.selected_features_).issubset(set(X.columns))


def test_tree_models_include_custom_macro_callables() -> None:
    X, y = _xy()

    sgt = mf.models.slow_growing_tree(X, y, max_depth=2, min_leaf_size=3)
    qrf = mf.models.quantile_regression_forest(
        X,
        y,
        n_estimators=8,
        min_samples_leaf=2,
        random_state=0,
    )

    assert len(sgt.predict(X.iloc[:4])) == 4
    quantiles = qrf.estimator.predict_quantiles(X.iloc[:4], levels=(0.1, 0.9))
    assert set(quantiles) == {0.1, 0.9}
    assert quantiles[0.1].shape == (4,)


def test_bagging_and_booging_are_callable_with_small_budgets() -> None:
    X, y = _xy()

    bag = mf.models.bagging(X, y, base="ridge", n_estimators=4, max_samples=0.7, random_state=0)
    boo = mf.models.booging(
        X,
        y,
        B=3,
        inner_n_estimators=5,
        sample_frac=0.8,
        random_state=0,
    )

    assert len(bag.predict(X.iloc[:3])) == 3
    assert len(boo.predict(X.iloc[:3])) == 3


@pytest.mark.parametrize(
    ("module", "extra", "fit_call"),
    [
        ("xgboost", "xgboost", lambda X, y: mf.models.xgboost(X, y)),
        ("lightgbm", "lightgbm", lambda X, y: mf.models.lightgbm(X, y)),
        ("catboost", "catboost", lambda X, y: mf.models.catboost(X, y)),
        ("pyearth", "mars", lambda X, y: mf.models.mars(X, y)),
        ("arch", "arch", lambda X, y: mf.models.garch11(y)),
        ("matplotlib", "macro_random_forest", lambda X, y: mf.models.macro_random_forest(X, y)),
    ],
)
def test_optional_external_models_fail_lazily_when_missing(
    monkeypatch: pytest.MonkeyPatch,
    module: str,
    extra: str,
    fit_call,
) -> None:
    X, y = _xy()

    def fail_import(name: str, package: str | None = None):  # noqa: ARG001
        if name == module:
            raise ImportError("blocked")
        return __import__(name)

    monkeypatch.setattr("macroforecast.models.utils.import_module", fail_import)

    with pytest.raises(ImportError, match=f"macroforecast\\[{extra}\\]"):
        fit_call(X, y)


def test_pcr_is_not_public_model_family() -> None:
    assert not hasattr(mf.models, "pcr")
    assert "pcr" not in mf.models.__all__
    assert "pcr" not in mf.models.MODEL_SPECS


def test_macro_random_forest_adapter_wires_reference_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    X, y = _xy(36)
    calls = {}

    class FakeMRF:
        def __init__(self, **kwargs):
            calls.update(kwargs)

        def _ensemble_loop(self):
            n = len(calls["oos_pos"])
            return {"pred_ensemble": pd.DataFrame({"Ensembled_Prediction": np.arange(n, dtype=float)})}

    monkeypatch.setattr(
        "macroforecast.models.tree.MacroRandomForestRegressor._import_external",
        staticmethod(lambda: FakeMRF),
    )

    fit = mf.models.macro_random_forest(
        X.iloc[:30],
        y.iloc[:30],
        x_columns=["x1"],
        S_columns=["x1", "x2"],
        B=2,
        print_b=False,
        parallelise=False,
    )
    pred = fit.predict(X.iloc[30:33])

    assert list(pred) == [0.0, 1.0, 2.0]
    assert calls["y_pos"] == 0
    assert calls["x_pos"].tolist() == [1]
    assert calls["S_pos"].tolist() == [1, 2]
    assert fit.metadata["B"] == 2
    assert fit.metadata["x_columns"] == ("x1",)


def test_macro_random_forest_vendored_backend_smoke() -> None:
    pytest.importorskip("joblib")
    pytest.importorskip("matplotlib")
    X, y = _xy(48)

    fit = mf.models.macro_random_forest(
        X.iloc[:36],
        y.iloc[:36],
        B=1,
        minsize=10,
        mtry_frac=1.0,
        S_columns=["x1", "x2"],
        parallelise=False,
        print_b=False,
    )
    pred = fit.predict(X.iloc[36:38])

    assert len(pred) == 2
    assert np.isfinite(pred).all()
    assert fit.estimator.output_ is not None
    assert "pred" in fit.estimator.output_


def test_model_metadata_records_all_fit_params() -> None:
    X, y = _xy()

    rf = mf.models.random_forest(
        X,
        y,
        n_estimators=7,
        max_depth=2,
        min_samples_leaf=3,
        bootstrap=False,
        n_jobs=1,
    )
    gb = mf.models.gradient_boosting(X, y, n_estimators=7, loss="absolute_error")

    assert rf.metadata["n_estimators"] == 7
    assert rf.metadata["min_samples_leaf"] == 3
    assert rf.metadata["bootstrap"] is False
    assert gb.metadata["loss"] == "absolute_error"


def test_ar_and_var_fit() -> None:
    X, y = _xy()
    panel = pd.concat([y, X], axis=1)

    ar_fit = mf.models.ar(y, n_lag=2)
    var_fit = mf.models.var(panel, target="y", n_lag=1)

    assert len(ar_fit.predict(pd.DataFrame(index=y.index[-4:]))) == 4
    assert len(var_fit.predict(panel.iloc[-4:])) == 4
