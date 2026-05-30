from __future__ import annotations

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


def test_model_spec_can_fit_like_model_callable() -> None:
    X, y = _xy()
    spec = mf.models.get_model("ridge", params={"alpha": 0.5})

    fit = spec(X, y)

    assert isinstance(fit, mf.models.ModelFit)
    assert fit.metadata["alpha"] == 0.5


def test_pcr_and_far_fit() -> None:
    X, y = _xy()

    pcr_fit = mf.models.pcr(X, y, n_components=1)
    far_fit = mf.models.far(X, y, n_factors=1, n_lag=1)

    assert len(pcr_fit.predict(X.iloc[-3:])) == 3
    assert len(far_fit.predict(X.iloc[-3:])) == 3


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


def test_optional_external_models_fail_lazily_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    X, y = _xy()

    def fail_import(name: str, package: str | None = None):
        if name == "xgboost":
            raise ImportError("blocked")
        return __import__(name)

    monkeypatch.setattr("macroforecast.models.utils.import_module", fail_import)

    with pytest.raises(ImportError, match="macroforecast\\[xgboost\\]"):
        mf.models.xgboost(X, y)


def test_ar_and_var_fit() -> None:
    X, y = _xy()
    panel = pd.concat([y, X], axis=1)

    ar_fit = mf.models.ar(y, n_lag=2)
    var_fit = mf.models.var(panel, target="y", n_lag=1)

    assert len(ar_fit.predict(pd.DataFrame(index=y.index[-4:]))) == 4
    assert len(var_fit.predict(panel.iloc[-4:])) == 4
