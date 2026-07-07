"""Information-criterion order selection for the autoregression and factor model.

The paper selects the AR and FM order by BIC rather than by cross-validation, so
these models must select their order from the training sample alone, with no
validation split. This guards both the standalone selector and the spec wiring.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.model_selection import SearchSpec, select_by_information_criterion
from macroforecast.models import get_model


def _ar2(n: int = 300, seed: int = 0, sigma: float = 1.0) -> pd.Series:
    rng = np.random.RandomState(seed)
    e = rng.randn(n) * float(sigma)
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = 0.6 * y[t - 1] - 0.25 * y[t - 2] + e[t]
    idx = pd.date_range("1990-01-01", periods=n, freq="MS")
    return pd.Series(y, index=idx, name="Y")


def test_ic_selector_recovers_true_ar_order() -> None:
    y = _ar2()
    spec = SearchSpec(method="grid", param_grid={"n_lag": (1, 2, 4, 6, 12)})
    result = select_by_information_criterion("ar", y, search=spec, criterion="bic")
    assert result.best_params["n_lag"] == 2
    assert result.metadata["validation"] == "none"
    assert result.method == "information_criterion:bic"


def test_explicit_ic_search_spec_selects_ar2_and_bic_is_stricter() -> None:
    y = _ar2(n=600, seed=7, sigma=0.5)
    spec = SearchSpec(
        method="information_criterion",
        criterion="aic",
        param_grid={"n_lag": (1, 2, 4, 6, 12)},
    )

    aic = mf.model_selection.select_params("ar", y, search=spec)
    direct = select_by_information_criterion("ar", y, search=spec, criterion="bic")
    bic = mf.model_selection.select_params(
        "ar",
        y,
        search=SearchSpec(
            method="information_criterion",
            criterion="bic",
            param_grid={"n_lag": (1, 2, 4, 6, 12)},
        ),
    )

    assert aic.best_params["n_lag"] == 2
    assert direct.method == "information_criterion:aic"
    assert direct.best_params["n_lag"] == 2
    assert bic.best_params["n_lag"] <= aic.best_params["n_lag"]
    assert aic.method == "information_criterion:aic"
    assert bic.method == "information_criterion:bic"


def test_information_criterion_search_errors_for_non_ic_model() -> None:
    X, y = pd.DataFrame({"x": np.arange(30.0)}), pd.Series(np.arange(30.0))
    spec = SearchSpec(
        method="information_criterion",
        criterion="aic",
        param_grid={"alpha": (0.1,)},
    )

    with pytest.raises(mf.model_selection.SearchError, match="ssr_"):
        mf.model_selection.select_params("ridge", X, y, search=spec)


def test_forecasting_explicit_ic_route_obeys_retune_every(monkeypatch) -> None:
    idx = pd.date_range("2000-01-31", periods=48, freq="ME")
    t = np.arange(len(idx), dtype=float)
    panel = pd.DataFrame({"y": 0.4 * t + np.sin(t / 3.0), "x": t}, index=idx)
    features = mf.feature_engineering.feature_spec(
        target="y",
        predictors=["x"],
        lags=(0,),
        target_lags=(0, 1),
    )
    window = mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=24),
        val=mf.window.val_last_block(size=6, retune_every=3),
        test=mf.window.test_origins(
            first_origin=idx[30],
            last_origin=idx[36],
            horizon=1,
        ),
    )
    spec = SearchSpec(
        method="information_criterion",
        criterion="bic",
        param_grid={"n_lag": (1, 2, 4)},
    )

    import macroforecast.forecasting.policies.base as base_policy

    calls = 0
    original = base_policy.select_by_information_criterion

    def counting_selector(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        base_policy,
        "select_by_information_criterion",
        counting_selector,
    )

    result = mf.forecasting.run(
        panel,
        "ar",
        window=window,
        features=features,
        target="y",
        horizon=1,
        model_selection=spec,
        save_models=False,
    )

    retuned = [row["retuned"] for row in result.to_frame()["model_selection"]]
    assert calls == 3
    assert retuned == [True, False, False, True, False, False, True]


def test_ar_and_far_specs_declare_bic_selection() -> None:
    assert get_model("ar").selection_method == "bic"
    assert get_model("far").selection_method == "bic"
    assert get_model("random_forest").selection_method == "cv"


def test_ar_runs_with_window_without_validation_block() -> None:
    y = _ar2()
    panel = y.to_frame("Y")
    window = mf.window.from_cutoffs(
        estimation_start="1990-01", test_start="2010-01", test_end="2013-12",
        mode="expanding", horizon=1, retrain_every=1,  # no val_method / val_size
    )
    features = mf.feature_engineering.feature_spec(
        target="Y", predictors=[], target_lags=range(0, 13)
    )
    search = SearchSpec(method="grid", param_grid={"n_lag": (1, 2, 4, 6, 12)})
    result = mf.forecasting.run(
        panel, "ar", window=window, features=features, target="Y", horizon=1,
        model_selection=search, save_models=False,
    )
    text = repr(result.to_dict())
    assert "information_criterion" in text
