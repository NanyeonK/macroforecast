"""Information-criterion order selection for the autoregression and factor model.

The paper selects the AR and FM order by BIC rather than by cross-validation, so
these models must select their order from the training sample alone, with no
validation split. This guards both the standalone selector and the spec wiring.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import macroforecast as mf
from macroforecast.model_selection import SearchSpec, select_by_information_criterion
from macroforecast.models import get_model


def _ar2(n: int = 300, seed: int = 0) -> pd.Series:
    rng = np.random.RandomState(seed)
    e = rng.randn(n)
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
