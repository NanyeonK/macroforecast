from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf


def _aggregation_xy(n: int = 48) -> tuple[pd.DataFrame, pd.Series]:
    idx = pd.date_range("2000-01-31", periods=n, freq="ME")
    x1 = 1.0 + np.linspace(0.0, 1.0, n)
    x2 = 2.0 + np.sin(np.arange(n) / 5.0)
    x3 = 0.5 + np.cos(np.arange(n) / 7.0)
    X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3}, index=idx)
    y = (0.5 * X["x1"] + 0.3 * X["x2"] + 0.2 * X["x3"]).rename("future_agg")
    return X, y


def test_component_aggregation_is_nonnegative_simplex_weight_model() -> None:
    X, y = _aggregation_xy()

    fit = mf.models.component_aggregation(
        X,
        y,
        alpha=0.01,
        reference_weights={"x1": 0.4, "x2": 0.4, "x3": 0.2},
    )

    coef = fit.diagnostics["coefficients"]
    assert fit.model == "component_aggregation"
    assert coef.index.tolist() == ["x1", "x2", "x3"]
    assert (coef >= -1e-10).all()
    assert float(coef.sum()) == pytest.approx(1.0, abs=1e-7)
    assert "Maximally Forward-Looking Core Inflation" in fit.metadata["source_reference"]
    assert fit.predict(X.iloc[:3]).index.equals(X.index[:3])


def test_rank_aggregation_sorts_inputs_and_records_rank_weight_curve() -> None:
    X, y = _aggregation_xy()

    fit = mf.models.rank_aggregation(X, y, alpha=0.1)
    raw_prediction = fit.predict(X.iloc[:4])
    sorted_prediction = pd.Series(
        np.sort(X.iloc[:4].to_numpy(dtype=float), axis=1)
        @ fit.estimator.coef_,
        index=X.index[:4],
        name="prediction",
    )

    coef = fit.diagnostics["coefficients"]
    assert coef.index.tolist() == ["rank_1", "rank_2", "rank_3"]
    assert (coef >= -1e-10).all()
    assert raw_prediction.equals(sorted_prediction)
    assert fit.estimator.rank_weight_curve_["percentile"].tolist() == [
        pytest.approx(1 / 3),
        pytest.approx(2 / 3),
        pytest.approx(1.0),
    ]


def test_albacore_wrappers_and_solver_helpers_are_public() -> None:
    X, y = _aggregation_xy()

    comp = mf.models.albacore_components(
        X,
        y,
        reference_weights={"x1": 0.5, "x2": 0.3, "x3": 0.2},
        alpha=0.1,
    )
    ranks = mf.models.albacore_ranks(X, y, alpha=0.1)
    weights = mf.models.solve_target_shrinkage_ridge(
        X,
        y,
        reference_weights={"x1": 0.5, "x2": 0.3, "x3": 0.2},
        alpha=0.1,
    )

    assert comp.diagnostics["coefficients"].sum() == pytest.approx(1.0, abs=1e-7)
    assert ranks.diagnostics["coefficients"].index.tolist() == [
        "rank_1",
        "rank_2",
        "rank_3",
    ]
    assert weights.index.tolist() == ["x1", "x2", "x3"]
    assert mf.albacore_components is mf.models.albacore_components
    assert mf.albacore_ranks is mf.models.albacore_ranks
    assert mf.models.get_model("assemblage_regression").family == "assemblage"
    assert mf.models.get_model("albacore_components").family == "assemblage"
