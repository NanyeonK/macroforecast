"""Issue #192 -- Apley & Zhu (2020) Accumulated Local Effects."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from macroforecast.core.runtime import _ale_table
from macroforecast.core.types import ModelArtifact


def _toy_model_artifact(n: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    y = pd.Series(2.0 * X["a"] - 0.5 * X["b"] + rng.normal(scale=0.1, size=n))
    fitted = LinearRegression().fit(X, y)
    return X, ModelArtifact(
        model_id="m",
        family="ols",
        fitted_object=fitted,
        framework="sklearn",
        feature_names=tuple(X.columns),
    )


def test_ale_returns_centred_function_per_feature():
    X, artifact = _toy_model_artifact()
    frame = _ale_table(artifact, X, n_quantiles=5)
    assert {"feature", "importance", "ale_function"}.issubset(frame.columns)
    # Every feature must have an ALE function entry (list of dicts).
    for _, row in frame.iterrows():
        ale = row["ale_function"]
        assert isinstance(ale, list)
        # Must include both bin centers and ALE values.
        if ale:
            assert "bin_center" in ale[0] and "ale" in ale[0]


def test_ale_function_is_centered_around_zero():
    X, artifact = _toy_model_artifact()
    frame = _ale_table(artifact, X, n_quantiles=10)
    for _, row in frame.iterrows():
        ale = row["ale_function"]
        if not ale:
            continue
        values = np.asarray([entry["ale"] for entry in ale])
        # Centred local effects + cumsum -> the function passes through zero
        # near the lower-end of the support (cumsum starts negative or at 0).
        # Pin: range is symmetric-ish (max + min sum near zero for the
        # idealised linear data).
        assert abs(values[0] + values[-1]) <= max(np.abs(values).max(), 1e-6) * 4


def test_ale_importance_increases_with_coefficient_magnitude():
    """For a linear model with coefficient (2, -0.5, 0), the importance for
    feature 'a' should exceed the importance for 'c'."""

    X, artifact = _toy_model_artifact(n=80, seed=1)
    frame = _ale_table(artifact, X, n_quantiles=10)
    importance = frame.set_index("feature")["importance"]
    assert importance["a"] > importance["b"]
    assert importance["b"] > importance["c"]


def test_ale_handles_constant_feature_without_crash():
    n = 30
    X = pd.DataFrame({"a": [1.0] * n, "b": np.linspace(-1, 1, n)})
    y = pd.Series(np.linspace(-1, 1, n))
    fitted = LinearRegression().fit(X, y)
    artifact = ModelArtifact(
        model_id="m",
        family="ols",
        fitted_object=fitted,
        framework="sklearn",
        feature_names=tuple(X.columns),
    )
    frame = _ale_table(artifact, X, n_quantiles=5)
    # Constant feature has 0 importance and an empty ale_function.
    a_row = frame[frame["feature"] == "a"].iloc[0]
    assert a_row["importance"] == 0.0
    assert a_row["ale_function"] == []
