from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf


def _supervised_frame(n: int = 24, p: int = 4) -> tuple[pd.DataFrame, pd.Series]:
    idx = pd.date_range("2000-01-31", periods=n, freq="ME")
    base = np.linspace(-1.0, 1.0, n)
    data = {
        f"x{j}": np.sin(base * (j + 1)) + (j + 1) * base**2
        for j in range(p)
    }
    X = pd.DataFrame(data, index=idx)
    y = pd.Series(
        0.5
        + 0.8 * X["x0"]
        - 0.3 * X["x1"]
        + 0.1 * X["x2"] ** 2
        + 0.05 * np.arange(n),
        index=idx,
        name="y",
    )
    return X, y


def _ols_subset_prediction(
    X: pd.DataFrame, y: pd.Series, X_new: pd.DataFrame, subset: tuple[int, ...]
) -> np.ndarray:
    x_values = X.to_numpy(dtype=float)[:, subset]
    design = np.column_stack([np.ones(len(X), dtype=float), x_values])
    params = np.linalg.lstsq(design, y.to_numpy(dtype=float), rcond=None)[0]
    new_design = np.column_stack(
        [np.ones(len(X_new), dtype=float), X_new.to_numpy(dtype=float)[:, subset]]
    )
    return new_design @ params


def test_csr_exact_small_case_averages_all_subsets() -> None:
    X, y = _supervised_frame(n=18, p=4)
    X_new = X.iloc[-3:]

    fit = mf.models.csr(X, y, k=2, max_subsets=100, random_state=123)
    pred = fit.predict(X_new)

    expected = np.mean(
        [
            _ols_subset_prediction(X, y, X_new, subset)
            for subset in combinations(range(4), 2)
        ],
        axis=0,
    )
    np.testing.assert_allclose(pred.to_numpy(dtype=float), expected, rtol=1e-12, atol=1e-12)
    assert len(fit.estimator.subsets_) == 6
    assert fit.metadata["k"] == 2


def test_csr_subset_cap_is_seed_deterministic() -> None:
    X, y = _supervised_frame(n=30, p=8)

    first = mf.models.csr(X, y, k=3, max_subsets=10, random_state=7)
    second = mf.models.csr(X, y, k=3, max_subsets=10, random_state=7)

    assert first.estimator.subsets_ == second.estimator.subsets_
    np.testing.assert_allclose(first.predict(X).to_numpy(), second.predict(X).to_numpy())


def test_csr_rejects_subset_size_larger_than_predictor_count() -> None:
    X, y = _supervised_frame(n=12, p=3)

    with pytest.raises(ValueError, match="at least k=4 predictors; got p=3"):
        mf.models.csr(X, y, k=4)


def test_csr_model_spec_exposes_random_state_for_pipeline_seed_derivation() -> None:
    spec = mf.models.get_model("csr")

    assert spec.input_kind == "supervised"
    assert spec.default_params["random_state"] == 1071
