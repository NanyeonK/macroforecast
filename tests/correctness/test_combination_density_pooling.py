"""Batch C: density pooling (linear opinion pool, logarithmic opinion pool)."""
import numpy as np
import pandas as pd
import pytest

import macroforecast as mf


def _density(n=50, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-31", periods=n, freq="ME", name="date")
    means = pd.DataFrame({"A": rng.standard_normal(n), "B": rng.standard_normal(n) + 1.0}, index=idx)
    sds = pd.DataFrame({"A": np.full(n, 1.0), "B": np.full(n, 2.0)}, index=idx)
    return means, sds


def test_linear_pool_mean_and_mixture_variance():
    means, sds = _density()
    lp = mf.forecasting.combine_linear_pool(means, sds)
    # equal-weight pooled mean = average of component means
    np.testing.assert_allclose(lp["mean"].to_numpy(), means.mean(axis=1).to_numpy())
    # mixture variance >= weighted component variance (disagreement adds variance)
    w_comp_var = (sds ** 2).mean(axis=1)
    assert np.all(lp["variance"].to_numpy() >= w_comp_var.to_numpy() - 1e-9)


def test_log_pool_is_sharper_than_linear():
    means, sds = _density(seed=1)
    lp = mf.forecasting.combine_linear_pool(means, sds)
    gp = mf.forecasting.combine_log_pool(means, sds)
    # log pool (product of densities) is more confident -> smaller variance
    assert np.all(gp["variance"].to_numpy() <= lp["variance"].to_numpy() + 1e-9)


def test_log_pool_gaussian_formula():
    # two Gaussians, equal weight: precision adds -> sigma^2 = 1/(0.5/1 + 0.5/4)
    means = pd.DataFrame({"A": [0.0], "B": [4.0]})
    sds = pd.DataFrame({"A": [1.0], "B": [2.0]})
    gp = mf.forecasting.combine_log_pool(means, sds)
    tau = 0.5 / 1.0 + 0.5 / 4.0
    var = 1.0 / tau
    mean = var * (0.5 * 0.0 / 1.0 + 0.5 * 4.0 / 4.0)
    np.testing.assert_allclose(gp["variance"].iloc[0], var)
    np.testing.assert_allclose(gp["mean"].iloc[0], mean)


def test_custom_weights_and_point_linear_pool():
    means, sds = _density(seed=2)
    lp = mf.forecasting.combine_linear_pool(means, sds, weights=[0.8, 0.2])
    np.testing.assert_allclose(lp["mean"].to_numpy(), (0.8 * means["A"] + 0.2 * means["B"]).to_numpy())
    # sds omitted -> reduces to weighted mean, no variance column
    point = mf.forecasting.combine_linear_pool(means, weights=[0.8, 0.2])
    assert "variance" not in point.columns
    np.testing.assert_allclose(point["mean"].to_numpy(), lp["mean"].to_numpy())
