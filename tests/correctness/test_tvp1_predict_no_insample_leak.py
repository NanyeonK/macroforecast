"""Regression test for TVP-1.

TVPRidgeRegressor.predict() must always apply the leak-free terminal-coefficient
forecast rule, even when called on the training index. The old shortcut returned
full-sample-smoothed in-sample fitted values for the training index, leaking
future information if used as forecasts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import macroforecast as mf


def test_tvp_predict_on_training_index_uses_terminal_beta():
    rng = np.random.default_rng(0)
    n = 60
    idx = pd.date_range("2000-01-31", periods=n, freq="ME")
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)}, index=idx)
    y = pd.Series(0.5 * X["a"] - 0.3 * X["b"] + rng.normal(scale=0.1, size=n), index=idx)

    fit = mf.models.tvp_ridge(
        X, y, lambda_candidates=(0.1, 1.0), kfold=3, cv_2srr=False, use_garch=False
    )
    est = fit.estimator

    pred = np.asarray(est.predict(X)).reshape(-1)

    # Expected: terminal (time-T) coefficient applied to every row.
    x_aug = np.column_stack([np.ones(n), X.to_numpy(dtype=float)])
    beta_last = est.betas_2srr_[:, :, -1]
    expected_terminal = (x_aug @ beta_last.T).reshape(-1)

    smoothed = est.yhat_2srr_.to_numpy(dtype=float).reshape(-1)

    np.testing.assert_allclose(pred, expected_terminal, rtol=1e-9, atol=1e-9)
    # The smoothed (two-sided) in-sample fit must differ from the forecast rule
    # for a genuinely time-varying coefficient path; predict must NOT return it.
    assert not np.allclose(pred, smoothed)
