"""`far` could not say which factor-extraction convention it wanted.

It centered the predictor block and ran PCA on that -- covariance PCA -- with no
way to ask for the standardized/correlation convention that `pca_step` uses by
default. The two are not interchangeable: covariance PCA lets the
largest-variance series dominate the factors, and in a macro panel a level-coded
series can outweigh dozens of growth rates. `docs/replication/zww_2023_replication.md`
records a published paper whose headline result turned on exactly this choice.

Issue #495. The default stays covariance so no existing result moves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf


def _panel(n: int = 120, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """A panel where one predictor is on a wildly different scale.

    This is the FRED-MD shape that makes the convention matter: `big` has a
    standard deviation ~1000x the others, so under covariance PCA it IS the
    first factor, and under correlation PCA it is one predictor among many.
    """
    idx = pd.date_range("1990-01-31", periods=n, freq="ME", name="date")
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "big": rng.normal(scale=800.0, size=n),
            "a": rng.normal(scale=0.8, size=n),
            "b": rng.normal(scale=0.8, size=n),
            "c": rng.normal(scale=0.8, size=n),
        },
        index=idx,
    )
    y = pd.Series(0.5 * X["a"] + 0.4 * X["b"] + rng.normal(scale=0.3, size=n), index=idx, name="y")
    return X, y


def test_the_default_is_unchanged_covariance_pca() -> None:
    """Nothing existing may move: the default must reproduce plain centering."""
    X, y = _panel()
    fit = mf.models.far(X, y, n_factors=2, n_lag=1)
    assert fit.metadata["scale"] is False

    # Recompute the covariance-PCA factors by hand and compare the fitted values.
    from macroforecast.feature_engineering.shared import _deterministic_pca

    centered = (X - X.mean(axis=0)).fillna(0.0)
    pca = _deterministic_pca(2, *centered.shape, random_state=0)
    expected = pca.fit_transform(centered)
    got = fit.estimator._pca.transform(centered)
    np.testing.assert_allclose(np.abs(got), np.abs(expected), rtol=0, atol=1e-10)


def test_scale_true_gives_the_correlation_convention() -> None:
    X, y = _panel()
    fit = mf.models.far(X, y, n_factors=2, n_lag=1, scale=True)
    assert fit.metadata["scale"] is True

    from macroforecast.feature_engineering.shared import _deterministic_pca

    sd = X.std(axis=0, ddof=0)
    prepared = ((X - X.mean(axis=0)) / sd).fillna(0.0)
    pca = _deterministic_pca(2, *prepared.shape, random_state=0)
    expected = pca.fit_transform(prepared)
    got = fit.estimator._pca.transform(prepared)
    np.testing.assert_allclose(np.abs(got), np.abs(expected), rtol=0, atol=1e-10)


def test_the_two_conventions_actually_disagree() -> None:
    """If they agreed, the parameter would be pointless -- pin that they do not."""
    X, y = _panel()
    cov = mf.models.far(X, y, n_factors=2, n_lag=1).predict(X)
    cor = mf.models.far(X, y, n_factors=2, n_lag=1, scale=True).predict(X)
    assert not np.allclose(np.asarray(cov, dtype=float), np.asarray(cor, dtype=float), atol=1e-8), (
        "covariance and correlation PCA produced the same fit on a panel built to separate them"
    )


def test_covariance_pca_is_dominated_by_the_large_series() -> None:
    """The substantive reason the convention matters, stated as a test."""
    X, _ = _panel()
    from macroforecast.feature_engineering.shared import _deterministic_pca

    centered = (X - X.mean(axis=0)).fillna(0.0)
    cov_load = np.abs(_deterministic_pca(1, *centered.shape, random_state=0)
                      .fit(centered).components_[0])
    sd = X.std(axis=0, ddof=0)
    prepared = ((X - X.mean(axis=0)) / sd).fillna(0.0)
    cor_load = np.abs(_deterministic_pca(1, *prepared.shape, random_state=0)
                      .fit(prepared).components_[0])
    big = list(X.columns).index("big")

    assert cov_load[big] > 0.99, "covariance PC1 should be essentially the large series"
    assert cor_load[big] < 0.9, "correlation PC1 should not be dominated by it"


def test_a_zero_variance_column_does_not_poison_the_block() -> None:
    """Its divisor is pinned to 1.0 rather than dividing to NaN."""
    X, y = _panel()
    X = X.assign(flat=3.0)
    fit = mf.models.far(X, y, n_factors=2, n_lag=1, scale=True)
    assert np.isfinite(np.asarray(fit.predict(X), dtype=float)).all()


@pytest.mark.parametrize("direct", [False, True])
def test_both_fit_paths_honour_the_flag(direct: bool) -> None:
    """`far` has two PCA sites (direct projection and roll-forward); both must."""
    X, y = _panel()
    if direct:
        # A direct projection regresses the H-AHEAD target on the origin's own
        # lags plus the predictor block. The target must therefore be the FUTURE
        # value -- if it is the same period, `y_lag0` IS the answer, the fit is
        # exact, and the factors (the thing under test) stop mattering at all.
        H = 3
        X = X.assign(**{f"y_lag{k}": y.shift(k) for k in range(2)})
        y = y.shift(-H)
        keep = X.notna().all(axis=1) & y.notna()
        X, y = X.loc[keep], y.loc[keep]
    cov = mf.models.far(X, y, n_factors=2, n_lag=1, direct=direct)
    cor = mf.models.far(X, y, n_factors=2, n_lag=1, direct=direct, scale=True)
    assert cov.estimator._x_scale is None
    assert cor.estimator._x_scale is not None
    assert not np.allclose(
        np.asarray(cov.predict(X), dtype=float),
        np.asarray(cor.predict(X), dtype=float),
        atol=1e-8,
    )
