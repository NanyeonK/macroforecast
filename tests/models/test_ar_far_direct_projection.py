"""Direct-mode AR/FAR: a one-shot projection of the (h-ahead) target onto the
FRESH one-period lag features, NOT an iterated roll-forward from stale history.

Under the ``direct``/``direct_average`` forecast policy the regression target is a
pre-built h-ahead object whose only leak-free lag is origin-h stale, so the legacy
``_AR``/``_FAR`` (which autoregress the target's own lags and iterate forward from
the last training value) collapse to persistence of a stale value. In ``direct``
mode the model must instead regress the target on the provided ``Y_lag*`` feature
columns (EXCLUDING the contemporaneous ``Y_lag0``, which would be look-ahead) and
predict each test row independently.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from macroforecast.models.timeseries import _AR, _FAR


def _lag_frame(values, n_lags):
    """Build a Y_lag0..Y_lag{n_lags} design from a 1-D series (lag k = shift k)."""
    s = pd.Series(values, name="Y")
    cols = {f"Y_lag{k}": s.shift(k) for k in range(0, n_lags + 1)}
    return pd.DataFrame(cols, index=s.index)


def test_direct_ar_is_per_row_ols_on_the_n_most_recent_lags():
    rng = np.random.default_rng(0)
    n = 200
    series = np.cumsum(rng.normal(size=n))          # near-unit-root level
    target = pd.Series(series, name="target")        # stand-in h-ahead target
    X = _lag_frame(series, n_lags=4)

    fit = _AR(n_lag=3, direct=True).fit(X, target)

    # Reference: OLS of target on the 3 MOST RECENT observed lags Y_lag0..Y_lag2
    # (lag 0 is observed at the origin, not look-ahead), on the rows where all those
    # lags are observed.
    use_cols = ["Y_lag0", "Y_lag1", "Y_lag2"]
    design = X[use_cols]
    mask = design.notna().all(axis=1) & target.notna()
    ref = LinearRegression().fit(design[mask].to_numpy(), target[mask].to_numpy())

    X_test = X.iloc[-5:]
    got = np.asarray(fit.predict(X_test), dtype=float)
    expected = ref.predict(X_test[use_cols].fillna(0.0).to_numpy())

    # Per-row projection, NOT an iterated path: every test row maps through the
    # same fitted coefficients on its OWN fresh lags.
    assert got.shape == (5,)
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-8)

    # IC plumbing preserved.
    assert fit.ssr_ is not None and fit.nobs_ is not None
    assert fit.n_params_ == len(use_cols) + 1  # 3 lags (0,1,2) + intercept


def test_direct_ar_does_not_persist_stale_value():
    """The legacy bug: prediction ~ a stale constant. Direct mode must vary with X."""
    rng = np.random.default_rng(1)
    series = np.cumsum(rng.normal(size=160))
    target = pd.Series(series, name="target")
    X = _lag_frame(series, n_lags=2)
    fit = _AR(n_lag=2, direct=True).fit(X, target)
    preds = np.asarray(fit.predict(X.iloc[-10:]), dtype=float)
    # Distinct test rows -> distinct predictions (not a single persisted constant).
    assert np.std(preds) > 1e-6


def test_direct_far_projects_on_factors_plus_lags_no_iteration():
    rng = np.random.default_rng(2)
    n = 200
    # predictor block (drives factors) + the target's own lags
    P = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"P{i}" for i in range(6)])
    series = np.cumsum(rng.normal(size=n))
    lags = _lag_frame(series, n_lags=2)
    X = pd.concat([P, lags], axis=1)
    target = pd.Series(series, name="target")

    fit = _FAR(n_factors=3, n_lag=2, direct=True).fit(X, target)
    preds = np.asarray(fit.predict(X.iloc[-8:]), dtype=float)
    assert preds.shape == (8,)
    assert np.std(preds) > 1e-6          # varies per row, not a persisted constant
    assert fit.ssr_ is not None and fit.nobs_ is not None
