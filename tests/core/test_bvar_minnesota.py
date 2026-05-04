"""Issues #185 / #186 -- BVAR Minnesota and Normal-Inverse-Wishart prior
estimators are operational.

The closed-form posterior mean is

    β̂ = (V⁻¹ + X'X)⁻¹ (V⁻¹ m + X'y)

with ``m`` placing unit weight on the first own-lag column when present
and ``V`` shrinking higher lags via the Litterman (1986) decay scheme.

Pins:

* Both families pass the L4 validator (operational status).
* ``_BayesianVAR.fit`` produces a closed-form posterior mean -- not the
  plain VAR coefficient -- by checking that strong-prior settings pull
  predictions toward the random-walk forecast.
* NIW differs from Minnesota in the σ² hyperparameter (heavier tails)
  -- their predictions diverge under the same data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from macrocast.core.runtime import _BayesianVAR
from macrocast.core.ops.l4_ops import (
    OPERATIONAL_MODEL_FAMILIES,
    FUTURE_MODEL_FAMILIES,
    get_family_status,
)


def test_bvar_minnesota_is_operational():
    assert "bvar_minnesota" in OPERATIONAL_MODEL_FAMILIES
    assert "bvar_minnesota" not in FUTURE_MODEL_FAMILIES
    assert get_family_status("bvar_minnesota") == "operational"


def test_bvar_normal_inverse_wishart_is_operational():
    assert "bvar_normal_inverse_wishart" in OPERATIONAL_MODEL_FAMILIES
    assert "bvar_normal_inverse_wishart" not in FUTURE_MODEL_FAMILIES
    assert get_family_status("bvar_normal_inverse_wishart") == "operational"


def _toy_data(n: int = 60, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    y = pd.Series(rng.normal(size=n).cumsum(), name="y")
    X = pd.DataFrame(
        {
            "y_lag1": y.shift(1),
            "x1_lag1": rng.normal(size=n),
            "x1_lag2": rng.normal(size=n),
        }
    )
    return X, y


def test_minnesota_random_walk_anchor_pulls_lag1_coef_toward_one():
    """Strong-prior limit (very small λ₁ → tight prior) should pull the
    own-lag-1 coefficient toward the random-walk anchor m=1."""

    X, y = _toy_data()
    loose = _BayesianVAR(prior="bvar_minnesota", lambda1=10.0).fit(X, y)
    tight = _BayesianVAR(prior="bvar_minnesota", lambda1=1e-3).fit(X, y)
    # ``y_lag1`` is the first column; the random-walk anchor sits there.
    assert loose._coef is not None and tight._coef is not None
    # As the prior tightens, the coefficient on y_lag1 must move toward 1.
    assert abs(tight._coef[0] - 1.0) < abs(loose._coef[0] - 1.0)


def test_minnesota_predicts_finite_values():
    X, y = _toy_data()
    model = _BayesianVAR(prior="bvar_minnesota").fit(X, y)
    preds = model.predict(X.fillna(0.0))
    assert preds.shape == (len(X),)
    assert np.all(np.isfinite(preds))


def test_niw_differs_from_minnesota_under_same_data():
    """Normal-Inverse-Wishart bumps λ₁ by the documented factor; the
    posterior mean coefficients (and therefore predictions) must differ
    from the plain Minnesota fit."""

    X, y = _toy_data()
    a = _BayesianVAR(prior="bvar_minnesota", lambda1=0.2).fit(X, y)
    b = _BayesianVAR(prior="bvar_normal_inverse_wishart", lambda1=0.2).fit(X, y)
    # Predictions should differ because λ₁ scaling is different.
    pa = a.predict(X.fillna(0.0))
    pb = b.predict(X.fillna(0.0))
    assert not np.allclose(pa, pb)


def test_bvar_handles_missing_columns_at_predict_time():
    X, y = _toy_data()
    model = _BayesianVAR(prior="bvar_minnesota").fit(X, y)
    # Predict with a frame missing one column -- should silently zero it.
    X_partial = X.drop(columns=["x1_lag2"]).fillna(0.0)
    preds = model.predict(X_partial)
    assert preds.shape == (len(X_partial),)
    assert np.all(np.isfinite(preds))
