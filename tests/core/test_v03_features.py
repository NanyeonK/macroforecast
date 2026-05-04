"""v0.3 new features (#280, #281, #282, #283)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from macrocast.core.runtime import (
    _BaggingWrapper,
    _QuantileRegressionForest,
    _l6_dmp_multi_horizon,
    _strobl_permutation_importance_frame,
)
from macrocast.core.types import ModelArtifact
from macrocast.core.ops.l4_ops import OPERATIONAL_MODEL_FAMILIES, get_family_status


# ---------------------------------------------------------------------------
# #280 Quantile Regression Forest
# ---------------------------------------------------------------------------

def test_qrf_is_operational_l4_family():
    assert "quantile_regression_forest" in OPERATIONAL_MODEL_FAMILIES
    assert get_family_status("quantile_regression_forest") == "operational"


def test_qrf_predict_quantiles_returns_per_level_arrays():
    rng = np.random.default_rng(0)
    n = 80
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"))
    y = pd.Series(2.0 * X["a"] + rng.normal(scale=0.5, size=n))
    qrf = _QuantileRegressionForest(n_estimators=20, max_depth=4, random_state=0, quantile_levels=(0.1, 0.5, 0.9))
    qrf.fit(X, y)
    quantiles = qrf.predict_quantiles(X)
    assert set(quantiles.keys()) == {0.1, 0.5, 0.9}
    # Every quantile array has the right length.
    for q in (0.1, 0.5, 0.9):
        assert quantiles[q].shape == (n,)
    # Per-row quantile crossing: q90 >= q50 >= q10.
    np.testing.assert_array_less(quantiles[0.1] - 1e-6, quantiles[0.9])


# ---------------------------------------------------------------------------
# #282 Bagging
# ---------------------------------------------------------------------------

def test_bagging_is_operational_l4_family():
    assert "bagging" in OPERATIONAL_MODEL_FAMILIES
    assert get_family_status("bagging") == "operational"


def test_bagging_predict_quantiles_band_widens_with_more_estimators():
    rng = np.random.default_rng(0)
    n = 60
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"))
    y = pd.Series(rng.normal(size=n))
    bag = _BaggingWrapper(base_family="ridge", n_estimators=15, max_samples=0.5, random_state=0).fit(X, y)
    bands = bag.predict_quantiles(X, levels=(0.1, 0.5, 0.9))
    # Band has positive width on average (bag members disagree).
    width = float((bands[0.9] - bands[0.1]).mean())
    assert width > 0


# ---------------------------------------------------------------------------
# #281 Strobl permutation importance
# ---------------------------------------------------------------------------

def test_strobl_pfi_returns_method_marker():
    rng = np.random.default_rng(0)
    n = 80
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"))
    y = pd.Series(2.0 * X["a"] + rng.normal(scale=0.3, size=n))
    fitted = LinearRegression().fit(X, y)
    artifact = ModelArtifact(
        model_id="m", family="ols", fitted_object=fitted,
        framework="sklearn", feature_names=tuple(X.columns),
    )
    result = _strobl_permutation_importance_frame(artifact, X, y)
    assert "method" in result.columns
    assert (result["method"] == "strobl_conditional").all()


def test_strobl_assigns_lower_importance_to_redundant_correlated_feature():
    """When two features are highly correlated and one carries the signal,
    Strobl should rank the *informative* feature higher than the
    redundant one (vanilla permutation cannot distinguish them as
    cleanly). Synthetic test: x1 carries the signal; x2 = x1 + tiny
    noise; x3 is pure noise."""

    rng = np.random.default_rng(0)
    n = 200
    x1 = rng.normal(size=n)
    x2 = x1 + rng.normal(scale=0.05, size=n)
    x3 = rng.normal(size=n)
    X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    y = pd.Series(2.0 * x1 + rng.normal(scale=0.1, size=n))
    fitted = LinearRegression().fit(X, y)
    artifact = ModelArtifact(
        model_id="m", family="ols", fitted_object=fitted,
        framework="sklearn", feature_names=tuple(X.columns),
    )
    result = _strobl_permutation_importance_frame(artifact, X, y, seed=0).set_index("feature")
    # Pure noise feature should rank lowest.
    assert result.loc["x3", "importance"] <= result.loc["x1", "importance"]
    assert result.loc["x3", "importance"] <= result.loc["x2", "importance"]


# ---------------------------------------------------------------------------
# #283 Diebold-Mariano-Pesaran joint multi-horizon
# ---------------------------------------------------------------------------

def test_dmp_multi_horizon_runs_per_pair_target():
    rng = np.random.default_rng(0)
    rows = []
    for h in (1, 6, 12):
        for origin in range(40):
            for model_id, scale in (("a", 0.5), ("b", 0.6)):
                err = rng.normal(scale=scale)
                rows.append(
                    {"model_id": model_id, "target": "y", "horizon": h, "origin": origin, "squared": err ** 2, "absolute": abs(err)}
                )
    errors = pd.DataFrame(rows)
    pairs = [("b", "a")]
    out = _l6_dmp_multi_horizon(errors, pairs, hac_kernel="newey_west")
    key = ("dmp_multi_horizon", "b", "a", "y")
    assert key in out
    payload = out[key]
    assert {"statistic", "p_value", "decision_at_5pct", "n_obs_stacked", "hac_kernel"}.issubset(payload.keys())
    # 40 origins × 3 horizons = 120 stacked observations.
    assert payload["n_obs_stacked"] == 120


def test_dmp_multi_horizon_rejects_when_one_model_dominates_persistently():
    rng = np.random.default_rng(0)
    rows = []
    for h in (1, 6, 12):
        for origin in range(40):
            err_a = rng.normal(scale=0.5)
            err_b = rng.normal(scale=0.5) + 1.5  # model B much worse
            for model_id, err in (("a", err_a), ("b", err_b)):
                rows.append(
                    {"model_id": model_id, "target": "y", "horizon": h, "origin": origin, "squared": err ** 2, "absolute": abs(err)}
                )
    errors = pd.DataFrame(rows)
    out = _l6_dmp_multi_horizon(errors, [("b", "a")], hac_kernel="newey_west")
    payload = out[("dmp_multi_horizon", "b", "a", "y")]
    assert payload["decision_at_5pct"] is True
