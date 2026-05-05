"""Issue #195 -- L1.G ``estimated_markov_switching`` runs the real
Hamilton (1989) Markov regression via statsmodels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from macroforecast.core.runtime import _estimate_markov_switching_regime


def test_hamilton_ms_uses_statsmodels_when_target_values_supplied():
    rng = np.random.default_rng(0)
    n = 120
    idx = pd.date_range("2010-01-01", periods=n, freq="MS")
    # Simulate two regimes with different means.
    series = np.where(np.arange(n) < n // 2, rng.normal(0.0, 1.0, n), rng.normal(2.5, 1.0, n))
    leaf = {"regime_target_values": series.tolist()}
    labels, probs, transition_matrix, metadata = _estimate_markov_switching_regime(
        idx, target_name="y", n_regimes=2, leaf_config=leaf
    )
    assert metadata["method"] == "hamilton_1989_markov_regression"
    assert isinstance(probs, pd.DataFrame)
    assert probs.shape == (n, 2)
    assert isinstance(transition_matrix, pd.DataFrame)
    # Posterior probabilities must sum to ~1 row-wise (after dropping NaN
    # padding from the dropna step).
    finite = probs.dropna()
    np.testing.assert_allclose(finite.sum(axis=1).to_numpy(), 1.0, atol=1e-6)
    # log-likelihood is reported.
    assert "log_likelihood" in metadata


def test_hamilton_ms_produces_label_per_observation():
    rng = np.random.default_rng(0)
    n = 100
    idx = pd.date_range("2010-01-01", periods=n, freq="MS")
    series = rng.normal(size=n)
    leaf = {"regime_target_values": series.tolist()}
    labels, _, _, _ = _estimate_markov_switching_regime(
        idx, target_name="y", n_regimes=2, leaf_config=leaf
    )
    assert len(labels) == n
    assert set(labels.unique()).issubset({"regime_0", "regime_1"})


def test_hamilton_ms_falls_back_when_series_missing():
    n = 60
    idx = pd.date_range("2010-01-01", periods=n, freq="MS")
    labels, probs, transition_matrix, metadata = _estimate_markov_switching_regime(
        idx, target_name="y", n_regimes=2, leaf_config={}
    )
    # Without ``regime_target_values`` we get the documented fallback path
    # (still produces valid labels for downstream layers).
    assert metadata["method"].startswith("fallback")
    assert probs is None and transition_matrix is None
    assert len(labels) == n


def test_hamilton_ms_falls_back_when_too_few_observations():
    idx = pd.date_range("2010-01-01", periods=8, freq="MS")
    leaf = {"regime_target_values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]}
    _, probs, transition_matrix, metadata = _estimate_markov_switching_regime(
        idx, target_name="y", n_regimes=2, leaf_config=leaf
    )
    assert metadata["method"].startswith("fallback")
    assert probs is None and transition_matrix is None
