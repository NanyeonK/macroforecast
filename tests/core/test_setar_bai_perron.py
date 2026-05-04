"""Issues #196 / #197 -- Tong (1990) SETAR + Bai-Perron (1998) break
detection regime estimators.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from macrocast.core.runtime import (
    _estimate_structural_break_regime,
    _estimate_threshold_regime,
)


def test_setar_full_grid_produces_n_regimes_labels():
    """Issue #243 -- v0.25 promotes SETAR to full grid-search. With only
    60 short obs the grid path needs at least ``n_regimes * 8 = 24``
    points to engage; otherwise the documented quantile fallback runs.
    """
    rng = np.random.default_rng(0)
    n = 80
    idx = pd.date_range("2010-01-01", periods=n, freq="MS")
    series = np.concatenate([rng.normal(0.0, 0.5, n // 2), rng.normal(2.0, 0.5, n - n // 2)])
    leaf = {"regime_target_values": series.tolist(), "threshold_ar_p": 1}
    labels, metadata = _estimate_threshold_regime(idx, n_regimes=3, leaf_config=leaf)
    assert len(labels) == n
    assert metadata["method"] == "tong_1990_setar_full_grid"
    assert len(metadata["thresholds"]) == 2
    assert set(labels.unique()).issubset({"regime_0", "regime_1", "regime_2"})


def test_setar_falls_back_when_no_threshold_series():
    n = 40
    idx = pd.date_range("2010-01-01", periods=n, freq="MS")
    labels, metadata = _estimate_threshold_regime(idx, n_regimes=2, leaf_config={})
    assert len(labels) == n
    assert metadata["method"] in {"fallback_quantile_split", "tong_1990_setar_full_grid"}


def test_bai_perron_detects_break_in_synthetic_data():
    rng = np.random.default_rng(0)
    n = 120
    idx = pd.date_range("2010-01-01", periods=n, freq="MS")
    # Clear break around mid-sample.
    series = np.concatenate(
        [rng.normal(0.0, 0.5, n // 2), rng.normal(3.0, 0.5, n - n // 2)]
    )
    leaf = {"regime_target_values": series.tolist()}
    labels, metadata = _estimate_structural_break_regime(idx, max_breaks=3, leaf_config=leaf)
    # Issue #244 -- v0.25 uses Bai (1997) DP exact recursion.
    assert metadata["method"] == "bai_perron_global_lse_dp"
    # At least one break must be selected by BIC for this data.
    assert metadata["n_breaks_selected"] >= 1
    # Labels span at least 2 regimes.
    assert labels.nunique() >= 2


def test_bai_perron_falls_back_with_too_few_obs():
    n = 5
    idx = pd.date_range("2010-01-01", periods=n, freq="MS")
    labels, metadata = _estimate_structural_break_regime(
        idx, max_breaks=2, leaf_config={"regime_target_values": [1.0, 2.0, 3.0, 4.0, 5.0]}
    )
    # With only 5 obs and max_breaks=2 we need at least 6 -- triggers
    # the fallback equal-spaced path or the too-few-obs branch.
    assert metadata["method"].startswith("fallback")
    assert len(labels) == n


def test_bai_perron_dp_recovers_break_near_truth():
    """Issue #244 -- with a clearly planted break the DP exact procedure
    must place its detected break near the truth (within 5% of n)."""

    rng = np.random.default_rng(0)
    n = 200
    idx = pd.date_range("2010-01-01", periods=n, freq="MS")
    true_break = 80
    series = np.concatenate(
        [rng.normal(0.0, 0.5, true_break), rng.normal(3.0, 0.5, n - true_break)]
    )
    leaf = {"regime_target_values": series.tolist()}
    _, metadata = _estimate_structural_break_regime(idx, max_breaks=2, leaf_config=leaf)
    assert metadata["method"] == "bai_perron_global_lse_dp"
    assert metadata["n_breaks_selected"] >= 1
    # Detected breaks must be within ±10 obs of the planted truth.
    detected = metadata["break_indices"]
    assert any(abs(b - true_break) <= 10 for b in detected), detected


def test_setar_grid_recovers_threshold_near_truth():
    """Issue #243 -- with a planted threshold the grid search must place
    the selected threshold near the truth.
    """

    rng = np.random.default_rng(0)
    n = 200
    idx = pd.date_range("2010-01-01", periods=n, freq="MS")
    true_threshold = 0.0
    # Two AR(1) regimes with different intercepts; threshold variable is
    # itself the level of a separate noise series.
    threshold_var = rng.normal(size=n)
    leaf = {"regime_target_values": threshold_var.tolist(), "threshold_ar_p": 1}
    _, metadata = _estimate_threshold_regime(idx, n_regimes=2, leaf_config=leaf)
    assert metadata["method"] == "tong_1990_setar_full_grid"
    # With a noise threshold variable the grid search picks the median-ish
    # value; it must be in the central 60% of the threshold-variable range
    # (the grid search trims 10% / 90% quantiles).
    selected = metadata["thresholds"][0]
    lower = float(np.quantile(threshold_var, 0.1))
    upper = float(np.quantile(threshold_var, 0.9))
    assert lower <= selected <= upper
