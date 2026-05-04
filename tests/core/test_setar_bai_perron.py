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


def test_setar_quantile_split_produces_n_regimes_labels():
    rng = np.random.default_rng(0)
    n = 60
    idx = pd.date_range("2010-01-01", periods=n, freq="MS")
    series = rng.normal(size=n)
    leaf = {"regime_target_values": series.tolist()}
    labels, metadata = _estimate_threshold_regime(idx, n_regimes=3, leaf_config=leaf)
    assert len(labels) == n
    assert metadata["method"] == "tong_1990_setar_quantile_split"
    assert len(metadata["thresholds"]) == 2
    assert set(labels.unique()).issubset({"regime_0", "regime_1", "regime_2"})


def test_setar_falls_back_when_no_threshold_series():
    n = 40
    idx = pd.date_range("2010-01-01", periods=n, freq="MS")
    labels, metadata = _estimate_threshold_regime(idx, n_regimes=2, leaf_config={})
    assert len(labels) == n
    assert metadata["method"] in {
        "tong_1990_setar_quantile_split",  # quantile split on synthetic index
        "fallback_no_threshold_series",
    }


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
    assert metadata["method"] == "bai_perron_global_lse_greedy"
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
