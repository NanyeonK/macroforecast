"""Regression test for MCS-001.

The exact Model Confidence Set (Hansen, Lunde, Nason 2011) must decide
membership using the cumulative (running-maximum) MCS p-value, not the raw
per-step elimination p-value. Once an elimination step fails to reject
(p > alpha), every still-active model belongs to the confidence set; a model
eliminated later with a small raw p-value but a cumulative MCS p-value above
alpha must remain IN the set (nestedness).

This test scripts a non-monotone p-value sequence (0.00, 0.225, 0.015) that
exposes the defect: with the raw-p decision the model eliminated at the third
step is wrongly rejected even though the cumulative MCS p-value is 0.225 > 0.10.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import macroforecast.tests as mft


def _scripted_statistic_factory(p_sequence):
    """Return a drop-in _mcs_statistic that eliminates the last active column
    and yields a scripted raw p-value at each call."""
    calls = {"i": 0}

    def _stub(matrix, *, n_boot, block_length, method, statistic, rng):
        idx = calls["i"]
        calls["i"] += 1
        p = p_sequence[idx]
        n_active = matrix.shape[1]
        eliminate_pos = n_active - 1  # always drop the last active column
        observed = 0.0
        n = 1000
        n_above = int(round(p * n))
        boot = np.full(n, -1.0)
        boot[:n_above] = 1.0  # mean(boot > observed=0) == p
        scores = np.zeros(n_active)
        return observed, eliminate_pos, scores, boot

    return _stub


def test_mcs_membership_uses_cumulative_pvalue(monkeypatch):
    # Columns ordered so "last active" elimination removes D, then C, then B,
    # leaving A as the surviving best model.
    wide = pd.DataFrame(
        {
            "A": np.zeros(50),
            "B": np.zeros(50),
            "C": np.zeros(50),
            "D": np.zeros(50),
        }
    )
    # raw p per elimination step: D=0.00 (reject), C=0.225 (keep, lifts cum max),
    # B=0.015 (raw < alpha but cumulative max stays 0.225 -> must KEEP).
    monkeypatch.setattr(
        mft, "_mcs_statistic", _scripted_statistic_factory([0.00, 0.225, 0.015])
    )

    result = mft._iterative_mcs_wide(
        wide,
        alpha=0.10,
        n_boot=10,
        block_length="auto",
        bootstrap_method="mcs_fixed_block",
        statistic="max",
        rng=np.random.default_rng(0),
    )

    # Canonical confidence set: D rejected; B, C, A retained (nested set).
    assert set(result["included_models"]) == {"A", "B", "C"}
    assert set(result["rejected_models"]) == {"D"}
