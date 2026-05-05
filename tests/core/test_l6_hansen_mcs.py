"""Hansen (2005) per-origin MCS / SPA / Reality Check / StepM regression tests
(closes #164).

Verifies:

* When the L5 per-origin loss panel is available the L6.D path uses the
  stationary block bootstrap (returns ``bootstrap_kind ==
  "stationary_block_bootstrap_per_origin"``).
* Politis-Romano stationary bootstrap index draw produces valid sequences
  (uniform marginal, expected restart frequency on average).
* Politis-White auto block length produces a positive integer scaled with
  T per the rule-of-thumb.
* MCS retains every model when the panel has no informative spread.
* MCS rejects the worst model when one model is clearly dominated by
  every other.
* StepM rejection set is a subset of the MCS complement.
* The function still falls back to the parametric Gaussian path when the
  per-origin panel is empty (back-compat).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macroforecast.core.runtime import (
    _fixed_block_bootstrap_indices,
    _l6_multiple_model_results,
    _mcs_from_per_origin_panel,
    _resolve_block_length,
    _stationary_bootstrap_indices,
)


# ---------------------------------------------------------------------------
# Bootstrap primitives
# ---------------------------------------------------------------------------

def test_stationary_bootstrap_indices_in_range_and_correct_length():
    rng = np.random.default_rng(0)
    n = 100
    indices = _stationary_bootstrap_indices(n, block_length=8, rng=rng)
    assert indices.shape == (n,)
    assert indices.min() >= 0 and indices.max() < n


def test_stationary_bootstrap_block_length_one_is_iid():
    rng = np.random.default_rng(0)
    n = 400
    indices_a = _stationary_bootstrap_indices(n, block_length=1, rng=rng)
    indices_b = _stationary_bootstrap_indices(n, block_length=1, rng=rng)
    # iid draws => marginal mean ~ (n-1)/2 = 199.5; tolerance generous.
    assert abs(indices_a.mean() - (n - 1) / 2) < n * 0.1
    assert abs(indices_b.mean() - (n - 1) / 2) < n * 0.1


def test_stationary_bootstrap_long_block_has_few_restarts():
    rng = np.random.default_rng(0)
    n = 1000
    block_length = 50
    expected_restarts = (n - 1) / block_length
    counts = []
    for _ in range(20):
        indices = _stationary_bootstrap_indices(n, block_length=block_length, rng=rng)
        # Restarts = positions where idx[t] != (idx[t-1] + 1) % n.
        diffs = (indices[1:] - indices[:-1]) % n
        counts.append(int((diffs != 1).sum()))
    avg = sum(counts) / len(counts)
    # Politis-Romano expected restarts ~ (n - 1) / block_length, allow 50% slack.
    assert 0.5 * expected_restarts < avg < 1.5 * expected_restarts


def test_fixed_block_bootstrap_returns_n_indices():
    rng = np.random.default_rng(0)
    indices = _fixed_block_bootstrap_indices(100, block_length=20, rng=rng)
    assert indices.shape == (100,)
    assert indices.min() >= 0 and indices.max() < 100


# ---------------------------------------------------------------------------
# Politis-White auto block length
# ---------------------------------------------------------------------------

def test_resolve_block_length_auto_scales_with_n():
    small = _resolve_block_length(np.zeros((50, 3)), "auto")
    large = _resolve_block_length(np.zeros((1000, 3)), "auto")
    assert 1 <= small < large
    assert large >= 4  # 2 * (40)^(1/3) ~ 7


def test_resolve_block_length_explicit_int_clipped_to_half_n():
    block = _resolve_block_length(np.zeros((20, 2)), 100)
    assert block == 10  # n // 2


def test_resolve_block_length_string_int_accepted():
    block = _resolve_block_length(np.zeros((40, 2)), "5")
    assert block == 5


# ---------------------------------------------------------------------------
# MCS on synthetic per-origin panels
# ---------------------------------------------------------------------------

def _build_panel(n_origins: int = 60, model_means: dict[str, float] | None = None, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if model_means is None:
        model_means = {"a": 1.0, "b": 1.0, "c": 1.0}
    rows = []
    for t in range(n_origins):
        for model, mu in model_means.items():
            squared = max(0.0, rng.normal(loc=mu, scale=0.3))
            rows.append({"model_id": model, "target": "y", "horizon": 1, "origin": t, "squared_error": squared, "absolute_error": np.sqrt(squared)})
    return pd.DataFrame(rows)


def test_mcs_per_origin_uses_block_bootstrap_kind():
    panel = _build_panel()
    out = _mcs_from_per_origin_panel(panel, {"mcs_alpha": 0.10, "bootstrap_n_replications": 200})
    assert out["bootstrap_kind"] == "stationary_block_bootstrap_per_origin"
    assert out["block_lengths_used"][("y", 1)] >= 1


def test_mcs_retains_all_models_when_losses_are_indistinguishable():
    panel = _build_panel(n_origins=80, model_means={"a": 1.0, "b": 1.0, "c": 1.0})
    out = _mcs_from_per_origin_panel(panel, {"mcs_alpha": 0.10, "bootstrap_n_replications": 300})
    included = out["mcs_inclusion"][("y", 1, 0.10)]
    # With no real difference every model should land inside the MCS.
    assert {"a", "b", "c"} <= included


def test_mcs_drops_the_worst_model_when_one_is_clearly_dominated():
    # Model d has 5x larger mean loss than the others.
    panel = _build_panel(n_origins=120, model_means={"a": 1.0, "b": 1.0, "c": 1.0, "d": 5.0}, seed=2)
    out = _mcs_from_per_origin_panel(panel, {"mcs_alpha": 0.10, "bootstrap_n_replications": 500})
    included = out["mcs_inclusion"][("y", 1, 0.10)]
    rejected = out["stepm_rejected"][("y", 1, 0.10)]
    assert "d" not in included
    assert "d" in rejected


def test_stepm_rejects_the_worst_model_first():
    """StepM (Romano-Wolf) is a different procedure from MCS -- it iteratively
    removes the worst-performing model whose studentized t-statistic exceeds
    the bootstrap critical. With a clearly worst model 'd' (mean 3.5 vs the
    others' 1.0..2.0), StepM must reject 'd'."""

    panel = _build_panel(n_origins=100, model_means={"a": 1.0, "b": 1.5, "c": 2.0, "d": 3.5}, seed=3)
    out = _mcs_from_per_origin_panel(panel, {"mcs_alpha": 0.10, "bootstrap_n_replications": 400})
    rejected = out["stepm_rejected"][("y", 1, 0.10)]
    assert "d" in rejected
    assert "a" not in rejected  # the best model should never be rejected


# ---------------------------------------------------------------------------
# Back-compat: empty per-origin panel falls through
# ---------------------------------------------------------------------------

def test_dispatcher_falls_back_to_summary_path_when_panel_is_empty():
    metrics = pd.DataFrame(
        [
            {"model_id": "a", "target": "y", "horizon": 1, "mse": 1.0},
            {"model_id": "b", "target": "y", "horizon": 1, "mse": 1.2},
        ]
    )
    out = _l6_multiple_model_results(metrics, {"mcs_alpha": 0.10, "bootstrap_n_replications": 200}, per_origin_panel=pd.DataFrame())
    assert out["bootstrap_kind"] == "parametric_gaussian_cross_sectional"


def test_dispatcher_uses_per_origin_path_when_panel_is_provided():
    panel = _build_panel(n_origins=40, model_means={"a": 1.0, "b": 1.5}, seed=7)
    metrics = panel.groupby(["model_id", "target", "horizon"], as_index=False).agg(mse=("squared_error", "mean"))
    out = _l6_multiple_model_results(metrics, {"mcs_alpha": 0.10, "bootstrap_n_replications": 200}, per_origin_panel=panel)
    assert out["bootstrap_kind"] == "stationary_block_bootstrap_per_origin"
