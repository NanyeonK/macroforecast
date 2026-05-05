"""Issue #199 -- Giacomini & Rossi (2010) fluctuation test."""
from __future__ import annotations

import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from macroforecast.core.runtime import _l6_cpa_results
from macroforecast.core.types import L4ModelArtifactsArtifact


def _build_errors_panel(
    a_loss: np.ndarray, b_loss: np.ndarray, target: str = "y", horizon: int = 1
) -> pd.DataFrame:
    n = a_loss.size
    rows = []
    for t in range(n):
        rows.append(
            {"model_id": "a", "target": target, "horizon": horizon, "origin": t, "squared": float(a_loss[t])}
        )
        rows.append(
            {"model_id": "b", "target": target, "horizon": horizon, "origin": t, "squared": float(b_loss[t])}
        )
    return pd.DataFrame(rows)


def _l4_models() -> L4ModelArtifactsArtifact:
    return L4ModelArtifactsArtifact(
        artifacts={"a": MagicMock(), "b": MagicMock()},
        is_benchmark={"a": True, "b": False},
    )


def test_giacomini_rossi_returns_supremum_statistic_and_critical_value():
    rng = np.random.default_rng(0)
    n = 80
    a_loss = rng.uniform(0.5, 1.5, n)
    b_loss = a_loss + rng.normal(scale=0.05, size=n)  # essentially equal
    errors = _build_errors_panel(a_loss, b_loss)
    result = _l6_cpa_results(
        errors,
        sub={"cpa_test": "giacomini_rossi_2010", "cpa_window_ratio": 0.25, "cpa_alpha": 0.05},
        l4_models=_l4_models(),
    )
    key = ("giacomini_rossi_2010", ("b", "a"), "y", 1)
    assert key in result, list(result.keys())
    payload = result[key]
    assert payload["statistic"] is not None
    # v0.25 (#248) replaces the constant ``k_alpha = 2.7727`` with a
    # simulated supremum-of-Brownian-bridge quantile per (m/T, alpha).
    # The simulated value is in the published [2.5, 3.5] range for 5%.
    assert 2.5 < payload["critical_value"] < 3.5
    assert payload["window_size"] >= 4
    assert payload["decision"] is False  # essentially equal -> should not reject


def test_giacomini_rossi_rejects_when_one_model_dominates_in_a_window():
    n = 100
    rng = np.random.default_rng(1)
    a_loss = rng.uniform(0.5, 0.6, n)
    b_loss = a_loss.copy()
    # Inject sustained large loss for b in second half so the rolling
    # standardised mean exceeds the critical value somewhere.
    b_loss[n // 2 :] += 2.0
    errors = _build_errors_panel(a_loss, b_loss)
    result = _l6_cpa_results(
        errors,
        sub={"cpa_test": "giacomini_rossi_2010", "cpa_window_ratio": 0.25, "cpa_alpha": 0.05},
        l4_models=_l4_models(),
    )
    payload = result[("giacomini_rossi_2010", ("b", "a"), "y", 1)]
    assert payload["statistic"] > payload["critical_value"]
    assert payload["decision"] is True


def test_gr_critical_value_varies_with_window_ratio():
    """Issue #248 -- critical values come from a Monte Carlo simulation
    keyed on (window_ratio, alpha). Different ratios produce different
    quantiles in the published [2, 4] band; alpha tightens the quantile.
    """

    from macroforecast.core.runtime import _gr_critical_value

    cv_005_at_25 = _gr_critical_value(0.25, 0.05)
    cv_010_at_25 = _gr_critical_value(0.25, 0.10)
    cv_005_at_10 = _gr_critical_value(0.10, 0.05)
    # 10% rejection threshold must be lower than 5% at the same ratio.
    assert cv_010_at_25 < cv_005_at_25
    # Different ratios -> different critical values.
    assert cv_005_at_25 != cv_005_at_10
    # All values lie in the published [2, 4.5] band for typical alpha.
    for cv in (cv_005_at_25, cv_010_at_25, cv_005_at_10):
        assert 2.0 < cv < 4.5


def test_gr_critical_value_cached_returns_same_value():
    """Cache hit returns identical (no random redraw)."""

    from macroforecast.core.runtime import _gr_critical_value

    a = _gr_critical_value(0.25, 0.05)
    b = _gr_critical_value(0.25, 0.05)
    assert a == b


def test_rossi_sekhposyan_recursive_variant_runs():
    rng = np.random.default_rng(0)
    n = 60
    a_loss = rng.uniform(0.5, 1.5, n)
    b_loss = a_loss + rng.normal(scale=0.1, size=n)
    errors = _build_errors_panel(a_loss, b_loss)
    result = _l6_cpa_results(
        errors,
        sub={"cpa_test": "rossi_sekhposyan", "cpa_window_ratio": 0.2, "cpa_alpha": 0.05},
        l4_models=_l4_models(),
    )
    key = ("rossi_sekhposyan", ("b", "a"), "y", 1)
    assert key in result
    assert result[key]["statistic"] is not None
