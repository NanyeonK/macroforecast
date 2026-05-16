"""Cycle 15.6: Unit tests for BIC K_max intractability warning in _bic_select_k."""
import warnings
import numpy as np
import pandas as pd
import pytest

from macroforecast.core.runtime import _bic_select_k


def _make_inputs(freq_ratio: int, n_obs_lf: int = 40):
    """Construct minimal valid frame_hf + y_lf for _bic_select_k."""
    rng = np.random.default_rng(42)
    # High-freq index: n_obs_lf * freq_ratio observations
    n_hf = n_obs_lf * freq_ratio
    hf_idx = pd.date_range("2000-01-01", periods=n_hf, freq="ME")
    frame_hf = pd.DataFrame({"x": rng.standard_normal(n_hf)}, index=hf_idx)

    # Low-freq index: one observation per freq_ratio high-freq periods
    lf_idx = hf_idx[freq_ratio - 1 :: freq_ratio]
    y_lf = pd.Series(rng.standard_normal(len(lf_idx)), index=lf_idx, name="y")
    return frame_hf, y_lf


def test_bic_kmax_warns_when_large_freq_ratio():
    """Cycle 15.6: _bic_select_k emits UserWarning when K_max > 30 (freq_ratio=22 → K_max=33)."""
    freq_ratio = 22  # K_max = ceil(1.5 * 22) = 33 > 30 → should warn
    frame_hf, y_lf = _make_inputs(freq_ratio=freq_ratio, n_obs_lf=80)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        _bic_select_k(frame_hf, y_lf, freq_ratio=freq_ratio, include_y_lag=False)
    bic_msgs = [
        w for w in ws
        if issubclass(w.category, UserWarning) and "BIC" in str(w.message)
    ]
    assert bic_msgs, (
        f"Expected UserWarning about BIC K_max but got: {[str(w.message) for w in ws]}"
    )
    assert "K_max=33" in str(bic_msgs[0].message), str(bic_msgs[0].message)


def test_bic_kmax_silent_below_threshold():
    """Cycle 15.6: _bic_select_k does NOT warn when K_max <= 30 (freq_ratio=3 → K_max=5)."""
    freq_ratio = 3  # K_max = ceil(1.5 * 3) = 5 <= 30 → no warn
    frame_hf, y_lf = _make_inputs(freq_ratio=freq_ratio, n_obs_lf=80)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        _bic_select_k(frame_hf, y_lf, freq_ratio=freq_ratio, include_y_lag=False)
    bic_msgs = [
        w for w in ws
        if issubclass(w.category, UserWarning) and "BIC" in str(w.message)
    ]
    assert not bic_msgs, (
        f"Unexpected BIC warning at small K_max: {[str(w.message) for w in bic_msgs]}"
    )


def test_bic_kmax_boundary_exactly_30():
    """Cycle 15.6: K_max == 30 (freq_ratio=20 → ceil(30.0)=30) does NOT warn."""
    freq_ratio = 20  # K_max = ceil(1.5 * 20) = 30 — at boundary, no warn
    frame_hf, y_lf = _make_inputs(freq_ratio=freq_ratio, n_obs_lf=80)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        _bic_select_k(frame_hf, y_lf, freq_ratio=freq_ratio, include_y_lag=False)
    bic_msgs = [
        w for w in ws
        if issubclass(w.category, UserWarning) and "BIC" in str(w.message)
    ]
    assert not bic_msgs, (
        f"Expected no BIC warning at K_max=30 boundary: {[str(w.message) for w in bic_msgs]}"
    )


def test_bic_kmax_boundary_just_over_30():
    """Cycle 15.6: K_max == 31 (freq_ratio=21 → ceil(31.5)=32) DOES warn."""
    freq_ratio = 21  # K_max = ceil(1.5 * 21) = ceil(31.5) = 32 > 30 → warn
    frame_hf, y_lf = _make_inputs(freq_ratio=freq_ratio, n_obs_lf=80)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        _bic_select_k(frame_hf, y_lf, freq_ratio=freq_ratio, include_y_lag=False)
    bic_msgs = [
        w for w in ws
        if issubclass(w.category, UserWarning) and "BIC" in str(w.message)
    ]
    assert bic_msgs, (
        f"Expected UserWarning at K_max=32 but got: {[str(w.message) for w in ws]}"
    )
