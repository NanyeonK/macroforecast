"""Tests for _enforce_fred_sd_frequency_policy (Cycle 17.5 LOW-A2 fix).

These tests call the helper directly with synthetic series_frequencies to
avoid the overhead of a full mf.run integration path.  The helper is a
module-level function in macroforecast.core.runtime.
"""

import pytest

from macroforecast.core.runtime import _enforce_fred_sd_frequency_policy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_freq_map(**kwargs):
    """Return a simple {col: freq_str} dict from keyword args."""
    return dict(kwargs)


# ---------------------------------------------------------------------------
# Test 1: reject_mixed_known_frequency raises when two frequencies present
# ---------------------------------------------------------------------------

def test_reject_mixed_known_frequency_raises():
    freq_map = _make_freq_map(A="monthly", B="quarterly", C="monthly")
    cols = ["A", "B", "C"]
    with pytest.raises(ValueError, match="reject_mixed_known_frequency"):
        _enforce_fred_sd_frequency_policy(
            "reject_mixed_known_frequency", freq_map, cols
        )


# ---------------------------------------------------------------------------
# Test 2: reject_mixed_known_frequency passes when all columns are the same freq
# ---------------------------------------------------------------------------

def test_reject_mixed_known_frequency_single_freq_ok():
    freq_map = _make_freq_map(A="monthly", B="monthly")
    cols = ["A", "B"]
    # Must not raise
    _enforce_fred_sd_frequency_policy(
        "reject_mixed_known_frequency", freq_map, cols
    )


# ---------------------------------------------------------------------------
# Test 3: reject_mixed_known_frequency passes when unknown columns present
#         (unknown freqs are ignored by this policy -- only known freqs matter)
# ---------------------------------------------------------------------------

def test_reject_mixed_known_frequency_unknown_ignored():
    freq_map = _make_freq_map(A="monthly", B="")  # B has no known freq
    cols = ["A", "B"]
    # Only one *known* frequency (monthly), so must not raise
    _enforce_fred_sd_frequency_policy(
        "reject_mixed_known_frequency", freq_map, cols
    )


# ---------------------------------------------------------------------------
# Test 4: require_single_known_frequency raises on unknown column
# ---------------------------------------------------------------------------

def test_require_single_known_frequency_raises_on_unknown():
    freq_map = _make_freq_map(A="monthly", B="")  # B has unknown freq
    cols = ["A", "B"]
    with pytest.raises(ValueError, match="require_single_known_frequency"):
        _enforce_fred_sd_frequency_policy(
            "require_single_known_frequency", freq_map, cols
        )


# ---------------------------------------------------------------------------
# Test 5: require_single_known_frequency raises on mixed known frequencies
# ---------------------------------------------------------------------------

def test_require_single_known_frequency_raises_on_mixed():
    freq_map = _make_freq_map(A="monthly", B="quarterly")
    cols = ["A", "B"]
    with pytest.raises(ValueError, match="require_single_known_frequency"):
        _enforce_fred_sd_frequency_policy(
            "require_single_known_frequency", freq_map, cols
        )


# ---------------------------------------------------------------------------
# Test 6: require_single_known_frequency passes when single freq + no unknowns
# ---------------------------------------------------------------------------

def test_require_single_known_frequency_ok():
    freq_map = _make_freq_map(A="quarterly", B="quarterly", C="quarterly")
    cols = ["A", "B", "C"]
    _enforce_fred_sd_frequency_policy(
        "require_single_known_frequency", freq_map, cols
    )


# ---------------------------------------------------------------------------
# Test 7: allow_mixed_frequency never raises regardless of content
# ---------------------------------------------------------------------------

def test_allow_mixed_frequency_no_raise():
    freq_map = _make_freq_map(A="monthly", B="quarterly", C="", D="annual")
    cols = ["A", "B", "C", "D"]
    _enforce_fred_sd_frequency_policy(
        "allow_mixed_frequency", freq_map, cols
    )


# ---------------------------------------------------------------------------
# Test 8: report_only never raises regardless of content
# ---------------------------------------------------------------------------

def test_report_only_no_raise():
    freq_map = _make_freq_map(A="monthly", B="quarterly", C="")
    cols = ["A", "B", "C"]
    _enforce_fred_sd_frequency_policy(
        "report_only", freq_map, cols
    )
