"""Regression tests for v0.1.1 hot-patch fixes.

These pin down two bugs raised in the PR #163 codex review that escaped
the original v0.1 merge:

* P1 -- ``set`` iteration order in ``_stable_repr`` was non-deterministic
  across Python processes (depends on ``PYTHONHASHSEED``), which broke the
  bit-exact replicate guarantee for any sink that surfaced a ``set``-valued
  field.
* P2 -- ``_build_nber_regime_series`` used a hard-coded ``"-28"`` suffix to
  build the recession-end timestamp, which silently truncated the last 1-3
  days of every 30/31-day month and mislabeled them as expansion.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from macrocast.core.execution import _stable_repr, _hash_sink
from macrocast.core.runtime import _build_nber_regime_series, _NBER_RECESSIONS


# ---------------------------------------------------------------------------
# P1 -- set hashing determinism
# ---------------------------------------------------------------------------

def test_stable_repr_sorts_set_values_for_deterministic_hashing():
    payload_a = {"models": {"ridge", "ar_p", "xgboost", "random_forest"}}
    payload_b = {"models": {"random_forest", "xgboost", "ar_p", "ridge"}}
    assert _stable_repr(payload_a) == _stable_repr(payload_b)


def test_stable_repr_sorts_frozenset_values():
    payload_a = {"included": frozenset(["a", "b", "c"])}
    payload_b = {"included": frozenset(["c", "b", "a"])}
    assert _stable_repr(payload_a) == _stable_repr(payload_b)


def test_hash_sink_with_set_payload_is_stable_within_process():
    payload = {"selected": {"ridge", "lasso", "xgboost"}, "scalar": 1.5}
    h1 = _hash_sink(payload)
    h2 = _hash_sink(payload)
    assert h1 == h2


def test_hash_sink_with_set_payload_is_stable_across_processes(tmp_path):
    """Spawn two child processes with different PYTHONHASHSEED values and
    verify they produce identical sink hashes for an identical set payload.

    Before the v0.1.1 fix, set iteration order varied with PYTHONHASHSEED
    so the two children would print different hashes.
    """

    script = textwrap.dedent(
        """
        import sys, json
        sys.path.insert(0, %r)
        from macrocast.core.execution import _hash_sink
        # A set big enough that hash-seed-driven iteration order matters.
        payload = {"models": {"ridge", "lasso", "ar_p", "xgboost", "random_forest", "knn", "huber", "ols"},
                   "horizons": {1, 3, 6, 12}}
        print(_hash_sink(payload))
        """
    ) % str(Path(__file__).resolve().parent.parent.parent)
    script_path = tmp_path / "hash_check.py"
    script_path.write_text(script)

    hashes = []
    for seed in ("0", "1", "12345", "67890"):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = seed
        out = subprocess.check_output([sys.executable, str(script_path)], env=env, text=True)
        hashes.append(out.strip())
    assert len(set(hashes)) == 1, f"hash drifted across PYTHONHASHSEED values: {hashes}"


# ---------------------------------------------------------------------------
# P2 -- NBER end-of-month boundary
# ---------------------------------------------------------------------------

def test_nber_regime_includes_30th_and_31st_of_recession_end_month():
    """A daily index spanning 2009-06-25..2009-07-05 must label every June
    day as recession (NBER 2009 recession ended in 2009-06 inclusive)."""

    index = pd.date_range("2009-06-25", "2009-07-05", freq="D")
    labels = _build_nber_regime_series(index)
    june_days = labels.loc[index <= "2009-06-30"]
    july_days = labels.loc[index >= "2009-07-01"]
    assert (june_days == "recession").all(), june_days.to_dict()
    assert (july_days == "expansion").all(), july_days.to_dict()


def test_nber_regime_handles_30day_end_months_too():
    """The 1990-1991 recession ends in 1991-03 (31 days). The 1980 recession
    ends in 1980-07 (31 days). Verify the last day of each end month is
    still recession.
    """

    cases = [("1991-03-30", "recession"), ("1991-03-31", "recession"), ("1991-04-01", "expansion"),
             ("1980-07-30", "recession"), ("1980-07-31", "recession"), ("1980-08-01", "expansion")]
    index = pd.DatetimeIndex([c[0] for c in cases])
    labels = _build_nber_regime_series(index)
    for date_str, expected in cases:
        assert labels.loc[pd.Timestamp(date_str)] == expected, (date_str, expected, labels.loc[pd.Timestamp(date_str)])


def test_nber_regime_monthly_panel_unchanged():
    """Monthly first-of-month indices should produce the same labels before
    and after the fix (they were never affected by the -28 truncation).
    """

    index = pd.date_range("2007-01-01", "2010-12-01", freq="MS")
    labels = _build_nber_regime_series(index)
    # NBER: 2007-12 to 2009-06 inclusive => Dec 2007, Jan-Dec 2008, Jan-Jun 2009 = 1 + 12 + 6 = 19 months
    assert int((labels == "recession").sum()) == 19
