"""Regression tests for the TGT-7 and DL-1 usage guards.

TGT-7: relying on the change-based default target_transform (which differences
the target) must warn, because on an already-transformed panel it double-
differences. Passing an explicit value-based transform must not warn.

DL-1: load_fred_*(vintage=None) resolves to revised current.csv and must warn so
a pseudo-out-of-sample replication does not silently use revised data.
"""
from __future__ import annotations

import warnings

import pytest

from macroforecast.forecasting.runner import _target_transform_for_policy
from macroforecast.data.loaders import _version_request


def test_tgt7_change_based_default_warns_on_stationary_panel():
    # Already-stationary features ('change') + change-based default target =
    # double-difference risk -> must warn.
    with pytest.warns(UserWarning, match="double-differences"):
        out = _target_transform_for_policy(
            "direct_average", feature_transform="change", explicit=None
        )
    assert out == "average_change"


def test_tgt7_no_warning_on_raw_or_level_panel():
    # Raw / level panels: change-based default is CORRECT, no false-positive warn.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _target_transform_for_policy(
            "direct_average", feature_transform=None, explicit=None
        ) == "average_change"
        assert _target_transform_for_policy(
            "path_average", feature_transform="level", explicit=None
        ) == "change"


def test_tgt7_explicit_value_transform_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = _target_transform_for_policy(
            "direct_average", feature_transform=None, explicit="average_value"
        )
    assert out == "average_value"


def test_dl1_vintage_none_warns():
    with pytest.warns(UserWarning, match="revised data"):
        req = _version_request("fred_md", vintage=None)
    assert req.mode == "current"


def test_dl1_explicit_vintage_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        req = _version_request("fred_md", vintage="2018-01")
    assert req.mode == "vintage"
