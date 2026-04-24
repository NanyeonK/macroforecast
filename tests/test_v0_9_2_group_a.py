"""v0.9.2 Group A batch 1: y-domain transform + train-window normalization.

Metrics are computed on the transformed scale (inverse-transform-to-raw-units
is deferred to v1.0+). This batch covers the 6 target-side axes flipped in
the v0.9.2 sweep.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macrocast.execution.build import (
    _apply_target_transform_and_normalization,
    _fit_target_normalization_for_window,
)
from macrocast.preprocessing.build import PreprocessContract


def _contract(**over) -> PreprocessContract:
    base = dict(
        target_transform_policy="raw_level",
        x_transform_policy="raw_level",
        tcode_policy="extra_preprocess_without_tcode",
        target_missing_policy="none",
        x_missing_policy="none",
        target_outlier_policy="none",
        x_outlier_policy="none",
        scaling_policy="none",
        dimensionality_reduction_policy="none",
        feature_selection_policy="none",
        preprocess_order="extra_only",
        preprocess_fit_scope="train_only",
        inverse_transform_policy="none",
        evaluation_scale="raw_level",
    )
    base.update(over)
    return PreprocessContract(**base)


def _series(n=24) -> pd.Series:
    return pd.Series(np.arange(1, n + 1, dtype=float), name="y")


def test_target_transform_difference_applied():
    s = _series()
    out = _apply_target_transform_and_normalization(s, _contract(target_transform="difference"))
    assert len(out) == len(s) - 1
    assert np.allclose(out.to_numpy(), np.ones(len(s) - 1))


def test_target_transform_log_applied():
    s = _series()
    out = _apply_target_transform_and_normalization(s, _contract(target_transform="log"))
    assert np.allclose(out.to_numpy(), np.log(s.to_numpy()))


def test_target_transform_log_difference_applied():
    s = _series()
    out = _apply_target_transform_and_normalization(s, _contract(target_transform="log_difference"))
    expected = np.diff(np.log(s.to_numpy()))
    assert np.allclose(out.to_numpy(), expected)


def test_target_transform_growth_rate_applied():
    s = _series()
    out = _apply_target_transform_and_normalization(s, _contract(target_transform="growth_rate"))
    expected = (s.iloc[1:].to_numpy() / s.iloc[:-1].to_numpy()) - 1.0
    assert np.allclose(out.to_numpy(), expected)


def test_target_transform_log_rejects_nonpositive():
    s = pd.Series([1.0, 0.0, -1.0])
    from macrocast.execution.errors import ExecutionError
    with pytest.raises(ExecutionError, match="strictly positive"):
        _apply_target_transform_and_normalization(s, _contract(target_transform="log"))


def test_target_normalization_zscore_fit_on_train_window():
    s = _series()
    out, state = _fit_target_normalization_for_window(s, _contract(target_normalization="zscore_train_only"))
    assert abs(out.mean()) < 1e-10
    assert abs(float(out.std(ddof=0)) - 1.0) < 1e-10
    assert state["fit_scope"] == "train_only"


def test_target_normalization_robust_zscore_fit_on_train_window():
    s = _series()
    out, state = _fit_target_normalization_for_window(s, _contract(target_normalization="robust_zscore"))
    # Median centred; MAD-scaled — median of out ≈ 0
    assert abs(out.median()) < 1e-10
    assert state["fit_scope"] == "train_only"


def test_level_default_is_identity():
    s = _series()
    out = _apply_target_transform_and_normalization(s, _contract())
    pd.testing.assert_series_equal(out, s.astype(float))


def test_target_transform_registry_statuses_are_operational():
    from macrocast.registry.build import _discover_axis_definitions

    defs = _discover_axis_definitions()

    def _status(axis, value):
        return next(e.status for e in defs[axis].entries if e.id == value)

    for v in ("difference", "log", "log_difference", "growth_rate"):
        assert _status("target_transform", v) == "operational"
    for v in ("zscore_train_only", "robust_zscore"):
        assert _status("target_normalization", v) == "operational"
