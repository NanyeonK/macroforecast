"""v0.91 interim: smoke tests for preprocessing planned-status promotions.

Each test exercises a newly-operational value through the X-side
preprocessing pipeline. The assertions are intentionally shallow (shape,
no-raise, semantic invariant) — the goal is to prove the runtime branch
exists and produces the contractually-expected output, not to validate
every downstream statistical property.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macrocast.execution.build import (
    _apply_missing_policy,
    _apply_outlier_policy,
    _apply_scaling_policy,
)
from macrocast.preprocessing.build import PreprocessContract


def _make_contract(**overrides) -> PreprocessContract:
    defaults = dict(
        target_transform_policy="raw_level",
        x_transform_policy="raw_level",
        tcode_policy="none",
        target_missing_policy="none",
        x_missing_policy="none",
        target_outlier_policy="none",
        x_outlier_policy="none",
        scaling_policy="none",
        dimensionality_reduction_policy="none",
        feature_selection_policy="none",
        preprocess_order="none",
        preprocess_fit_scope="train_only",
        inverse_transform_policy="none",
        evaluation_scale="raw_level",
    )
    defaults.update(overrides)
    return PreprocessContract(**defaults)


@pytest.fixture
def sample_frame():
    rng = np.random.default_rng(0)
    X_train = pd.DataFrame(rng.standard_normal((40, 4)), columns=list("abcd"))
    X_pred = pd.DataFrame(rng.standard_normal((5, 4)), columns=list("abcd"))
    return X_train, X_pred


def test_scaling_demean_only_centers_but_preserves_variance(sample_frame):
    X_train, X_pred = sample_frame
    c = _make_contract(scaling_policy="demean_only")
    Xt, Xp = _apply_scaling_policy(X_train, X_pred, c)
    assert np.allclose(Xt.mean().to_numpy(), 0.0, atol=1e-8)
    # variance of the training set should equal pre-scaling variance (up to float noise)
    assert np.allclose(Xt.std(ddof=1).to_numpy(), X_train.std(ddof=1).to_numpy(), atol=1e-8)


def test_scaling_unit_variance_only_scales_but_preserves_mean(sample_frame):
    X_train, X_pred = sample_frame
    c = _make_contract(scaling_policy="unit_variance_only")
    Xt, Xp = _apply_scaling_policy(X_train, X_pred, c)
    # sklearn StandardScaler uses ddof=0; test accordingly
    assert np.allclose(Xt.std(ddof=0).to_numpy(), 1.0, atol=1e-8)
    assert not np.allclose(Xt.mean().to_numpy(), 0.0, atol=1e-2)


def test_outlier_trim_clips_to_extreme_quantiles(sample_frame):
    X_train, X_pred = sample_frame
    X_train = X_train.copy()
    X_train.iloc[0, 0] = 10_000.0
    c = _make_contract(x_outlier_policy="trim")
    Xt, _ = _apply_outlier_policy(X_train, X_pred, c)
    assert Xt.iloc[0, 0] < 10_000.0


def test_outlier_mad_clip_clips_extremes(sample_frame):
    X_train, X_pred = sample_frame
    X_train = X_train.copy()
    X_train.iloc[0, 0] = 1_000.0
    c = _make_contract(x_outlier_policy="mad_clip")
    Xt, _ = _apply_outlier_policy(X_train, X_pred, c)
    assert Xt.iloc[0, 0] < 1_000.0


def test_outlier_to_missing_converts_extremes_to_nan(sample_frame):
    X_train, X_pred = sample_frame
    X_train = X_train.copy()
    X_train.iloc[0, 0] = 1_000.0
    c = _make_contract(x_outlier_policy="outlier_to_missing")
    Xt, _ = _apply_outlier_policy(X_train, X_pred, c)
    assert pd.isna(Xt.iloc[0, 0])


def test_missing_drop_columns_removes_columns_with_any_nan(sample_frame):
    X_train, X_pred = sample_frame
    X_train = X_train.copy()
    X_train.iloc[0, 0] = np.nan
    c = _make_contract(x_missing_policy="drop_columns")
    Xt, Xp = _apply_missing_policy(X_train, X_pred, c)
    assert "a" not in Xt.columns
    assert set(Xt.columns) == set(Xp.columns) == {"b", "c", "d"}


def test_missing_drop_if_above_threshold_keeps_low_missing(sample_frame):
    X_train, X_pred = sample_frame
    X_train = X_train.copy()
    X_train.iloc[:35, 0] = np.nan  # column 'a' >30% missing
    X_train.iloc[0, 1] = np.nan     # column 'b' <30% missing — keep
    c = _make_contract(x_missing_policy="drop_if_above_threshold")
    Xt, Xp = _apply_missing_policy(X_train, X_pred, c)
    assert "a" not in Xt.columns
    assert "b" in Xt.columns


def test_missing_indicator_appends_indicator_columns(sample_frame):
    X_train, X_pred = sample_frame
    X_train = X_train.copy()
    X_train.iloc[0, 0] = np.nan
    c = _make_contract(x_missing_policy="missing_indicator")
    Xt, Xp = _apply_missing_policy(X_train, X_pred, c)
    assert "a__missing" in Xt.columns
    assert Xt.loc[0, "a"] == 0.0  # NaN filled with 0
    assert Xt.loc[0, "a__missing"] == 1.0


def test_registry_promotions_are_operational():
    from macrocast.registry.build import _discover_axis_definitions

    defs = _discover_axis_definitions()

    def _status(axis: str, value: str) -> str:
        return next(e.status for e in defs[axis].entries if e.id == value)

    # scaling_policy
    assert _status("scaling_policy", "demean_only") == "operational"
    assert _status("scaling_policy", "unit_variance_only") == "operational"
    # x_outlier_policy
    assert _status("x_outlier_policy", "trim") == "operational"
    assert _status("x_outlier_policy", "mad_clip") == "operational"
    assert _status("x_outlier_policy", "outlier_to_missing") == "operational"
    # x_missing_policy
    for v in ("drop_rows", "drop_columns", "drop_if_above_threshold", "missing_indicator"):
        assert _status("x_missing_policy", v) == "operational", f"{v} not flipped"
