"""v0.9.2 Group B: preprocessing contract relaxations + runtime branches.

This batch adds two genuine runtime implementations. More Group-B axes
(tcode_application_scope, representation_policy:tcode_only, cv_select_lags)
require deeper infrastructure and land in a follow-up batch.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from macrocast.execution.build import (
    _apply_additional_preprocessing,
    _apply_tcode_preprocessing,
    _apply_x_lag_creation,
    _build_raw_panel_training_data,
)
from macrocast.preprocessing.build import PreprocessContract


def _contract(**overrides) -> PreprocessContract:
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
    base.update(overrides)
    return PreprocessContract(**base)


def test_fixed_x_lags_adds_lag1_columns():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [10.0, 20.0, 30.0, 40.0, 50.0]})
    Xp = pd.DataFrame({"a": [6.0], "b": [60.0]})
    c = _contract(x_lag_creation="fixed_x_lags")
    Xt, Xp2 = _apply_x_lag_creation(X, Xp, c)
    assert {"a", "b", "a__lag1", "b__lag1"}.issubset(set(Xt.columns))
    assert Xt["a__lag1"].iloc[1] == 1.0
    assert Xt["a__lag1"].iloc[0] == 0.0  # filled NaN
    assert Xt.shape == (5, 4)
    assert Xp2.shape == (1, 4)


def test_raw_panel_fixed_x_lag_prediction_uses_origin_history():
    frame = pd.DataFrame(
        {
            "target": [10.0, 11.0, 12.0, 13.0, 14.0],
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    c = _contract(x_lag_creation="fixed_x_lags")

    X_train, y_train, X_pred = _build_raw_panel_training_data(
        frame,
        "target",
        horizon=1,
        start_idx=0,
        origin_idx=3,
        contract=c,
        predictor_family="all_macro_vars",
    )

    assert X_train.tolist() == [[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]]
    assert y_train.tolist() == [11.0, 12.0, 13.0]
    assert X_pred.tolist() == [[4.0, 3.0]]


def test_raw_panel_target_level_addback_uses_origin_target_history():
    frame = pd.DataFrame(
        {
            "target": [10.0, 11.0, 12.0, 13.0, 14.0],
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    c = _contract()

    X_train, y_train, X_pred = _build_raw_panel_training_data(
        frame,
        "target",
        horizon=1,
        start_idx=0,
        origin_idx=3,
        contract=c,
        predictor_family="all_macro_vars",
        level_feature_block="target_level_addback",
    )

    assert X_train.tolist() == [[1.0, 10.0], [2.0, 11.0], [3.0, 12.0]]
    assert y_train.tolist() == [11.0, 12.0, 13.0]
    assert X_pred.tolist() == [[4.0, 13.0]]


def test_raw_panel_x_level_addback_uses_preserved_level_source():
    frame = pd.DataFrame(
        {
            "target": [10.0, 11.0, 12.0, 13.0, 14.0],
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    frame.attrs["macrocast_level_source_frame"] = pd.DataFrame(
        {
            "target": [100.0, 110.0, 120.0, 130.0, 140.0],
            "a": [101.0, 102.0, 103.0, 104.0, 105.0],
        }
    )
    c = _contract()

    X_train, y_train, X_pred = _build_raw_panel_training_data(
        frame,
        "target",
        horizon=1,
        start_idx=0,
        origin_idx=3,
        contract=c,
        predictor_family="all_macro_vars",
        level_feature_block="x_level_addback",
    )

    assert X_train.tolist() == [[1.0, 101.0], [2.0, 102.0], [3.0, 103.0]]
    assert y_train.tolist() == [11.0, 12.0, 13.0]
    assert X_pred.tolist() == [[4.0, 104.0]]


def test_raw_panel_selected_level_addbacks_use_requested_level_columns():
    frame = pd.DataFrame(
        {
            "target": [10.0, 11.0, 12.0, 13.0, 14.0],
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [6.0, 7.0, 8.0, 9.0, 10.0],
        }
    )
    frame.attrs["macrocast_level_source_frame"] = pd.DataFrame(
        {
            "target": [100.0, 110.0, 120.0, 130.0, 140.0],
            "a": [101.0, 102.0, 103.0, 104.0, 105.0],
            "b": [201.0, 202.0, 203.0, 204.0, 205.0],
        }
    )
    c = _contract()

    X_train, y_train, X_pred = _build_raw_panel_training_data(
        frame,
        "target",
        horizon=1,
        start_idx=0,
        origin_idx=3,
        contract=c,
        predictor_family="all_macro_vars",
        level_feature_block="selected_level_addbacks",
        spec={"selected_level_addback_columns": ["b"]},
    )

    assert X_train.tolist() == [[1.0, 6.0, 201.0], [2.0, 7.0, 202.0], [3.0, 8.0, 203.0]]
    assert y_train.tolist() == [11.0, 12.0, 13.0]
    assert X_pred.tolist() == [[4.0, 9.0, 204.0]]


def test_raw_panel_level_growth_pairs_use_requested_level_counterparts():
    frame = pd.DataFrame(
        {
            "target": [10.0, 11.0, 12.0, 13.0, 14.0],
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [6.0, 7.0, 8.0, 9.0, 10.0],
        }
    )
    frame.attrs["macrocast_level_source_frame"] = pd.DataFrame(
        {
            "target": [100.0, 110.0, 120.0, 130.0, 140.0],
            "a": [101.0, 102.0, 103.0, 104.0, 105.0],
            "b": [201.0, 202.0, 203.0, 204.0, 205.0],
        }
    )
    c = _contract()

    X_train, y_train, X_pred = _build_raw_panel_training_data(
        frame,
        "target",
        horizon=1,
        start_idx=0,
        origin_idx=3,
        contract=c,
        predictor_family="all_macro_vars",
        level_feature_block="level_growth_pairs",
        spec={"level_growth_pair_columns": ["a"]},
    )

    assert X_train.tolist() == [[1.0, 6.0, 101.0], [2.0, 7.0, 102.0], [3.0, 8.0, 103.0]]
    assert y_train.tolist() == [11.0, 12.0, 13.0]
    assert X_pred.tolist() == [[4.0, 9.0, 104.0]]


def test_tcode_preprocessing_preserves_pre_tcode_level_source():
    frame = pd.DataFrame(
        {
            "target": [10.0, 11.0, 12.0],
            "a": [1.0, 2.0, 4.0],
        }
    )
    raw_result = SimpleNamespace(data=frame, transform_codes={"target": 1, "a": 2})
    recipe = SimpleNamespace(
        data_task_spec={
            "official_transform_policy": "dataset_tcode",
            "official_transform_scope": "apply_tcode_to_X",
        }
    )

    transformed = _apply_tcode_preprocessing(raw_result, recipe, _contract(), target="target")

    values = transformed.data["a"].tolist()
    assert np.isnan(values[0])
    assert values[1:] == [1.0, 2.0]
    source = transformed.data.attrs["macrocast_level_source_frame"]
    assert source["a"].tolist() == [1.0, 2.0, 4.0]


def test_raw_panel_moving_average_features_use_trailing_origin_history():
    frame = pd.DataFrame(
        {
            "target": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    c = _contract()

    X_train, y_train, X_pred = _build_raw_panel_training_data(
        frame,
        "target",
        horizon=1,
        start_idx=0,
        origin_idx=4,
        contract=c,
        predictor_family="all_macro_vars",
        temporal_feature_block="moving_average_features",
    )

    assert X_train.tolist() == [[1.0, 1.0], [2.0, 1.5], [3.0, 2.0], [4.0, 3.0]]
    assert y_train.tolist() == [11.0, 12.0, 13.0, 14.0]
    assert X_pred.tolist() == [[5.0, 4.0]]


def test_raw_panel_volatility_features_use_trailing_origin_history():
    frame = pd.DataFrame(
        {
            "target": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    c = _contract()

    X_train, y_train, X_pred = _build_raw_panel_training_data(
        frame,
        "target",
        horizon=1,
        start_idx=0,
        origin_idx=4,
        contract=c,
        predictor_family="all_macro_vars",
        temporal_feature_block="volatility_features",
    )

    trailing_vol = float(np.std([1.0, 2.0, 3.0], ddof=0))
    assert np.allclose(
        X_train,
        [[1.0, 0.0], [2.0, 0.5], [3.0, trailing_vol], [4.0, trailing_vol]],
    )
    assert y_train.tolist() == [11.0, 12.0, 13.0, 14.0]
    assert np.allclose(X_pred, [[5.0, trailing_vol]])


def test_raw_panel_rolling_moments_features_use_trailing_origin_history():
    frame = pd.DataFrame(
        {
            "target": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    c = _contract()

    X_train, y_train, X_pred = _build_raw_panel_training_data(
        frame,
        "target",
        horizon=1,
        start_idx=0,
        origin_idx=4,
        contract=c,
        predictor_family="all_macro_vars",
        temporal_feature_block="rolling_moments",
    )

    trailing_var = float(np.var([1.0, 2.0, 3.0], ddof=0))
    assert np.allclose(
        X_train,
        [
            [1.0, 1.0, 0.0],
            [2.0, 1.5, 0.25],
            [3.0, 2.0, trailing_var],
            [4.0, 3.0, trailing_var],
        ],
    )
    assert y_train.tolist() == [11.0, 12.0, 13.0, 14.0]
    assert np.allclose(X_pred, [[5.0, 4.0, trailing_var]])


def test_no_x_lags_is_identity():
    X = pd.DataFrame({"a": [1.0, 2.0]})
    Xp = pd.DataFrame({"a": [3.0]})
    c = _contract()
    Xt, Xp2 = _apply_x_lag_creation(X, Xp, c)
    assert Xt.equals(X)
    assert Xp2.equals(Xp)


def test_hp_filter_shifts_column_mean_toward_zero():
    rng = np.random.default_rng(0)
    # Construct a series with a strong trend — HP filter removes trend
    trend = np.linspace(0, 100, 60)
    noise = rng.standard_normal(60) * 0.5
    X = pd.DataFrame({"a": trend + noise, "b": trend * 0.5 + noise})
    Xp = pd.DataFrame({"a": [50.0], "b": [25.0]})
    c = _contract(additional_preprocessing="hp_filter")
    Xt, _ = _apply_additional_preprocessing(X, Xp, c)
    # cycle component has mean near zero (vs original ~50)
    assert abs(Xt["a"].mean()) < 1.0
    assert abs(X["a"].mean()) > 40.0  # sanity: original had big mean


def test_hp_filter_none_is_identity():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    Xp = pd.DataFrame({"a": [4.0]})
    c = _contract()
    Xt, Xp2 = _apply_additional_preprocessing(X, Xp, c)
    assert Xt.equals(X)
    assert Xp2.equals(Xp)


def test_registry_promotions_are_operational():
    from macrocast.registry.build import _discover_axis_definitions

    defs = _discover_axis_definitions()

    def _status(axis, value):
        return next(e.status for e in defs[axis].entries if e.id == value)

    assert _status("additional_preprocessing", "hp_filter") == "operational"
    assert _status("x_lag_creation", "fixed_x_lags") == "operational"


def test_contract_accepts_hp_filter():
    from macrocast.preprocessing.build import is_operational_preprocess_contract

    c = _contract(additional_preprocessing="hp_filter")
    assert is_operational_preprocess_contract(c)


def test_contract_accepts_fixed_x_lags():
    from macrocast.preprocessing.build import is_operational_preprocess_contract

    c = _contract(x_lag_creation="fixed_x_lags")
    assert is_operational_preprocess_contract(c)
