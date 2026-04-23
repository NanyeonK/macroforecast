"""v0.9.2 Group B: preprocessing contract relaxations + runtime branches.

This batch adds two genuine runtime implementations. More Group-B axes
(tcode_application_scope, representation_policy:tcode_only, cv_select_lags)
require deeper infrastructure and land in a follow-up batch.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from macrocast import clear_custom_extensions, target_transformer
from macrocast.execution.build import (
    _apply_additional_preprocessing,
    _apply_tcode_preprocessing,
    _apply_x_lag_creation,
    _build_predictions,
    _build_lagged_supervised_matrix,
    _build_raw_panel_training_data,
    _compute_minimal_importance,
    _feature_runtime_builder,
    _importance_feature_names,
    _lag_order,
    _model_spec,
    _raw_panel_feature_names,
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


def _dispatch_recipe(*, model_family: str, feature_recipes=(), blocks: dict | None = None):
    return SimpleNamespace(
        stage0=SimpleNamespace(
            fixed_design=SimpleNamespace(sample_split="rolling_window_oos", benchmark="zero_change"),
            varying_design=SimpleNamespace(
                model_families=(model_family,),
                feature_recipes=tuple(feature_recipes),
            ),
        ),
        benchmark_config={},
        data_task_spec={},
        training_spec={},
        layer2_representation_spec={"feature_blocks": dict(blocks or {})},
        target="target",
        horizons=(1,),
    )


def test_layer2_blocks_drive_raw_panel_runtime_dispatch_without_legacy_feature_recipe():
    recipe = _dispatch_recipe(
        model_family="ridge",
        blocks={
            "feature_block_set": {"value": "high_dimensional_x"},
            "x_lag_feature_block": {"value": "fixed_x_lags"},
        },
    )

    spec = _model_spec(recipe)

    assert _feature_runtime_builder(recipe) == "raw_feature_panel"
    assert spec["feature_builder"] == "autoreg_lagged_target"
    assert spec["feature_runtime_builder"] == "raw_feature_panel"
    assert spec["feature_runtime"] == "raw_panel_v1"
    assert spec["executor_name"] == "ridge_raw_feature_panel_v0"


def test_layer2_target_lag_block_drives_autoreg_runtime_dispatch():
    recipe = _dispatch_recipe(
        model_family="ar",
        blocks={
            "feature_block_set": {"value": "target_lags_only"},
            "target_lag_block": {"value": "fixed_target_lags"},
        },
    )

    spec = _model_spec(recipe)

    assert _feature_runtime_builder(recipe) == "autoreg_lagged_target"
    assert spec["feature_runtime_builder"] == "autoreg_lagged_target"
    assert spec["feature_runtime"] == "autoreg_lagged_target_v1"
    assert spec["executor_name"] == "ar_bic_autoreg_v0"


def test_target_transformer_gate_uses_layer2_feature_runtime_not_legacy_bridge():
    clear_custom_extensions()

    @target_transformer("identity_target_runtime_gate")
    class IdentityTargetRuntimeGate:
        def fit(self, target_train, context):
            return self

        def transform(self, target, context):
            return target

        def inverse_transform_prediction(self, target_pred, context):
            return target_pred

    frame = pd.DataFrame(
        {
            "target": [1.0, 1.2, 1.4, 1.8, 2.1, 2.4, 2.7, 3.0],
            "a": [2.0, 2.1, 2.2, 2.5, 2.8, 3.1, 3.2, 3.3],
        },
        index=pd.date_range("2000-01-01", periods=8, freq="MS"),
    )
    recipe = _dispatch_recipe(
        model_family="ridge",
        feature_recipes=("factor_pca",),
        blocks={
            "feature_block_set": {"value": "high_dimensional_x"},
            "x_lag_feature_block": {"value": "fixed_x_lags"},
        },
    )
    recipe.training_spec = {"target_transformer": "identity_target_runtime_gate"}
    recipe.benchmark_config = {"minimum_train_size": 5, "rolling_window_size": 5}
    recipe.data_task_spec = {"predictor_family": "all_macro_vars"}

    predictions, _ = _build_predictions(
        frame,
        frame["target"],
        recipe,
        _contract(),
    )

    assert _feature_runtime_builder(recipe) == "raw_feature_panel"
    assert set(predictions["target_transformer"]) == {"identity_target_runtime_gate"}
    assert set(predictions["model_target_scale"]) == {"transformed"}


def test_target_lag_block_lag_order_matches_legacy_max_ar_lag():
    train = pd.Series([1.0, 1.2, 1.1, 1.5, 1.7, 1.8])
    legacy = _dispatch_recipe(model_family="ridge", feature_recipes=("autoreg_lagged_target",))
    legacy.benchmark_config = {"max_ar_lag": 2}
    explicit = _dispatch_recipe(
        model_family="ridge",
        blocks={
            "feature_block_set": {"value": "target_lags_only"},
            "target_lag_block": {
                "value": "fixed_target_lags",
                "lag_orders": [1, 2],
                "feature_names": ["target_lag_1", "target_lag_2"],
                "runtime_block": {
                    "matrix_composition": "fixed_target_lags",
                    "lag_count": 2,
                },
            },
        },
    )
    explicit.benchmark_config = {"max_ar_lag": 5}

    legacy_lag_order = _lag_order(legacy, train)
    explicit_lag_order = _lag_order(explicit, train)
    legacy_X, legacy_y = _build_lagged_supervised_matrix(train, legacy_lag_order)
    explicit_X, explicit_y = _build_lagged_supervised_matrix(train, explicit_lag_order)

    assert legacy_lag_order == 2
    assert explicit_lag_order == 2
    assert np.allclose(legacy_X, explicit_X)
    assert np.allclose(legacy_y, explicit_y)


def test_importance_feature_names_follow_runtime_feature_blocks():
    frame = pd.DataFrame(
        {
            "target": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )
    recipe = _dispatch_recipe(
        model_family="ridge",
        feature_recipes=("autoreg_lagged_target",),
        blocks={
            "feature_block_set": {"value": "high_dimensional_x"},
            "x_lag_feature_block": {"value": "fixed_x_lags"},
        },
    )
    recipe.benchmark_config = {"minimum_train_size": 5, "rolling_window_size": 5}

    feature_names, X_train, y_train, X_pred = _importance_feature_names(
        recipe,
        frame,
        frame["target"],
        _contract(),
    )

    assert feature_names == ["a", "a_lag_1"]
    assert X_train.shape[1] == 2
    assert len(y_train) == X_train.shape[0]
    assert X_pred.shape == (1, 2)


def test_minimal_importance_uses_runtime_feature_builder_metadata():
    frame = pd.DataFrame(
        {
            "target": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )
    recipe = _dispatch_recipe(
        model_family="ridge",
        feature_recipes=("autoreg_lagged_target",),
        blocks={
            "feature_block_set": {"value": "high_dimensional_x"},
            "x_lag_feature_block": {"value": "fixed_x_lags"},
        },
    )
    recipe.benchmark_config = {"minimum_train_size": 5, "rolling_window_size": 5}

    payload = _compute_minimal_importance(recipe=recipe, raw_frame=frame, target_series=frame["target"], contract=_contract())

    assert payload["feature_builder"] == "raw_feature_panel"
    assert payload["feature_runtime_builder"] == "raw_feature_panel"
    assert payload["legacy_feature_builder"] == "autoreg_lagged_target"
    assert payload["feature_dispatch_source"] == "layer2_feature_blocks"
    assert {item["feature"] for item in payload["feature_importance"]} == {"a", "a_lag_1"}


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


def test_raw_panel_x_lag_feature_block_matches_legacy_bridge():
    frame = pd.DataFrame(
        {
            "target": [10.0, 11.0, 12.0, 13.0, 14.0],
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    legacy = _build_raw_panel_training_data(
        frame,
        "target",
        horizon=1,
        start_idx=0,
        origin_idx=3,
        contract=_contract(x_lag_creation="fixed_x_lags"),
        predictor_family="all_macro_vars",
    )
    explicit = _build_raw_panel_training_data(
        frame,
        "target",
        horizon=1,
        start_idx=0,
        origin_idx=3,
        contract=_contract(x_lag_creation="no_x_lags"),
        predictor_family="all_macro_vars",
        x_lag_feature_block="fixed_x_lags",
    )

    for legacy_arr, explicit_arr in zip(legacy, explicit):
        assert np.allclose(legacy_arr, explicit_arr)


def test_raw_panel_factor_feature_block_matches_legacy_dimred_bridge():
    frame = pd.DataFrame(
        {
            "target": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "b": [2.0, 3.0, 5.0, 7.0, 11.0, 13.0],
            "c": [1.5, 1.0, 2.5, 3.0, 3.5, 5.0],
        }
    )
    legacy_fit_state: list[dict[str, object]] = []
    explicit_fit_state: list[dict[str, object]] = []

    legacy = _build_raw_panel_training_data(
        frame,
        "target",
        horizon=1,
        start_idx=0,
        origin_idx=4,
        contract=_contract(dimensionality_reduction_policy="pca"),
        predictor_family="all_macro_vars",
        fit_state_sink=legacy_fit_state,
    )
    explicit = _build_raw_panel_training_data(
        frame,
        "target",
        horizon=1,
        start_idx=0,
        origin_idx=4,
        contract=_contract(dimensionality_reduction_policy="none"),
        predictor_family="all_macro_vars",
        fit_state_sink=explicit_fit_state,
        factor_feature_block="pca_static_factors",
    )

    for legacy_arr, explicit_arr in zip(legacy, explicit):
        assert np.allclose(legacy_arr, explicit_arr)
    assert legacy_fit_state[-1]["runtime_policy"] == "pca"
    assert explicit_fit_state[-1]["runtime_policy"] == "pca"


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


def test_raw_panel_moving_average_rotation_uses_trailing_origin_history():
    frame = pd.DataFrame(
        {
            "target": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "b": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
        }
    )
    c = _contract()

    X_train, y_train, X_pred = _build_raw_panel_training_data(
        frame,
        "target",
        horizon=1,
        start_idx=0,
        origin_idx=6,
        contract=c,
        predictor_family="all_macro_vars",
        rotation_feature_block="moving_average_rotation",
    )

    assert np.allclose(
        X_train,
        [
            [1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
            [2.0, 4.0, 1.5, 3.0, 1.5, 3.0],
            [3.0, 6.0, 2.0, 4.0, 2.0, 4.0],
            [4.0, 8.0, 3.0, 6.0, 2.5, 5.0],
            [5.0, 10.0, 4.0, 8.0, 3.0, 6.0],
            [6.0, 12.0, 5.0, 10.0, 3.5, 7.0],
        ],
    )
    assert y_train.tolist() == [11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
    assert np.allclose(X_pred, [[7.0, 14.0, 6.0, 12.0, 4.5, 9.0]])


def test_raw_panel_append_blocks_compose_with_fixed_x_lags():
    frame = pd.DataFrame(
        {
            "target": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    c = _contract(x_lag_creation="fixed_x_lags")

    X_train, y_train, X_pred = _build_raw_panel_training_data(
        frame,
        "target",
        horizon=1,
        start_idx=0,
        origin_idx=4,
        contract=c,
        predictor_family="all_macro_vars",
        temporal_feature_block="moving_average_features",
        rotation_feature_block="moving_average_rotation",
    )

    assert np.allclose(
        X_train,
        [
            [1.0, 0.0, 1.0, 1.0, 1.0],
            [2.0, 1.0, 1.5, 1.5, 1.5],
            [3.0, 2.0, 2.0, 2.0, 2.0],
            [4.0, 3.0, 3.0, 3.0, 2.5],
        ],
    )
    assert y_train.tolist() == [11.0, 12.0, 13.0, 14.0]
    assert np.allclose(X_pred, [[5.0, 4.0, 4.0, 4.0, 3.0]])


def test_raw_panel_feature_names_order_fixed_x_lags_before_append_blocks():
    frame = pd.DataFrame(
        {
            "target": [10.0, 11.0, 12.0],
            "a": [1.0, 2.0, 3.0],
        }
    )
    recipe = SimpleNamespace(
        data_task_spec={"predictor_family": "all_macro_vars"},
        preprocess_contract=_contract(x_lag_creation="fixed_x_lags"),
        layer2_representation_spec={
            "feature_blocks": {
                "temporal_feature_block": {"value": "moving_average_features"},
                "rotation_feature_block": {"value": "moving_average_rotation"},
                "level_feature_block": {"value": "none"},
            }
        },
    )

    assert _raw_panel_feature_names(frame, "target", recipe) == [
        "a",
        "a_lag_1",
        "a_ma3",
        "a_rotma3",
        "a_rotma6",
    ]


def test_raw_panel_feature_names_prefer_explicit_x_lag_block_over_bridge():
    frame = pd.DataFrame(
        {
            "target": [10.0, 11.0, 12.0],
            "a": [1.0, 2.0, 3.0],
        }
    )
    recipe = SimpleNamespace(
        data_task_spec={"predictor_family": "all_macro_vars"},
        preprocess_contract=_contract(x_lag_creation="no_x_lags"),
        layer2_representation_spec={
            "feature_blocks": {
                "x_lag_feature_block": {"value": "fixed_x_lags"},
                "temporal_feature_block": {"value": "none"},
                "rotation_feature_block": {"value": "none"},
                "level_feature_block": {"value": "none"},
            }
        },
    )

    assert _raw_panel_feature_names(frame, "target", recipe) == ["a", "a_lag_1"]


def test_raw_panel_marx_rotation_replaces_lag_polynomial_basis_with_origin_history():
    frame = pd.DataFrame(
        {
            "target": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "b": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
        }
    )
    c = _contract()

    X_train, y_train, X_pred = _build_raw_panel_training_data(
        frame,
        "target",
        horizon=1,
        start_idx=0,
        origin_idx=6,
        contract=c,
        predictor_family="all_macro_vars",
        rotation_feature_block="marx_rotation",
        marx_max_lag=3,
    )

    assert np.allclose(
        X_train,
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.5, 1.0 / 3.0, 2.0, 1.0, 2.0 / 3.0],
            [2.0, 1.5, 1.0, 4.0, 3.0, 2.0],
            [3.0, 2.5, 2.0, 6.0, 5.0, 4.0],
            [4.0, 3.5, 3.0, 8.0, 7.0, 6.0],
            [5.0, 4.5, 4.0, 10.0, 9.0, 8.0],
        ],
    )
    assert y_train.tolist() == [11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
    assert np.allclose(X_pred, [[6.0, 5.5, 5.0, 12.0, 11.0, 10.0]])


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


def test_raw_panel_local_temporal_factors_use_trailing_origin_history():
    frame = pd.DataFrame(
        {
            "target": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "b": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
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
        temporal_feature_block="local_temporal_factors",
    )

    assert np.allclose(
        X_train,
        [
            [1.0, 2.0, 1.5, 0.5],
            [2.0, 4.0, 2.25, 0.75],
            [3.0, 6.0, 3.0, 1.0],
            [4.0, 8.0, 4.5, 1.5],
        ],
    )
    assert y_train.tolist() == [11.0, 12.0, 13.0, 14.0]
    assert np.allclose(X_pred, [[5.0, 10.0, 6.0, 2.0]])


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
