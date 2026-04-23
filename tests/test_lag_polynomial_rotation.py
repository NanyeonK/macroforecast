import pytest
import pandas as pd

from macrocast.execution.lag_polynomial_rotation import (
    build_marx_rotation_contract,
    build_marx_rotation_frame,
    marx_rotation_public_feature_name,
    marx_rotation_runtime_feature_name,
)


def test_marx_rotation_names_are_stable_and_explicit() -> None:
    assert marx_rotation_public_feature_name("INDPRO", 3) == "INDPRO_marx_ma_lag1_to_lag3"
    assert marx_rotation_runtime_feature_name("INDPRO", 3) == "INDPRO__marx_ma_lag1_to_lag3"


def test_marx_rotation_contract_builds_predictor_major_names() -> None:
    contract = build_marx_rotation_contract(max_lag=3, predictors=["a", "b"])

    assert contract["schema_version"] == "lag_polynomial_rotation_contract_v1"
    assert contract["rotation_block"] == "marx_rotation"
    assert contract["composer_contract"] == "lag_polynomial_rotation_block_composer"
    assert contract["runtime_status"] == "operational"
    assert contract["runtime_builder"] == "build_marx_rotation_frame"
    assert contract["source_feature_name_pattern"] == "{predictor}_lag_{k}"
    assert contract["source_runtime_feature_name_pattern"] == "{predictor}__lag{k}"
    assert contract["rotated_feature_name_pattern"] == "{predictor}_marx_ma_lag1_to_lag{p}"
    assert contract["rotated_runtime_feature_name_pattern"] == "{predictor}__marx_ma_lag1_to_lag{p}"
    assert contract["rotation_orders"] == [1, 2, 3]
    assert contract["feature_order"] == "predictor_major_then_rotation_order"
    assert contract["basis_policy"] == "replace_lag_polynomial_basis"
    assert contract["initial_lag_fill_policy"] == "zero_fill_before_start"
    assert contract["feature_names"] == [
        "a_marx_ma_lag1_to_lag1",
        "a_marx_ma_lag1_to_lag2",
        "a_marx_ma_lag1_to_lag3",
        "b_marx_ma_lag1_to_lag1",
        "b_marx_ma_lag1_to_lag2",
        "b_marx_ma_lag1_to_lag3",
    ]
    assert contract["runtime_feature_names"] == [
        "a__marx_ma_lag1_to_lag1",
        "a__marx_ma_lag1_to_lag2",
        "a__marx_ma_lag1_to_lag3",
        "b__marx_ma_lag1_to_lag1",
        "b__marx_ma_lag1_to_lag2",
        "b__marx_ma_lag1_to_lag3",
    ]
    assert contract["alignment"] == {
        "train_row_t_uses": "X_{t-1}, ..., X_{t-p} for each rotated order p",
        "prediction_origin_uses": "X_{origin-1}, ..., X_{origin-p} for each rotated order p",
        "lookahead": "forbidden",
    }
    assert "do_not_append_source_lag_columns" in contract["duplicate_base_policy"]


def test_build_marx_rotation_frame_uses_zero_filled_origin_history() -> None:
    source = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [2.0, 4.0, 6.0, 8.0],
        }
    )

    rotated = build_marx_rotation_frame(source, max_lag=3)

    assert rotated.columns.tolist() == [
        "a__marx_ma_lag1_to_lag1",
        "a__marx_ma_lag1_to_lag2",
        "a__marx_ma_lag1_to_lag3",
        "b__marx_ma_lag1_to_lag1",
        "b__marx_ma_lag1_to_lag2",
        "b__marx_ma_lag1_to_lag3",
    ]
    assert rotated.to_numpy().tolist() == [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.5, 1.0 / 3.0, 2.0, 1.0, 2.0 / 3.0],
        [2.0, 1.5, 1.0, 4.0, 3.0, 2.0],
        [3.0, 2.5, 2.0, 6.0, 5.0, 4.0],
    ]


def test_marx_rotation_contract_schema_can_be_emitted_before_lag_order_is_known() -> None:
    contract = build_marx_rotation_contract()

    assert contract["runtime_status"] == "skeleton_only"
    assert contract["rotation_orders"] == "required_from_recipe"
    assert "feature_names" not in contract
    assert "runtime_feature_names" not in contract


def test_marx_rotation_contract_requires_lag_order_for_concrete_names() -> None:
    with pytest.raises(ValueError, match="predictors require max_lag"):
        build_marx_rotation_contract(predictors=["a"])

    with pytest.raises(ValueError, match="max_lag must be positive"):
        build_marx_rotation_contract(max_lag=0)

    with pytest.raises(ValueError, match="rotation_order must be positive"):
        marx_rotation_public_feature_name("a", 0)
