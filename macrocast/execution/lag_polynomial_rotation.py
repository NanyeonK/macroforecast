"""Lag-polynomial rotation contracts and builders for Layer 2 feature blocks."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import pandas as pd


LAG_POLYNOMIAL_ROTATION_SCHEMA_VERSION: Final = "lag_polynomial_rotation_contract_v1"
MARX_ROTATION_BLOCK_ID: Final = "marx_rotation"
MARX_COMPOSER_CONTRACT: Final = "lag_polynomial_rotation_block_composer"
MARX_SOURCE_PUBLIC_FEATURE_NAME_PATTERN: Final = "{predictor}_lag_{k}"
MARX_SOURCE_RUNTIME_FEATURE_NAME_PATTERN: Final = "{predictor}__lag{k}"
MARX_PUBLIC_FEATURE_NAME_PATTERN: Final = "{predictor}_marx_ma_lag1_to_lag{p}"
MARX_RUNTIME_FEATURE_NAME_PATTERN: Final = "{predictor}__marx_ma_lag1_to_lag{p}"


def _positive_int(value: int, *, name: str) -> int:
    value_i = int(value)
    if value_i <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return value_i


def _predictor_name(value: str) -> str:
    value_s = str(value)
    if not value_s:
        raise ValueError("predictor names must be non-empty strings")
    return value_s


def marx_rotation_public_feature_name(predictor: str, rotation_order: int) -> str:
    """Return the public feature name for one MARX rotated-lag coordinate."""
    predictor = _predictor_name(predictor)
    rotation_order = _positive_int(rotation_order, name="rotation_order")
    return f"{predictor}_marx_ma_lag1_to_lag{rotation_order}"


def marx_rotation_runtime_feature_name(predictor: str, rotation_order: int) -> str:
    """Return the runtime feature name for one MARX rotated-lag coordinate."""
    predictor = _predictor_name(predictor)
    rotation_order = _positive_int(rotation_order, name="rotation_order")
    return f"{predictor}__marx_ma_lag1_to_lag{rotation_order}"


def _feature_names(
    predictors: Sequence[str],
    rotation_orders: Sequence[int],
    *,
    runtime: bool,
) -> list[str]:
    name_fn = marx_rotation_runtime_feature_name if runtime else marx_rotation_public_feature_name
    return [
        name_fn(str(predictor), int(rotation_order))
        for predictor in predictors
        for rotation_order in rotation_orders
    ]


def build_marx_rotation_frame(source: pd.DataFrame, *, max_lag: int) -> pd.DataFrame:
    """Build predictor-major MARX rotated-lag features from a source X panel."""
    max_lag_i = _positive_int(max_lag, name="max_lag")
    source = source.astype(float)
    feature_frames: list[pd.Series] = []
    for predictor in source.columns:
        predictor_name = _predictor_name(str(predictor))
        lagged = [source[predictor].shift(lag).fillna(0.0) for lag in range(1, max_lag_i + 1)]
        for rotation_order in range(1, max_lag_i + 1):
            rotated = sum(lagged[:rotation_order]) / float(rotation_order)
            feature_frames.append(
                rotated.rename(marx_rotation_runtime_feature_name(predictor_name, rotation_order))
            )
    if not feature_frames:
        return pd.DataFrame(index=source.index)
    return pd.concat(feature_frames, axis=1)


def build_marx_rotation_contract(
    *,
    max_lag: int | None = None,
    predictors: Sequence[str] = (),
) -> dict[str, object]:
    """Build the MARX lag-polynomial rotation contract.

    ``max_lag`` is optional because compiler boundary metadata can be emitted
    before an executable recipe supplies the lag order. When predictors are
    supplied, ``max_lag`` is required so concrete feature names can be built.
    """
    predictor_names = tuple(_predictor_name(predictor) for predictor in predictors)
    if max_lag is None:
        if predictor_names:
            raise ValueError("predictors require max_lag so concrete MARX feature names can be built")
        rotation_orders: list[int] | str = "required_from_recipe"
    else:
        max_lag_i = _positive_int(max_lag, name="max_lag")
        rotation_orders = list(range(1, max_lag_i + 1))

    payload: dict[str, object] = {
        "schema_version": LAG_POLYNOMIAL_ROTATION_SCHEMA_VERSION,
        "rotation_block": MARX_ROTATION_BLOCK_ID,
        "composer_contract": MARX_COMPOSER_CONTRACT,
        "runtime_status": "skeleton_only" if max_lag is None else "operational",
        "runtime_builder": "build_marx_rotation_frame",
        "source_block": "x_lag_feature_block",
        "source_feature_name_pattern": MARX_SOURCE_PUBLIC_FEATURE_NAME_PATTERN,
        "source_runtime_feature_name_pattern": MARX_SOURCE_RUNTIME_FEATURE_NAME_PATTERN,
        "rotation_matrix": "lower_triangular_cumulative_average",
        "rotation_orders": rotation_orders,
        "rotated_feature_name_pattern": MARX_PUBLIC_FEATURE_NAME_PATTERN,
        "rotated_runtime_feature_name_pattern": MARX_RUNTIME_FEATURE_NAME_PATTERN,
        "feature_order": "predictor_major_then_rotation_order",
        "basis_policy": "replace_lag_polynomial_basis",
        "duplicate_base_policy": (
            "do_not_append_source_lag_columns when the rotated MARX basis is active; "
            "the p=1 rotated coordinate is retained as the first rotated-basis coordinate"
        ),
        "initial_lag_fill_policy": "zero_fill_before_start",
        "formula": "Z_{i,p,t} = p^{-1} * sum_{j=1}^{p} X_{i,t-j}",
        "alignment": {
            "train_row_t_uses": "X_{t-1}, ..., X_{t-p} for each rotated order p",
            "prediction_origin_uses": "X_{origin-1}, ..., X_{origin-p} for each rotated order p",
            "lookahead": "forbidden",
        },
        "composer_requirements": [
            "build an explicit X-lag block before rotation",
            "apply cumulative lower-triangular moving-average rotation over the lag polynomial",
            "replace the source lag-polynomial basis with the rotated basis in final Z",
            "write stable public/runtime feature names in predictor-major, rotation-order ascending sequence",
            "prove row-date and prediction-origin no-lookahead alignment",
        ],
    }
    if max_lag is not None:
        payload["max_lag"] = int(max_lag)
    if predictor_names:
        assert isinstance(rotation_orders, list)
        payload["predictors"] = list(predictor_names)
        payload["feature_names"] = _feature_names(predictor_names, rotation_orders, runtime=False)
        payload["runtime_feature_names"] = _feature_names(predictor_names, rotation_orders, runtime=True)
    return payload


__all__ = [
    "LAG_POLYNOMIAL_ROTATION_SCHEMA_VERSION",
    "MARX_COMPOSER_CONTRACT",
    "MARX_PUBLIC_FEATURE_NAME_PATTERN",
    "MARX_RUNTIME_FEATURE_NAME_PATTERN",
    "build_marx_rotation_contract",
    "build_marx_rotation_frame",
    "marx_rotation_public_feature_name",
    "marx_rotation_runtime_feature_name",
]
