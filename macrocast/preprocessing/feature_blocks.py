from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .errors import PreprocessValidationError


CUSTOM_FEATURE_BLOCK_CONTRACT_VERSION = "custom_feature_block_callable_v1"
CUSTOM_FEATURE_COMBINER_CONTRACT_VERSION = "custom_feature_combiner_v1"
CUSTOM_FINAL_Z_SELECTION_CONTRACT_VERSION = "custom_final_z_selection_v1"


@dataclass(frozen=True)
class FeatureBlockCallableContext:
    block_kind: str
    fit_scope: str = "train_only"
    horizon: int | None = None
    forecast_origin: Any | None = None
    feature_namespace: str = "custom"
    X_train: Any | None = None
    X_pred: Any | None = None
    y_train: Any | None = None
    source_frame: Any | None = None
    predictors: tuple[str, ...] = ()
    train_index: Any | None = None
    pred_index: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FeatureBlockCallableResult:
    train_features: Any
    pred_features: Any
    feature_names: tuple[str, ...]
    runtime_feature_names: tuple[str, ...] = ()
    fit_state: Mapping[str, Any] = field(default_factory=dict)
    leakage_metadata: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    contract_version: str = CUSTOM_FEATURE_BLOCK_CONTRACT_VERSION


class FeatureBlockCallable(Protocol):
    def __call__(self, context: FeatureBlockCallableContext) -> FeatureBlockCallableResult:
        ...


@dataclass(frozen=True)
class FeatureCombinerCallableContext:
    blocks_train: Mapping[str, Any]
    blocks_pred: Mapping[str, Any]
    y_train: Any
    feature_names: tuple[str, ...]
    block_roles: Mapping[str, str] = field(default_factory=dict)
    fit_scope: str = "train_only"
    horizon: int | None = None
    forecast_origin: Any | None = None
    train_index: Any | None = None
    pred_index: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FeatureCombinerCallableResult:
    Z_train: Any
    Z_pred: Any
    feature_names: tuple[str, ...]
    block_roles: Mapping[str, str] = field(default_factory=dict)
    fit_state: Mapping[str, Any] = field(default_factory=dict)
    leakage_metadata: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    contract_version: str = CUSTOM_FEATURE_COMBINER_CONTRACT_VERSION


class FeatureCombinerCallable(Protocol):
    def __call__(self, context: FeatureCombinerCallableContext) -> FeatureCombinerCallableResult:
        ...


def custom_feature_block_required_fields() -> tuple[str, ...]:
    return (
        "train_features",
        "pred_features",
        "feature_names",
        "runtime_feature_names",
        "fit_state",
        "leakage_metadata",
        "provenance",
    )


def custom_feature_combiner_required_fields() -> tuple[str, ...]:
    return (
        "Z_train",
        "Z_pred",
        "feature_names",
        "block_roles",
        "fit_state",
        "leakage_metadata",
        "provenance",
    )


def custom_final_z_selection_required_fields() -> tuple[str, ...]:
    return (
        "candidate_feature_names",
        "selected_feature_names",
        "dropped_feature_names",
        "feature_selection_policy",
        "fit_state",
        "leakage_metadata",
    )


def validate_feature_block_callable_result(result: FeatureBlockCallableResult) -> None:
    if result.contract_version != CUSTOM_FEATURE_BLOCK_CONTRACT_VERSION:
        raise PreprocessValidationError(
            f"custom feature block result must use contract_version={CUSTOM_FEATURE_BLOCK_CONTRACT_VERSION!r}"
        )
    if not result.feature_names:
        raise PreprocessValidationError("custom feature block result requires at least one feature name")
    if len(set(result.feature_names)) != len(result.feature_names):
        raise PreprocessValidationError("custom feature block feature_names must be unique")
    if result.runtime_feature_names and len(result.runtime_feature_names) != len(result.feature_names):
        raise PreprocessValidationError("runtime_feature_names must be empty or match feature_names length")
    if result.runtime_feature_names and len(set(result.runtime_feature_names)) != len(result.runtime_feature_names):
        raise PreprocessValidationError("custom feature block runtime_feature_names must be unique")
    if "lookahead" not in result.leakage_metadata:
        raise PreprocessValidationError("custom feature block leakage_metadata must record lookahead policy")


def validate_feature_combiner_callable_result(result: FeatureCombinerCallableResult) -> None:
    if result.contract_version != CUSTOM_FEATURE_COMBINER_CONTRACT_VERSION:
        raise PreprocessValidationError(
            f"custom feature combiner result must use contract_version={CUSTOM_FEATURE_COMBINER_CONTRACT_VERSION!r}"
        )
    if not result.feature_names:
        raise PreprocessValidationError("custom feature combiner result requires at least one feature name")
    if len(set(result.feature_names)) != len(result.feature_names):
        raise PreprocessValidationError("custom feature combiner feature_names must be unique")
    if "lookahead" not in result.leakage_metadata:
        raise PreprocessValidationError("custom feature combiner leakage_metadata must record lookahead policy")
    unknown_roles = set(result.block_roles) - set(result.feature_names)
    if unknown_roles:
        raise PreprocessValidationError(
            f"custom feature combiner block_roles contains names not present in feature_names: {sorted(unknown_roles)}"
        )


def custom_feature_block_contract_metadata(*, block_kind: str) -> dict[str, Any]:
    return {
        "schema_version": CUSTOM_FEATURE_BLOCK_CONTRACT_VERSION,
        "block_kind": block_kind,
        "required_fields": list(custom_feature_block_required_fields()),
        "fit_scope": "train_window_only",
        "required_leakage_metadata": ["lookahead"],
        "required_name_properties": ["stable_public_names", "stable_runtime_names", "unique_names"],
    }


def custom_feature_combiner_contract_metadata() -> dict[str, Any]:
    return {
        "schema_version": CUSTOM_FEATURE_COMBINER_CONTRACT_VERSION,
        "layer": 2,
        "callable": "fn(FeatureCombinerCallableContext) -> FeatureCombinerCallableResult",
        "required_fields": list(custom_feature_combiner_required_fields()),
        "fit_scope": "train_window_only",
        "required_leakage_metadata": ["lookahead"],
        "input_blocks": "ordered train/pred feature frames grouped by Layer 2 block role plus candidate_z",
        "required_name_properties": ["stable_public_names", "unique_names"],
    }


def custom_final_z_selection_contract_metadata() -> dict[str, Any]:
    return {
        "schema_version": CUSTOM_FINAL_Z_SELECTION_CONTRACT_VERSION,
        "layer": 2,
        "scope": "feature selection after custom feature blocks or a custom feature combiner has produced final Z candidates",
        "required_fields": list(custom_final_z_selection_required_fields()),
        "fit_scope": "train_window_only",
        "required_leakage_metadata": ["lookahead"],
        "required_name_properties": ["preserve_custom_feature_provenance", "record_selected_and_dropped_names"],
    }
