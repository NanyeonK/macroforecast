from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..preprocessing import PreprocessContract
from ..raw import RawLoadResult
from ..recipes import RecipeSpec, RunSpec


@dataclass(frozen=True)
class ExecutionSpec:
    recipe: RecipeSpec
    run: RunSpec
    preprocess: PreprocessContract


@dataclass(frozen=True)
class ExecutionResult:
    spec: ExecutionSpec
    run: RunSpec
    raw_result: RawLoadResult
    artifact_dir: str


FORECAST_PAYLOAD_CONTRACT_VERSION = "forecast_payload_v1"
DIRECTION_FORECAST_PAYLOAD_CONTRACT_VERSION = "direction_forecast_payload_v1"
INTERVAL_FORECAST_PAYLOAD_CONTRACT_VERSION = "interval_forecast_payload_v1"
DENSITY_FORECAST_PAYLOAD_CONTRACT_VERSION = "density_forecast_payload_v1"
LAYER2_REPRESENTATION_CONTRACT_VERSION = "layer2_representation_v1"
PREDICTION_ROW_SCHEMA_VERSION = "prediction_row_schema_v1"
LAYER2_REPRESENTATION_REQUIRED_FIELDS = (
    "Z_train",
    "y_train",
    "Z_pred",
    "feature_names",
    "block_order",
    "block_roles",
    "fit_state",
    "alignment",
    "leakage_contract",
)


@dataclass(frozen=True)
class ForecastPayload:
    y_pred: float
    selected_lag: int
    selected_bic: float
    tuning_payload: dict[str, Any] = field(default_factory=dict)
    contract_version: str = FORECAST_PAYLOAD_CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        tuning_payload = dict(self.tuning_payload)
        tuning_payload.setdefault("forecast_payload_contract", self.contract_version)
        return {
            "y_pred": self.y_pred,
            "selected_lag": self.selected_lag,
            "selected_bic": self.selected_bic,
            "tuning_payload": tuning_payload,
            "contract_version": self.contract_version,
        }


@dataclass(frozen=True)
class DirectionForecastPayload:
    direction: str
    up_probability: float
    threshold: float = 0.0
    contract_version: str = DIRECTION_FORECAST_PAYLOAD_CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "up_probability": self.up_probability,
            "threshold": self.threshold,
            "contract_version": self.contract_version,
        }


@dataclass(frozen=True)
class IntervalForecastPayload:
    lower: float
    upper: float
    coverage: float
    center: float
    method: str = "gaussian_train_std_symmetric_v1"
    contract_version: str = INTERVAL_FORECAST_PAYLOAD_CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "coverage": self.coverage,
            "center": self.center,
            "method": self.method,
            "contract_version": self.contract_version,
        }


@dataclass(frozen=True)
class DensityForecastPayload:
    mean: float
    variance: float
    distribution: str = "gaussian_train_std_v1"
    contract_version: str = DENSITY_FORECAST_PAYLOAD_CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean,
            "variance": self.variance,
            "distribution": self.distribution,
            "contract_version": self.contract_version,
        }


@dataclass(frozen=True)
class Layer2Representation:
    Z_train: np.ndarray
    y_train: np.ndarray
    Z_pred: np.ndarray
    feature_names: tuple[str, ...]
    block_order: tuple[str, ...] = ()
    block_roles: dict[str, str] = field(default_factory=dict)
    fit_state: tuple[dict[str, Any], ...] = ()
    alignment: dict[str, Any] = field(default_factory=dict)
    leakage_contract: str = "forecast_origin_only"
    feature_builder: str = ""
    feature_runtime_builder: str = ""
    legacy_feature_builder: str = ""
    feature_dispatch_source: str = "layer2_feature_blocks"
    contract_version: str = LAYER2_REPRESENTATION_CONTRACT_VERSION

    @property
    def latest_fit_state(self) -> dict[str, Any] | None:
        if not self.fit_state:
            return None
        return self.fit_state[-1]

    def contract_metadata(self) -> dict[str, Any]:
        z_train = np.asarray(self.Z_train)
        y_train = np.asarray(self.y_train)
        z_pred = np.asarray(self.Z_pred)
        return {
            "schema_version": self.contract_version,
            "contract_version": self.contract_version,
            "required_fields": list(LAYER2_REPRESENTATION_REQUIRED_FIELDS),
            "matrix_shapes": {
                "Z_train": list(z_train.shape),
                "y_train": list(y_train.shape),
                "Z_pred": list(z_pred.shape),
            },
            "feature_count": len(self.feature_names),
            "feature_names": list(self.feature_names),
            "block_order": list(self.block_order),
            "block_roles": dict(self.block_roles),
            "alignment": dict(self.alignment),
            "leakage_contract": self.leakage_contract,
            "feature_runtime_builder": self.feature_runtime_builder,
            "legacy_feature_builder": self.legacy_feature_builder,
            "feature_dispatch_source": self.feature_dispatch_source,
        }

    def runtime_context(self, *, mode: str) -> dict[str, Any]:
        return {
            "layer2_representation_contract": self.contract_version,
            "feature_builder": self.feature_builder,
            "feature_runtime_builder": self.feature_runtime_builder,
            "legacy_feature_builder": self.legacy_feature_builder,
            "feature_dispatch_source": self.feature_dispatch_source,
            "mode": mode,
            "feature_names": list(self.feature_names),
            "block_order": list(self.block_order),
            "block_roles": dict(self.block_roles),
            "alignment": dict(self.alignment),
            "leakage_contract": self.leakage_contract,
        }
