from __future__ import annotations

from .errors import DesignNormalizationError
from .types import ComparisonContract, FixedDesign, ReplicationInput, ResearchDesign, VaryingDesign
from ..registry.naming import canonical_axis_value

_ALLOWED_RESEARCH_DESIGNS: tuple[ResearchDesign, ...] = (
    "single_forecast_run",
    "controlled_variation",
    "study_bundle",
    "replication_recipe",
)


def _tupleize(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        if all(isinstance(item, str) for item in value):
            return value
        raise DesignNormalizationError("tuple values must contain only strings")
    if isinstance(value, list):
        if all(isinstance(item, str) for item in value):
            return tuple(value)
        raise DesignNormalizationError("list values must contain only strings")
    raise DesignNormalizationError(f"expected tuple/list of strings, got {type(value).__name__}")


def normalize_research_design(value: str) -> str:
    value = canonical_axis_value("research_design", value)
    if value not in _ALLOWED_RESEARCH_DESIGNS:
        raise DesignNormalizationError(
            f"unknown research_design={value!r}; expected one of {_ALLOWED_RESEARCH_DESIGNS}"
        )
    return value


def normalize_fixed_design(value: FixedDesign | dict) -> FixedDesign:
    if isinstance(value, FixedDesign):
        return value
    if not isinstance(value, dict):
        raise DesignNormalizationError("fixed_design must be FixedDesign or dict")
    return FixedDesign(**value)


def normalize_varying_design(value: VaryingDesign | dict | None) -> VaryingDesign:
    if value is None:
        return VaryingDesign()
    if isinstance(value, VaryingDesign):
        return value
    if not isinstance(value, dict):
        raise DesignNormalizationError("varying_design must be VaryingDesign, dict, or None")
    payload = dict(value)
    for key in (
        "model_families",
        "feature_recipes",
        "preprocess_variants",
        "tuning_variants",
        "horizons",
    ):
        payload[key] = _tupleize(payload.get(key))
    return VaryingDesign(**payload)


def normalize_comparison_contract(value: ComparisonContract | dict) -> ComparisonContract:
    if isinstance(value, ComparisonContract):
        return value
    if not isinstance(value, dict):
        raise DesignNormalizationError("comparison_contract must be ComparisonContract or dict")
    return ComparisonContract(**value)


def normalize_replication_input(value: ReplicationInput | dict | None) -> ReplicationInput | None:
    if value is None:
        return None
    if isinstance(value, ReplicationInput):
        return value
    if not isinstance(value, dict):
        raise DesignNormalizationError("replication_input must be ReplicationInput, dict, or None")
    payload = dict(value)
    payload["locked_constraints"] = _tupleize(payload.get("locked_constraints"))
    return ReplicationInput(**payload)
