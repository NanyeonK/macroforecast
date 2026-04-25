from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from .errors import CompileValidationError
from .types import CompiledRecipeSpec, CompileResult
from ..execution import execute_recipe
from ..execution.horizon_target import (
    build_path_average_target_protocol as _build_path_average_target_protocol,
    construction_scale as _target_construction_scale,
    is_path_average_construction as _is_path_average_construction,
)
from ..execution.lag_polynomial_rotation import (
    build_marx_rotation_contract as _build_marx_rotation_contract,
)
from ..preprocessing import (
    CUSTOM_FEATURE_BLOCK_CONTRACT_VERSION,
    build_preprocess_contract,
    build_target_scale_contract,
    check_preprocess_governance,
    custom_feature_block_contract_metadata,
    custom_feature_combiner_contract_metadata,
    custom_final_z_selection_contract_metadata,
    is_operational_preprocess_contract,
    preprocess_to_dict,
)
from ..recipes import build_recipe_spec, build_run_spec
from ..registry import AxisSelection, get_axis_registry, get_canonical_layer_order
from ..registry.stage0.experiment_unit import derive_experiment_unit_default, get_experiment_unit_entry
from ..design import build_design_frame, resolve_route_owner, design_to_dict
from ..custom import (
    get_custom_feature_block,
    get_custom_target_transformer,
    get_custom_preprocessor,
    is_custom_feature_block,
    is_custom_feature_combiner,
    is_custom_model,
    is_custom_preprocessor,
    is_custom_target_transformer,
)

_ALLOWED_SELECTION_MODES = ("fixed_axes", "sweep_axes", "conditional_axes", "leaf_config")

_AXIS_NAME_ALIASES = {
    "info_set": "information_set_type",
    "dataset_source": "source_adapter",
    "task": "target_structure",
}

_AXIS_VALUE_ALIASES = {
    ("horizon_target_construction", "future_level_y_t_plus_h"): "future_target_level_t_plus_h",
}

_DATASET_DEFAULT_FREQUENCY = {
    "fred_md": "monthly",
    "fred_qd": "quarterly",
    "fred_sd": "monthly",
    "fred_md+fred_sd": "monthly",
    "fred_qd+fred_sd": "quarterly",
}

_COMPOSITE_DATASET_FREQUENCY = {
    "fred_md+fred_sd": "monthly",
    "fred_qd+fred_sd": "quarterly",
}

_X_IMPUTATION_METHODS = {"mean", "median", "ffill", "bfill"}

_MULTI_BENCHMARK_ALLOWED_MEMBERS = {
    "historical_mean",
    "zero_change",
    "ar_bic",
    "rolling_mean",
    "ar_fixed_p",
    "ardi",
}

_TARGET_TRANSFORMER_FEATURE_RUNTIMES = {"autoreg_lagged_target", "raw_feature_panel"}
_TARGET_TRANSFORMER_RAW_PANEL_MODELS = {"ols", "ridge", "lasso", "elasticnet"}
_RAW_PANEL_FEATURE_BUILDERS = {"raw_feature_panel", "raw_X_only", "factor_pca", "factors_plus_AR"}
_RAW_PANEL_FEATURE_BLOCK_SETS = {
    "transformed_x",
    "transformed_x_lags",
    "factors_plus_target_lags",
    "factor_blocks_only",
    "high_dimensional_x",
    "selected_sparse_x",
    "level_augmented_x",
    "rotation_augmented_x",
    "mixed_blocks",
    "custom_blocks",
}
_MARX_COMPOSITION_MODES = {
    "operational": [
        "replace_lag_polynomial_basis",
        "marx_append_to_x",
        "marx_then_factor",
        "factor_then_marx",
        "marx_with_external_x_lag_append",
        "marx_with_temporal_append",
    ],
    "gated": [],
}
_MAF_COMPOSITION_MODES = {
    "operational": [
        "factor_then_maf",
    ],
    "gated": [],
}
_TARGET_LAG_BLOCK_TO_SELECTION = {
    "none": "none",
    "fixed_target_lags": "fixed",
    "ic_selected_target_lags": "ic_select",
    "horizon_specific_target_lags": "horizon_specific",
    "custom_target_lags": "custom",
}
_TARGET_LAG_SELECTION_TO_BLOCK = {
    "none": "none",
    "fixed": "fixed_target_lags",
    "ic_select": "ic_selected_target_lags",
    "cv_select": "custom_target_lags",
    "horizon_specific": "horizon_specific_target_lags",
    "custom": "custom_target_lags",
}
_TARGET_LAG_SELECTION_TO_LEGACY_Y = {
    "none": "fixed",
    "fixed": "fixed",
    "ic_select": "IC_select",
    "cv_select": "cv_select",
    "horizon_specific": "fixed",
    "custom": "model_specific",
}
_X_LAG_FEATURE_BLOCK_TO_CREATION = {
    "none": "no_x_lags",
    "fixed_x_lags": "fixed_x_lags",
    "variable_specific_x_lags": "variable_specific_lags",
    "category_specific_x_lags": "category_specific_lags",
    "cv_selected_x_lags": "cv_selected_x_lags",
    "custom_x_lags": "no_x_lags",
}
_DIMRED_TO_FACTOR_FEATURE_BLOCK = {
    "none": "none",
    "pca": "pca_static_factors",
    "static_factor": "pca_static_factors",
    "custom": "custom_factors",
}
_FACTOR_FEATURE_BLOCK_COMPATIBLE_DIMRED = {
    "none": {"none"},
    "pca_static_factors": {"none", "pca", "static_factor"},
}
_FACTOR_BRIDGE_BUILDERS = {"factor_pca", "factors_plus_AR"}
_FACTOR_DIMRED_BRIDGES = {"pca", "static_factor"}


def load_recipe_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _normalize_layer_spec(layer_spec: dict[str, Any] | None) -> dict[str, Any]:
    layer_spec = layer_spec or {}
    return {key: dict(layer_spec.get(key, {})) for key in _ALLOWED_SELECTION_MODES}


def _canonical_axis_name(axis_name: str) -> str:
    return _AXIS_NAME_ALIASES.get(axis_name, axis_name)


def _canonical_axis_value(axis_name: str, value: str) -> str:
    return _AXIS_VALUE_ALIASES.get((axis_name, value), value)


def _build_axis_selections(recipe_dict: dict[str, Any]) -> tuple[AxisSelection, ...]:
    registry = get_axis_registry()
    path = recipe_dict.get("path", {})
    selections: list[AxisSelection] = []
    for layer in get_canonical_layer_order():
        layer_spec = _normalize_layer_spec(path.get(layer))
        for selection_mode in ("fixed_axes", "sweep_axes", "conditional_axes"):
            mode_name = selection_mode.replace("_axes", "")
            for raw_axis_name, raw_value in layer_spec[selection_mode].items():
                axis_name = _canonical_axis_name(raw_axis_name)
                if axis_name not in registry:
                    raise CompileValidationError(f"unknown registry axis {raw_axis_name!r}")
                entry = registry[axis_name]
                raw_values = tuple(raw_value) if isinstance(raw_value, list) else (raw_value,)
                values = tuple(_canonical_axis_value(axis_name, str(value)) for value in raw_values)
                for value in values:
                    dynamic_allowed = (
                        (axis_name == "model_family" and is_custom_model(value))
                        or (axis_name == "custom_preprocessor" and is_custom_preprocessor(value))
                        or (axis_name == "target_transformer" and is_custom_target_transformer(value))
                    )
                    if value not in entry.allowed_values and not dynamic_allowed:
                        raise CompileValidationError(
                            f"axis {raw_axis_name!r} received unknown value {value!r}"
                        )
                selected_status = {value: entry.current_status.get(value, "operational") for value in values}
                selections.append(
                    AxisSelection(
                        axis_name=axis_name,
                        layer=layer,
                        selection_mode=mode_name,
                        selected_values=values,
                        selected_status=selected_status,
                    )
                )
    return tuple(selections)


def _rule_experiment_unit_default(
    *,
    selection_map: dict[str, AxisSelection],
    leaf_config: dict[str, Any],
) -> str:
    """Derivation rule: default experiment_unit from recipe shape.

    Thin wrapper around macrocast.registry.stage0.experiment_unit.
    derive_experiment_unit_default. Registered under the rule key
    experiment_unit_default in :data:`DERIVATION_RULES`.
    """

    def _sv(name: str, default: str | None = None) -> str | None:
        if name in selection_map:
            return selection_map[name].selected_values[0]
        return default

    research_design = _sv("research_design", "single_path_benchmark")
    task = (
        _sv("target_structure")
        or leaf_config.get("target_structure")
        or leaf_config.get("task")
        or "single_target_point_forecast"
    )
    model_sel = selection_map.get("model_family")
    feature_sel = selection_map.get("feature_builder")
    return derive_experiment_unit_default(
        research_design=research_design or "single_path_benchmark",
        task=task,
        model_axis_mode=model_sel.selection_mode if model_sel else "fixed",
        feature_axis_mode=feature_sel.selection_mode if feature_sel else "fixed",
        wrapper_family=leaf_config.get("wrapper_family"),
    )


DERIVATION_RULES: dict[str, Any] = {
    "experiment_unit_default": _rule_experiment_unit_default,
}


def _resolve_derived_axes(
    recipe_dict: dict[str, Any],
    selection_map: dict[str, AxisSelection],
    leaf_config: dict[str, Any],
) -> list[AxisSelection]:
    """Parse recipe's derived_axes sections and resolve each via a registered rule.

    Shape: derived_axes: {axis_name: rule_name} at any layer block.
    Conflict if the axis also appears in fixed/sweep/conditional; unknown
    rule raises CompileValidationError.
    """
    registry = get_axis_registry()
    additions: list[AxisSelection] = []
    for layer in get_canonical_layer_order():
        layer_block = recipe_dict.get("path", {}).get(layer) or {}
        if not isinstance(layer_block, dict):
            continue
        derived = layer_block.get("derived_axes") or {}
        if not isinstance(derived, dict):
            raise CompileValidationError(
                f"layer {layer!r}: derived_axes must be a mapping of axis_name -> rule_name"
            )
        for raw_axis_name, rule_name in derived.items():
            axis_name = _canonical_axis_name(raw_axis_name)
            if axis_name not in registry:
                raise CompileValidationError(
                    f"layer {layer!r}: unknown axis {raw_axis_name!r} in derived_axes"
                )
            if axis_name in selection_map:
                raise CompileValidationError(
                    f"axis {raw_axis_name!r} declared as derived but also appears "
                    "in fixed/sweep/conditional_axes"
                )
            if not isinstance(rule_name, str) or rule_name not in DERIVATION_RULES:
                raise CompileValidationError(
                    f"unknown derivation rule {rule_name!r} for axis {raw_axis_name!r}"
                )
            rule = DERIVATION_RULES[rule_name]
            value = rule(selection_map=selection_map, leaf_config=leaf_config)
            entry = registry[axis_name]
            if value not in entry.allowed_values:
                raise CompileValidationError(
                    f"derivation rule {rule_name!r} produced value {value!r} which is "
                    f"not an allowed value of axis {raw_axis_name!r}"
                )
            additions.append(
                AxisSelection(
                    axis_name=axis_name,
                    layer=layer,
                    selection_mode="derived",
                    selected_values=(value,),
                    selected_status={value: entry.current_status[value]},
                )
            )
    return additions


def _leaf_config(recipe_dict: dict[str, Any]) -> dict[str, Any]:
    path = recipe_dict.get("path", {})
    leaf: dict[str, Any] = {}
    for layer in get_canonical_layer_order():
        leaf.update(_normalize_layer_spec(path.get(layer))["leaf_config"])
    return leaf


def _ensure_unique_axis_selections(selections: tuple[AxisSelection, ...]) -> None:
    seen: dict[str, AxisSelection] = {}
    for selection in selections:
        if selection.axis_name in seen:
            previous = seen[selection.axis_name]
            if previous.selected_values != selection.selected_values or previous.selection_mode != selection.selection_mode or previous.layer != selection.layer:
                raise CompileValidationError(
                    f"axis {selection.axis_name!r} was specified more than once through canonical/legacy aliases"
                )
        else:
            seen[selection.axis_name] = selection


def _selection_map(selections: tuple[AxisSelection, ...]) -> dict[str, AxisSelection]:
    return {selection.axis_name: selection for selection in selections}


def _selected_value_or_none(selection_map: dict[str, AxisSelection], axis_name: str) -> str | None:
    selection = selection_map.get(axis_name)
    if selection is None or len(selection.selected_values) != 1:
        return None
    return selection.selected_values[0]


def _infer_feature_builder_bridge(selection_map: dict[str, AxisSelection]) -> str | None:
    if "feature_builder" in selection_map:
        return None
    block_set = _selected_value_or_none(selection_map, "feature_block_set")
    if block_set in {"target_lags_only"}:
        return "autoreg_lagged_target"
    if block_set in {"factors_plus_target_lags"}:
        return "factors_plus_AR"
    if block_set in {"factor_blocks_only"}:
        return "factor_pca"
    if block_set in {
        "transformed_x",
        "transformed_x_lags",
        "high_dimensional_x",
        "selected_sparse_x",
        "level_augmented_x",
        "rotation_augmented_x",
        "mixed_blocks",
        "custom_blocks",
    }:
        return "raw_feature_panel"

    target_lag_block = _selected_value_or_none(selection_map, "target_lag_block")
    factor_block = _selected_value_or_none(selection_map, "factor_feature_block")
    raw_block_values = [
        _selected_value_or_none(selection_map, "x_lag_feature_block"),
        _selected_value_or_none(selection_map, "level_feature_block"),
        _selected_value_or_none(selection_map, "temporal_feature_block"),
        _selected_value_or_none(selection_map, "rotation_feature_block"),
    ]
    has_raw_block = any(value not in {None, "none"} for value in raw_block_values)
    if factor_block not in {None, "none"} and target_lag_block not in {None, "none"}:
        return "factors_plus_AR"
    if factor_block not in {None, "none"}:
        return "factor_pca"
    if has_raw_block:
        return "raw_feature_panel"
    if target_lag_block not in {None, "none"}:
        return "autoreg_lagged_target"
    return None


def _append_feature_builder_bridge_if_needed(
    selections: tuple[AxisSelection, ...],
    selection_map: dict[str, AxisSelection],
) -> tuple[AxisSelection, ...]:
    inferred = _infer_feature_builder_bridge(selection_map)
    if inferred is None:
        return selections
    entry = get_axis_registry()["feature_builder"]
    return selections + (
        AxisSelection(
            axis_name="feature_builder",
            layer=entry.layer,
            selection_mode="derived",
            selected_values=(inferred,),
            selected_status={inferred: entry.current_status.get(inferred, "operational")},
        ),
    )


def _selection_value(selection_map: dict[str, AxisSelection], axis_name: str, default: str | None = None) -> str:
    if axis_name not in selection_map:
        if default is None:
            raise CompileValidationError(f"missing required axis {axis_name!r}")
        return default
    values = selection_map[axis_name].selected_values
    if len(values) != 1:
        raise CompileValidationError(f"axis {axis_name!r} must be fixed for direct single-run compilation")
    return values[0]


_CUSTOM_FEATURE_BLOCK_AXES = {
    "temporal_feature_block": ("custom_temporal_features", "temporal", "custom_temporal_feature_block"),
    "rotation_feature_block": ("custom_rotation", "rotation", "custom_rotation_feature_block"),
    "factor_feature_block": ("custom_factors", "factor", "custom_factor_feature_block"),
}


def _custom_feature_block_name_from_leaf(leaf_config: dict[str, Any], block_kind: str) -> str | None:
    custom_blocks = leaf_config.get("custom_feature_blocks") or {}
    if not isinstance(custom_blocks, dict):
        custom_blocks = {}
    for key in (
        block_kind,
        f"{block_kind}_feature_block",
        f"custom_{block_kind}_feature_block",
        f"custom_{block_kind}_block",
    ):
        value = custom_blocks.get(key)
        if value:
            return str(value)
    value = leaf_config.get(f"custom_{block_kind}_feature_block") or leaf_config.get(f"custom_{block_kind}_block")
    return str(value) if value else None


def _custom_feature_block_axis_is_registered(selection: AxisSelection, leaf_config: dict[str, Any]) -> bool:
    spec = _CUSTOM_FEATURE_BLOCK_AXES.get(selection.axis_name)
    if spec is None or len(selection.selected_values) != 1:
        return False
    axis_value, block_kind, _field = spec
    if selection.selected_values[0] != axis_value:
        return False
    block_name = _custom_feature_block_name_from_leaf(leaf_config, block_kind)
    return bool(block_name and is_custom_feature_block(block_name, block_kind=block_kind))


def _custom_feature_combiner_name_from_leaf(leaf_config: dict[str, Any]) -> str | None:
    custom_blocks = leaf_config.get("custom_feature_blocks") or {}
    if not isinstance(custom_blocks, dict):
        custom_blocks = {}
    for key in ("combiner", "feature_combiner", "custom_combiner", "custom_feature_combiner"):
        value = custom_blocks.get(key)
        if value:
            return str(value)
    value = (
        leaf_config.get("custom_feature_combiner")
        or leaf_config.get("custom_combiner")
        or leaf_config.get("custom_feature_block_combiner")
    )
    return str(value) if value else None


def _custom_feature_combiner_axis_is_registered(selection: AxisSelection, leaf_config: dict[str, Any]) -> bool:
    if selection.axis_name != "feature_block_combination" or len(selection.selected_values) != 1:
        return False
    if selection.selected_values[0] != "custom_combiner":
        return False
    combiner_name = _custom_feature_combiner_name_from_leaf(leaf_config)
    return bool(combiner_name and is_custom_feature_combiner(combiner_name))


def _target_structure(selection_map: dict[str, AxisSelection], default: str = "single_target_point_forecast") -> str:
    return _selection_value(selection_map, "target_structure", default=default)


_LEGACY_OFFICIAL_TRANSFORM_BRIDGE_AXES = {
    "target_transform_policy",
    "x_transform_policy",
    "tcode_policy",
    "preprocess_order",
    "representation_policy",
    "tcode_application_scope",
}
_OFFICIAL_TCODE_POLICIES = {"tcode_only", "tcode_then_extra_preprocess", "extra_then_tcode"}


def _has_legacy_official_transform_bridge(selection_map: dict[str, AxisSelection]) -> bool:
    return any(axis in selection_map for axis in _LEGACY_OFFICIAL_TRANSFORM_BRIDGE_AXES)


def _legacy_official_transform_scope(selection_map: dict[str, AxisSelection]) -> str:
    if "tcode_application_scope" in selection_map:
        return _selection_value(selection_map, "tcode_application_scope")
    target_tcode = _selection_value(selection_map, "target_transform_policy", default="raw_level") == "tcode_transformed"
    x_tcode = _selection_value(selection_map, "x_transform_policy", default="raw_level") == "dataset_tcode_transformed"
    if target_tcode and x_tcode:
        return "apply_tcode_to_both"
    if target_tcode:
        return "apply_tcode_to_target"
    if x_tcode:
        return "apply_tcode_to_X"
    if _selection_value(selection_map, "tcode_policy", default="raw_only") in _OFFICIAL_TCODE_POLICIES:
        return "apply_tcode_to_both"
    return "apply_tcode_to_none"


def _legacy_official_transform_policy(selection_map: dict[str, AxisSelection]) -> str:
    tcode_policy = _selection_value(selection_map, "tcode_policy", default="raw_only")
    target_policy = _selection_value(selection_map, "target_transform_policy", default="raw_level")
    x_policy = _selection_value(selection_map, "x_transform_policy", default="raw_level")
    scope = _legacy_official_transform_scope(selection_map)
    if (
        tcode_policy in _OFFICIAL_TCODE_POLICIES
        or target_policy == "tcode_transformed"
        or x_policy == "dataset_tcode_transformed"
        or scope != "apply_tcode_to_none"
    ):
        return "dataset_tcode"
    return "raw_official_frame"


def _official_transform_policy(selection_map: dict[str, AxisSelection]) -> str:
    return _selection_value(
        selection_map,
        "official_transform_policy",
        default=_legacy_official_transform_policy(selection_map),
    )


def _official_transform_scope(selection_map: dict[str, AxisSelection]) -> str:
    return _selection_value(
        selection_map,
        "official_transform_scope",
        default=_legacy_official_transform_scope(selection_map),
    )


def _official_transform_source_payload(selection_map: dict[str, AxisSelection]) -> dict[str, object]:
    legacy_axes = sorted(
        axis for axis in _LEGACY_OFFICIAL_TRANSFORM_BRIDGE_AXES if axis in selection_map
    )

    def _source(axis_name: str) -> str:
        if axis_name in selection_map:
            return "layer1_axis"
        if legacy_axes:
            return "legacy_layer2_tcode_bridge"
        return "compiler_default"

    return {
        "policy_source": _source("official_transform_policy"),
        "scope_source": _source("official_transform_scope"),
        "legacy_bridge_axes": legacy_axes,
    }


def _validate_official_transform_contract(selection_map: dict[str, AxisSelection]) -> None:
    """Keep the new Layer 1 official-transform axes aligned with legacy Layer 2 bridge axes."""

    policy = _official_transform_policy(selection_map)
    scope = _official_transform_scope(selection_map)
    if policy == "raw_official_frame" and scope != "apply_tcode_to_none":
        raise CompileValidationError(
            "official_transform_policy='raw_official_frame' requires official_transform_scope='apply_tcode_to_none'"
        )
    if policy == "dataset_tcode" and scope == "apply_tcode_to_none":
        raise CompileValidationError(
            "official_transform_policy='dataset_tcode' requires an official_transform_scope other than 'apply_tcode_to_none'"
        )

    has_legacy_bridge = _has_legacy_official_transform_bridge(selection_map)
    legacy_policy = _legacy_official_transform_policy(selection_map)
    legacy_scope = _legacy_official_transform_scope(selection_map)
    if has_legacy_bridge and "official_transform_policy" in selection_map and policy != legacy_policy:
        raise CompileValidationError(
            "official_transform_policy conflicts with legacy Layer 2 t-code representation axes; "
            f"got official_transform_policy={policy!r}, legacy policy={legacy_policy!r}"
        )
    if has_legacy_bridge and "official_transform_scope" in selection_map and scope != legacy_scope:
        raise CompileValidationError(
            "official_transform_scope conflicts with legacy Layer 2 tcode_application_scope; "
            f"got official_transform_scope={scope!r}, tcode_application_scope={legacy_scope!r}"
        )


def _extra_preprocessing_requested(selection_map: dict[str, AxisSelection]) -> bool:
    neutral_values = {
        "target_missing_policy": "none",
        "x_missing_policy": "none",
        "target_outlier_policy": "none",
        "x_outlier_policy": "none",
        "scaling_policy": "none",
        "dimensionality_reduction_policy": "none",
        "feature_selection_policy": "none",
        "feature_selection_semantics": "select_before_factor",
        "additional_preprocessing": "none",
        "x_lag_creation": "no_x_lags",
        "x_lag_feature_block": "none",
    }
    for axis, neutral in neutral_values.items():
        selection = selection_map.get(axis)
        if selection is not None and selection.selected_values[0] != neutral:
            return True
    return False


def _x_lag_creation_from_feature_block(block: str) -> str:
    return _X_LAG_FEATURE_BLOCK_TO_CREATION.get(block, "custom_lags")


def _x_lag_creation_value(selection_map: dict[str, AxisSelection], *, default: str = "no_x_lags") -> str:
    block = (
        _selection_value(selection_map, "x_lag_feature_block")
        if "x_lag_feature_block" in selection_map
        else None
    )
    block_bridge = _x_lag_creation_from_feature_block(str(block)) if block is not None else default
    value = _selection_value(selection_map, "x_lag_creation", default=block_bridge)
    if block is not None and value != block_bridge:
        raise CompileValidationError(
            "x_lag_feature_block conflicts with legacy x_lag_creation bridge; "
            f"got x_lag_feature_block={block!r} -> x_lag_creation={block_bridge!r}, "
            f"but x_lag_creation={value!r}"
        )
    return value


def _official_preprocess_bridge_defaults(selection_map: dict[str, AxisSelection]) -> dict[str, str]:
    policy = _official_transform_policy(selection_map)
    scope = _official_transform_scope(selection_map)
    extra_requested = _extra_preprocessing_requested(selection_map)
    if policy == "raw_official_frame":
        return {
            "target_transform_policy": "raw_level",
            "x_transform_policy": "raw_level",
            "tcode_policy": "extra_preprocess_without_tcode" if extra_requested else "raw_only",
            "preprocess_order": "extra_only" if extra_requested else "none",
            "representation_policy": "raw_only",
            "tcode_application_scope": "apply_tcode_to_none",
        }

    target_policy = "tcode_transformed" if scope in {"apply_tcode_to_target", "apply_tcode_to_both"} else "raw_level"
    x_policy = "dataset_tcode_transformed" if scope in {"apply_tcode_to_X", "apply_tcode_to_both"} else "raw_level"
    return {
        "target_transform_policy": target_policy,
        "x_transform_policy": x_policy,
        "tcode_policy": "tcode_then_extra_preprocess" if extra_requested else "tcode_only",
        "preprocess_order": "tcode_then_extra" if extra_requested else "tcode_only",
        "representation_policy": "tcode_only",
        "tcode_application_scope": scope,
    }


def _build_preprocess_contract(selection_map: dict[str, AxisSelection]) -> Any:
    required = {
        "target_missing_policy",
        "x_missing_policy",
        "target_outlier_policy",
        "x_outlier_policy",
        "scaling_policy",
        "dimensionality_reduction_policy",
        "feature_selection_policy",
        "preprocess_fit_scope",
        "inverse_transform_policy",
        "evaluation_scale",
    }
    bridge_defaults = _official_preprocess_bridge_defaults(selection_map)
    bridge_axes = {
        "target_transform_policy",
        "x_transform_policy",
        "tcode_policy",
        "preprocess_order",
        "representation_policy",
        "tcode_application_scope",
    }
    defaults = {
        "target_transform": "level",
        "target_normalization": "none",
        "target_domain": "unconstrained",
        "scaling_scope": "columnwise",
        "additional_preprocessing": "none",
        "x_lag_creation": _x_lag_creation_value(selection_map),
        "feature_grouping": "none",
        "feature_selection_semantics": "select_before_factor",
    }
    missing = sorted(axis for axis in required if axis not in selection_map)
    if missing:
        raise CompileValidationError(f"preprocessing layer missing required axes: {missing}")
    payload = {axis: selection_map[axis].selected_values[0] for axis in required}
    payload.update(
        {
            axis: _selection_value(selection_map, axis, default=bridge_defaults[axis])
            for axis in bridge_axes
        }
    )
    payload.update({axis: _selection_value(selection_map, axis, default=value) for axis, value in defaults.items()})
    try:
        contract = build_preprocess_contract(**payload)
        check_preprocess_governance(
            contract,
            preprocessing_sweep=any(
                selection.layer == "2_preprocessing" and selection.selection_mode == "sweep"
                for selection in selection_map.values()
            ),
            model_sweep=(
                "model_family" in selection_map and selection_map["model_family"].selection_mode == "sweep"
            ),
        )
    except Exception as exc:
        raise CompileValidationError(str(exc)) from exc
    return contract


def _benchmark_spec(selection_map: dict[str, AxisSelection], leaf_config: dict[str, Any]) -> dict[str, Any]:
    benchmark_family = _selection_value(selection_map, "benchmark_family")
    benchmark_config = dict(leaf_config.get("benchmark_config", {}))
    training_cfg = dict(leaf_config.get("training_config", {}))
    if (
        _target_lag_block_value(selection_map) == "fixed_target_lags"
        and "max_ar_lag" not in benchmark_config
    ):
        target_lag_count, _source = _target_lag_count_config(leaf_config)
        benchmark_config["max_ar_lag"] = int(target_lag_count)
    if benchmark_family == "custom_benchmark":
        plugin_path = benchmark_config.get("plugin_path")
        callable_name = benchmark_config.get("callable_name")
        missing = [key for key, value in {"plugin_path": plugin_path, "callable_name": callable_name}.items() if not value]
        if missing:
            raise CompileValidationError(
                f"custom_benchmark requires benchmark_config fields: {missing}"
            )
    return {
        "benchmark_family": benchmark_family,
        **benchmark_config,
    }


def _model_spec(selection_map: dict[str, AxisSelection]) -> dict[str, Any]:
    model_values = selection_map["model_family"].selected_values
    feature_values = selection_map["feature_builder"].selected_values
    return {
        "model_family": model_values[0],
        "feature_builder": feature_values[0],
        "framework": _selection_value(selection_map, "framework"),
        "model_family_values": list(model_values),
        "feature_builder_values": list(feature_values),
    }


def _selection_values(selection: AxisSelection) -> Any:
    if selection.selection_mode == "fixed" and len(selection.selected_values) == 1:
        return selection.selected_values[0]
    return list(selection.selected_values)


def _json_like(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_like(item) for item in value]
    if isinstance(value, list):
        return [_json_like(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_like(item) for key, item in value.items()}
    return value


def _build_tree_context(
    stage0,
    run_spec: RunSpec,
    selections: tuple[AxisSelection, ...],
    leaf_config: dict[str, Any],
) -> dict[str, Any]:
    stage0_payload = design_to_dict(stage0)
    fixed_axes: dict[str, Any] = {}
    sweep_axes: dict[str, Any] = {}
    conditional_axes: dict[str, Any] = {}
    axis_layers: dict[str, str] = {}
    for selection in selections:
        axis_layers[selection.axis_name] = selection.layer
        value = _selection_values(selection)
        if selection.selection_mode == "fixed":
            fixed_axes[selection.axis_name] = value
        elif selection.selection_mode == "sweep":
            sweep_axes[selection.axis_name] = value
        elif selection.selection_mode == "conditional":
            conditional_axes[selection.axis_name] = value
    reproducibility_mode = leaf_config.get("reproducibility_mode_override")
    if reproducibility_mode is None:
        reproducibility_mode = next((selection.selected_values[0] for selection in selections if selection.axis_name == "reproducibility_mode"), "best_effort")
    failure_policy = next((selection.selected_values[0] for selection in selections if selection.axis_name == "failure_policy"), "fail_fast")
    variation_axes = _variation_axes(selections)
    return {
        "research_design": stage0.research_design,
        "design_shape": stage0.design_shape,
        "execution_posture": stage0.execution_posture,
        "experiment_unit": stage0.experiment_unit,
        "route_contract": _route_contract(stage0, run_spec, selections),
        "variation_axes": list(variation_axes),
        "controlled_axis_kind": _controlled_axis_kind(variation_axes),
        "reproducibility_mode": reproducibility_mode,
        "failure_policy": failure_policy,
        "compute_mode": next((selection.selected_values[0] for selection in selections if selection.axis_name == "compute_mode"), "serial"),
        "route_owner": resolve_route_owner(stage0),
        "fixed_design": _json_like(stage0_payload["fixed_design"]),
        "varying_design": _json_like(stage0_payload["varying_design"]),
        "comparison_contract": _json_like(stage0_payload["comparison_contract"]),
        "fixed_axes": _json_like(fixed_axes),
        "sweep_axes": _json_like(sweep_axes),
        "conditional_axes": _json_like(conditional_axes),
        "axis_layers": _json_like(axis_layers),
        "leaf_config": _json_like(dict(leaf_config)),
    }


def _variation_axes(selections: tuple[AxisSelection, ...]) -> tuple[str, ...]:
    return tuple(
        f"{selection.layer}.{selection.axis_name}"
        for selection in selections
        if selection.selection_mode in {"sweep", "conditional"} and len(selection.selected_values) > 1
    )


def _controlled_axis_kind(variation_axes: tuple[str, ...]) -> str:
    if not variation_axes:
        return "none"
    kinds: list[str] = []
    for axis in variation_axes:
        axis_name = axis.rsplit(".", 1)[-1]
        if axis_name == "model_family":
            kinds.append("model")
        elif axis_name == "feature_builder":
            kinds.append("feature")
        elif axis.startswith("2_preprocessing."):
            kinds.append("preprocessing")
        elif axis_name in {"hp_space_style", "search_algorithm", "validation_splitter"}:
            kinds.append("tuning")
        else:
            kinds.append(axis_name)
    unique = tuple(dict.fromkeys(kinds))
    if len(unique) == 1:
        return unique[0]
    return "multi_axis"


def _route_contract(stage0, run_spec: RunSpec, selections: tuple[AxisSelection, ...]) -> str:
    route_owner = run_spec.route_owner
    if route_owner == "single_run":
        if _variation_axes(selections):
            return "sweep_runner_executable"
        return "single_run_executable"
    if route_owner == "wrapper":
        return "wrapper_handoff"
    if route_owner == "replication":
        return "replication_handoff"
    if route_owner == "orchestrator":
        return "orchestrator_handoff"
    return "not_supported_route"


def _tree_context_summary(tree_context: dict[str, Any]) -> str:
    fixed_names = ",".join(sorted(tree_context["fixed_axes"])) or "none"
    sweep_names = ",".join(sorted(tree_context["sweep_axes"])) or "none"
    conditional_names = ",".join(sorted(tree_context["conditional_axes"])) or "none"
    return (
        f"tree_context=route_owner={tree_context['route_owner']}; "
        f"execution_posture={tree_context['execution_posture']}; "
        f"fixed_axes=[{fixed_names}]; "
        f"sweep_axes=[{sweep_names}]; "
        f"conditional_axes=[{conditional_names}]"
    )


def _first_selected_value(selection_map: dict[str, AxisSelection], axis_name: str, default: str) -> str:
    selection = selection_map.get(axis_name)
    if selection is None or not selection.selected_values:
        return default
    return selection.selected_values[0]


def _is_non_empty_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) > 0


def _require_non_empty_sequence(leaf_config: dict[str, Any], key: str, context: str) -> Sequence:
    value = leaf_config.get(key)
    if not _is_non_empty_sequence(value):
        raise CompileValidationError(f"{context} requires leaf_config.{key} as a non-empty list")
    return value


def _require_non_empty_mapping(leaf_config: dict[str, Any], key: str, context: str) -> Mapping:
    value = leaf_config.get(key)
    if not isinstance(value, Mapping) or not value:
        raise CompileValidationError(f"{context} requires leaf_config.{key} as a non-empty dict")
    return value


def _targets_for_layer1_contract(selection_map: dict[str, AxisSelection], leaf_config: dict[str, Any]) -> tuple[str, ...]:
    target_structure = _target_structure(selection_map)
    if target_structure == "multi_target_point_forecast":
        targets = leaf_config.get("targets")
        if not _is_non_empty_sequence(targets):
            return ()
        return tuple(str(target) for target in targets)
    target = leaf_config.get("target")
    return (str(target),) if target else ()


def _require_mapping_entries_for_targets(
    leaf_config: dict[str, Any],
    key: str,
    context: str,
    targets: tuple[str, ...],
) -> Mapping:
    mapping = _require_non_empty_mapping(leaf_config, key, context)
    missing = [target for target in targets if target not in mapping]
    if missing:
        raise CompileValidationError(f"{context} requires leaf_config.{key} entries for targets: {missing}")
    return mapping


def _validate_layer1_data_task_contract(
    selection_map: dict[str, AxisSelection],
    leaf_config: dict[str, Any],
) -> None:
    """Close Layer 1 values whose runtime semantics require leaf_config inputs."""

    _validate_official_transform_contract(selection_map)

    dataset = _selection_value(selection_map, "dataset")
    if dataset == "fred_sd" and "frequency" not in selection_map:
        raise CompileValidationError("dataset='fred_sd' requires explicit frequency ('monthly' or 'quarterly')")
    if dataset in _COMPOSITE_DATASET_FREQUENCY:
        expected = _COMPOSITE_DATASET_FREQUENCY[dataset]
        frequency = _selection_value(selection_map, "frequency", default=expected)
        if frequency != expected:
            raise CompileValidationError(
                f"dataset={dataset!r} requires frequency={expected!r}; got {frequency!r}"
            )

    targets = _targets_for_layer1_contract(selection_map, leaf_config)

    variable_universe = _selection_value(selection_map, "variable_universe", default="all_variables")
    if variable_universe == "handpicked_set":
        _require_non_empty_sequence(
            leaf_config,
            "variable_universe_columns",
            "variable_universe='handpicked_set'",
        )
    elif variable_universe == "category_subset":
        _require_non_empty_mapping(
            leaf_config,
            "variable_universe_category_columns",
            "variable_universe='category_subset'",
        )
        if not leaf_config.get("variable_universe_category"):
            raise CompileValidationError(
                "variable_universe='category_subset' requires leaf_config.variable_universe_category"
            )
    elif variable_universe == "target_specific_subset":
        _require_mapping_entries_for_targets(
            leaf_config,
            "target_specific_columns",
            "variable_universe='target_specific_subset'",
            targets,
        )

    predictor_family = _selection_value(selection_map, "predictor_family", default="target_lags_only")
    if predictor_family == "handpicked_set":
        _require_non_empty_sequence(
            leaf_config,
            "handpicked_columns",
            "predictor_family='handpicked_set'",
        )
    elif predictor_family == "category_based":
        _require_non_empty_mapping(
            leaf_config,
            "predictor_category_columns",
            "predictor_family='category_based'",
        )
        if not leaf_config.get("predictor_category"):
            raise CompileValidationError("predictor_family='category_based' requires leaf_config.predictor_category")

    deterministic_components = _selection_value(selection_map, "deterministic_components", default="none")
    if deterministic_components == "break_dummies":
        _require_non_empty_sequence(
            leaf_config,
            "break_dates",
            "deterministic_components='break_dummies'",
        )

    benchmark_family = _selection_value(selection_map, "benchmark_family")
    if benchmark_family == "multi_benchmark_suite":
        suite = _require_non_empty_sequence(
            leaf_config,
            "benchmark_suite",
            "benchmark_family='multi_benchmark_suite'",
        )
        unknown = sorted(set(suite) - _MULTI_BENCHMARK_ALLOWED_MEMBERS)
        if unknown:
            raise CompileValidationError(
                "benchmark_family='multi_benchmark_suite' supports benchmark_suite members "
                f"{sorted(_MULTI_BENCHMARK_ALLOWED_MEMBERS)}; got unsupported members {unknown}"
            )
    elif benchmark_family == "paper_specific_benchmark":
        _require_mapping_entries_for_targets(
            leaf_config,
            "paper_forecast_series",
            "benchmark_family='paper_specific_benchmark'",
            targets,
        )
    elif benchmark_family == "survey_forecast":
        _require_mapping_entries_for_targets(
            leaf_config,
            "survey_forecast_series",
            "benchmark_family='survey_forecast'",
            targets,
        )
    elif benchmark_family == "expert_benchmark":
        benchmark_config = leaf_config.get("benchmark_config", {})
        if not isinstance(benchmark_config, Mapping) or not benchmark_config.get("expert_callable"):
            raise CompileValidationError(
                "benchmark_family='expert_benchmark' requires leaf_config.benchmark_config.expert_callable"
            )

    missing_availability = _selection_value(selection_map, "missing_availability", default="zero_fill_before_start")
    if missing_availability == "x_impute_only":
        method = leaf_config.get("x_imputation")
        if method not in _X_IMPUTATION_METHODS:
            raise CompileValidationError(
                "missing_availability='x_impute_only' requires leaf_config.x_imputation "
                f"in {sorted(_X_IMPUTATION_METHODS)}"
            )

    raw_missing_policy = _selection_value(selection_map, "raw_missing_policy", default="preserve_raw_missing")
    if raw_missing_policy == "x_impute_raw":
        method = leaf_config.get("raw_x_imputation")
        if method not in _X_IMPUTATION_METHODS:
            raise CompileValidationError(
                "raw_missing_policy='x_impute_raw' requires leaf_config.raw_x_imputation "
                f"in {sorted(_X_IMPUTATION_METHODS)}"
            )

    raw_outlier_policy = _selection_value(selection_map, "raw_outlier_policy", default="preserve_raw_outliers")
    raw_outlier_columns = leaf_config.get("raw_outlier_columns")
    if raw_outlier_policy != "preserve_raw_outliers" and raw_outlier_columns is not None:
        _require_non_empty_sequence(
            leaf_config,
            "raw_outlier_columns",
            f"raw_outlier_policy={raw_outlier_policy!r}",
        )

    release_lag_rule = _selection_value(selection_map, "release_lag_rule", default="ignore_release_lag")
    if release_lag_rule == "series_specific_lag":
        _require_non_empty_mapping(
            leaf_config,
            "release_lag_per_series",
            "release_lag_rule='series_specific_lag'",
        )


def _validate_layer2_feature_block_contract(
    selection_map: dict[str, AxisSelection],
    leaf_config: dict[str, Any],
) -> None:
    level_feature_block = _selection_value(selection_map, "level_feature_block", default="none")
    if level_feature_block == "selected_level_addbacks":
        _require_non_empty_sequence(
            leaf_config,
            "selected_level_addback_columns",
            "level_feature_block='selected_level_addbacks'",
        )
    if level_feature_block == "level_growth_pairs":
        _require_non_empty_sequence(
            leaf_config,
            "level_growth_pair_columns",
            "level_feature_block='level_growth_pairs'",
        )
    rotation_feature_block = _selection_value(selection_map, "rotation_feature_block", default="none")
    if rotation_feature_block == "marx_rotation":
        if leaf_config.get("marx_max_lag") is None:
            raise CompileValidationError(
                "rotation_feature_block='marx_rotation' requires leaf_config.marx_max_lag"
            )
        try:
            marx_max_lag = int(leaf_config["marx_max_lag"])
        except (TypeError, ValueError) as exc:
            raise CompileValidationError(
                "rotation_feature_block='marx_rotation' requires leaf_config.marx_max_lag to be a positive integer"
            ) from exc
        if marx_max_lag <= 0:
            raise CompileValidationError(
                "rotation_feature_block='marx_rotation' requires leaf_config.marx_max_lag to be a positive integer"
            )


def _data_task_spec(selection_map: dict[str, AxisSelection], leaf_config: dict[str, Any]) -> dict[str, Any]:
    dataset = _first_selected_value(selection_map, "dataset", "fred_md")
    source_adapter = _selection_value(selection_map, "source_adapter", default=dataset)
    target_structure = _first_selected_value(selection_map, "target_structure", "single_target_point_forecast")
    feature_builder = _first_selected_value(selection_map, "feature_builder", "autoreg_lagged_target")
    information_set_type = _first_selected_value(selection_map, "information_set_type", "revised")
    custom_blocks = leaf_config.get("custom_feature_blocks") or {}
    if not isinstance(custom_blocks, dict):
        custom_blocks = {}
    return {
        "custom_data_path": leaf_config.get("custom_data_path"),
        "source_adapter": source_adapter,
        "target_structure": target_structure,
        "official_transform_policy": _official_transform_policy(selection_map),
        "official_transform_scope": _official_transform_scope(selection_map),
        "official_transform_source": _official_transform_source_payload(selection_map),
        "frequency": _selection_value(selection_map, "frequency", default=_DATASET_DEFAULT_FREQUENCY.get(dataset, "monthly")),
        "information_set_type": information_set_type,
        "overlap_handling": _selection_value(selection_map, "overlap_handling", default="allow_overlap"),
        "sample_start_date": leaf_config.get("sample_start_date"),
        "sample_end_date": leaf_config.get("sample_end_date"),
        # 1.4 variable_universe input channels
        "variable_universe_category": leaf_config.get("variable_universe_category"),
        "variable_universe_category_columns": leaf_config.get("variable_universe_category_columns"),
        "target_specific_columns": leaf_config.get("target_specific_columns"),
        "variable_universe_columns": leaf_config.get("variable_universe_columns"),
        # 1.4 predictor_family input channels
        "handpicked_columns": leaf_config.get("handpicked_columns"),
        "predictor_category": leaf_config.get("predictor_category"),
        "predictor_category_columns": leaf_config.get("predictor_category_columns"),
        # Layer 2 feature-block input channels
        "selected_level_addback_columns": leaf_config.get("selected_level_addback_columns"),
        "level_growth_pair_columns": leaf_config.get("level_growth_pair_columns"),
        "marx_max_lag": leaf_config.get("marx_max_lag"),
        "custom_feature_blocks": leaf_config.get("custom_feature_blocks"),
        "custom_temporal_feature_block": leaf_config.get("custom_temporal_feature_block") or custom_blocks.get("temporal"),
        "custom_rotation_feature_block": leaf_config.get("custom_rotation_feature_block") or custom_blocks.get("rotation"),
        "custom_factor_feature_block": leaf_config.get("custom_factor_feature_block") or custom_blocks.get("factor"),
        "custom_feature_combiner": (
            leaf_config.get("custom_feature_combiner")
            or leaf_config.get("custom_combiner")
            or leaf_config.get("custom_feature_block_combiner")
            or custom_blocks.get("combiner")
            or custom_blocks.get("feature_combiner")
        ),
        # 1.4 benchmark_family input channels
        "benchmark_suite": leaf_config.get("benchmark_suite"),
        "paper_forecast_series": leaf_config.get("paper_forecast_series"),
        "survey_forecast_series": leaf_config.get("survey_forecast_series"),
        # 1.4 deterministic_components input channels
        "break_dates": leaf_config.get("break_dates"),
        # 1.5 release_lag_rule + missing_availability + contemporaneous_x_rule input channels
        "release_lag_per_series": leaf_config.get("release_lag_per_series"),
        "x_imputation": leaf_config.get("x_imputation"),
        # Layer 1 full raw-source cleaning before official transforms/T-codes
        "raw_x_imputation": leaf_config.get("raw_x_imputation"),
        "raw_outlier_columns": leaf_config.get("raw_outlier_columns"),
        # FRED-SD inferred t-codes are opt-in research metadata, not source
        # metadata. Runtime consumes these fields before t-code preprocessing.
        "sd_tcode_policy": leaf_config.get("sd_tcode_policy", "none"),
        "sd_tcode_map_version": leaf_config.get("sd_tcode_map_version"),
        "sd_tcode_allowed_statuses": leaf_config.get("sd_tcode_allowed_statuses"),
        # Compatibility mirror: `oos_period` is a Layer 4 evaluation axis.
        # Keep it in data_task_spec for older runtime/readers until the
        # migration window closes, but treat evaluation_spec as canonical.
        "oos_period": _selection_value(selection_map, "oos_period", default="all_oos_data"),
        "missing_availability": _selection_value(selection_map, "missing_availability", default="zero_fill_before_start"),
        "raw_missing_policy": _selection_value(selection_map, "raw_missing_policy", default="preserve_raw_missing"),
        "raw_outlier_policy": _selection_value(selection_map, "raw_outlier_policy", default="preserve_raw_outliers"),
        "release_lag_rule": _selection_value(selection_map, "release_lag_rule", default="ignore_release_lag"),
        "benchmark_family": _selection_value(selection_map, "benchmark_family"),
        "data_vintage": leaf_config.get("data_vintage"),
        "exogenous_x_path_policy": leaf_config.get("exogenous_x_path_policy")
        or leaf_config.get("future_x_path_policy"),
        "scheduled_known_future_x_columns": leaf_config.get("scheduled_known_future_x_columns")
        or leaf_config.get("known_future_x_columns"),
        "recursive_x_model_family": leaf_config.get("recursive_x_model_family")
        or leaf_config.get("future_x_model_family"),
    }


def _target_lag_block_value(selection_map: dict[str, AxisSelection]) -> str | None:
    return (
        _selection_value(selection_map, "target_lag_block")
        if "target_lag_block" in selection_map
        else None
    )


def _target_lag_selection_from_block(block: str | None) -> str | None:
    if block is None:
        return None
    return _TARGET_LAG_BLOCK_TO_SELECTION.get(block, "custom")


def _target_lag_block_from_selection_value(selection: str) -> str:
    return _TARGET_LAG_SELECTION_TO_BLOCK.get(selection, "custom_target_lags")


def _legacy_y_lag_count_default(
    selection_map: dict[str, AxisSelection],
    *,
    model_family: str,
) -> str:
    selection_from_block = _target_lag_selection_from_block(_target_lag_block_value(selection_map))
    if selection_from_block is not None:
        return _TARGET_LAG_SELECTION_TO_LEGACY_Y.get(selection_from_block, "model_specific")
    return "IC_select" if model_family == "ar" else "fixed"


def _target_lag_selection_value(
    selection_map: dict[str, AxisSelection],
    *,
    legacy_y_lag_count: str,
) -> str:
    target_lag_block = _target_lag_block_value(selection_map)
    selection_default = (
        _target_lag_selection_from_block(target_lag_block)
        or _target_lag_selection_from_legacy_y_lag_count(str(legacy_y_lag_count))
    )
    selection = _selection_value(selection_map, "target_lag_selection", default=selection_default)
    if target_lag_block is not None:
        expected_block = _target_lag_block_from_selection_value(selection)
        if target_lag_block != expected_block:
            raise CompileValidationError(
                "target_lag_block conflicts with target_lag_selection; "
                f"got target_lag_block={target_lag_block!r}, "
                f"target_lag_selection={selection!r} -> {expected_block!r}"
            )
    return selection


def _training_spec(selection_map: dict[str, AxisSelection], leaf_config: dict[str, Any]) -> dict[str, Any]:
    framework = _first_selected_value(selection_map, "framework", "expanding")
    model_family = _first_selected_value(selection_map, "model_family", "ar")
    feature_builder = _first_selected_value(selection_map, "feature_builder", "autoreg_lagged_target")
    feature_runtime = _feature_runtime_for_validation(selection_map, fallback_feature_builder=str(feature_builder))
    forecast_type_default = _layer3_forecast_type_default(feature_runtime)
    training_cfg = dict(leaf_config.get("training_config", {}))
    custom_preprocessor = _selection_value(selection_map, "custom_preprocessor", default="none")
    if custom_preprocessor != "none":
        get_custom_preprocessor(custom_preprocessor)
    target_transformer = _selection_value(selection_map, "target_transformer", default="none")
    if target_transformer != "none":
        get_custom_target_transformer(target_transformer)
    legacy_y_lag_count = _selection_value(
        selection_map,
        "y_lag_count",
        default=_legacy_y_lag_count_default(selection_map, model_family=model_family),
    )
    _target_lag_selection_value(
        selection_map,
        legacy_y_lag_count=str(legacy_y_lag_count),
    )
    _validate_legacy_factor_ar_lags(training_cfg)
    return {
        "outer_window": _selection_value(selection_map, "outer_window", default=framework),
        "refit_policy": _selection_value(selection_map, "refit_policy", default="refit_every_step"),
        "sequence_framework": _selection_value(selection_map, "sequence_framework", default="not_sequence"),
        "horizon_modelization": _selection_value(selection_map, "horizon_modelization", default="separate_model_per_h"),
        "validation_size_rule": _selection_value(selection_map, "validation_size_rule", default="ratio"),
        "validation_location": _selection_value(selection_map, "validation_location", default="last_block"),
        "embargo_gap": _selection_value(selection_map, "embargo_gap", default="none"),
        "split_family": _selection_value(selection_map, "split_family", default="time_split"),
        "shuffle_rule": _selection_value(selection_map, "shuffle_rule", default="forbidden_for_time_series"),
        "alignment_fairness": _selection_value(selection_map, "alignment_fairness", default="same_split_across_models"),
        "search_algorithm": _selection_value(selection_map, "search_algorithm", default="grid_search"),
        "tuning_objective": _selection_value(selection_map, "tuning_objective", default="validation_mse"),
        "tuning_budget": _selection_value(selection_map, "tuning_budget", default="max_trials"),
        "hp_space_style": _selection_value(selection_map, "hp_space_style", default="discrete_grid"),
        "seed_policy": _selection_value(selection_map, "seed_policy", default="fixed_seed"),
        "early_stopping": _selection_value(selection_map, "early_stopping", default="none"),
        "convergence_handling": _selection_value(selection_map, "convergence_handling", default="mark_fail"),
        "y_lag_count": legacy_y_lag_count,
        "lookback": _selection_value(selection_map, "lookback", default="fixed_lookback"),
        "logging_level": _selection_value(selection_map, "logging_level", default="silent"),
        "checkpointing": _selection_value(selection_map, "checkpointing", default="none"),
        "cache_policy": _selection_value(selection_map, "cache_policy", default="no_cache"),
        "execution_backend": _selection_value(selection_map, "execution_backend", default="local_cpu"),
        "forecast_type": _selection_value(selection_map, "forecast_type", default=forecast_type_default),
        "forecast_object": _selection_value(selection_map, "forecast_object", default="point_mean"),
        "interval_coverage": leaf_config.get("interval_coverage", 0.9),
        "min_train_size": _selection_value(selection_map, "min_train_size", default="fixed_n_obs"),
        "training_start_rule": _selection_value(selection_map, "training_start_rule", default="earliest_possible"),
        "training_start_date": leaf_config.get("training_start_date"),
        "quantile_level": leaf_config.get("quantile_level", 0.5),
        "validation_ratio": training_cfg.get("validation_ratio", 0.2),
        "validation_n": training_cfg.get("validation_n", 5),
        "validation_years": training_cfg.get("validation_years", 1),
        "obs_per_year": training_cfg.get("obs_per_year", 12),
        "max_trials": training_cfg.get("max_trials", 6),
        "max_time_seconds": training_cfg.get("max_time_seconds", 15.0),
        "early_stop_trials": training_cfg.get("early_stop_trials", 3),
        "early_stop_min_delta": training_cfg.get("early_stop_min_delta", 1e-4),
        "embargo_gap_size": training_cfg.get("embargo_gap_size", 0),
        "refit_k_steps": training_cfg.get("refit_k_steps", 3),
        "anchored_max_window_size": training_cfg.get("anchored_max_window_size", 60),
        "random_seed": leaf_config.get("random_seed", 42),
    }


def _validate_legacy_factor_ar_lags(training_cfg: Mapping[str, Any]) -> None:
    if "target_lag_count" in training_cfg and "factor_ar_lags" in training_cfg:
        if int(training_cfg["target_lag_count"]) != int(training_cfg["factor_ar_lags"]):
            raise CompileValidationError(
                "training_config.factor_ar_lags is a legacy target-lag alias; "
                "use training_config.factor_lag_count for factor lag features when it differs "
                "from training_config.target_lag_count"
            )


def _lag_count_from_training_config(
    leaf_config: Mapping[str, Any],
    training_spec: Mapping[str, Any] | None,
    *,
    primary_key: str,
    legacy_key: str = "factor_ar_lags",
) -> tuple[Any, str]:
    training_cfg = dict(leaf_config.get("training_config", {}) or {})
    legacy_training = dict(training_spec or {})
    _validate_legacy_factor_ar_lags(training_cfg)
    if primary_key in training_cfg:
        return training_cfg[primary_key], primary_key
    if legacy_key in training_cfg:
        return training_cfg[legacy_key], f"legacy_{legacy_key}"
    if primary_key in legacy_training:
        return legacy_training[primary_key], f"legacy_training_spec.{primary_key}"
    if legacy_key in legacy_training:
        return legacy_training[legacy_key], f"legacy_training_spec.{legacy_key}"
    return 1, "default"


def _target_lag_count_config(
    leaf_config: Mapping[str, Any],
    training_spec: Mapping[str, Any] | None = None,
) -> tuple[Any, str]:
    return _lag_count_from_training_config(
        leaf_config,
        training_spec,
        primary_key="target_lag_count",
    )


def _factor_lag_count_config(
    leaf_config: Mapping[str, Any],
    training_spec: Mapping[str, Any] | None = None,
) -> tuple[Any, str]:
    return _lag_count_from_training_config(
        leaf_config,
        training_spec,
        primary_key="factor_lag_count",
    )


def _factor_count_config(
    selection_map: dict[str, AxisSelection],
    leaf_config: dict[str, Any],
    training_spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    training_cfg = dict(leaf_config.get("training_config", {}) or {})
    legacy_training = dict(training_spec or {})
    return {
        "mode": _selection_value(
            selection_map,
            "factor_count",
            default=legacy_training.get("factor_count", "fixed"),
        ),
        "fixed_factor_count": int(training_cfg.get("fixed_factor_count", legacy_training.get("fixed_factor_count", 3))),
        "max_factors": int(training_cfg.get("max_factors", legacy_training.get("max_factors", 5))),
        "selection_scope": "train_window",
    }


def _target_lag_selection_from_legacy_y_lag_count(value: str) -> str:
    mapping = {
        "fixed": "fixed",
        "IC_select": "ic_select",
        "cv_select": "cv_select",
        "model_specific": "custom",
    }
    return mapping.get(value, "custom")


def _target_lag_block_from_selection(
    selection: str,
    *,
    source_axis: str,
    source_value: Any,
    lag_count: Any = 1,
) -> dict[str, Any]:
    block = _target_lag_block_from_selection_value(selection)
    payload: dict[str, Any] = {
        "value": block,
        "source_axis": source_axis,
        "source_value": source_value,
        "target_lag_selection": selection,
        "runtime_bridge": {
            "target_lag_count": lag_count,
        },
    }
    try:
        lag_count_i = int(lag_count)
    except (TypeError, ValueError):
        lag_count_i = 0
    if block == "fixed_target_lags":
        payload.update(
            {
                "lag_orders": list(range(1, lag_count_i + 1)),
                "feature_names": [f"target_lag_{lag}" for lag in range(1, lag_count_i + 1)],
                "runtime_block": {
                    "matrix_composition": "fixed_target_lags",
                    "lag_count": lag_count_i,
                },
                "alignment": {
                    "train_row_t_uses": "target_{origin_t-k+1}",
                    "prediction_origin_uses": "target_{origin-k+1}",
                    "lookahead": "forbidden",
                },
            }
        )
    if selection in {"cv_select", "custom"}:
        payload["note"] = (
            "target_lag_block does not yet have a dedicated CV/custom runtime path; "
            "explicit target-lag feature execution remains a later Layer 2 block task"
        )
    return payload


def _x_lag_block_from_bridge(value: str, *, source_axis: str = "x_lag_creation") -> dict[str, Any]:
    mapping = {
        "no_x_lags": "none",
        "fixed_x_lags": "fixed_x_lags",
        "cv_selected_x_lags": "cv_selected_x_lags",
        "variable_specific_lags": "variable_specific_x_lags",
        "category_specific_lags": "category_specific_x_lags",
    }
    block = mapping.get(value, "custom_x_lags")
    payload: dict[str, Any] = {
        "value": block,
        "source_axis": source_axis,
        "source_value": value,
    }
    if block == "fixed_x_lags":
        payload.update(
            {
                "lag_orders": [1],
                "feature_name_pattern": "{predictor}_lag_{k}",
                "runtime_feature_name_pattern": "{predictor}__lag{k}",
                "alignment": {
                    "train_row_t_uses": "X_{t-k}",
                    "prediction_origin_uses": "X_{origin-k}",
                    "lookahead": "forbidden",
                },
                "runtime_bridge": {"x_lag_creation": "fixed_x_lags"},
            }
        )
    return payload


def _x_lag_block_from_selection(selection_map: dict[str, AxisSelection], x_lag_creation: str) -> dict[str, Any]:
    if "x_lag_feature_block" not in selection_map:
        return _x_lag_block_from_bridge(x_lag_creation)
    block = _selection_value(selection_map, "x_lag_feature_block")
    return _x_lag_block_from_bridge(
        _x_lag_creation_from_feature_block(block),
        source_axis="x_lag_feature_block",
    ) | {"source_value": block}


def _factor_feature_block_value(selection_map: dict[str, AxisSelection]) -> str | None:
    return (
        _selection_value(selection_map, "factor_feature_block")
        if "factor_feature_block" in selection_map
        else None
    )


def _level_feature_block_value(selection_map: dict[str, AxisSelection]) -> str | None:
    return (
        _selection_value(selection_map, "level_feature_block")
        if "level_feature_block" in selection_map
        else None
    )


def _level_block_from_selection(
    selection_map: dict[str, AxisSelection],
    data_task_spec: Mapping[str, Any],
) -> dict[str, Any]:
    explicit_block = _level_feature_block_value(selection_map)
    block = explicit_block or "none"
    if block == "target_level_addback":
        return {
            "value": "target_level_addback",
            "source_axis": "level_feature_block",
            "source_value": "target_level_addback",
            "feature_names": ["target_level_origin"],
            "runtime_feature_name": "__target_level_origin",
            "alignment": {
                "train_row_t_uses": "target_t",
                "prediction_origin_uses": "target_origin",
                "lookahead": "forbidden",
            },
            "runtime_bridge": {"raw_panel_level_addback": "target_level_addback"},
        }
    if block == "x_level_addback":
        return {
            "value": "x_level_addback",
            "source_axis": "level_feature_block",
            "source_value": "x_level_addback",
            "feature_name_pattern": "{predictor}_level",
            "runtime_feature_name_pattern": "{predictor}__level",
            "level_source": "Layer 1 H after raw missing/outlier policy and before official transforms/T-codes",
            "alignment": {
                "train_row_t_uses": "H_{t}",
                "prediction_origin_uses": "H_{origin}",
                "lookahead": "forbidden",
            },
            "runtime_bridge": {"raw_panel_level_addback": "x_level_addback"},
        }
    if block == "selected_level_addbacks":
        selected_columns = [str(column) for column in data_task_spec.get("selected_level_addback_columns") or ()]
        return {
            "value": "selected_level_addbacks",
            "source_axis": "level_feature_block",
            "source_value": "selected_level_addbacks",
            "selected_columns": selected_columns,
            "feature_names": [f"{column}_level" for column in selected_columns],
            "runtime_feature_names": [f"{column}__level" for column in selected_columns],
            "level_source": "Layer 1 H after raw missing/outlier policy and before official transforms/T-codes",
            "alignment": {
                "train_row_t_uses": "selected H_{t}",
                "prediction_origin_uses": "selected H_{origin}",
                "lookahead": "forbidden",
            },
            "runtime_bridge": {"raw_panel_level_addback": "selected_level_addbacks"},
        }
    if block == "level_growth_pairs":
        pair_columns = [str(column) for column in data_task_spec.get("level_growth_pair_columns") or ()]
        return {
            "value": "level_growth_pairs",
            "source_axis": "level_feature_block",
            "source_value": "level_growth_pairs",
            "pair_columns": pair_columns,
            "transformed_feature_names": pair_columns,
            "level_feature_names": [f"{column}_level" for column in pair_columns],
            "runtime_level_feature_names": [f"{column}__level" for column in pair_columns],
            "level_source": "Layer 1 H after raw missing/outlier policy and before official transforms/T-codes",
            "pair_semantics": "existing transformed predictor column paired with raw-level H counterpart",
            "alignment": {
                "train_row_t_uses": "X_{t} and H_{t}",
                "prediction_origin_uses": "X_{origin} and H_{origin}",
                "lookahead": "forbidden",
            },
            "runtime_bridge": {"raw_panel_level_addback": "level_growth_pairs"},
        }
    if explicit_block is not None:
        return {
            "value": block,
            "source_axis": "level_feature_block",
            "source_value": explicit_block,
        }
    return {"value": "none", "source": "not_wired"}


def _temporal_feature_block_value(selection_map: dict[str, AxisSelection]) -> str | None:
    return (
        _selection_value(selection_map, "temporal_feature_block")
        if "temporal_feature_block" in selection_map
        else None
    )


def _rotation_feature_block_value(selection_map: dict[str, AxisSelection]) -> str | None:
    return (
        _selection_value(selection_map, "rotation_feature_block")
        if "rotation_feature_block" in selection_map
        else None
    )


def _rotation_block_from_selection(
    selection_map: dict[str, AxisSelection],
    data_task_spec: dict[str, Any],
) -> dict[str, Any]:
    explicit_block = _rotation_feature_block_value(selection_map)
    block = explicit_block or "none"
    if block == "none" and explicit_block is not None:
        return {
            "value": "none",
            "source_axis": "rotation_feature_block",
            "source_value": "none",
        }
    if block == "moving_average_rotation":
        return {
            "value": "moving_average_rotation",
            "source_axis": "rotation_feature_block",
            "source_value": "moving_average_rotation",
            "windows": [3, 6],
            "rotation_construction": "deterministic trailing moving-average rotations of active X columns",
            "feature_name_patterns": ["{predictor}_rotma3", "{predictor}_rotma6"],
            "runtime_feature_name_patterns": ["{predictor}__rotma3", "{predictor}__rotma6"],
            "alignment": {
                "train_row_t_uses": "X_t and trailing observed X history within each rotation window",
                "prediction_origin_uses": "X_origin and trailing observed X history within each rotation window",
                "lookahead": "forbidden",
            },
            "runtime_bridge": {"raw_panel_rotation_features": "moving_average_rotation"},
            "scope_note": "generic moving-average rotation primitive; MARX uses lag-polynomial basis replacement while MAF/custom rotations remain separate blocks",
        }
    if block == "marx_rotation":
        max_lag = int(data_task_spec["marx_max_lag"])
        contract = _build_marx_rotation_contract(max_lag=max_lag)
        return {
            "value": "marx_rotation",
            "source_axis": "rotation_feature_block",
            "source_value": "marx_rotation",
            "runtime_status": "operational",
            "max_lag": max_lag,
            "rotation_orders": list(range(1, max_lag + 1)),
            "feature_name_pattern": contract["rotated_feature_name_pattern"],
            "runtime_feature_name_pattern": contract["rotated_runtime_feature_name_pattern"],
            "required_runtime_contract": contract["composer_contract"],
            "composer_contract": contract,
            "alignment": contract["alignment"],
            "basis_policy": contract["basis_policy"],
            "duplicate_base_policy": contract["duplicate_base_policy"],
            "initial_lag_fill_policy": contract["initial_lag_fill_policy"],
            "composition_modes": _MARX_COMPOSITION_MODES,
            "runtime_bridge": {"raw_panel_rotation_features": "marx_rotation"},
            "scope_note": "MARX is a preset over lag-polynomial rotation; it replaces the X lag-polynomial basis rather than appending generic moving-average rotation features",
        }
    if block == "maf_rotation":
        contract = {
            "schema_version": "factor_score_rotation_contract_v1",
            "runtime_status": "operational",
            "runtime_builder": "build_factor_score_moving_average_rotation",
            "source_block": "pca_static_factors",
            "windows": [3, 6],
            "source_feature_name_pattern": "factor_{k}",
            "rotated_feature_name_pattern": "factor_{k}_maf_ma{window}",
            "rotated_runtime_feature_name_pattern": "factor_{k}__maf_ma{window}",
            "alignment": {
                "train_row_t_uses": "factor scores estimated from X_t under train-window factor loadings",
                "prediction_origin_uses": "factor scores estimated from X_origin under train-window factor loadings",
                "lookahead": "forbidden",
            },
            "basis_policy": "factor_score_trailing_moving_average",
        }
        return {
            "value": "maf_rotation",
            "source_axis": "rotation_feature_block",
            "source_value": "maf_rotation",
            "runtime_status": "operational",
            "windows": [3, 6],
            "feature_name_patterns": ["factor_{k}_maf_ma3", "factor_{k}_maf_ma6"],
            "runtime_feature_name_patterns": ["factor_{k}__maf_ma3", "factor_{k}__maf_ma6"],
            "required_runtime_contract": contract["schema_version"],
            "composer_contract": contract,
            "alignment": contract["alignment"],
            "basis_policy": contract["basis_policy"],
            "composition_modes": _MAF_COMPOSITION_MODES,
            "runtime_bridge": {
                "raw_panel_rotation_features": "maf_rotation",
                "source_block": "factor_scores",
            },
            "required_semantics": [
                "fit factor scores on the training window",
                "rotate or smooth factor-score histories with explicit fit/apply provenance",
                "compose factor and rotation blocks without leaking prediction-origin information",
            ],
            "scope_note": "MAF requires factor-to-rotation block composition and is not a raw-X moving-average append",
        }
    if block == "custom_rotation":
        custom_name = data_task_spec.get("custom_rotation_feature_block")
        registered = bool(custom_name and is_custom_feature_block(str(custom_name), block_kind="rotation"))
        payload = {
            "value": "custom_rotation",
            "source_axis": "rotation_feature_block",
            "source_value": "custom_rotation",
            "runtime_status": "operational" if registered else "registry_only",
            "required_runtime_contract": CUSTOM_FEATURE_BLOCK_CONTRACT_VERSION,
            "custom_feature_block": str(custom_name) if custom_name else None,
            "callable_contract": custom_feature_block_contract_metadata(block_kind="rotation"),
            "required_semantics": [
                "return train and prediction rotation feature frames",
                "return stable public and runtime feature names",
                "return fit-state provenance and leakage metadata",
            ],
            "scope_note": "custom_rotation needs a block-local callable contract; the broad custom_preprocessor hook is not enough",
        }
        if registered:
            payload["runtime_bridge"] = {"custom_feature_block": str(custom_name), "block_kind": "rotation"}
        return payload
    if explicit_block is not None:
        return {
            "value": block,
            "source_axis": "rotation_feature_block",
            "source_value": explicit_block,
        }
    return {"value": "none", "source": "not_wired"}


def _factor_rotation_order_from_selection(
    selection_map: dict[str, AxisSelection],
) -> dict[str, Any]:
    explicit_value = (
        _selection_value(selection_map, "factor_rotation_order", default=None)
        if "factor_rotation_order" in selection_map
        else None
    )
    rotation_feature_block = _selection_value(selection_map, "rotation_feature_block", default="none")
    value = str(explicit_value or ("factor_then_rotation" if rotation_feature_block == "maf_rotation" else "rotation_then_factor"))
    return {
        "value": value,
        "source_axis": "factor_rotation_order" if explicit_value is not None else "default_by_rotation_feature_block",
        "source_value": explicit_value if explicit_value is not None else value,
        "rule": (
            "Layer 2 owns whether factor and rotation blocks are composed as rotation-then-factor "
            "or factor-then-rotation; Layer 3 consumes only the resulting Z matrix"
        ),
    }


def _temporal_block_from_selection(selection_map: dict[str, AxisSelection], data_task_spec: dict[str, Any]) -> dict[str, Any]:
    explicit_block = _temporal_feature_block_value(selection_map)
    block = explicit_block or "none"
    if block == "local_temporal_factors":
        return {
            "value": "local_temporal_factors",
            "source_axis": "temporal_feature_block",
            "source_value": "local_temporal_factors",
            "window": 3,
            "factor_construction": "deterministic cross-sectional summaries with trailing time smoothing",
            "feature_names": ["local_temporal_factor_mean3", "local_temporal_factor_dispersion3"],
            "runtime_feature_names": ["__local_temporal_factor_mean3", "__local_temporal_factor_dispersion3"],
            "alignment": {
                "train_row_t_uses": "X_{t}, X_{t-1}, X_{t-2}",
                "prediction_origin_uses": "X_{origin}, X_{origin-1}, X_{origin-2}",
                "lookahead": "forbidden",
            },
            "runtime_bridge": {"raw_panel_temporal_features": "local_temporal_factors"},
        }
    if block == "rolling_moments":
        return {
            "value": "rolling_moments",
            "source_axis": "temporal_feature_block",
            "source_value": "rolling_moments",
            "window": 3,
            "moments": ["mean", "variance"],
            "feature_name_patterns": ["{predictor}_mean3", "{predictor}_var3"],
            "runtime_feature_name_patterns": ["{predictor}__mean3", "{predictor}__var3"],
            "alignment": {
                "train_row_t_uses": "X_{t}, X_{t-1}, X_{t-2}",
                "prediction_origin_uses": "X_{origin}, X_{origin-1}, X_{origin-2}",
                "lookahead": "forbidden",
            },
            "runtime_bridge": {"raw_panel_temporal_features": "rolling_moments"},
        }
    if block in {"moving_average_features", "volatility_features"}:
        suffix = "ma3" if block == "moving_average_features" else "vol3"
        bridge = "moving_average_features" if block == "moving_average_features" else "volatility_features"
        return {
            "value": block,
            "source_axis": "temporal_feature_block",
            "source_value": block,
            "window": 3,
            "feature_name_pattern": "{predictor}_" + suffix,
            "runtime_feature_name_pattern": "{predictor}__" + suffix,
            "alignment": {
                "train_row_t_uses": "X_{t}, X_{t-1}, X_{t-2}",
                "prediction_origin_uses": "X_{origin}, X_{origin-1}, X_{origin-2}",
                "lookahead": "forbidden",
            },
            "runtime_bridge": {"raw_panel_temporal_features": bridge},
        }
    if block == "custom_temporal_features":
        custom_name = data_task_spec.get("custom_temporal_feature_block")
        registered = bool(custom_name and is_custom_feature_block(str(custom_name), block_kind="temporal"))
        return {
            "value": "custom_temporal_features",
            "source_axis": "temporal_feature_block",
            "source_value": "custom_temporal_features",
            "runtime_status": "operational" if registered else "registry_only",
            "required_runtime_contract": CUSTOM_FEATURE_BLOCK_CONTRACT_VERSION,
            "custom_feature_block": str(custom_name) if custom_name else None,
            "callable_contract": custom_feature_block_contract_metadata(block_kind="temporal"),
            "required_semantics": [
                "return train and prediction temporal feature frames",
                "return stable public and runtime feature names",
                "return fit-state provenance and leakage metadata",
            ],
            "runtime_bridge": (
                {"custom_feature_block": str(custom_name), "block_kind": "temporal"}
                if registered
                else {}
            ),
            "scope_note": "custom_temporal_features is a block-local callable contract; custom_preprocessor is a broader matrix hook",
        }
    if explicit_block is not None:
        return {
            "value": block,
            "source_axis": "temporal_feature_block",
            "source_value": explicit_block,
        }
    return {"value": "none", "source": "not_wired"}


def _factor_block_from_bridge(
    *,
    feature_builder: str,
    dimred: str,
    factor_count_config: Mapping[str, Any],
    factor_lag_count: Any,
    factor_lag_count_source: str,
    data_task_spec: dict[str, Any],
    preprocess_contract,
    explicit_block: str | None,
    rotation_feature_block: str = "none",
    factor_rotation_order: str = "rotation_then_factor",
) -> dict[str, Any]:
    inferred = "pca_static_factors" if feature_builder in _FACTOR_BRIDGE_BUILDERS else _DIMRED_TO_FACTOR_FEATURE_BLOCK.get(dimred, "custom_factors")
    block = explicit_block or inferred
    builtin_factor_blocks = {"pca_static_factors", "pca_factor_lags", "supervised_factors"}
    runtime_bridge: dict[str, Any] = {}
    if feature_builder in _FACTOR_BRIDGE_BUILDERS:
        runtime_bridge["feature_builder"] = feature_builder
    if dimred in _FACTOR_DIMRED_BRIDGES:
        runtime_bridge["dimensionality_reduction_policy"] = dimred
    composition_modes = {"operational": [], "gated": []}
    supported_rotation_semantics: list[str] = []
    active_rotation_semantic = "none"
    rotation_rule = "factor and rotation blocks are inactive or not composed"
    if block == "pca_static_factors" and rotation_feature_block == "marx_rotation":
        composition_modes = _MARX_COMPOSITION_MODES
        supported_rotation_semantics = ["marx_then_factor", "factor_then_marx"]
        active_rotation_semantic = (
            "factor_then_marx"
            if factor_rotation_order == "factor_then_rotation"
            else "marx_then_factor"
        )
        rotation_rule = (
            "when rotation_feature_block='marx_rotation' is active with pca_static_factors, "
            "factor_rotation_order selects whether MARX replaces the X lag-polynomial basis before "
            "static factors are fit, or static factor-score histories are rotated by the MARX basis"
        )
    elif block == "pca_static_factors" and rotation_feature_block == "maf_rotation":
        composition_modes = _MAF_COMPOSITION_MODES
        supported_rotation_semantics = ["factor_then_maf"]
        active_rotation_semantic = "factor_then_maf"
        rotation_rule = (
            "maf_rotation is defined as a factor-score rotation: static factor-score histories are "
            "estimated first and then smoothed by trailing moving-average factor rotations"
        )

    payload: dict[str, Any] = {
        "value": block,
        "source_axis": "factor_feature_block" if explicit_block is not None else "feature_builder/dimensionality_reduction_policy",
        "source_value": explicit_block if explicit_block is not None else {"feature_builder": feature_builder, "dimensionality_reduction_policy": dimred},
        "runtime_bridge": runtime_bridge,
        "runtime_block": (
            {"matrix_composition": "pca_static_factors", "default_dimensionality_reduction_policy": "pca"}
            if block == "pca_static_factors"
            else (
                {
                    "matrix_composition": "pca_factor_lags",
                    "default_dimensionality_reduction_policy": "pca",
                    "factor_lag_count": int(factor_lag_count),
                    "factor_lag_count_source": factor_lag_count_source,
                }
                if block == "pca_factor_lags"
                else {}
            )
        ),
        "feature_selection_interaction": {
            "feature_selection_policy": getattr(preprocess_contract, "feature_selection_policy", "none"),
            "supported_semantics": ["select_before_factor", "select_after_factor"],
            "gated_semantics": [
                "select_after_custom_blocks",
            ],
            "active_semantic": (
                getattr(preprocess_contract, "feature_selection_semantics", "select_before_factor")
                if block in builtin_factor_blocks and getattr(preprocess_contract, "feature_selection_policy", "none") != "none"
                else "none"
            ),
            "rule": (
                "when feature_selection_policy is active with an executable built-in factor block, the runtime supports "
                "select_before_factor (select raw predictor X, then fit factors) and "
                "select_after_factor (fit factors, optionally append target lags and deterministic columns, "
                "then select among final Z columns)"
            ),
        },
        "rotation_interaction": {
            "rotation_feature_block": rotation_feature_block,
            "factor_rotation_order": factor_rotation_order,
            "composition_modes": composition_modes,
            "supported_semantics": supported_rotation_semantics,
            "active_semantic": active_rotation_semantic,
            "rule": rotation_rule,
        },
    }
    if block == "pca_static_factors":
        fixed_count = int(factor_count_config.get("fixed_factor_count", 3))
        max_factors = int(factor_count_config.get("max_factors", 5))
        payload.update(
            {
                "factor_count": {
                    "mode": factor_count_config.get("mode", "fixed"),
                    "fixed_factor_count": fixed_count,
                    "max_factors": max_factors,
                    "selection_scope": "train_window",
                },
                "feature_name_pattern": "factor_{k}",
                "feature_names": [f"factor_{idx}" for idx in range(1, fixed_count + 1)],
                "loadings_artifact": "feature_representation_fit_state.json",
                "alignment": {
                    "train_window_fit": "fit factor loadings on Z_train source X only",
                    "prediction_origin_apply": "apply fitted loadings to X at the prediction origin",
                    "lookahead": "forbidden",
                },
                "fit_scope": getattr(preprocess_contract, "preprocess_fit_scope", "train_only"),
            }
        )
    if block in {"pca_factor_lags", "supervised_factors", "custom_factors"}:
        required_contract = {
            "pca_factor_lags": "factor_lag_block_composer_v1",
            "supervised_factors": "supervised_factor_block_contract_v1",
            "custom_factors": CUSTOM_FEATURE_BLOCK_CONTRACT_VERSION,
        }[block]
        custom_name = data_task_spec.get("custom_factor_feature_block")
        custom_registered = block == "custom_factors" and bool(
            custom_name and is_custom_feature_block(str(custom_name), block_kind="factor")
        )
        builtin_operational = block in {"pca_factor_lags", "supervised_factors"}
        payload.update(
            {
                "runtime_status": "operational" if (custom_registered or builtin_operational) else "registry_only",
                "required_runtime_contract": required_contract,
                "custom_feature_block": str(custom_name) if custom_name else None,
                **(
                    {
                        "factor_lag_count": int(factor_lag_count),
                        "factor_lag_count_source": factor_lag_count_source,
                    }
                    if block == "pca_factor_lags"
                    else {}
                ),
                "runtime_bridge": (
                    {"custom_feature_block": str(custom_name), "block_kind": "factor"}
                    if custom_registered
                    else runtime_bridge
                ),
                "callable_contract": (
                    custom_feature_block_contract_metadata(block_kind="factor")
                    if block == "custom_factors"
                    else {}
                ),
                "note": (
                    "factor_feature_block has a dedicated train-window fit/apply runtime path"
                    if builtin_operational or custom_registered
                    else "factor_feature_block is representable, but this value does not yet "
                    "have a dedicated train-window fit/apply runtime path"
                ),
            }
        )
    return payload


def _feature_block_set_from_bridge(feature_builder: str, data_richness_mode: str) -> dict[str, Any]:
    if feature_builder == "autoreg_lagged_target":
        value = "target_lags_only"
    elif feature_builder == "factors_plus_AR":
        value = "factors_plus_target_lags"
    elif feature_builder in {"raw_feature_panel", "raw_X_only"}:
        value = {
            "target_lags_only": "target_lags_only",
            "factor_plus_lags": "factors_plus_target_lags",
            "full_high_dimensional_X": "high_dimensional_x",
            "selected_sparse_X": "selected_sparse_x",
            "mixed_mode": "mixed_blocks",
        }.get(data_richness_mode, "transformed_x")
    elif feature_builder == "factor_pca":
        value = "factor_blocks_only"
    else:
        value = "legacy_feature_builder_bridge"
    return {
        "value": value,
        "source_axes": ["feature_builder", "data_richness_mode"],
        "source_values": {
            "feature_builder": feature_builder,
            "data_richness_mode": data_richness_mode,
        },
    }


def _feature_block_combination_from_bridge(
    feature_builder: str,
    x_lag_creation: str,
    *,
    explicit_value: str | None = None,
    data_task_spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if explicit_value is not None:
        value = explicit_value
    elif feature_builder == "factors_plus_AR":
        value = "append_to_target_lags"
    elif x_lag_creation != "no_x_lags":
        value = "append_to_base_x"
    elif feature_builder in {"raw_feature_panel", "raw_X_only", "factor_pca", "autoreg_lagged_target"}:
        value = "replace_with_blocks"
    else:
        value = "concatenate_named_blocks"
    operational_values = {
        "replace_with_blocks",
        "append_to_base_x",
        "append_to_target_lags",
        "concatenate_named_blocks",
    }
    runtime_status = "operational" if value in operational_values else "registry_only"
    payload: dict[str, Any] = {
        "value": value,
        "source_axes": ["feature_builder", "x_lag_creation"],
        "source_values": {
            "feature_builder": feature_builder,
            "x_lag_creation": x_lag_creation,
        },
        "explicit_axis_value": explicit_value,
        "runtime_status": runtime_status,
        "sweep_axis_status": "public_axis_with_runtime_pruning",
        "governance_note": (
            "feature_block_combination summarizes the effective composer today; "
            "invalid compositions are rejected by compiler/runtime compatibility gates"
        ),
    }
    if value == "custom_combiner":
        spec = dict(data_task_spec or {})
        custom_name = spec.get("custom_feature_combiner")
        registered = bool(custom_name and is_custom_feature_combiner(str(custom_name)))
        payload.update(
            {
                "runtime_status": "operational" if registered else "registry_only",
                "required_runtime_contract": "custom_feature_combiner_v1",
                "custom_feature_combiner": str(custom_name) if custom_name else None,
                "callable_contract": custom_feature_combiner_contract_metadata(),
                "required_semantics": [
                    "receive named train and prediction block frames",
                    "return final Z_train/Z_pred with stable public feature names",
                    "return block roles, fit-state provenance, and leakage metadata",
                ],
                "scope_note": "custom_combiner owns final Layer 2 block composition before Layer 3 consumes Z",
            }
        )
        if registered:
            payload["runtime_bridge"] = {"custom_feature_combiner": str(custom_name)}
    return payload


def _deterministic_block_from_selection(
    *,
    deterministic_components: str,
    structural_break_segmentation: str,
    data_task_spec: Mapping[str, Any],
) -> dict[str, Any]:
    active = deterministic_components != "none" or structural_break_segmentation != "none"
    return {
        "value": "deterministic_features" if active else "none",
        "deterministic_components": deterministic_components,
        "structural_break_segmentation": structural_break_segmentation,
        "break_dates": data_task_spec.get("break_dates"),
        "runtime_status": "operational",
        "source_axes": ["deterministic_components", "structural_break_segmentation"],
    }


def _feature_runtime_for_validation(
    selection_map: dict[str, AxisSelection],
    *,
    fallback_feature_builder: str | None,
) -> str | None:
    """Return the runtime feature family implied by Layer 2 block selections.

    Compiler validation still accepts legacy bridge axes, but gates that refer
    to runtime support should classify the selected feature path the same way
    execution does: explicit blocks first, source bridge only as fallback.
    """
    block_set = _selection_value(selection_map, "feature_block_set", default="")
    raw_block_axes = (
        "x_lag_feature_block",
        "factor_feature_block",
        "level_feature_block",
        "temporal_feature_block",
        "rotation_feature_block",
    )
    raw_block_values = (
        _selection_value(selection_map, axis_name, default="none")
        for axis_name in raw_block_axes
    )
    if block_set in _RAW_PANEL_FEATURE_BLOCK_SETS or any(value != "none" for value in raw_block_values):
        return "raw_feature_panel"

    if fallback_feature_builder in _RAW_PANEL_FEATURE_BUILDERS:
        return "raw_feature_panel"

    target_lag_block = _selection_value(selection_map, "target_lag_block", default="none")
    if block_set == "target_lags_only" or target_lag_block != "none":
        return "autoreg_lagged_target"

    return fallback_feature_builder


def _layer3_forecast_type_default(feature_runtime: str | None) -> str:
    return "iterated" if feature_runtime == "autoreg_lagged_target" else "direct"


def _layer3_capability_rejections(
    *,
    model_family: str | None,
    feature_runtime: str | None,
    forecast_type: str,
    forecast_object: str,
    horizon_target_construction: str = "future_target_level_t_plus_h",
    exogenous_x_path_policy: str = "unavailable",
    target_lag_block: str | None = None,
    scheduled_known_future_x_columns: Sequence[Any] | None = None,
    recursive_x_model_family: str | None = None,
) -> tuple[str, ...]:
    blocked: list[str] = []
    if model_family == "ar" and feature_runtime == "raw_feature_panel":
        blocked.append("raw_feature_panel is not compatible with model_family='ar' in the current runtime slice")
    if model_family == "quantile_linear" and forecast_object not in {"point_median", "quantile"}:
        blocked.append("model_family='quantile_linear' requires forecast_object='point_median' or 'quantile'")
    if forecast_object in {"point_median", "quantile"} and model_family != "quantile_linear":
        blocked.append(
            f"forecast_object={forecast_object!r} requires model_family='quantile_linear' in the current runtime slice"
        )
    if forecast_object in {"direction", "interval", "density"} and model_family == "quantile_linear":
        blocked.append(
            f"forecast_object={forecast_object!r} uses the scalar point payload wrapper and is not compatible "
            "with model_family='quantile_linear' in the current runtime slice"
        )
    if forecast_object == "sequence" and not _is_path_average_construction(horizon_target_construction):
        blocked.append(
            "forecast_object='sequence' requires a path-average target construction or the future "
            "sequence_representation_contract_v1"
        )
    if feature_runtime == "raw_feature_panel" and forecast_type == "iterated":
        if model_family in {"pcr", "pls", "factor_augmented_linear", "lstm", "gru", "tcn"}:
            blocked.append(
                "forecast_type='iterated' for raw-panel feature runtime currently supports scalar tabular "
                "generators only, not factor/deep generator families"
            )
        if str(exogenous_x_path_policy) not in {
            "hold_last_observed",
            "observed_future_x",
            "scheduled_known_future_x",
            "recursive_x_model",
        }:
            blocked.append(
                "forecast_type='iterated' for raw-panel feature runtime requires "
                "leaf_config.exogenous_x_path_policy in "
                "{'hold_last_observed', 'observed_future_x', 'scheduled_known_future_x', "
                "'recursive_x_model'}"
            )
        if str(exogenous_x_path_policy) == "scheduled_known_future_x" and not scheduled_known_future_x_columns:
            blocked.append(
                "forecast_type='iterated' with exogenous_x_path_policy='scheduled_known_future_x' "
                "requires leaf_config.scheduled_known_future_x_columns"
            )
        if str(exogenous_x_path_policy) == "recursive_x_model" and str(recursive_x_model_family or "") != "ar1":
            blocked.append(
                "forecast_type='iterated' with exogenous_x_path_policy='recursive_x_model' "
                "requires leaf_config.recursive_x_model_family='ar1'"
            )
        if str(target_lag_block or "none") != "fixed_target_lags":
            blocked.append(
                "forecast_type='iterated' for raw-panel feature runtime requires "
                "target_lag_block='fixed_target_lags' so recursive target-history updates are explicit"
            )
        if forecast_object != "point_mean":
            blocked.append(
                "forecast_type='iterated' for raw-panel feature runtime currently supports forecast_object='point_mean' only"
            )
    if feature_runtime == "autoreg_lagged_target" and forecast_type == "direct":
        blocked.append(
            "forecast_type='direct' is not implemented for the target-lag-only feature runtime in v1.0 "
            "(the operational path is iterated); use forecast_type='iterated' or leave unset to take the dynamic default"
        )
    return tuple(blocked)


_LAYER3_CAPABILITY_STATUS_CATALOG = {
    "operational": "the selected cell has a runtime contract and can execute",
    "operational_narrow": "the selected cell executes for a named narrow contract slice while broader variants remain gated",
    "blocked_by_incompatibility": "the selected values are valid individually but cannot compose in the current runtime",
    "not_supported_yet": "the cell is reserved for a future runtime contract and is not accepted as an executable recipe value",
    "registry_only": "the registry names a placeholder or extension hook without a built-in runtime contract",
}

_LAYER3_PAYLOAD_CONTRACTS = {
    "point_mean": "forecast_payload_v1",
    "point_median": "forecast_payload_v1",
    "quantile": "forecast_payload_v1",
    "direction": "direction_forecast_payload_v1",
    "interval": "interval_forecast_payload_v1",
    "density": "density_forecast_payload_v1",
    "sequence": "sequence_forecast_payload_v1",
}


_LAYER3_FUTURE_CONTRACT_REQUIREMENTS = {
    "sequence_representation_contract_v1": {
        "owner_layer": "2_preprocessing",
        "producer": "sequence_or_tensor_representation_builder",
        "consumer": "sequence_or_tensor_forecast_generator",
        "contract_status": "gated_named",
        "required_fields": [
            "origin_index",
            "sample_axis",
            "lookback_axis",
            "channel_names",
            "target_alignment",
            "fit_state",
            "leakage_metadata",
            "missing_release_lag_handling",
        ],
        "validation_gates": [
            "sample_origin_alignment_test",
            "lookback_no_future_leakage_test",
            "channel_name_schema_test",
            "generator_payload_shape_test",
        ],
    },
    "sequence_forecast_payload_v1": {
        "owner_layer": "3_training",
        "producer": "sequence_or_tensor_forecast_generator",
        "consumer": "artifact_writer_and_evaluation",
        "contract_status": "gated_named",
        "required_fields": [
            "origin_index",
            "horizon",
            "path_or_vector_payload",
            "step_rows",
            "aggregation_rule",
            "payload_metrics",
        ],
        "validation_gates": [
            "jsonl_schema_test",
            "prediction_row_projection_test",
            "metric_aggregation_test",
        ],
    },
    "exogenous_x_path_contract_v1": {
        "owner_layer": "layer1_layer2_layer3_boundary",
        "producer": "scenario_or_future_x_provider",
        "consumer": "raw_panel_iterated_forecast_generator",
        "contract_status": "gated_named",
        "path_kinds": [
            "observed_future_x",
            "scheduled_known_future_x",
            "hold_last_observed",
            "recursive_x_model",
            "unavailable",
        ],
        "required_fields": [
            "path_kind",
            "origin_index",
            "horizon_steps",
            "predictor_names",
            "x_path_frame_or_assumption",
            "availability_mask",
            "vintage_cutoff",
            "release_lag_policy",
            "no_lookahead_evidence",
        ],
        "validation_gates": [
            "future_x_availability_test",
            "release_lag_mask_test",
            "origin_index_alignment_test",
            "scenario_assumption_manifest_test",
        ],
    },
    "multi_step_raw_panel_payload_v1": {
        "owner_layer": "3_training",
        "producer": "raw_panel_iterated_forecast_generator",
        "consumer": "artifact_writer_and_evaluation",
        "contract_status": "gated_named",
        "required_fields": [
            "origin_index",
            "horizon",
            "step_predictions",
            "final_horizon_prediction",
            "target_history_updates",
            "exogenous_x_path_ref",
            "recursive_state_trace",
            "payload_metrics",
        ],
        "validation_gates": [
            "step_trace_schema_test",
            "final_prediction_projection_test",
            "recursive_target_history_test",
            "jsonl_schema_test",
        ],
    },
}


_LAYER3_FUTURE_CAPABILITY_CELLS = (
    {
        "cell_id": "feature_runtime.sequence_tensor",
        "dimension": "feature_runtime",
        "runtime_status": "not_supported_yet",
        "owner_layer": "2_preprocessing",
        "upstream_contract": "sequence_representation_contract_v1",
        "payload_contract": "sequence_forecast_payload_v1",
        "required_contracts": [
            "sequence_representation_contract_v1",
            "sequence_forecast_payload_v1",
        ],
        "contract_requirements": {
            "sequence_representation_contract_v1": _LAYER3_FUTURE_CONTRACT_REQUIREMENTS[
                "sequence_representation_contract_v1"
            ],
            "sequence_forecast_payload_v1": _LAYER3_FUTURE_CONTRACT_REQUIREMENTS[
                "sequence_forecast_payload_v1"
            ],
        },
        "opening_rule": "open only after Layer 2 emits an explicit sequence/tensor representation and Layer 3 writes typed sequence payloads",
        "requires": [
            "Layer 2 sequence/tensor representation handoff",
            "Layer 3 sequence forecast payload coercion",
        ],
    },
    {
        "cell_id": "forecast_type.raw_panel_iterated",
        "dimension": "forecast_type x feature_runtime",
        "runtime_status": "operational_narrow",
        "owner_layer": "3_training",
        "scenario_contract": "exogenous_x_path_contract_v1",
        "payload_contract": "multi_step_raw_panel_payload_v1",
        "required_contracts": [
            "exogenous_x_path_contract_v1",
            "multi_step_raw_panel_payload_v1",
        ],
        "contract_requirements": {
            "exogenous_x_path_contract_v1": _LAYER3_FUTURE_CONTRACT_REQUIREMENTS[
                "exogenous_x_path_contract_v1"
            ],
            "multi_step_raw_panel_payload_v1": _LAYER3_FUTURE_CONTRACT_REQUIREMENTS[
                "multi_step_raw_panel_payload_v1"
            ],
        },
        "opening_rule": "hold_last_observed, observed_future_x, scheduled_known_future_x, and recursive_x_model(ar1) are operational; broader recursively forecast future-X paths remain gated",
        "requires": [
            "explicit hold_last_observed, observed_future_x, scheduled_known_future_x, or recursive_x_model(ar1) exogenous-X scenario",
            "fixed target-lag recursive history",
            "multi-step raw-panel forecast payload artifacts",
        ],
    },
)


def _layer3_capability_matrix(
    selection_map: dict[str, AxisSelection],
    leaf_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    leaf_config = dict(leaf_config or {})
    feature_builder = _selected_value_or_none(selection_map, "feature_builder")
    model_family = _selected_value_or_none(selection_map, "model_family")
    feature_runtime = _feature_runtime_for_validation(
        selection_map,
        fallback_feature_builder=feature_builder,
    )
    forecast_type = _selection_value(
        selection_map,
        "forecast_type",
        default=_layer3_forecast_type_default(feature_runtime),
    )
    forecast_object = _selection_value(selection_map, "forecast_object", default="point_mean")
    horizon_target_construction = _selection_value(
        selection_map,
        "horizon_target_construction",
        default="future_target_level_t_plus_h",
    )
    blocked = _layer3_capability_rejections(
        model_family=model_family,
        feature_runtime=feature_runtime,
        forecast_type=forecast_type,
        forecast_object=forecast_object,
        horizon_target_construction=horizon_target_construction,
        exogenous_x_path_policy=str(
            leaf_config.get("exogenous_x_path_policy")
            or leaf_config.get("future_x_path_policy")
            or "unavailable"
        ),
        target_lag_block=_selection_value(selection_map, "target_lag_block", default="none"),
        scheduled_known_future_x_columns=leaf_config.get("scheduled_known_future_x_columns")
        or leaf_config.get("known_future_x_columns"),
        recursive_x_model_family=leaf_config.get("recursive_x_model_family")
        or leaf_config.get("future_x_model_family"),
    )
    return {
        "schema_version": "layer3_capability_matrix_v1",
        "schema_revision": 6,
        "dimensions": ["model_family", "feature_runtime", "forecast_type", "forecast_object"],
        "canonical_dimensions": [
            "forecast_generator_family",
            "representation_runtime",
            "forecast_protocol",
            "forecast_object",
        ],
        "dimension_aliases": {
            "model_family": "forecast_generator_family",
            "benchmark_family": "baseline_forecast_generator_role",
            "feature_runtime": "representation_runtime",
            "forecast_type": "forecast_protocol",
        },
        "status_catalog": dict(_LAYER3_CAPABILITY_STATUS_CATALOG),
        "future_cells": [dict(cell) for cell in _LAYER3_FUTURE_CAPABILITY_CELLS],
        "rules": {
            "feature_runtime": {
                "ar": {
                    "operational": ["autoreg_lagged_target"],
                    "blocked": {"raw_feature_panel": "AR runtime consumes target-lag-only representation"},
                },
                "default_non_ar": {
                    "operational": ["raw_feature_panel", "autoreg_lagged_target"],
                    "blocked": {},
                },
            },
            "forecast_type": {
                "autoreg_lagged_target": {
                    "default": "iterated",
                    "operational": ["iterated"],
                    "blocked": {"direct": "direct target-lag-only runtime is not implemented"},
                },
                "raw_feature_panel": {
                    "default": "direct",
                    "operational": ["direct"],
                    "conditional_operational": {
                        "iterated": {
                            "requires": [
                                "leaf_config.exogenous_x_path_policy in {'hold_last_observed', 'observed_future_x', 'scheduled_known_future_x', 'recursive_x_model'}",
                                "target_lag_block='fixed_target_lags'",
                                "forecast_object='point_mean'",
                            ],
                            "runtime_contract": "raw_panel_iterated_future_x_path_v1",
                            "payload_contract": "multi_step_raw_panel_payload_v1",
                        }
                    },
                    "blocked": {"iterated": "raw-panel iterated runtime requires an explicit future-X scenario"},
                },
            },
            "forecast_object": {
                "point_mean": {
                    "model_family": "any_non_quantile_linear",
                    "runtime_status": "operational",
                },
                "point_median": {
                    "model_family": "quantile_linear",
                    "runtime_status": "operational",
                },
                "quantile": {
                    "model_family": "quantile_linear",
                    "runtime_status": "operational",
                },
                "direction": {
                    "model_family": "scalar_point_generator",
                    "runtime_status": "operational",
                    "payload_contract": "direction_forecast_payload_v1",
                },
                "interval": {
                    "model_family": "scalar_point_generator",
                    "runtime_status": "operational",
                    "payload_contract": "interval_forecast_payload_v1",
                },
                "density": {
                    "model_family": "scalar_point_generator",
                    "runtime_status": "operational",
                    "payload_contract": "density_forecast_payload_v1",
                },
            },
        },
        "active_cell": {
            "model_family": model_family,
            "feature_builder": feature_builder,
            "feature_runtime": feature_runtime,
            "forecast_type": forecast_type,
            "forecast_object": forecast_object,
            "payload_contract": _LAYER3_PAYLOAD_CONTRACTS.get(forecast_object, "forecast_payload_v1"),
            "runtime_status": "blocked_by_incompatibility" if blocked else "operational",
            "blocked_reasons": list(blocked),
        },
        "canonical_active_cell": {
            "forecast_generator_family": model_family,
            "representation_runtime": feature_runtime,
            "forecast_protocol": forecast_type,
            "forecast_object": forecast_object,
            "payload_contract": _LAYER3_PAYLOAD_CONTRACTS.get(forecast_object, "forecast_payload_v1"),
            "runtime_status": "blocked_by_incompatibility" if blocked else "operational",
            "blocked_reasons": list(blocked),
        },
    }


def _compatibility_source_payload(
    *,
    feature_builder: str,
    predictor_family: str,
    data_richness_mode: str,
    factor_count_config: Mapping[str, Any],
    target_lag_selection: str,
    target_lag_count: Any,
    factor_lag_count: Any,
    legacy_y_lag_count: Any,
) -> dict[str, Any]:
    return {
        "source_kind": "legacy_bridge",
        "feature_builder": feature_builder,
        "predictor_family": predictor_family,
        "data_richness_mode": data_richness_mode,
        "factor_count": factor_count_config.get("mode", "fixed"),
        "target_lag_selection": target_lag_selection,
        "target_lag_count": target_lag_count,
        "factor_lag_count": factor_lag_count,
        "legacy_y_lag_count": legacy_y_lag_count,
        "legacy_manifest_alias": "source_bridge",
    }


def _path_average_protocols_for_horizons(construction: str, horizons: Sequence[int]) -> dict[str, Any] | None:
    if not _is_path_average_construction(construction):
        return None
    return {
        "runtime_effect": "layer3_stepwise_execution",
        "formula_owner": "2_preprocessing",
        "execution_owner": "3_training",
        "layer3_runtime": "operational",
        "layer3_runtime_contract": "path_average_stepwise_execution_v1",
        "protocols_by_horizon": {
            str(int(horizon)): _build_path_average_target_protocol(construction, int(horizon))
            for horizon in horizons
        },
    }


def _layer2_representation_spec(
    selection_map: dict[str, AxisSelection],
    leaf_config: dict[str, Any],
    preprocess_contract,
    *,
    data_task_spec: dict[str, Any],
    training_spec: dict[str, Any],
) -> dict[str, Any]:
    feature_builder = _first_selected_value(selection_map, "feature_builder", "autoreg_lagged_target")
    data_richness_mode = _selection_value(
        selection_map,
        "data_richness_mode",
        default=("target_lags_only" if feature_builder == "autoreg_lagged_target" else "full_high_dimensional_X"),
    )
    feature_runtime = _feature_runtime_for_validation(selection_map, fallback_feature_builder=str(feature_builder))
    predictor_family_default = "target_lags_only" if feature_runtime == "autoreg_lagged_target" else "all_macro_vars"
    predictor_family = _selection_value(selection_map, "predictor_family", default=predictor_family_default)
    contemporaneous_x_rule = _selection_value(selection_map, "contemporaneous_x_rule", default="forbid_contemporaneous")
    deterministic_components = _selection_value(selection_map, "deterministic_components", default="none")
    structural_break_segmentation = _selection_value(selection_map, "structural_break_segmentation", default="none")
    y_lag_count = training_spec.get("y_lag_count", "fixed")
    target_lag_selection = _target_lag_selection_value(
        selection_map,
        legacy_y_lag_count=str(y_lag_count),
    )
    if "target_lag_selection" in selection_map:
        target_lag_selection_source_axis = "target_lag_selection"
        target_lag_selection_source_value = target_lag_selection
    elif "target_lag_block" in selection_map:
        target_lag_selection_source_axis = "target_lag_block"
        target_lag_selection_source_value = _selection_value(selection_map, "target_lag_block")
    else:
        target_lag_selection_source_axis = "y_lag_count"
        target_lag_selection_source_value = y_lag_count
    target_lag_count, target_lag_count_source = _target_lag_count_config(leaf_config, training_spec)
    factor_lag_count, factor_lag_count_source = _factor_lag_count_config(leaf_config, training_spec)
    factor_count_config = _factor_count_config(selection_map, leaf_config, training_spec)
    x_lag_creation = getattr(preprocess_contract, "x_lag_creation", "no_x_lags")
    dimred = getattr(preprocess_contract, "dimensionality_reduction_policy", "none")
    horizon_target_construction = _selection_value(selection_map, "horizon_target_construction", default="future_target_level_t_plus_h")
    custom_preprocessor = _selection_value(selection_map, "custom_preprocessor", default="none")
    target_transformer = _selection_value(selection_map, "target_transformer", default="none")
    horizons = tuple(int(h) for h in leaf_config.get("horizons", [1]))
    path_average_protocol = _path_average_protocols_for_horizons(str(horizon_target_construction), horizons)
    explicit_factor_feature_block = _factor_feature_block_value(selection_map)
    has_explicit_target_lag_block = "target_lag_block" in selection_map
    target_lag_block = (
        _target_lag_block_from_selection(
            str(target_lag_selection),
            source_axis=target_lag_selection_source_axis,
            source_value=target_lag_selection_source_value,
            lag_count=target_lag_count,
        )
        if has_explicit_target_lag_block or feature_builder in {"autoreg_lagged_target", "factors_plus_AR"}
        else {"value": "none", "source_axis": "feature_builder", "source_value": feature_builder}
    )
    compatibility_source = _compatibility_source_payload(
        feature_builder=str(feature_builder),
        predictor_family=str(predictor_family),
        data_richness_mode=str(data_richness_mode),
        factor_count_config=factor_count_config,
        target_lag_selection=str(target_lag_selection),
        target_lag_count=target_lag_count,
        factor_lag_count=factor_lag_count,
        legacy_y_lag_count=y_lag_count,
    )
    return {
        "schema_version": "layer2_representation_v1",
        "runtime_effect": "provenance_plus_runtime_block_dispatch",
        "compatibility_source": compatibility_source,
        "source_bridge": {
            key: value
            for key, value in compatibility_source.items()
            if key not in {"source_kind", "legacy_manifest_alias"}
        },
        "target_lag_config": {
            "selection": target_lag_selection,
            "selection_source_axis": target_lag_selection_source_axis,
            "selection_source_value": target_lag_selection_source_value,
            "count": target_lag_count,
            "count_source": target_lag_count_source,
        },
        "target_representation": {
            "horizon_target_construction": horizon_target_construction,
            "target_construction_scale": _target_construction_scale(str(horizon_target_construction)),
            "path_average_protocol": path_average_protocol,
            "target_transform": getattr(preprocess_contract, "target_transform", "level"),
            "target_normalization": getattr(preprocess_contract, "target_normalization", "none"),
            "target_domain": getattr(preprocess_contract, "target_domain", "unconstrained"),
            "target_missing_policy": getattr(preprocess_contract, "target_missing_policy", "none"),
            "target_outlier_policy": getattr(preprocess_contract, "target_outlier_policy", "none"),
            "target_transformer": target_transformer,
            "inverse_transform_policy": getattr(preprocess_contract, "inverse_transform_policy", "none"),
            "evaluation_scale": getattr(preprocess_contract, "evaluation_scale", "raw_level"),
            "target_scale_contract": build_target_scale_contract(
                preprocess_contract,
                target_transformer=str(target_transformer),
            ),
        },
        "input_panel": {
            "predictor_family": predictor_family,
            "contemporaneous_x_rule": contemporaneous_x_rule,
        },
        "feature_blocks": {
            "feature_block_set": _feature_block_set_from_bridge(str(feature_builder), str(data_richness_mode)),
            "target_lag_block": target_lag_block,
            "x_lag_feature_block": _x_lag_block_from_selection(selection_map, str(x_lag_creation)),
            "factor_rotation_order": _factor_rotation_order_from_selection(selection_map),
            "factor_feature_block": _factor_block_from_bridge(
                feature_builder=str(feature_builder),
                dimred=str(dimred),
                factor_count_config=factor_count_config,
                factor_lag_count=factor_lag_count,
                factor_lag_count_source=factor_lag_count_source,
                data_task_spec=data_task_spec,
                preprocess_contract=preprocess_contract,
                explicit_block=explicit_factor_feature_block,
                rotation_feature_block=_selection_value(selection_map, "rotation_feature_block", default="none"),
                factor_rotation_order=_factor_rotation_order_from_selection(selection_map)["value"],
            ),
            "level_feature_block": _level_block_from_selection(selection_map, data_task_spec),
            "deterministic_feature_block": _deterministic_block_from_selection(
                deterministic_components=str(deterministic_components),
                structural_break_segmentation=str(structural_break_segmentation),
                data_task_spec=data_task_spec,
            ),
            "rotation_feature_block": _rotation_block_from_selection(selection_map, data_task_spec),
            "temporal_feature_block": _temporal_block_from_selection(selection_map, data_task_spec),
            "feature_block_combination": _feature_block_combination_from_bridge(
                str(feature_builder),
                str(x_lag_creation),
                explicit_value=(
                    _selection_value(selection_map, "feature_block_combination", default=None)
                    if "feature_block_combination" in selection_map
                    else None
                ),
                data_task_spec=data_task_spec,
            ),
        },
        "frame_conditioning": {
            "contemporaneous_x_rule": contemporaneous_x_rule,
            "x_missing_policy": getattr(preprocess_contract, "x_missing_policy", "none"),
            "x_outlier_policy": getattr(preprocess_contract, "x_outlier_policy", "none"),
            "scaling_policy": getattr(preprocess_contract, "scaling_policy", "none"),
            "scaling_scope": getattr(preprocess_contract, "scaling_scope", "columnwise"),
            "additional_preprocessing": getattr(preprocess_contract, "additional_preprocessing", "none"),
            "dimensionality_reduction_policy": dimred,
            "feature_selection_policy": getattr(preprocess_contract, "feature_selection_policy", "none"),
            "feature_selection_semantics": getattr(
                preprocess_contract,
                "feature_selection_semantics",
                "select_before_factor",
            ),
            "custom_final_z_selection_contract": (
                custom_final_z_selection_contract_metadata()
                if getattr(preprocess_contract, "feature_selection_semantics", "select_before_factor")
                == "select_after_custom_blocks"
                else {}
            ),
            "feature_grouping": getattr(preprocess_contract, "feature_grouping", "none"),
            "custom_preprocessor": custom_preprocessor,
            "preprocess_order": getattr(preprocess_contract, "preprocess_order", "none"),
            "preprocess_fit_scope": getattr(preprocess_contract, "preprocess_fit_scope", "not_applicable"),
            "separation_rule": _selection_value(selection_map, "separation_rule", default="strict_separation"),
        },
        "compatibility_notes": [
            "Feature-block specs drive executor-family dispatch, fixed target-lag matrix composition, fixed X-lag matrix composition, PCA static-factor matrix composition, and fixed target-lag concatenation with raw-panel/factor-panel direct Z.",
            "Legacy y_lag_count remains accepted for target-lag selection; legacy factor_ar_lags remains accepted as a target-lag-count fallback. Use factor_lag_count for factor-lag feature depth.",
        ],
    }


def _evaluation_spec(selection_map: dict[str, AxisSelection], leaf_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "primary_metric": _selection_value(selection_map, "primary_metric", default="msfe"),
        "point_metrics": _selection_value(selection_map, "point_metrics", default="MSFE"),
        "relative_metrics": _selection_value(selection_map, "relative_metrics", default="relative_MSFE"),
        "direction_metrics": _selection_value(selection_map, "direction_metrics", default="directional_accuracy"),
        "density_metrics": _selection_value(selection_map, "density_metrics", default="pinball_loss"),
        "economic_metrics": _selection_value(selection_map, "economic_metrics", default="utility_gain"),
        "benchmark_window": _selection_value(selection_map, "benchmark_window", default="expanding"),
        "benchmark_scope": _selection_value(selection_map, "benchmark_scope", default="same_for_all"),
        "agg_time": _selection_value(selection_map, "agg_time", default="full_oos_average"),
        "agg_horizon": _selection_value(selection_map, "agg_horizon", default="equal_weight"),
        "agg_target": _selection_value(selection_map, "agg_target", default="report_separately_only"),
        "ranking": _selection_value(selection_map, "ranking", default="mean_metric_rank"),
        "report_style": _selection_value(selection_map, "report_style", default="tidy_dataframe"),
        "regime_definition": _selection_value(selection_map, "regime_definition", default="none"),
        "regime_use": _selection_value(selection_map, "regime_use", default="eval_only"),
        "regime_metrics": _selection_value(selection_map, "regime_metrics", default="all_main_metrics_by_regime"),
        "decomposition_target": _selection_value(selection_map, "decomposition_target", default="preprocessing_effect"),
        "decomposition_order": _selection_value(selection_map, "decomposition_order", default="marginal_effect_only"),
        "oos_period": _selection_value(selection_map, "oos_period", default="all_oos_data"),
        "regime_start": leaf_config.get("regime_start"),
        "regime_end": leaf_config.get("regime_end"),
    }


def _build_stage0_and_recipe(
    recipe_dict: dict[str, Any],
    selection_map: dict[str, AxisSelection],
    leaf_config: dict[str, Any],
    preprocess_contract,
):
    research_design = _selection_value(selection_map, "research_design")
    dataset = _selection_value(selection_map, "dataset")
    information_set_type = _selection_value(selection_map, "information_set_type")
    target_structure = _target_structure(selection_map)
    benchmark = _selection_value(selection_map, "benchmark_family")
    framework = _selection_value(selection_map, "framework")
    target = leaf_config.get("target", "")
    targets = tuple(leaf_config.get("targets", ()))
    horizons = tuple(leaf_config["horizons"])
    data_vintage = leaf_config.get("data_vintage")
    model_axis = selection_map["model_family"]
    feature_axis = selection_map["feature_builder"]
    feature_builders = feature_axis.selected_values
    wrapper_family = leaf_config.get("wrapper_family")

    if target_structure == "multi_target_point_forecast":
        if len(targets) < 2:
            raise CompileValidationError("target_structure='multi_target_point_forecast' requires leaf_config.targets with at least two entries")
    else:
        if not target:
            raise CompileValidationError("single-target recipes require leaf_config.target")

    derived_experiment_unit = derive_experiment_unit_default(
        research_design=research_design,
        task=target_structure,
        model_axis_mode=model_axis.selection_mode,
        feature_axis_mode=feature_axis.selection_mode,
        wrapper_family=wrapper_family,
    )
    experiment_unit_explicit = "experiment_unit" in selection_map
    experiment_unit = _selection_value(selection_map, "experiment_unit", default=derived_experiment_unit)
    if experiment_unit_explicit:
        unit_entry = get_experiment_unit_entry(experiment_unit)
        if experiment_unit != derived_experiment_unit:
            raise CompileValidationError(
                f"experiment_unit={experiment_unit!r} conflicts with current recipe shape; implied unit is {derived_experiment_unit!r}"
            )
        if unit_entry.requires_multi_target and target_structure != "multi_target_point_forecast":
            raise CompileValidationError(
                f"experiment_unit={experiment_unit!r} requires target_structure='multi_target_point_forecast'"
            )
        if not unit_entry.requires_multi_target and target_structure == "multi_target_point_forecast":
            raise CompileValidationError(
                f"experiment_unit={experiment_unit!r} is incompatible with target_structure='multi_target_point_forecast'"
            )

    sample_split = {
        "expanding": "expanding_window_oos",
        "rolling": "rolling_window_oos",
    }[framework]
    info_set_token = {
        "revised": "revised_monthly",
        "pseudo_oos_revised": "pseudo_oos_revised",
    }.get(information_set_type, information_set_type)

    stage0 = build_design_frame(
        research_design=research_design,
        experiment_unit=experiment_unit if experiment_unit_explicit else None,
        fixed_design={
            "dataset_adapter": dataset,
            "information_set": info_set_token,
            "sample_split": sample_split,
            "benchmark": benchmark,
            "evaluation_protocol": "point_forecast_core",
            "forecast_task": target_structure,
        },
        comparison_contract={
            "information_set_policy": "identical",
            "sample_split_policy": "identical",
            "benchmark_policy": "identical",
            "evaluation_policy": "identical",
        },
        varying_design={
            "model_families": model_axis.selected_values,
            "feature_recipes": feature_builders,
            "horizons": tuple(f"h{h}" for h in horizons),
        },
    )
    benchmark_spec = _benchmark_spec(selection_map, leaf_config)
    if framework == "rolling":
        rolling_window_size = int(benchmark_spec.get("rolling_window_size", benchmark_spec.get("minimum_train_size", 5)))
        minimum_train_size = int(benchmark_spec.get("minimum_train_size", 5))
        if rolling_window_size < minimum_train_size:
            raise CompileValidationError("rolling_window_size must be at least minimum_train_size for rolling framework")
    data_task_spec = _data_task_spec(selection_map, leaf_config)
    training_spec = _training_spec(selection_map, leaf_config)
    recipe_spec = build_recipe_spec(
        recipe_id=recipe_dict["recipe_id"],
        stage0=stage0,
        target=target,
        horizons=horizons,
        raw_dataset=dataset,
        benchmark_config=benchmark_spec,
        data_task_spec=data_task_spec,
        training_spec=training_spec,
        layer2_representation_spec=_layer2_representation_spec(
            selection_map,
            leaf_config,
            preprocess_contract,
            data_task_spec=data_task_spec,
            training_spec=training_spec,
        ),
        data_vintage=data_vintage,
        targets=targets,
    )
    run_spec = build_run_spec(recipe_spec)
    return stage0, recipe_spec, run_spec


def _execution_status(
    selections: tuple[AxisSelection, ...],
    preprocess_contract,
    leaf_config: dict[str, Any] | None = None,
    stage0=None,
    run_spec: RunSpec | None = None,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    warnings: list[str] = []
    blocked: list[str] = []
    not_supported: list[str] = []
    leaf_config = dict(leaf_config or {})
    selection_map = _selection_map(selections)
    registry = get_axis_registry()
    has_runner_managed_sweep = any(
        selection.selection_mode in {"sweep", "conditional"} and len(selection.selected_values) > 1
        for selection in selections
    )

    failure_policy = _selection_value(selection_map, "failure_policy", default="fail_fast")

    for selection in selections:
        entry = registry[selection.axis_name]
        if selection.selection_mode == "sweep" and entry.default_policy == "fixed":
            warnings.append(
                f"fixed-policy axis '{selection.axis_name}' placed in sweep_axes; grammar accepts it, but governance expects a fixed selection for this axis"
            )
        if selection.selection_mode in {"sweep", "conditional"} and len(selection.selected_values) > 1:
            warnings.append(
                f"axis {selection.axis_name} uses runner-managed sweep values {selection.selected_values}; execute via compile_sweep_plan/execute_sweep"
            )
        for value, status in selection.selected_status.items():
            if status in {"registry_only", "planned", "external_plugin", "not_supported_yet"}:
                if _custom_feature_block_axis_is_registered(selection, leaf_config):
                    continue
                if _custom_feature_combiner_axis_is_registered(selection, leaf_config):
                    continue
                not_supported.append(
                    f"axis {selection.axis_name} value {value} is not supported by the current runtime (status={status})"
                )

    horizon_target_construction = _selection_value(
        selection_map,
        "horizon_target_construction",
        default="future_target_level_t_plus_h",
    )
    if _is_path_average_construction(str(horizon_target_construction)):
        if str(getattr(preprocess_contract, "target_transform", "level")) != "level":
            not_supported.append(
                "path-average target construction currently requires target_transform_policy='raw_level'"
            )
        if str(getattr(preprocess_contract, "target_normalization", "none")) != "none":
            not_supported.append(
                "path-average target construction currently requires target_normalization='none'"
            )
        target_transformer = _selection_value(selection_map, "target_transformer", default="none")
        if str(target_transformer) != "none":
            not_supported.append(
                "path-average target construction currently does not support custom target_transformer"
            )

    if not is_operational_preprocess_contract(preprocess_contract):
        not_supported.append("preprocessing contract is not supported by the current runtime slice")

    if preprocess_contract.tcode_policy == "raw_only" and preprocess_contract.preprocess_order != "none":
        blocked.append("raw_only tcode_policy cannot be paired with non-none preprocess_order")

    model_family = _selection_value(selection_map, "model_family") if "model_family" in selection_map and len(selection_map["model_family"].selected_values) == 1 else None
    feature_builder = _selection_value(selection_map, "feature_builder") if "feature_builder" in selection_map and len(selection_map["feature_builder"].selected_values) == 1 else None
    feature_runtime = _feature_runtime_for_validation(
        selection_map,
        fallback_feature_builder=feature_builder,
    )
    forecast_type = _selection_value(
        selection_map,
        "forecast_type",
        default=_layer3_forecast_type_default(feature_runtime),
    )
    forecast_object = _selection_value(selection_map, "forecast_object", default="point_mean")
    horizon_target_construction_for_l3 = _selection_value(
        selection_map,
        "horizon_target_construction",
        default="future_target_level_t_plus_h",
    )
    blocked.extend(
        _layer3_capability_rejections(
            model_family=model_family,
            feature_runtime=feature_runtime,
            forecast_type=forecast_type,
            forecast_object=forecast_object,
            horizon_target_construction=horizon_target_construction_for_l3,
            exogenous_x_path_policy=str(
                leaf_config.get("exogenous_x_path_policy")
                or leaf_config.get("future_x_path_policy")
                or "unavailable"
            ),
            target_lag_block=_selection_value(selection_map, "target_lag_block", default="none"),
            scheduled_known_future_x_columns=leaf_config.get("scheduled_known_future_x_columns")
            or leaf_config.get("known_future_x_columns"),
            recursive_x_model_family=leaf_config.get("recursive_x_model_family")
            or leaf_config.get("future_x_model_family"),
        )
    )
    if feature_runtime == "raw_feature_panel" and forecast_type == "iterated" and not blocked:
        if str(getattr(preprocess_contract, "target_transform", "level")) != "level":
            not_supported.append(
                "raw-panel iterated operational future-X paths currently require target_transform_policy='raw_level'"
            )
        if str(getattr(preprocess_contract, "target_normalization", "none")) != "none":
            not_supported.append(
                "raw-panel iterated operational future-X paths currently require target_normalization='none'"
            )
        if _selection_value(selection_map, "target_transformer", default="none") != "none":
            not_supported.append(
                "raw-panel iterated operational future-X paths currently does not support custom target_transformer"
            )
        if str(getattr(preprocess_contract, "tcode_policy", "raw_only")) != "raw_only":
            not_supported.append(
                "raw-panel iterated operational future-X paths currently require tcode_policy='raw_only'"
            )
        if str(getattr(preprocess_contract, "x_transform", "level")) != "level":
            not_supported.append(
                "raw-panel iterated operational future-X paths currently require x_transform_policy='raw_level'"
            )
        if str(getattr(preprocess_contract, "preprocess_order", "none")) != "none":
            not_supported.append(
                "raw-panel iterated operational future-X paths currently require preprocess_order='none'"
            )
        if str(getattr(preprocess_contract, "scaling_policy", "none")) != "none":
            not_supported.append(
                "raw-panel iterated operational future-X paths currently require scaling_policy='none'"
            )
        if str(getattr(preprocess_contract, "dimensionality_reduction_policy", "none")) != "none":
            not_supported.append(
                "raw-panel iterated operational future-X paths currently require dimensionality_reduction_policy='none'"
            )
        if str(getattr(preprocess_contract, "feature_selection_policy", "none")) != "none":
            not_supported.append(
                "raw-panel iterated operational future-X paths currently require feature_selection_policy='none'"
            )
    if forecast_object in {"interval", "density"}:
        if str(getattr(preprocess_contract, "target_transform", "level")) != "level":
            not_supported.append(
                f"forecast_object={forecast_object!r} currently requires target_transform_policy='raw_level'"
            )
        if str(getattr(preprocess_contract, "target_normalization", "none")) != "none":
            not_supported.append(
                f"forecast_object={forecast_object!r} currently requires target_normalization='none'"
            )
        target_transformer = _selection_value(selection_map, "target_transformer", default="none")
        if str(target_transformer) != "none":
            not_supported.append(
                f"forecast_object={forecast_object!r} currently does not support custom target_transformer"
            )

    # 1.3 training_start_rule=fixed_start requires leaf_config.training_start_date
    if feature_builder is not None:
        _ts_rule = _selection_value(selection_map, "training_start_rule", default="earliest_possible")
        if _ts_rule == "fixed_start" and not leaf_config.get("training_start_date"):
            blocked.append("training_start_rule='fixed_start' requires leaf_config.training_start_date (ISO date string)")

    # 1.3 overlap_handling=evaluate_with_hac compatibility (v1.0)
    _overlap = _selection_value(selection_map, "overlap_handling", default="allow_overlap")
    if _overlap == "evaluate_with_hac":
        _stat_test = _selection_value(selection_map, "stat_test", default="none")
        _hac_compatible = {"dm_hln", "dm_modified", "spa", "mcs", "cw", "cpa"}
        if _stat_test not in _hac_compatible and _stat_test != "none":
            blocked.append(
                f"overlap_handling='evaluate_with_hac' requires a HAC-capable stat_test "
                f"(one of {sorted(_hac_compatible)}); got stat_test={_stat_test!r}"
            )

    if feature_runtime is not None:
        predictor_family = _selection_value(selection_map, "predictor_family", default=("target_lags_only" if feature_runtime == "autoreg_lagged_target" else "all_macro_vars"))
        if predictor_family == "target_lags_only" and feature_runtime != "autoreg_lagged_target":
            blocked.append("predictor_family='target_lags_only' requires the target-lag-only feature runtime in the current runtime slice")
        if predictor_family == "all_macro_vars" and feature_runtime != "raw_feature_panel":
            blocked.append("predictor_family='all_macro_vars' requires a macro-X or factor feature runtime in the current runtime slice")
        target_lag_block = _selection_value(selection_map, "target_lag_block", default="none")
        if target_lag_block not in {"none", "fixed_target_lags"}:
            not_supported.append(
                f"target_lag_block={target_lag_block!r} is not executable in the current runtime slice"
            )
        x_lag_feature_block = _selection_value(selection_map, "x_lag_feature_block", default="none")
        if x_lag_feature_block != "none" and feature_runtime != "raw_feature_panel":
            not_supported.append(
                "x_lag_feature_block is currently executable only in feature runtimes that build from macro-X panels; "
                "target-lag-only runtime has no X-block composer"
            )
        level_feature_block = _selection_value(selection_map, "level_feature_block", default="none")
        level_block_active = level_feature_block in {
            "target_level_addback",
            "x_level_addback",
            "selected_level_addbacks",
            "level_growth_pairs",
        }
        if level_block_active and feature_runtime != "raw_feature_panel":
            not_supported.append(
                f"level_feature_block={level_feature_block!r} is currently executable only with "
                "raw-panel feature runtimes; factor and target-lag "
                "composition requires a dedicated block composer"
            )
        if (
            level_block_active
            and _selection_value(selection_map, "contemporaneous_x_rule", default="forbid_contemporaneous")
            != "forbid_contemporaneous"
        ):
            not_supported.append(
                f"level_feature_block={level_feature_block!r} requires "
                "contemporaneous_x_rule='forbid_contemporaneous' so level features are observed at the forecast origin"
            )
        temporal_feature_block = _selection_value(selection_map, "temporal_feature_block", default="none")
        temporal_block_active = temporal_feature_block in {
            "local_temporal_factors",
            "moving_average_features",
            "rolling_moments",
            "volatility_features",
            "custom_temporal_features",
        }
        rotation_feature_block = _selection_value(selection_map, "rotation_feature_block", default="none")
        rotation_block_active = rotation_feature_block in {"moving_average_rotation", "marx_rotation", "maf_rotation", "custom_rotation"}
        factor_rotation_order = _factor_rotation_order_from_selection(selection_map)["value"]
        feature_block_combination = _selection_value(selection_map, "feature_block_combination", default="replace_with_blocks")
        marx_append_mode = feature_block_combination in {"append_to_base_x", "concatenate_named_blocks"}
        custom_combiner_name = _custom_feature_combiner_name_from_leaf(leaf_config)
        custom_combiner_registered = bool(
            custom_combiner_name and is_custom_feature_combiner(custom_combiner_name)
        )
        if feature_block_combination == "custom_combiner":
            if feature_runtime != "raw_feature_panel":
                not_supported.append(
                    "feature_block_combination='custom_combiner' is currently executable only with "
                    "raw-panel Layer 2 tabular feature runtimes"
                )
            if not custom_combiner_registered:
                not_supported.append(
                    "feature_block_combination='custom_combiner' requires a registered "
                    "leaf_config.custom_feature_combiner callable"
                )
        if temporal_block_active and feature_runtime != "raw_feature_panel":
            not_supported.append(
                f"temporal_feature_block={temporal_feature_block!r} is currently executable only with "
                "raw-panel feature runtimes"
            )
        if rotation_block_active and feature_runtime != "raw_feature_panel":
            not_supported.append(
                f"rotation_feature_block={rotation_feature_block!r} is currently executable only with "
                "raw-panel feature runtimes"
            )
        if (
            rotation_feature_block == "marx_rotation"
            and getattr(preprocess_contract, "x_lag_creation", "no_x_lags") != "no_x_lags"
            and not marx_append_mode
        ):
            not_supported.append(
                f"rotation_feature_block={rotation_feature_block!r} cannot yet be combined with "
                "x_lag_feature_block or x_lag_creation unless feature_block_combination appends named blocks"
            )
        if (
            rotation_block_active
            and _selection_value(selection_map, "contemporaneous_x_rule", default="forbid_contemporaneous")
            != "forbid_contemporaneous"
        ):
            not_supported.append(
                f"rotation_feature_block={rotation_feature_block!r} requires "
                "contemporaneous_x_rule='forbid_contemporaneous' so rotation features use forecast-origin history"
            )
        if rotation_feature_block == "marx_rotation" and temporal_block_active and not marx_append_mode:
            not_supported.append(
                f"rotation_feature_block={rotation_feature_block!r} cannot yet be combined with "
                "temporal_feature_block unless feature_block_combination appends named blocks"
            )
        dimred = getattr(preprocess_contract, "dimensionality_reduction_policy", "none")
        feature_selection = getattr(preprocess_contract, "feature_selection_policy", "none")
        feature_selection_semantics = getattr(
            preprocess_contract,
            "feature_selection_semantics",
            "select_before_factor",
        )
        explicit_factor_block = _factor_feature_block_value(selection_map)
        factor_bridge_active = feature_builder in _FACTOR_BRIDGE_BUILDERS or dimred in _FACTOR_DIMRED_BRIDGES
        operational_factor_blocks = {"pca_static_factors", "pca_factor_lags", "supervised_factors"}
        factor_block_active = explicit_factor_block in operational_factor_blocks or (explicit_factor_block is None and factor_bridge_active)
        registered_custom_factor = (
            explicit_factor_block == "custom_factors"
            and _custom_feature_block_axis_is_registered(selection_map["factor_feature_block"], leaf_config)
            if "factor_feature_block" in selection_map
            else False
        )
        custom_block_active = (
            temporal_feature_block == "custom_temporal_features"
            or rotation_feature_block == "custom_rotation"
            or explicit_factor_block == "custom_factors"
            or feature_block_combination == "custom_combiner"
        )
        deterministic_components = _selection_value(selection_map, "deterministic_components", default="none")
        structural_break_segmentation = _selection_value(selection_map, "structural_break_segmentation", default="none")
        if temporal_block_active and (factor_block_active or dimred != "none"):
            not_supported.append(
                f"temporal_feature_block={temporal_feature_block!r} cannot yet be combined with "
                "factor_feature_block or dimensionality_reduction_policy; temporal-to-factor composition requires a block composer"
            )
        factor_then_rotation_requested = factor_rotation_order == "factor_then_rotation" or rotation_feature_block == "maf_rotation"
        pca_static_factor_active = (
            explicit_factor_block == "pca_static_factors"
            or (
                explicit_factor_block is None
                and factor_bridge_active
                and (dimred in {"pca", "static_factor"} or feature_builder in _FACTOR_BRIDGE_BUILDERS)
            )
        )
        if factor_then_rotation_requested and rotation_feature_block not in {"marx_rotation", "maf_rotation"}:
            not_supported.append(
                "factor_rotation_order='factor_then_rotation' currently requires "
                "rotation_feature_block='marx_rotation' or 'maf_rotation'"
            )
        if rotation_feature_block == "maf_rotation" and not pca_static_factor_active:
            not_supported.append(
                "rotation_feature_block='maf_rotation' requires factor_feature_block='pca_static_factors' "
                "or an equivalent pca/static_factor bridge"
            )
        if (
            factor_then_rotation_requested
            and factor_block_active
            and not pca_static_factor_active
        ):
            not_supported.append(
                "factor_rotation_order='factor_then_rotation' is currently executable only with "
                "pca_static_factors or an equivalent pca/static_factor bridge"
            )
        if factor_then_rotation_requested and getattr(preprocess_contract, "x_lag_creation", "no_x_lags") != "no_x_lags":
            not_supported.append(
                "factor_rotation_order='factor_then_rotation' currently requires x_lag_feature_block='none' "
                "and x_lag_creation='no_x_lags'"
            )
        if factor_then_rotation_requested and level_block_active:
            not_supported.append(
                "factor_rotation_order='factor_then_rotation' currently cannot be combined with level_feature_block; "
                "level addbacks remain available with rotation_then_factor or non-factor rotations"
            )
        marx_then_factor_allowed = (
            rotation_feature_block == "marx_rotation"
            and factor_block_active
            and factor_rotation_order == "rotation_then_factor"
        )
        factor_then_rotation_allowed = (
            rotation_feature_block in {"marx_rotation", "maf_rotation"}
            and factor_block_active
            and factor_then_rotation_requested
            and pca_static_factor_active
        )
        if (
            rotation_block_active
            and (factor_block_active or dimred != "none")
            and not (marx_then_factor_allowed or factor_then_rotation_allowed)
        ):
            not_supported.append(
                f"rotation_feature_block={rotation_feature_block!r} cannot yet be combined with "
                "factor_feature_block or dimensionality_reduction_policy; rotation-to-factor composition requires a block composer"
            )
        if explicit_factor_block == "none" and factor_bridge_active:
            not_supported.append(
                "factor_feature_block='none' conflicts with an active factor compatibility bridge "
                f"(legacy feature builder={feature_builder!r}, dimensionality_reduction_policy={dimred!r})"
            )
        if (
            feature_selection != "none"
            and explicit_factor_block not in {None, "none", *operational_factor_blocks}
            and feature_selection_semantics != "select_after_custom_blocks"
        ):
            not_supported.append(
                "feature_selection_policy is operational with factor blocks only for "
                "built-in executable factor_feature_block values; custom factor selection remains gated"
            )
        if (
            feature_selection != "none"
            and feature_selection_semantics == "select_after_custom_blocks"
            and not custom_block_active
        ):
            not_supported.append(
                "feature_selection_semantics='select_after_custom_blocks' requires a custom feature block "
                "or feature_block_combination='custom_combiner'"
            )
        if (
            feature_selection != "none"
            and feature_selection_semantics == "select_after_custom_blocks"
            and explicit_factor_block == "custom_factors"
            and not registered_custom_factor
        ):
            not_supported.append(
                "feature_selection_semantics='select_after_custom_blocks' with custom_factors requires "
                "a registered custom factor feature block"
            )
        if feature_selection != "none" and feature_selection_semantics == "select_after_factor" and not factor_block_active:
            not_supported.append(
                "feature_selection_semantics='select_after_factor' requires "
                "an executable factor_feature_block or an equivalent pca/static_factor bridge"
            )

    target_transformer = _selection_value(selection_map, "target_transformer", default="none")
    if target_transformer != "none":
        feature_runtime = _feature_runtime_for_validation(
            selection_map,
            fallback_feature_builder=feature_builder,
        )
        if feature_runtime not in _TARGET_TRANSFORMER_FEATURE_RUNTIMES:
            blocked.append(
                "target_transformer is currently executable only with supported target-lag or raw-panel feature runtimes "
                f"{sorted(_TARGET_TRANSFORMER_FEATURE_RUNTIMES)}"
            )
        if (
            feature_runtime == "raw_feature_panel"
            and model_family is not None
            and model_family not in _TARGET_TRANSFORMER_RAW_PANEL_MODELS
            and not is_custom_model(model_family)
        ):
            blocked.append(
                "target_transformer raw-panel runtime currently supports "
                f"model_family in {sorted(_TARGET_TRANSFORMER_RAW_PANEL_MODELS)} or a registered custom model; "
                f"got {model_family!r}"
            )
        if getattr(preprocess_contract, "target_transform", "level") != "level":
            blocked.append("target_transformer requires target_transform='level' until built-in and custom target transforms are composed")
        if getattr(preprocess_contract, "target_normalization", "none") != "none":
            blocked.append("target_transformer requires target_normalization='none' until normalization composition is implemented")
        if getattr(preprocess_contract, "inverse_transform_policy", "none") != "none":
            blocked.append("target_transformer requires inverse_transform_policy='none'; the plugin performs prediction inverse-transform")
        if getattr(preprocess_contract, "evaluation_scale", "raw_level") not in {"raw_level", "original_scale"}:
            blocked.append("target_transformer runtime currently supports raw-scale evaluation only")

    if failure_policy not in {"fail_fast", "skip_failed_cell", "skip_failed_model", "save_partial_results", "warn_only"}:
        not_supported.append(
            f"failure_policy {failure_policy!r} is not supported by the current runtime slice"
        )
    experiment_unit = (
        stage0.experiment_unit
        if stage0 is not None
        else (
            _selection_value(selection_map, "experiment_unit", default="single_target_single_model")
            if "experiment_unit" in selection_map
            else "single_target_single_model"
        )
    )
    compute_mode = _selection_value(selection_map, "compute_mode", default="serial")
    if compute_mode not in {"serial", "parallel_by_model", "parallel_by_horizon", "parallel_by_target", "parallel_by_oos_date"}:
        not_supported.append(
            f"compute_mode {compute_mode!r} is not supported by the current runtime slice"
        )
    route_owner = run_spec.route_owner if run_spec is not None else None
    if route_owner in {"wrapper", "replication", "orchestrator"} and has_runner_managed_sweep:
        not_supported.append(
            f"route_owner={route_owner!r} cannot contain sweep_axes/conditional_axes until a composed runner contract is implemented"
        )
    runner_ready_status: str | None = None
    if route_owner == "wrapper":
        if experiment_unit == "multi_target_separate_runs":
            runner_ready_status = "ready_for_wrapper_runner"
            warnings.append(
                "route_owner='wrapper' requires execute_separate_runs; direct run_compiled_recipe/execute_recipe is not supported"
            )
        else:
            not_supported.append(
                f"experiment_unit={experiment_unit!r} has no executable wrapper runner contract in the current runtime"
            )
    elif route_owner == "replication":
        if experiment_unit == "replication_recipe":
            runner_ready_status = "ready_for_replication_runner"
            warnings.append(
                "route_owner='replication' requires execute_replication; direct run_compiled_recipe/execute_recipe is not supported"
            )
        else:
            not_supported.append(
                f"experiment_unit={experiment_unit!r} cannot use route_owner='replication'"
            )
    elif route_owner == "orchestrator":
        not_supported.append(
            "route_owner='orchestrator' has no executable runner contract in the current runtime"
        )
    elif route_owner == "single_run" and has_runner_managed_sweep:
        runner_ready_status = "ready_for_sweep_runner"

    if blocked:
        return "blocked_by_incompatibility", tuple(warnings + not_supported), tuple(blocked)

    if not_supported:
        return "not_supported", tuple(warnings + not_supported), ()
    if runner_ready_status is not None:
        return runner_ready_status, tuple(warnings), ()
    if not warnings:
        return "executable", (), ()
    return "not_supported", tuple(warnings), ()


def _build_wrapper_handoff(
    stage0,
    recipe_spec: RecipeSpec,
    run_spec: RunSpec,
    leaf_config: dict[str, Any],
    *,
    experiment_unit_explicit: bool,
) -> dict[str, Any]:
    if run_spec.route_owner != "wrapper":
        return {}
    wrapper_family = leaf_config.get("wrapper_family")
    bundle_label = leaf_config.get("bundle_label")
    if experiment_unit_explicit:
        wrapper_family = wrapper_family or stage0.experiment_unit
        bundle_label = bundle_label or f"{recipe_spec.recipe_id}-{wrapper_family}"
    if wrapper_family not in {
        "single_target_full_sweep",
        "multi_target_separate_runs",
        "multi_target_shared_design",
        "benchmark_suite",
        "ablation_study",
    }:
        raise CompileValidationError(
            "wrapper_bundle_plan requires a wrapper family in {'single_target_full_sweep', 'multi_target_separate_runs', 'multi_target_shared_design', 'benchmark_suite', 'ablation_study'}"
        )
    if not isinstance(bundle_label, str) or not bundle_label.strip():
        raise CompileValidationError("wrapper_bundle_plan requires non-empty leaf_config.bundle_label")
    return {
        "wrapper_family": wrapper_family,
        "bundle_label": bundle_label,
        "route_owner": run_spec.route_owner,
        "execution_posture": stage0.execution_posture,
        "experiment_unit": stage0.experiment_unit,
        "recipe_id": recipe_spec.recipe_id,
        "artifact_subdir": run_spec.artifact_subdir,
        "targets": list(recipe_spec.targets),
        "horizons": list(recipe_spec.horizons),
    }


def compile_recipe_dict(recipe_dict: dict[str, Any]) -> CompileResult:
    from macrocast.compiler.migrations import migrate_legacy_stat_test
    recipe_dict = migrate_legacy_stat_test(recipe_dict)
    if not recipe_dict.get("recipe_id"):
        raise CompileValidationError("recipe_id is required")
    selections = _build_axis_selections(recipe_dict)
    _ensure_unique_axis_selections(selections)
    selection_map = _selection_map(selections)
    selections = _append_feature_builder_bridge_if_needed(selections, selection_map)
    _ensure_unique_axis_selections(selections)
    selection_map = _selection_map(selections)
    required_axes = {"research_design", "dataset", "information_set_type", "target_structure", "framework", "benchmark_family", "model_family", "feature_builder"}
    missing_axes = sorted(axis for axis in required_axes if axis not in selection_map)
    if missing_axes:
        raise CompileValidationError(f"recipe missing required axes: {missing_axes}")
    leaf_config = _leaf_config(recipe_dict)
    if "horizons" not in leaf_config:
        raise CompileValidationError("recipe leaf_config missing 'horizons'")
    # Resolve declarative derived_axes (axis_type=derived). Each derivation rule
    # computes a concrete value from the current selection_map + leaf_config and
    # is appended to the selection tuple with selection_mode="derived".
    derived_additions = _resolve_derived_axes(recipe_dict, selection_map, leaf_config)
    if derived_additions:
        selections = selections + tuple(derived_additions)
        _ensure_unique_axis_selections(selections)
        selection_map = _selection_map(selections)
        selections = _append_feature_builder_bridge_if_needed(selections, selection_map)
        _ensure_unique_axis_selections(selections)
        selection_map = _selection_map(selections)
    target_structure_value = _target_structure(selection_map)
    experiment_unit_explicit = "experiment_unit" in selection_map
    if target_structure_value == "multi_target_point_forecast":
        if "targets" not in leaf_config:
            raise CompileValidationError("recipe leaf_config missing 'targets'")
    else:
        if "target" not in leaf_config:
            raise CompileValidationError("recipe leaf_config missing 'target'")

    # Custom source adapter validation: custom_csv / custom_parquet require leaf_config.custom_data_path.
    source_adapter_choice = selection_map["source_adapter"].selected_values[0] if "source_adapter" in selection_map else None
    if source_adapter_choice in {"custom_csv", "custom_parquet"} and not leaf_config.get("custom_data_path"):
        raise CompileValidationError(
            f"source_adapter={source_adapter_choice!r} requires leaf_config.custom_data_path"
        )
    _validate_layer1_data_task_contract(selection_map, leaf_config)
    _validate_layer2_feature_block_contract(selection_map, leaf_config)
    reproducibility_mode = _selection_value(selection_map, "reproducibility_mode", default="best_effort")
    random_seed = leaf_config.get("random_seed")
    if reproducibility_mode in {"strict_reproducible", "seeded_reproducible"} and random_seed is None:
        raise CompileValidationError(
            f"reproducibility_mode={reproducibility_mode!r} requires leaf_config.random_seed"
        )
    failure_policy = _selection_value(selection_map, "failure_policy", default="fail_fast")
    compute_mode = _selection_value(selection_map, "compute_mode", default="serial")

    preprocess_contract = _build_preprocess_contract(selection_map)
    stage0, recipe_spec, run_spec = _build_stage0_and_recipe(
        recipe_dict,
        selection_map,
        leaf_config,
        preprocess_contract,
    )
    execution_status, warnings, blocked = _execution_status(
        selections,
        preprocess_contract,
        leaf_config=leaf_config,
        stage0=stage0,
        run_spec=run_spec,
    )
    tree_context = _build_tree_context(stage0, run_spec, selections, leaf_config)
    wrapper_handoff = _build_wrapper_handoff(
        stage0,
        recipe_spec,
        run_spec,
        leaf_config,
        experiment_unit_explicit=experiment_unit_explicit,
    )

    compiled = CompiledRecipeSpec(
        recipe_id=recipe_dict["recipe_id"],
        layer_order=get_canonical_layer_order(),
        axis_selections=selections,
        leaf_config=leaf_config,
        preprocess_contract=preprocess_contract,
        stage0=stage0,
        recipe_spec=recipe_spec,
        run_spec=run_spec,
        execution_status=execution_status,
        warnings=warnings,
        blocked_reasons=blocked,
        tree_context=tree_context,
        wrapper_handoff=wrapper_handoff,
    )
    manifest = compiled_spec_to_dict(compiled)
    return CompileResult(compiled=compiled, manifest=manifest)


def compile_recipe_yaml(path: str | Path) -> CompileResult:
    return compile_recipe_dict(load_recipe_yaml(path))



def _output_spec(selection_map):
    return {
        "export_format": _selection_value(selection_map, "export_format", default="json"),
        "saved_objects": _selection_value(selection_map, "saved_objects", default="full_bundle"),
        "provenance_fields": _selection_value(selection_map, "provenance_fields", default="full"),
        "artifact_granularity": _selection_value(selection_map, "artifact_granularity", default="aggregated"),
    }
def compiled_spec_to_dict(compiled: CompiledRecipeSpec) -> dict[str, Any]:
    selection_map = {selection.axis_name: selection for selection in compiled.axis_selections}
    return {
        "recipe_id": compiled.recipe_id,
        "layer_order": list(compiled.layer_order),
        "execution_status": compiled.execution_status,
        "warnings": list(compiled.warnings),
        "blocked_reasons": list(compiled.blocked_reasons),
        "leaf_config": dict(compiled.leaf_config),
        "benchmark_spec": dict(compiled.recipe_spec.benchmark_config),
        "model_spec": _model_spec(selection_map),
        "reproducibility_spec": {
            "reproducibility_mode": _selection_value(selection_map, "reproducibility_mode", default="best_effort"),
            "random_seed": compiled.leaf_config.get("random_seed"),
        },
        "failure_policy_spec": {
            "failure_policy": _selection_value(selection_map, "failure_policy", default="fail_fast"),
        },
        "compute_mode_spec": {
            "compute_mode": _selection_value(selection_map, "compute_mode", default="serial"),
        },
        "data_task_spec": _data_task_spec(selection_map, compiled.leaf_config),
        "training_spec": _training_spec(selection_map, compiled.leaf_config),
        "layer2_representation_spec": dict(compiled.recipe_spec.layer2_representation_spec),
        "layer3_capability_matrix": _layer3_capability_matrix(selection_map, compiled.leaf_config),
        "evaluation_spec": _evaluation_spec(selection_map, compiled.leaf_config),
        "stat_test_spec": {
            "stat_test": _selection_value(selection_map, "stat_test", default="none"),
            "dependence_correction": _selection_value(selection_map, "dependence_correction", default="none"),
        },
        "importance_spec": {
            "importance_method": _selection_value(selection_map, "importance_method"),
        },
        "output_spec": _output_spec(selection_map),
        "preprocess_contract": preprocess_to_dict(compiled.preprocess_contract),
        "axis_selections": [
            {
                "axis_name": selection.axis_name,
                "layer": selection.layer,
                "selection_mode": selection.selection_mode,
                "selected_values": list(selection.selected_values),
                "selected_status": dict(selection.selected_status),
            }
            for selection in compiled.axis_selections
        ],
        "run_spec": {
            "run_id": compiled.run_spec.run_id,
            "artifact_subdir": compiled.run_spec.artifact_subdir,
            "route_owner": compiled.run_spec.route_owner,
        },
        "tree_context": dict(compiled.tree_context),
        "wrapper_handoff": dict(compiled.wrapper_handoff),
    }


def run_compiled_recipe(
    compiled: CompiledRecipeSpec,
    *,
    output_root: str | Path,
    local_raw_source: str | Path | None = None,
):
    if compiled.run_spec.route_owner != "single_run":
        raise CompileValidationError(
            f"compiled recipe route_owner={compiled.run_spec.route_owner!r} requires a dedicated runner; "
            "run_compiled_recipe only executes route_owner='single_run'"
        )
    if compiled.execution_status != "executable":
        raise CompileValidationError(
            f"compiled recipe is not executable: {compiled.execution_status}; warnings={compiled.warnings}; blocked={compiled.blocked_reasons}"
        )
    return execute_recipe(
        recipe=compiled.recipe_spec,
        preprocess=compiled.preprocess_contract,
        output_root=output_root,
        local_raw_source=local_raw_source,
        provenance_payload={"compiler": compiled_spec_to_dict(compiled), "tree_context": dict(compiled.tree_context)},
    )
