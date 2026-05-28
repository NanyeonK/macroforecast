from __future__ import annotations

from calendar import monthrange
from datetime import date
import re
from typing import Any, Literal


CustomSourcePolicy = Literal["official_only", "custom_panel_only", "official_plus_custom"]
Dataset = Literal["fred_md", "fred_qd", "fred_sd", "fred_md+fred_sd", "fred_qd+fred_sd"]
Frequency = Literal["monthly", "quarterly"]
VintagePolicy = Literal["current_vintage", "real_time_alfred"]
TargetStructure = Literal["single_target", "multi_target"]
VariableUniverse = Literal[
    "all_variables",
    "core_variables",
    "category_variables",
    "target_specific_variables",
    "explicit_variable_list",
]
RegimeDefinition = Literal[
    "none",
    "external_nber",
    "external_user_provided",
    "estimated_markov_switching",
    "estimated_threshold",
    "estimated_structural_break",
]
RegimeTemporalRule = Literal["expanding_window_per_origin", "rolling_window_per_origin", "block_recompute"]

DATASET_OPTIONS: tuple[Dataset, ...] = ("fred_md", "fred_qd", "fred_sd", "fred_md+fred_sd", "fred_qd+fred_sd")
FRED_SD_DATASETS = frozenset({"fred_sd", "fred_md+fred_sd", "fred_qd+fred_sd"})
MONTHLY_DATASETS = frozenset({"fred_md", "fred_md+fred_sd"})
QUARTERLY_DATASETS = frozenset({"fred_qd", "fred_qd+fred_sd"})
REGIME_ESTIMATED_OPTIONS = frozenset(
    {"estimated_markov_switching", "estimated_threshold", "estimated_structural_break"}
)
REGIME_OPTIONS: tuple[RegimeDefinition, ...] = (
    "none",
    "external_nber",
    "external_user_provided",
    "estimated_markov_switching",
    "estimated_threshold",
    "estimated_structural_break",
)
REGIME_TEMPORAL_RULE_OPTIONS: tuple[RegimeTemporalRule, ...] = (
    "expanding_window_per_origin",
    "rolling_window_per_origin",
    "block_recompute",
)

DEFAULT_AXES: dict[str, Any] = {
    "panel_composition": "official_only",
    "dataset": "fred_md",
    "information_set_type": "final_revised_data",
    "vintage_policy": "current_vintage",
    "target_structure": "single_target",
    "variable_universe": "all_variables",
    "fred_sd_frequency_policy": "report_only",
    "state_selection": "all_states",
    "sd_variable_selection": "all_sd_variables",
    "missing_availability": "zero_fill_leading_predictor_gaps",
    "release_lag_rule": "ignore_release_lag",
    "contemporaneous_x_rule": "allow_same_period_predictors",
    "target_geography_policy": "all_states",
    "predictor_geography_policy": "match_target",
    "sample_start_rule": "max_balanced",
    "sample_end_rule": "latest_available",
    "regime_definition": "none",
}

OPTION_SETS: dict[str, set[Any]] = {
    "panel_composition": {"official_only", "custom_panel_only", "official_plus_custom"},
    "dataset": set(DATASET_OPTIONS),
    "frequency": {"monthly", "quarterly"},
    "information_set_type": {"final_revised_data", "pseudo_oos_on_revised_data"},
    "vintage_policy": {"current_vintage", "real_time_alfred"},
    "target_structure": {"single_target", "multi_target"},
    "variable_universe": {
        "all_variables",
        "core_variables",
        "category_variables",
        "target_specific_variables",
        "explicit_variable_list",
    },
    "fred_sd_frequency_policy": {
        "report_only",
        "allow_mixed_frequency",
        "reject_mixed_known_frequency",
        "require_single_known_frequency",
    },
    "fred_sd_state_group": {
        "all_states",
        "census_region_northeast",
        "census_region_midwest",
        "census_region_south",
        "census_region_west",
        "census_division_new_england",
        "census_division_middle_atlantic",
        "census_division_east_north_central",
        "census_division_west_north_central",
        "census_division_south_atlantic",
        "census_division_east_south_central",
        "census_division_west_south_central",
        "census_division_mountain",
        "census_division_pacific",
        "contiguous_48_plus_dc",
        "custom_state_group",
    },
    "state_selection": {"all_states", "selected_states"},
    "fred_sd_variable_group": {
        "all_sd_variables",
        "labor_market_core",
        "employment_sector",
        "gsp_output",
        "housing",
        "trade",
        "income",
        "direct_analog_high_confidence",
        "provisional_analog_medium",
        "semantic_review_outputs",
        "no_reliable_analog",
        "custom_sd_variable_group",
    },
    "sd_variable_selection": {"all_sd_variables", "selected_sd_variables"},
    "missing_availability": {
        "require_complete_rows",
        "keep_available_rows",
        "impute_predictors_only",
        "zero_fill_leading_predictor_gaps",
    },
    "release_lag_rule": {"ignore_release_lag", "fixed_lag_all_series", "series_specific_lag"},
    "contemporaneous_x_rule": {"allow_same_period_predictors", "forbid_same_period_predictors"},
    "target_geography_policy": {"single_state", "all_states", "selected_states"},
    "predictor_geography_policy": {"match_target", "all_states", "selected_states", "national_only"},
    "sample_start_rule": {"earliest_available", "fixed_date", "max_balanced"},
    "sample_end_rule": {"latest_available", "fixed_date"},
    "horizon_set": {"standard_md", "standard_qd", "single", "custom_list", "range_up_to_h"},
    "regime_definition": {
        "none",
        "external_nber",
        "external_user_provided",
        "estimated_markov_switching",
        "estimated_threshold",
        "estimated_structural_break",
    },
    "regime_estimation_temporal_rule": set(REGIME_TEMPORAL_RULE_OPTIONS),
}


def data(
    *,
    panel_composition: CustomSourcePolicy | None = None,
    dataset: Dataset | dict[str, Any] | None = None,
    frequency: Frequency | None = None,
    information_set_type: Literal["final_revised_data", "pseudo_oos_on_revised_data"] | None = None,
    vintage_policy: VintagePolicy | None = None,
    target: str | None = None,
    targets: list[str] | tuple[str, ...] | None = None,
    target_structure: TargetStructure | None = None,
    variable_universe: VariableUniverse | None = None,
    fred_sd_frequency_policy: str | None = None,
    fred_sd_state_group: str | None = None,
    state_selection: str | None = None,
    fred_sd_variable_group: str | None = None,
    sd_variable_selection: str | None = None,
    missing_availability: str | None = None,
    release_lag_rule: str | None = None,
    contemporaneous_x_rule: str | None = None,
    target_geography_policy: str | None = None,
    predictor_geography_policy: str | None = None,
    sample_start_rule: str | None = None,
    sample_end_rule: str | None = None,
    horizon_set: str | None = None,
    regime_definition: RegimeDefinition | None = None,
    regime_estimation_temporal_rule: RegimeTemporalRule | None = None,
    **leaf_config: Any,
) -> dict[str, dict[str, Any]]:
    """Return a validated data recipe block without loading a dataset."""

    block = build_data_block(
        panel_composition=panel_composition,
        dataset=dataset,
        frequency=frequency,
        information_set_type=information_set_type,
        vintage_policy=vintage_policy,
        target=target,
        targets=targets,
        target_structure=target_structure,
        variable_universe=variable_universe,
        fred_sd_frequency_policy=fred_sd_frequency_policy,
        fred_sd_state_group=fred_sd_state_group,
        state_selection=state_selection,
        fred_sd_variable_group=fred_sd_variable_group,
        sd_variable_selection=sd_variable_selection,
        missing_availability=missing_availability,
        release_lag_rule=release_lag_rule,
        contemporaneous_x_rule=contemporaneous_x_rule,
        target_geography_policy=target_geography_policy,
        predictor_geography_policy=predictor_geography_policy,
        sample_start_rule=sample_start_rule,
        sample_end_rule=sample_end_rule,
        horizon_set=horizon_set,
        regime_definition=regime_definition,
        regime_estimation_temporal_rule=regime_estimation_temporal_rule,
        **leaf_config,
    )
    errors = validate_data_block(block)
    if errors:
        raise ValueError("; ".join(errors))
    return {"data": block}


def build_data_block(
    *,
    panel_composition: CustomSourcePolicy | None = None,
    dataset: Dataset | dict[str, Any] | None = None,
    frequency: Frequency | None = None,
    information_set_type: Literal["final_revised_data", "pseudo_oos_on_revised_data"] | None = None,
    vintage_policy: VintagePolicy | None = None,
    target: str | None = None,
    targets: list[str] | tuple[str, ...] | None = None,
    target_structure: TargetStructure | None = None,
    variable_universe: VariableUniverse | None = None,
    fred_sd_frequency_policy: str | None = None,
    fred_sd_state_group: str | None = None,
    state_selection: str | None = None,
    fred_sd_variable_group: str | None = None,
    sd_variable_selection: str | None = None,
    missing_availability: str | None = None,
    release_lag_rule: str | None = None,
    contemporaneous_x_rule: str | None = None,
    target_geography_policy: str | None = None,
    predictor_geography_policy: str | None = None,
    sample_start_rule: str | None = None,
    sample_end_rule: str | None = None,
    horizon_set: str | None = None,
    regime_definition: RegimeDefinition | None = None,
    regime_estimation_temporal_rule: RegimeTemporalRule | None = None,
    **leaf_config: Any,
) -> dict[str, Any]:
    if target is not None and targets is not None:
        raise ValueError("provide either target or targets, not both")
    if targets is not None and target_structure is None:
        target_structure = "multi_target"

    axis_values: dict[str, Any | None] = {
        "panel_composition": panel_composition,
        "dataset": dataset,
        "frequency": frequency,
        "information_set_type": information_set_type,
        "vintage_policy": vintage_policy,
        "target_structure": target_structure,
        "variable_universe": variable_universe,
        "fred_sd_frequency_policy": fred_sd_frequency_policy,
        "fred_sd_state_group": fred_sd_state_group,
        "state_selection": state_selection,
        "fred_sd_variable_group": fred_sd_variable_group,
        "sd_variable_selection": sd_variable_selection,
        "missing_availability": missing_availability,
        "release_lag_rule": release_lag_rule,
        "contemporaneous_x_rule": contemporaneous_x_rule,
        "target_geography_policy": target_geography_policy,
        "predictor_geography_policy": predictor_geography_policy,
        "sample_start_rule": sample_start_rule,
        "sample_end_rule": sample_end_rule,
        "horizon_set": horizon_set,
        "regime_definition": regime_definition,
        "regime_estimation_temporal_rule": regime_estimation_temporal_rule,
    }
    fixed_axes = {key: value for key, value in axis_values.items() if value is not None}
    leaf = dict(leaf_config)
    if target is not None:
        leaf["target"] = target
    if targets is not None:
        leaf["targets"] = list(targets)
    block: dict[str, Any] = {"fixed_axes": fixed_axes}
    if leaf:
        block["leaf_config"] = leaf
    return block


def resolve_data_block(block: dict[str, Any]) -> dict[str, Any]:
    fixed_axes = block.get("fixed_axes", {}) or {}
    leaf_config = block.get("leaf_config", {}) or {}
    custom_policy = fixed_axes.get("panel_composition", DEFAULT_AXES["panel_composition"])
    dataset = None if custom_policy == "custom_panel_only" else fixed_axes.get("dataset", DEFAULT_AXES["dataset"])
    frequency = fixed_axes.get("frequency")
    if frequency is None:
        frequency = _derived_frequency(custom_policy, dataset)
    horizon_set = fixed_axes.get("horizon_set")
    if horizon_set is None and frequency in {"monthly", "quarterly"}:
        horizon_set = "standard_md" if frequency == "monthly" else "standard_qd"
    regime_definition = fixed_axes.get("regime_definition", DEFAULT_AXES["regime_definition"])
    regime_temporal_rule = fixed_axes.get("regime_estimation_temporal_rule")
    if regime_definition in REGIME_ESTIMATED_OPTIONS and regime_temporal_rule is None:
        regime_temporal_rule = "expanding_window_per_origin"

    resolved = {
        "panel_composition": custom_policy,
        "dataset": dataset,
        "frequency": frequency,
        "information_set_type": fixed_axes.get("information_set_type", DEFAULT_AXES["information_set_type"]),
        "vintage_policy": None if custom_policy == "custom_panel_only" else fixed_axes.get("vintage_policy", "current_vintage"),
        "target_structure": fixed_axes.get("target_structure", DEFAULT_AXES["target_structure"]),
        "variable_universe": fixed_axes.get("variable_universe", DEFAULT_AXES["variable_universe"]),
        "fred_sd_frequency_policy": fixed_axes.get("fred_sd_frequency_policy"),
        "fred_sd_state_group": fixed_axes.get("fred_sd_state_group"),
        "state_selection": fixed_axes.get("state_selection"),
        "fred_sd_variable_group": fixed_axes.get("fred_sd_variable_group"),
        "sd_variable_selection": fixed_axes.get("sd_variable_selection"),
        "missing_availability": fixed_axes.get("missing_availability", DEFAULT_AXES["missing_availability"]),
        "release_lag_rule": fixed_axes.get("release_lag_rule", DEFAULT_AXES["release_lag_rule"]),
        "contemporaneous_x_rule": fixed_axes.get("contemporaneous_x_rule", DEFAULT_AXES["contemporaneous_x_rule"]),
        "target_geography_policy": None,
        "predictor_geography_policy": None,
        "sample_start_rule": fixed_axes.get("sample_start_rule", DEFAULT_AXES["sample_start_rule"]),
        "sample_end_rule": fixed_axes.get("sample_end_rule", DEFAULT_AXES["sample_end_rule"]),
        "horizon_set": horizon_set,
        "regime_definition": regime_definition,
        "regime_estimation_temporal_rule": regime_temporal_rule,
    }
    if custom_policy == "custom_panel_only" or dataset == "fred_sd":
        resolved["variable_universe"] = fixed_axes.get("variable_universe") if "variable_universe" in fixed_axes else None
    if dataset in FRED_SD_DATASETS:
        resolved["fred_sd_frequency_policy"] = fixed_axes.get("fred_sd_frequency_policy", DEFAULT_AXES["fred_sd_frequency_policy"])
        resolved["fred_sd_state_group"] = fixed_axes.get("fred_sd_state_group")
        resolved["state_selection"] = fixed_axes.get("state_selection", DEFAULT_AXES["state_selection"])
        resolved["fred_sd_variable_group"] = fixed_axes.get("fred_sd_variable_group")
        resolved["sd_variable_selection"] = fixed_axes.get("sd_variable_selection", DEFAULT_AXES["sd_variable_selection"])
        resolved["target_geography_policy"] = fixed_axes.get("target_geography_policy", "all_states")
        resolved["predictor_geography_policy"] = fixed_axes.get("predictor_geography_policy", "match_target")
    return resolved


def validate_data_block(block: dict[str, Any]) -> list[str]:
    fixed_axes = block.get("fixed_axes", {}) or {}
    leaf_config = block.get("leaf_config", {}) or {}
    errors: list[str] = []
    if not isinstance(fixed_axes, dict):
        return ["data.fixed_axes must be a mapping"]
    if not isinstance(leaf_config, dict):
        return ["data.leaf_config must be a mapping"]
    for key, value in fixed_axes.items():
        allowed = OPTION_SETS.get(key)
        if allowed is None:
            errors.append(f"unknown data option {key!r}")
        elif not _is_sweep_marker(value) and value not in allowed:
            errors.append(f"{key} must be one of {sorted(allowed)}")
    resolved = resolve_data_block(block)
    errors.extend(_validate_source_selection(fixed_axes, leaf_config, resolved))
    errors.extend(_validate_target(leaf_config, resolved))
    errors.extend(_validate_variable_universe(leaf_config, resolved))
    errors.extend(_validate_horizons(leaf_config, resolved))
    errors.extend(_validate_sample_window(leaf_config, resolved))
    errors.extend(_validate_regime(leaf_config, resolved))
    return errors


def resolved_horizons(resolved: dict[str, Any], leaf_config: dict[str, Any]) -> tuple[int, ...]:
    horizon_set = resolved.get("horizon_set")
    if horizon_set == "standard_md":
        return (1, 3, 6, 12)
    if horizon_set == "standard_qd":
        return (1, 2, 4, 8)
    if horizon_set in {"single", "custom_list"} and _positive_int_list(leaf_config.get("target_horizons")):
        return tuple(leaf_config["target_horizons"])
    if horizon_set == "range_up_to_h" and isinstance(leaf_config.get("max_horizon"), int) and leaf_config["max_horizon"] > 0:
        return tuple(range(1, leaf_config["max_horizon"] + 1))
    return ()


def regime_metadata_from_resolved(resolved: dict[str, Any], leaf_config: dict[str, Any]) -> Any:
    from macroforecast.core.types import L1RegimeMetadataArtifact

    definition = resolved["regime_definition"]
    metadata: dict[str, Any] = {
        "leaf_config": {
            key: leaf_config[key]
            for key in (
                "regime_indicator_path",
                "regime_dates_list",
                "transition_variable",
                "threshold_variable",
                "n_thresholds",
                "max_breaks",
                "break_ic_criterion",
                "regime_rolling_window_size",
                "block_recompute_interval",
            )
            if key in leaf_config
        }
    }
    if definition == "external_nber":
        metadata["source_series"] = "USREC"
    if definition in REGIME_ESTIMATED_OPTIONS:
        metadata["runtime_status"] = "schema_valid_runtime_estimator_pending"
    return L1RegimeMetadataArtifact(
        definition=definition,
        n_regimes=leaf_config.get("n_regimes", 2),
        regime_label_series=None,
        regime_probabilities=None,
        transition_matrix=None,
        estimation_temporal_rule=resolved.get("regime_estimation_temporal_rule")
        if definition in REGIME_ESTIMATED_OPTIONS
        else None,
        estimation_metadata=metadata,
    )


def normalize_iso_partial(value: Any, *, end_of_period: bool = False) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        date.fromisoformat(value)
        return value
    except ValueError:
        pass
    if re.match(r"^\d{4}-\d{2}$", value):
        y, m = int(value[:4]), int(value[5:7])
        if not (1 <= m <= 12):
            return None
        d = monthrange(y, m)[1] if end_of_period else 1
        return f"{y:04d}-{m:02d}-{d:02d}"
    if re.match(r"^\d{4}$", value):
        return f"{value}-12-31" if end_of_period else f"{value}-01-01"
    return None


def _derived_frequency(custom_policy: Any, dataset: Any) -> str | None:
    if custom_policy == "custom_panel_only" or dataset == "fred_sd":
        return None
    if dataset in MONTHLY_DATASETS:
        return "monthly"
    if dataset in QUARTERLY_DATASETS:
        return "quarterly"
    return None


def _validate_source_selection(fixed_axes: dict[str, Any], leaf_config: dict[str, Any], resolved: dict[str, Any]) -> list[str]:
    errors = []
    custom_policy = resolved["panel_composition"]
    dataset = resolved["dataset"]
    if custom_policy == "custom_panel_only":
        if not any(key in leaf_config for key in ("custom_source_path", "custom_panel_inline", "custom_panel_records")):
            errors.append("custom_panel_only requires custom_source_path, custom_panel_inline, or custom_panel_records")
        if "dataset" in fixed_axes:
            errors.append("dataset is inactive when panel_composition=custom_panel_only")
    if custom_policy == "official_plus_custom":
        for key in ("custom_source_path", "custom_merge_rule"):
            if key not in leaf_config:
                errors.append(f"official_plus_custom requires {key}")
    frequency = fixed_axes.get("frequency")
    if dataset in MONTHLY_DATASETS and frequency not in {None, "monthly"}:
        errors.append("frequency must be monthly for FRED-MD datasets")
    if dataset in QUARTERLY_DATASETS and frequency not in {None, "quarterly"}:
        errors.append("frequency must be quarterly for FRED-QD datasets")
    if (dataset == "fred_sd" or custom_policy == "custom_panel_only") and frequency is None:
        errors.append("frequency must be explicitly set for fred_sd standalone or custom-only data")
    if resolved.get("vintage_policy") == "real_time_alfred":
        alfred_mode = leaf_config.get("alfred_mode", "local")
        if alfred_mode == "local" and not leaf_config.get("alfred_snapshot_dir"):
            errors.append("real_time_alfred with alfred_mode=local requires alfred_snapshot_dir")
    return errors


def _validate_target(leaf_config: dict[str, Any], resolved: dict[str, Any]) -> list[str]:
    if resolved["target_structure"] == "single_target":
        return [] if isinstance(leaf_config.get("target"), str) else ["single_target requires target string"]
    targets = leaf_config.get("targets")
    if not isinstance(targets, list) or not targets:
        return ["multi_target requires non-empty targets list"]
    return []


def _validate_variable_universe(leaf_config: dict[str, Any], resolved: dict[str, Any]) -> list[str]:
    errors = []
    variable_universe = resolved.get("variable_universe")
    if variable_universe == "category_variables":
        for key in ("variable_universe_category_columns", "variable_universe_category"):
            if key not in leaf_config:
                errors.append(f"category_variables requires {key}")
    if variable_universe == "target_specific_variables" and not isinstance(leaf_config.get("target_specific_columns"), dict):
        errors.append("target_specific_variables requires target_specific_columns")
    if variable_universe == "explicit_variable_list":
        columns = leaf_config.get("variable_universe_columns")
        if not isinstance(columns, list) or not columns:
            errors.append("explicit_variable_list requires non-empty variable_universe_columns")
    return errors


def _validate_horizons(leaf_config: dict[str, Any], resolved: dict[str, Any]) -> list[str]:
    horizon_set = resolved.get("horizon_set")
    horizons = leaf_config.get("target_horizons")
    if horizon_set == "single":
        if not isinstance(horizons, list) or not _positive_int_list(horizons) or len(horizons) != 1:
            return ["single horizon_set requires target_horizons list of length 1"]
    if horizon_set == "custom_list" and not _positive_int_list(horizons):
        return ["custom_list requires non-empty positive integer target_horizons"]
    if horizon_set == "range_up_to_h":
        max_horizon = leaf_config.get("max_horizon")
        if not isinstance(max_horizon, int) or max_horizon <= 0:
            return ["range_up_to_h requires positive integer max_horizon"]
    return []


def _validate_sample_window(leaf_config: dict[str, Any], resolved: dict[str, Any]) -> list[str]:
    errors = []
    start = leaf_config.get("sample_start_date")
    end = leaf_config.get("sample_end_date")
    if resolved["sample_start_rule"] == "fixed_date" and normalize_iso_partial(start) is None:
        errors.append("fixed_date sample_start_rule requires sample_start_date")
    if resolved["sample_end_rule"] == "fixed_date" and normalize_iso_partial(end) is None:
        errors.append("fixed_date sample_end_rule requires sample_end_date")
    start_iso = normalize_iso_partial(start)
    end_iso = normalize_iso_partial(end)
    if start_iso is not None and end_iso is not None:
        if date.fromisoformat(end_iso) < date.fromisoformat(start_iso):
            errors.append("sample_end_date must be >= sample_start_date")
    return errors


def _validate_regime(leaf_config: dict[str, Any], resolved: dict[str, Any]) -> list[str]:
    regime = resolved.get("regime_definition")
    if regime in {None, "none"}:
        return []
    errors = []
    if regime == "external_user_provided":
        has_path = "regime_indicator_path" in leaf_config
        has_dates = "regime_dates_list" in leaf_config
        if has_path == has_dates:
            errors.append("external_user_provided requires exactly one of regime_indicator_path or regime_dates_list")
    if regime == "estimated_threshold" and not isinstance(leaf_config.get("threshold_variable"), str):
        errors.append("estimated_threshold requires threshold_variable")
    temporal_rule = resolved.get("regime_estimation_temporal_rule")
    if regime in REGIME_ESTIMATED_OPTIONS and temporal_rule not in REGIME_TEMPORAL_RULE_OPTIONS:
        errors.append(f"regime_estimation_temporal_rule must be one of {sorted(REGIME_TEMPORAL_RULE_OPTIONS)}")
    return errors


def _positive_int_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(item, int) and item > 0 for item in value)


def _is_sweep_marker(value: Any) -> bool:
    return isinstance(value, dict) and "sweep" in value


__all__ = [
    "data",
    "build_data_block",
    "resolve_data_block",
    "validate_data_block",
    "resolved_horizons",
    "regime_metadata_from_resolved",
    "normalize_iso_partial",
]
