from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

from ..dag import DAG, GatePredicate, Node, NodeRef
from ..layer_specs import AxisSpec, LayerImplementationSpec, Option, SubLayerSpec
from ..types import L1DataDefinitionArtifact, L1RegimeMetadataArtifact


class L1Data:
    """Layer 1 Data implementation marker."""


CustomSourcePolicy = Literal["official_only", "custom_panel_only", "official_plus_custom"]
Dataset = Literal["fred_md", "fred_qd", "fred_sd", "fred_md+fred_sd", "fred_qd+fred_sd"]
Frequency = Literal["monthly", "quarterly"]
VintagePolicy = Literal["current_vintage", "real_time_alfred"]
TargetStructure = Literal["single_target", "multi_series_target"]
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
VALID_STATE_CODES = frozenset(
    {
        "AL",
        "AK",
        "AZ",
        "AR",
        "CA",
        "CO",
        "CT",
        "DE",
        "DC",
        "FL",
        "GA",
        "HI",
        "ID",
        "IL",
        "IN",
        "IA",
        "KS",
        "KY",
        "LA",
        "ME",
        "MD",
        "MA",
        "MI",
        "MN",
        "MS",
        "MO",
        "MT",
        "NE",
        "NV",
        "NH",
        "NJ",
        "NM",
        "NY",
        "NC",
        "ND",
        "OH",
        "OK",
        "OR",
        "PA",
        "RI",
        "SC",
        "SD",
        "TN",
        "TX",
        "UT",
        "VT",
        "VA",
        "WA",
        "WV",
        "WI",
        "WY",
        "US",
    }
)

DEFAULT_AXES: dict[str, Any] = {
    "custom_source_policy": "official_only",
    "dataset": "fred_md",
    "vintage_policy": "current_vintage",
    "target_structure": "single_target",
    "variable_universe": "all_variables",
    "target_geography_scope": "all_states",
    "predictor_geography_scope": "match_target",
    "sample_start_rule": "max_balanced",
    "sample_end_rule": "latest_available",
    "regime_definition": "none",
}

L1_AXIS_NAMES: tuple[str, ...] = (
    "custom_source_policy",
    "dataset",
    "frequency",
    "vintage_policy",
    "target_structure",
    "variable_universe",
    "target_geography_scope",
    "predictor_geography_scope",
    "sample_start_rule",
    "sample_end_rule",
    "horizon_set",
    "regime_definition",
    "regime_estimation_temporal_rule",
)

SWEEPABLE_AXES = frozenset({"dataset"})
REGIME_OPTIONS: tuple[RegimeDefinition, ...] = (
    "none",
    "external_nber",
    "external_user_provided",
    "estimated_markov_switching",
    "estimated_threshold",
    "estimated_structural_break",
)
REGIME_ESTIMATED_OPTIONS = frozenset(
    {"estimated_markov_switching", "estimated_threshold", "estimated_structural_break"}
)
REGIME_TEMPORAL_RULE_OPTIONS: tuple[RegimeTemporalRule, ...] = (
    "expanding_window_per_origin",
    "rolling_window_per_origin",
    "block_recompute",
)


@dataclass(frozen=True)
class ResolvedAxis:
    value: Any
    source: Literal["explicit", "derived", "dynamic_default", "package_default"]


@dataclass(frozen=True)
class L1LayerExecutionRecord:
    layer_id: Literal["l1"]
    status: Literal["completed", "failed", "skipped_disabled", "skipped_diagnostic_off"]
    artifact: L1DataDefinitionArtifact
    regime_artifact: L1RegimeMetadataArtifact
    resolved_axes: dict[str, ResolvedAxis]
    produced_sinks: tuple[str, ...] = ("l1_data_definition_v1", "l1_regime_metadata_v1")
    sink_hashes: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class L1Manifest:
    layer_execution_log: dict[str, L1LayerExecutionRecord]


@dataclass(frozen=True)
class L1Recipe:
    layer: Any


def parse_layer_yaml(yaml_text: str, layer_id: Literal["l1"] = "l1") -> Any:
    if layer_id != "l1":
        raise ValueError("L1 parser only accepts layer_id='l1'")
    from ..yaml import LayerYamlSpec, parse_recipe_yaml

    root = parse_recipe_yaml(yaml_text)
    raw = root.get("1_data", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("1_data: layer YAML must be a mapping")
    return LayerYamlSpec(layer_id="l1", raw_yaml=raw, enabled=bool(raw.get("enabled", True)))


def normalize_to_dag_form(layer: Any | dict[str, Any], layer_id: Literal["l1"] = "l1") -> DAG:
    if layer_id != "l1":
        raise ValueError("L1 normalizer only accepts layer_id='l1'")
    raw = _raw_layer(layer)
    fixed_axes = raw.get("fixed_axes", {}) or {}
    leaf_config = raw.get("leaf_config", {}) or {}
    resolved = resolve_axes_from_raw(fixed_axes, leaf_config, tolerate_invalid=True)

    nodes: dict[str, Node] = {}
    inputs: list[NodeRef] = []
    layer_globals: dict[str, Any] = {}
    for axis_name in _active_axis_names(resolved):
        value = resolved.get(axis_name)
        if value is None:
            continue
        node_id = f"axis_{axis_name}"
        nodes[node_id] = Node(
            id=node_id,
            type="axis",
            layer_id="l1",
            op=axis_name,
            params={"value": value},
            gates=_axis_gates(axis_name),
        )
        inputs.append(NodeRef(node_id))
        layer_globals[axis_name] = value

    aggregate = Node(
        id="data_definition_aggregate",
        type="combine",
        layer_id="l1",
        op="layer_meta_aggregate",
        params={"leaf_config": leaf_config},
        inputs=tuple(inputs),
    )
    nodes[aggregate.id] = aggregate
    sinks = {"l1_data_definition_v1": aggregate.id, "l1_regime_metadata_v1": aggregate.id}
    return DAG(layer_id="l1", nodes=nodes, sinks=sinks, layer_globals=layer_globals)


def resolve_axes(dag: DAG) -> dict[str, Any]:
    fixed_axes = {name.removeprefix("axis_"): node.params["value"] for name, node in dag.nodes.items() if name.startswith("axis_")}
    leaf_config = dict(dag.nodes["data_definition_aggregate"].params.get("leaf_config", {}))
    return resolve_axes_from_raw(fixed_axes, leaf_config)


def resolve_axes_from_raw(
    fixed_axes: dict[str, Any], leaf_config: dict[str, Any], *, tolerate_invalid: bool = False
) -> dict[str, Any]:
    custom_policy = fixed_axes.get("custom_source_policy", DEFAULT_AXES["custom_source_policy"])
    dataset = None if custom_policy == "custom_panel_only" else fixed_axes.get("dataset", DEFAULT_AXES["dataset"])
    frequency = fixed_axes.get("frequency")
    if _is_sweep_marker(dataset):
        dataset = DEFAULT_AXES["dataset"]
    if _is_sweep_marker(frequency):
        frequency = _derived_frequency(custom_policy, dataset)
    if frequency is None:
        frequency = _derived_frequency(custom_policy, dataset)
    horizon_set = fixed_axes.get("horizon_set")
    if _is_sweep_marker(horizon_set):
        horizon_set = None
    if horizon_set is None and frequency in {"monthly", "quarterly"}:
        horizon_set = "standard_md" if frequency == "monthly" else "standard_qd"
    regime_definition = fixed_axes.get("regime_definition", DEFAULT_AXES["regime_definition"])
    regime_temporal_rule = fixed_axes.get("regime_estimation_temporal_rule")
    if regime_definition in REGIME_ESTIMATED_OPTIONS and regime_temporal_rule is None:
        regime_temporal_rule = "expanding_window_per_origin"

    resolved = {
        "custom_source_policy": custom_policy,
        "dataset": dataset,
        "frequency": frequency,
        "vintage_policy": None if custom_policy == "custom_panel_only" else fixed_axes.get("vintage_policy", "current_vintage"),
        "target_structure": fixed_axes.get("target_structure", DEFAULT_AXES["target_structure"]),
        "variable_universe": fixed_axes.get("variable_universe", DEFAULT_AXES["variable_universe"]),
        "target_geography_scope": None,
        "predictor_geography_scope": None,
        "sample_start_rule": fixed_axes.get("sample_start_rule", DEFAULT_AXES["sample_start_rule"]),
        "sample_end_rule": fixed_axes.get("sample_end_rule", DEFAULT_AXES["sample_end_rule"]),
        "horizon_set": horizon_set,
        "regime_definition": regime_definition,
        "regime_estimation_temporal_rule": regime_temporal_rule,
    }
    if custom_policy == "custom_panel_only" or dataset == "fred_sd":
        resolved["variable_universe"] = None if "variable_universe" not in fixed_axes else fixed_axes["variable_universe"]
    if dataset in FRED_SD_DATASETS:
        resolved["target_geography_scope"] = fixed_axes.get("target_geography_scope", "all_states")
        resolved["predictor_geography_scope"] = fixed_axes.get("predictor_geography_scope", "match_target")
    if tolerate_invalid:
        return resolved
    return resolved


def validate_layer(layer: Any | dict[str, Any] | str) -> Any:
    from ..validator import Issue, Severity, ValidationReport

    if isinstance(layer, str):
        layer = parse_layer_yaml(layer)
    raw = _raw_layer(layer)
    issues: list[Issue] = []
    fixed_axes = raw.get("fixed_axes", {}) or {}
    leaf_config = raw.get("leaf_config", {}) or {}
    if not isinstance(fixed_axes, dict):
        return ValidationReport((_issue("l1.fixed_axes", "fixed_axes must be a mapping"),))
    if not isinstance(leaf_config, dict):
        return ValidationReport((_issue("l1.leaf_config", "leaf_config must be a mapping"),))

    for axis_name in fixed_axes:
        if axis_name not in L1_AXIS_NAMES:
            issues.append(_issue("l1.fixed_axes", f"unknown L1 axis {axis_name!r}"))
        elif _is_sweep_marker(fixed_axes[axis_name]) and axis_name not in SWEEPABLE_AXES:
            issues.append(_issue(f"l1.{axis_name}", f"L1 axis {axis_name} is not sweepable"))

    resolved = resolve_axes_from_raw(fixed_axes, leaf_config, tolerate_invalid=True)
    issues.extend(_validate_options(fixed_axes, resolved))
    issues.extend(_validate_source_selection(fixed_axes, leaf_config, resolved))
    issues.extend(_validate_target(leaf_config, resolved))
    issues.extend(_validate_variable_universe(leaf_config, resolved))
    issues.extend(_validate_geography(leaf_config, resolved))
    issues.extend(_validate_sample_window(leaf_config, resolved))
    issues.extend(_validate_horizons(leaf_config, resolved))
    issues.extend(_validate_regime(leaf_config, resolved))

    if resolved.get("custom_source_policy") == "custom_panel_only":
        issues.append(
            Issue(
                "l1_custom_source_policy",
                Severity.SOFT,
                "layer",
                "l1.custom_source_policy",
                "custom_panel_only makes FRED-specific axes inactive",
            )
        )
    if resolved.get("target_structure") == "multi_series_target" and len(leaf_config.get("targets", ()) or ()) == 1:
        issues.append(
            Issue(
                "l1_target_structure",
                Severity.SOFT,
                "layer",
                "l1.target_structure",
                "multi_series_target used with a single target; consider single_target",
            )
        )
    columns = leaf_config.get("variable_universe_columns")
    if resolved.get("variable_universe") == "explicit_variable_list" and isinstance(columns, list) and len(columns) < 5:
        issues.append(
            Issue(
                "l1_variable_universe",
                Severity.SOFT,
                "layer",
                "l1.variable_universe",
                "very few predictors; consider whether intended",
            )
        )
    if (
        resolved.get("target_geography_scope") == "single_state"
        and resolved.get("predictor_geography_scope") == "all_states"
    ):
        issues.append(
            Issue(
                "l1_geography_overfit_warning",
                Severity.SOFT,
                "layer",
                "l1.predictor_geography_scope",
                "many predictors for one target may overfit; consider match_target",
            )
        )

    return ValidationReport(tuple(issues))


def validate_regime_source_reference(layer: Any | dict[str, Any] | str, selector: Any) -> Any:
    from ..validator import ValidationReport

    if isinstance(layer, str):
        layer = parse_layer_yaml(layer)
    raw = _raw_layer(layer)
    resolved = resolve_axes_from_raw(raw.get("fixed_axes", {}) or {}, raw.get("leaf_config", {}) or {}, tolerate_invalid=True)
    if (
        getattr(selector, "layer_ref", None) == "l1"
        and getattr(selector, "sink_name", None) == "l1_regime_metadata_v1"
        and resolved.get("regime_definition") == "none"
    ):
        return ValidationReport(
            (
                _issue(
                    "l1.l1_regime_metadata_v1",
                    "l1_regime_metadata_v1 is inactive when regime_definition is none",
                ),
            )
        )
    return ValidationReport()


def build_recipe_with_l1_only(yaml_text: str) -> L1Recipe:
    return L1Recipe(layer=parse_layer_yaml(yaml_text))


def execute_recipe(recipe: L1Recipe) -> L1Manifest:
    report = validate_layer(recipe.layer)
    if report.has_hard_errors:
        raise ValueError("; ".join(issue.message for issue in report.hard_errors))
    dag = normalize_to_dag_form(recipe.layer)
    raw = recipe.layer.raw_yaml
    fixed_axes = raw.get("fixed_axes", {}) or {}
    leaf_config = raw.get("leaf_config", {}) or {}
    resolved = resolve_axes(dag)
    artifact = _artifact_from_resolved(resolved, leaf_config)
    record = L1LayerExecutionRecord(
        layer_id="l1",
        status="completed",
        artifact=artifact,
        regime_artifact=_regime_artifact_from_resolved(resolved, leaf_config),
        resolved_axes=_resolved_axis_entries(resolved, fixed_axes),
        produced_sinks=tuple(dag.sinks),
    )
    return L1Manifest(layer_execution_log={"l1": record})


def _artifact_from_resolved(resolved: dict[str, Any], leaf_config: dict[str, Any]) -> L1DataDefinitionArtifact:
    horizons = _resolved_horizons(resolved, leaf_config)
    target = leaf_config.get("target")
    targets = tuple(leaf_config.get("targets", ()) or ((target,) if target else ()))
    return L1DataDefinitionArtifact(
        custom_source_policy=resolved["custom_source_policy"],
        dataset=resolved["dataset"],
        frequency=resolved["frequency"],
        vintage_policy=resolved["vintage_policy"],
        target_structure=resolved["target_structure"],
        target=target,
        targets=targets,
        variable_universe=resolved["variable_universe"],
        target_geography_scope=resolved["target_geography_scope"],
        predictor_geography_scope=resolved["predictor_geography_scope"],
        sample_start_rule=resolved["sample_start_rule"],
        sample_end_rule=resolved["sample_end_rule"],
        horizon_set=resolved["horizon_set"],
        target_horizons=horizons,
        regime_definition=resolved["regime_definition"],
        leaf_config=leaf_config,
    )


def _regime_artifact_from_resolved(
    resolved: dict[str, Any], leaf_config: dict[str, Any]
) -> L1RegimeMetadataArtifact:
    definition = resolved["regime_definition"]
    n_regimes = leaf_config.get("n_regimes", 2)
    metadata = {
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
        n_regimes=n_regimes,
        regime_label_series=None,
        regime_probabilities=None,
        transition_matrix=None,
        estimation_temporal_rule=resolved.get("regime_estimation_temporal_rule")
        if definition in REGIME_ESTIMATED_OPTIONS
        else None,
        estimation_metadata=metadata,
    )


def _validate_options(fixed_axes: dict[str, Any], resolved: dict[str, Any]) -> list[Any]:
    issues = []
    option_sets = {
        "custom_source_policy": {"official_only", "custom_panel_only", "official_plus_custom"},
        "dataset": set(DATASET_OPTIONS),
        "frequency": {"monthly", "quarterly"},
        "vintage_policy": {"current_vintage", "real_time_alfred"},
        "target_structure": {"single_target", "multi_series_target"},
        "variable_universe": {
            "all_variables",
            "core_variables",
            "category_variables",
            "target_specific_variables",
            "explicit_variable_list",
        },
        "target_geography_scope": {"single_state", "all_states", "selected_states"},
        "predictor_geography_scope": {"match_target", "all_states", "selected_states", "national_only"},
        "sample_start_rule": {"earliest_available", "fixed_date", "max_balanced"},
        "sample_end_rule": {"latest_available", "fixed_date"},
        "horizon_set": {"standard_md", "standard_qd", "single", "custom_list", "range_up_to_h"},
        "regime_definition": set(REGIME_OPTIONS),
        "regime_estimation_temporal_rule": set(REGIME_TEMPORAL_RULE_OPTIONS),
    }
    for axis_name, allowed in option_sets.items():
        value = fixed_axes.get(axis_name, resolved.get(axis_name))
        if value is None or _is_sweep_marker(value):
            continue
        if value not in allowed:
            issues.append(_issue(f"l1.{axis_name}", f"{axis_name} must be one of {sorted(allowed)}"))
    return issues


def _validate_source_selection(fixed_axes: dict[str, Any], leaf_config: dict[str, Any], resolved: dict[str, Any]) -> list[Any]:
    issues = []
    custom_policy = resolved["custom_source_policy"]
    dataset = resolved["dataset"]
    if custom_policy == "custom_panel_only":
        if "custom_source_path" not in leaf_config:
            issues.append(_issue("l1.custom_source_path", "custom_panel_only requires leaf_config.custom_source_path"))
        if "dataset" in fixed_axes:
            issues.append(_issue("l1.dataset", "dataset is inactive when custom_source_policy=custom_panel_only"))
    if custom_policy == "official_plus_custom":
        for key in ("custom_source_path", "custom_merge_rule"):
            if key not in leaf_config:
                issues.append(_issue(f"l1.{key}", f"official_plus_custom requires leaf_config.{key}"))
    frequency = fixed_axes.get("frequency")
    if _is_sweep_marker(frequency):
        frequency = None
    if dataset in MONTHLY_DATASETS and frequency not in {None, "monthly"}:
        issues.append(_issue("l1.frequency", "frequency must be monthly for FRED-MD datasets"))
    if dataset in QUARTERLY_DATASETS and frequency not in {None, "quarterly"}:
        issues.append(_issue("l1.frequency", "frequency must be quarterly for FRED-QD datasets"))
    if (dataset == "fred_sd" or custom_policy == "custom_panel_only") and frequency is None:
        issues.append(_issue("l1.frequency", "frequency must be explicitly set for fred_sd standalone or custom-only data"))
    if resolved.get("vintage_policy") == "real_time_alfred":
        if "vintage_date_or_tag" not in leaf_config:
            issues.append(_issue("l1.vintage_date_or_tag", "real_time_alfred requires leaf_config.vintage_date_or_tag"))
        issues.append(_issue("l1.vintage_policy", "vintage_policy=real_time_alfred is future and not executable"))
    return issues


def _validate_target(leaf_config: dict[str, Any], resolved: dict[str, Any]) -> list[Any]:
    if resolved["target_structure"] == "single_target":
        return [] if isinstance(leaf_config.get("target"), str) else [_issue("l1.target", "single_target requires leaf_config.target string")]
    targets = leaf_config.get("targets")
    if not isinstance(targets, list) or not targets:
        return [_issue("l1.targets", "multi_series_target requires non-empty leaf_config.targets list")]
    return []


def _validate_variable_universe(leaf_config: dict[str, Any], resolved: dict[str, Any]) -> list[Any]:
    issues = []
    variable_universe = resolved.get("variable_universe")
    custom_policy = resolved.get("custom_source_policy")
    dataset = resolved.get("dataset")
    if (custom_policy == "custom_panel_only" or dataset == "fred_sd") and variable_universe is not None:
        issues.append(_issue("l1.variable_universe", "variable_universe is inactive for custom-only or standalone fred_sd"))
        return issues
    if variable_universe == "category_variables":
        for key in ("variable_universe_category_columns", "variable_universe_category"):
            if key not in leaf_config:
                issues.append(_issue(f"l1.{key}", f"category_variables requires leaf_config.{key}"))
    if variable_universe == "target_specific_variables":
        mapping = leaf_config.get("target_specific_columns")
        targets = _targets_from_leaf_config(leaf_config, resolved)
        if not isinstance(mapping, dict):
            issues.append(_issue("l1.target_specific_columns", "target_specific_variables requires leaf_config.target_specific_columns"))
        elif any(target not in mapping for target in targets):
            issues.append(_issue("l1.target_specific_columns", "target_specific_columns must cover all targets"))
    if variable_universe == "explicit_variable_list":
        columns = leaf_config.get("variable_universe_columns")
        if not isinstance(columns, list) or not columns:
            issues.append(_issue("l1.variable_universe_columns", "explicit_variable_list requires non-empty variable_universe_columns"))
    return issues


def _validate_geography(leaf_config: dict[str, Any], resolved: dict[str, Any]) -> list[Any]:
    issues = []
    dataset = resolved.get("dataset")
    if dataset not in FRED_SD_DATASETS:
        if "target_geography_scope" in resolved and resolved.get("target_geography_scope") is None:
            return issues
    target_scope = resolved.get("target_geography_scope")
    predictor_scope = resolved.get("predictor_geography_scope")
    if target_scope == "single_state":
        state = leaf_config.get("target_state")
        if state not in VALID_STATE_CODES:
            issues.append(_issue("l1.target_state", "single_state requires valid leaf_config.target_state"))
        if state == "US" and predictor_scope not in {"national_only", "match_target"}:
            issues.append(_issue("l1.predictor_geography_scope", "US target requires national_only or match_target predictors"))
    if target_scope == "selected_states":
        issues.extend(_validate_state_list("target_states", leaf_config.get("target_states")))
    if predictor_scope == "selected_states":
        issues.extend(_validate_state_list("predictor_states", leaf_config.get("predictor_states")))
    if predictor_scope == "national_only" and dataset == "fred_sd":
        issues.append(_issue("l1.predictor_geography_scope", "national_only requires a dataset with fred_md or fred_qd"))
    return issues


def _validate_sample_window(leaf_config: dict[str, Any], resolved: dict[str, Any]) -> list[Any]:
    issues = []
    start = leaf_config.get("sample_start_date")
    end = leaf_config.get("sample_end_date")
    if resolved["sample_start_rule"] == "fixed_date" and not _is_iso_date(start):
        issues.append(_issue("l1.sample_start_date", "fixed_date sample_start_rule requires ISO sample_start_date"))
    if resolved["sample_end_rule"] == "fixed_date" and not _is_iso_date(end):
        issues.append(_issue("l1.sample_end_date", "fixed_date sample_end_rule requires ISO sample_end_date"))
    if _is_iso_date(start) and _is_iso_date(end) and date.fromisoformat(end) < date.fromisoformat(start):
        issues.append(_issue("l1.sample_end_date", "sample_end_date must be >= sample_start_date"))
    return issues


def _validate_horizons(leaf_config: dict[str, Any], resolved: dict[str, Any]) -> list[Any]:
    issues = []
    horizon_set = resolved.get("horizon_set")
    horizons = leaf_config.get("target_horizons")
    if horizon_set == "single":
        if not _positive_int_list(horizons) or len(horizons) != 1:
            issues.append(_issue("l1.target_horizons", "single horizon_set requires target_horizons list of length 1"))
    if horizon_set == "custom_list" and not _positive_int_list(horizons):
        issues.append(_issue("l1.target_horizons", "custom_list requires non-empty positive integer target_horizons"))
    if horizon_set == "range_up_to_h":
        max_horizon = leaf_config.get("max_horizon")
        if not isinstance(max_horizon, int) or max_horizon <= 0:
            issues.append(_issue("l1.max_horizon", "range_up_to_h requires positive integer max_horizon"))
    resolved_horizons = _resolved_horizons(resolved, leaf_config)
    if resolved_horizons and max(resolved_horizons) > (12 if resolved.get("frequency") == "monthly" else 8):
        from ..validator import Issue, Severity

        issues.append(
            Issue(
                "l1_long_horizon_warning",
                Severity.SOFT,
                "layer",
                "l1.horizon_set",
                "long horizon forecast; ensure sample is large enough",
            )
        )
    return issues


def _validate_regime(leaf_config: dict[str, Any], resolved: dict[str, Any]) -> list[Any]:
    from ..validator import Issue, Severity

    regime = resolved.get("regime_definition")
    if regime in {None, "none"}:
        return []
    issues = []
    if regime == "external_user_provided":
        has_path = "regime_indicator_path" in leaf_config
        has_dates = "regime_dates_list" in leaf_config
        if has_path == has_dates:
            issues.append(
                _issue(
                    "l1.regime_definition",
                    "external_user_provided requires exactly one of regime_indicator_path or regime_dates_list",
                )
            )
    if regime in {"external_user_provided", "estimated_markov_switching"}:
        n_regimes = leaf_config.get("n_regimes", 2)
        if not isinstance(n_regimes, int) or n_regimes <= 0:
            issues.append(_issue("l1.n_regimes", "n_regimes must be a positive integer"))
    if regime == "estimated_threshold":
        if not isinstance(leaf_config.get("threshold_variable"), str):
            issues.append(_issue("l1.threshold_variable", "estimated_threshold requires leaf_config.threshold_variable"))
        if "n_thresholds" in leaf_config and (
            not isinstance(leaf_config["n_thresholds"], int) or leaf_config["n_thresholds"] <= 0
        ):
            issues.append(_issue("l1.n_thresholds", "n_thresholds must be a positive integer"))
    if regime == "estimated_structural_break":
        if "max_breaks" in leaf_config and (
            not isinstance(leaf_config["max_breaks"], int) or leaf_config["max_breaks"] <= 0
        ):
            issues.append(_issue("l1.max_breaks", "max_breaks must be a positive integer"))
        if leaf_config.get("break_ic_criterion", "bic") not in {"aic", "bic"}:
            issues.append(_issue("l1.break_ic_criterion", "break_ic_criterion must be one of ['aic', 'bic']"))
    temporal_rule = resolved.get("regime_estimation_temporal_rule")
    if temporal_rule == "full_sample_once":
        issues.append(_issue("l1.regime_estimation_temporal_rule", "full_sample_once is rejected because it leaks future data"))
    if regime in REGIME_ESTIMATED_OPTIONS:
        if temporal_rule not in REGIME_TEMPORAL_RULE_OPTIONS:
            issues.append(
                _issue(
                    "l1.regime_estimation_temporal_rule",
                    f"regime_estimation_temporal_rule must be one of {sorted(REGIME_TEMPORAL_RULE_OPTIONS)}",
                )
            )
        if temporal_rule == "rolling_window_per_origin":
            value = leaf_config.get(
                "regime_rolling_window_size",
                60 if resolved.get("frequency") == "monthly" else 20,
            )
            if not isinstance(value, int) or value <= 0:
                issues.append(_issue("l1.regime_rolling_window_size", "regime_rolling_window_size must be a positive integer"))
        if temporal_rule == "block_recompute":
            value = leaf_config.get(
                "block_recompute_interval",
                12 if resolved.get("frequency") == "monthly" else 4,
            )
            if not isinstance(value, int) or value <= 0:
                issues.append(_issue("l1.block_recompute_interval", "block_recompute_interval must be a positive integer"))
    elif temporal_rule is not None:
        issues.append(_issue("l1.regime_estimation_temporal_rule", "regime_estimation_temporal_rule is inactive unless regime_definition is estimated_*"))
    if regime == "external_nber" and resolved.get("frequency") != "monthly":
        issues.append(
            Issue(
                "l1_external_nber_frequency",
                Severity.SOFT,
                "layer",
                "l1.regime_definition",
                "USREC is a monthly series; quarterly analysis will use end-of-quarter values",
            )
        )
    return issues


def _resolved_horizons(resolved: dict[str, Any], leaf_config: dict[str, Any]) -> tuple[int, ...]:
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


def _active_axis_names(resolved: dict[str, Any]) -> tuple[str, ...]:
    return tuple(axis for axis in L1_AXIS_NAMES if resolved.get(axis) is not None)


def _axis_gates(axis_name: str) -> tuple[GatePredicate, ...]:
    if axis_name in {"dataset", "vintage_policy"}:
        return (GatePredicate(kind="axis_not_equals", target="custom_source_policy", value="custom_panel_only"),)
    if axis_name == "variable_universe":
        return (GatePredicate(kind="axis_not_in", target="dataset", value=["fred_sd"]),)
    if axis_name in {"target_geography_scope", "predictor_geography_scope"}:
        return (GatePredicate(kind="axis_in", target="dataset", value=list(FRED_SD_DATASETS)),)
    if axis_name == "regime_estimation_temporal_rule":
        return (GatePredicate(kind="axis_starts_with", target="regime_definition", value="estimated_"),)
    return ()


def _derived_frequency(custom_policy: Any, dataset: Any) -> str | None:
    if custom_policy == "custom_panel_only" or dataset == "fred_sd":
        return None
    if dataset in MONTHLY_DATASETS:
        return "monthly"
    if dataset in QUARTERLY_DATASETS:
        return "quarterly"
    return None


def _targets_from_leaf_config(leaf_config: dict[str, Any], resolved: dict[str, Any]) -> tuple[str, ...]:
    if resolved["target_structure"] == "single_target":
        return (leaf_config.get("target"),) if isinstance(leaf_config.get("target"), str) else ()
    return tuple(leaf_config.get("targets", ()) or ())


def _resolved_axis_entries(resolved: dict[str, Any], fixed_axes: dict[str, Any]) -> dict[str, ResolvedAxis]:
    entries = {}
    for axis_name, value in resolved.items():
        if value is None:
            continue
        if axis_name in fixed_axes:
            source = "explicit"
        elif axis_name in {"frequency", "horizon_set"}:
            source = "derived"
        else:
            source = "package_default"
        entries[axis_name] = ResolvedAxis(value, source)
    return entries


def _validate_state_list(key: str, states: Any) -> list[Any]:
    if not isinstance(states, list) or not states:
        return [_issue(f"l1.{key}", f"{key} must be a non-empty list of valid state codes")]
    invalid = [state for state in states if state not in VALID_STATE_CODES]
    if invalid:
        return [_issue(f"l1.{key}", f"{key} contains invalid state codes: {invalid}")]
    return []


def _positive_int_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(item, int) and item > 0 for item in value)


def _is_iso_date(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _is_sweep_marker(value: Any) -> bool:
    return isinstance(value, dict) and "sweep" in value


def _raw_layer(layer: Any | dict[str, Any]) -> dict[str, Any]:
    return layer.raw_yaml if hasattr(layer, "raw_yaml") else layer


def _issue(location: str, message: str) -> Any:
    from ..validator import Issue, Severity

    return Issue("l1_contract", Severity.HARD, "layer", location, message)


def _options(values: tuple[str, ...]) -> tuple[Option, ...]:
    return tuple(Option(value, value.replace("_", " ").title(), "") for value in values)


L1_LAYER_SPEC = LayerImplementationSpec(
    layer_id="l1",
    name="Data",
    category="construction",
    expected_inputs=(),
    produces=("l1_data_definition_v1", "l1_regime_metadata_v1"),
    ui_mode="list",
    layer_globals=(),
    sub_layers=(
        SubLayerSpec(id="l1_a", name="Source selection", axes=("custom_source_policy", "dataset", "frequency", "vintage_policy")),
        SubLayerSpec(id="l1_b", name="Target definition", axes=("target_structure",)),
        SubLayerSpec(id="l1_c", name="Predictor universe", axes=("variable_universe",)),
        SubLayerSpec(id="l1_d", name="Geography scope", axes=("target_geography_scope", "predictor_geography_scope")),
        SubLayerSpec(id="l1_e", name="Sample window", axes=("sample_start_rule", "sample_end_rule")),
        SubLayerSpec(id="l1_f", name="Horizon set", axes=("horizon_set",)),
        SubLayerSpec(id="l1_g", name="Regime definition", axes=("regime_definition", "regime_estimation_temporal_rule")),
    ),
    axes={
        "l1_a": {
            "custom_source_policy": AxisSpec("custom_source_policy", _options(("official_only", "custom_panel_only", "official_plus_custom")), "official_only", sweepable=False),
            "dataset": AxisSpec("dataset", _options(DATASET_OPTIONS), "fred_md", sweepable=True),
            "frequency": AxisSpec("frequency", _options(("monthly", "quarterly")), "derived", sweepable=False),
            "vintage_policy": AxisSpec(
                "vintage_policy",
                (
                    Option("current_vintage", "Current Vintage", "", status="operational"),
                    Option("real_time_alfred", "Real-Time ALFRED", "", status="future"),
                ),
                "current_vintage",
                sweepable=False,
            ),
        },
        "l1_b": {"target_structure": AxisSpec("target_structure", _options(("single_target", "multi_series_target")), "single_target", sweepable=False)},
        "l1_c": {
            "variable_universe": AxisSpec(
                "variable_universe",
                _options(("all_variables", "core_variables", "category_variables", "target_specific_variables", "explicit_variable_list")),
                "all_variables",
                sweepable=False,
            )
        },
        "l1_d": {
            "target_geography_scope": AxisSpec("target_geography_scope", _options(("single_state", "all_states", "selected_states")), "all_states", sweepable=False),
            "predictor_geography_scope": AxisSpec(
                "predictor_geography_scope",
                _options(("match_target", "all_states", "selected_states", "national_only")),
                "match_target",
                sweepable=False,
            ),
        },
        "l1_e": {
            "sample_start_rule": AxisSpec("sample_start_rule", _options(("earliest_available", "fixed_date", "max_balanced")), "max_balanced", sweepable=False),
            "sample_end_rule": AxisSpec("sample_end_rule", _options(("latest_available", "fixed_date")), "latest_available", sweepable=False),
        },
        "l1_f": {
            "horizon_set": AxisSpec("horizon_set", _options(("standard_md", "standard_qd", "single", "custom_list", "range_up_to_h")), "derived", sweepable=False)
        },
        "l1_g": {
            "regime_definition": AxisSpec("regime_definition", _options(REGIME_OPTIONS), "none", sweepable=False),
            "regime_estimation_temporal_rule": AxisSpec(
                "regime_estimation_temporal_rule",
                _options(REGIME_TEMPORAL_RULE_OPTIONS),
                "expanding_window_per_origin",
                sweepable=False,
                gate=GatePredicate(kind="axis_starts_with", target="regime_definition", value="estimated_"),
            ),
        },
    },
    ops=(),
)


__all__ = [
    "L1Data",
    "L1_LAYER_SPEC",
    "parse_layer_yaml",
    "normalize_to_dag_form",
    "resolve_axes",
    "validate_layer",
    "validate_regime_source_reference",
    "build_recipe_with_l1_only",
    "execute_recipe",
]
