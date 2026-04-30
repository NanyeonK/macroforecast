from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ..dag import DAG, Node, NodeRef, SourceSelector
from ..ops.l4_ops import FUTURE_MODEL_FAMILIES, MODEL_FAMILY_STATUS, OPERATIONAL_MODEL_FAMILIES, get_family_status


class L4ForecastingModel:
    """Layer 4 Forecasting Model implementation marker."""

    @classmethod
    def list_sublayers(cls) -> tuple[str, ...]:
        return ("L4.A", "L4.B", "L4.C", "L4.D")


class L4ResolvedAxes(dict):
    def get_active(self, key: str) -> bool:
        return bool(self.get("_active", {}).get(key, True))


def parse_layer_yaml(yaml_text: str, layer_id: Literal["l4"] = "l4") -> dict[str, Any]:
    if layer_id != "l4":
        raise ValueError("L4 parser only accepts layer_id='l4'")
    from ..yaml import parse_recipe_yaml

    root = parse_recipe_yaml(yaml_text)
    raw = root.get("4_forecasting_model", root)
    if not isinstance(raw, dict):
        raise ValueError("4_forecasting_model: layer YAML must be a mapping")
    return raw


def parse_recipe_yaml(yaml_text: str) -> dict[str, Any]:
    from ..yaml import parse_recipe_yaml as parse

    return parse(yaml_text)


def parse_dag_form(nodes: list[dict[str, Any]] | dict[str, Any]) -> DAG:
    layer_yaml = nodes if isinstance(nodes, dict) else {"nodes": nodes, "sinks": {}}
    if "sinks" not in layer_yaml:
        layer_yaml = {**layer_yaml, "sinks": {}}
    return normalize_to_dag_form(layer_yaml)


def normalize_to_dag_form(layer: dict[str, Any], layer_id: Literal["l4"] = "l4") -> DAG:
    if layer_id != "l4":
        raise ValueError("L4 normalizer only accepts layer_id='l4'")
    if "nodes" not in layer:
        raise ValueError("L4 supports DAG form only")

    nodes: dict[str, Node] = {}
    for raw_node in layer.get("nodes", ()):
        node = _parse_node(raw_node)
        if node.id in nodes:
            raise ValueError(f"l4.{node.id}: duplicate node id")
        nodes[node.id] = node

    sink_map: dict[str, str] = {}
    raw_sinks = layer.get("sinks", {}) or {}
    for sink_name, target in raw_sinks.items():
        if sink_name == "l4_model_artifacts_v1" and isinstance(target, list):
            aggregate_id = "sink:l4_model_artifacts_v1"
            nodes[aggregate_id] = Node(
                id=aggregate_id,
                type="combine",
                layer_id="l4",
                op="l4_model_artifacts_collect",
                inputs=tuple(NodeRef(item) for item in target),
            )
            sink_map[sink_name] = aggregate_id
        elif sink_name == "l4_training_metadata_v1" and target == "auto":
            fit_nodes = tuple(NodeRef(node.id) for node in nodes.values() if node.op == "fit_model")
            aggregate_id = "sink:l4_training_metadata_v1"
            nodes[aggregate_id] = Node(
                id=aggregate_id,
                type="combine",
                layer_id="l4",
                op="l4_training_metadata_build",
                inputs=fit_nodes[:1] if len(fit_nodes) == 1 else fit_nodes,
            )
            sink_map[sink_name] = aggregate_id
        else:
            sink_map[sink_name] = target

    return DAG(layer_id="l4", nodes=nodes, sinks=sink_map, layer_globals={"leaf_config": layer.get("leaf_config", {}) or {}})


def resolve_axes(dag: DAG) -> L4ResolvedAxes:
    values: dict[str, Any] = {"_active": {}}
    for node in dag.nodes.values():
        if node.op != "fit_model":
            continue
        for key, default in {
            "forecast_strategy": "direct",
            "training_start_rule": "expanding",
            "refit_policy": "every_origin",
            "search_algorithm": "none",
            "tuning_objective": "cv_mse",
            "validation_method": "expanding_walk_forward",
        }.items():
            values[key] = node.params.get(key, default)
        values["_active"]["tuning_objective"] = values.get("search_algorithm") != "none"
        values["_active"]["validation_method"] = values.get("search_algorithm") != "none"
        break
    return L4ResolvedAxes(values)


def validate_layer(layer: dict[str, Any] | str, recipe_context: dict[str, Any] | None = None) -> ValidationReport:
    from ..validator import Issue, Severity, ValidationReport, validate_dag

    raw = parse_layer_yaml(layer) if isinstance(layer, str) else layer
    issues: list[Issue] = []
    try:
        dag = normalize_to_dag_form(raw)
    except Exception as exc:
        return ValidationReport((_issue("l4", str(exc)),))

    issues.extend(_validate_sinks(dag))
    issues.extend(_validate_benchmark(raw))
    issues.extend(_validate_fit_nodes(raw))
    issues.extend(_validate_combine_nodes(raw))
    issues.extend(_validate_regime(raw, recipe_context))
    issues.extend(_validate_path_average(raw, recipe_context))

    dag_result = validate_dag(dag)
    issues.extend(
        Issue("l4_dag", Severity.HARD if issue.severity == "hard" else Severity.SOFT, "dag", issue.location, issue.message)
        for issue in dag_result.issues
    )
    return ValidationReport(tuple(issues))


def validate_recipe(recipe_yaml: dict[str, Any] | str) -> ValidationReport:
    from ..validator import ValidationReport

    root = parse_recipe_yaml(recipe_yaml) if isinstance(recipe_yaml, str) else recipe_yaml
    if "4_forecasting_model" not in root:
        return ValidationReport()
    return validate_layer(root["4_forecasting_model"], recipe_context=_recipe_context(root))


def execute_layer(layer: dict[str, Any] | str):
    raw = parse_layer_yaml(layer) if isinstance(layer, str) else layer
    families = [node.get("params", {}).get("family") for node in raw.get("nodes", ()) if node.get("op") == "fit_model"]
    if any(family in {"macroeconomic_random_forest", "dfm_mixed_mariano_murasawa"} for family in families):
        raise NotImplementedError("Phase 1 runtime: selected L4 model family execution is deferred")
    raise NotImplementedError("Phase 1 runtime: L4 execution is deferred")


def resolve_combine_node(layer: dict[str, Any] | str) -> dict[str, Any]:
    raw = parse_layer_yaml(layer) if isinstance(layer, str) else layer
    for node in raw.get("nodes", ()):
        if node.get("op") == "weighted_average_forecast":
            params = dict(node.get("params", {}) or {})
            if params.get("weights_method") == "inverse_msfe":
                params["weights_method"] = "dmsfe"
                params["dmsfe_theta"] = 1.0
            params.setdefault("weights_method", "dmsfe")
            params.setdefault("dmsfe_theta", 0.95)
            return params
    return {}


def make_l4_yaml(family: str = "ridge", **params: Any) -> str:
    params = {"family": family, "forecast_strategy": "direct", "training_start_rule": "expanding", "refit_policy": "every_origin", "search_algorithm": "none", **params}
    param_text = _yaml_inline(params)
    inputs = "[src_y]" if family == "ar_p" else "[src_X, src_y]"
    return f"""
4_forecasting_model:
  nodes:
    - {{id: src_X, type: source, selector: {{layer_ref: l3, sink_name: l3_features_v1, subset: {{component: X_final}}}}}}
    - {{id: src_y, type: source, selector: {{layer_ref: l3, sink_name: l3_features_v1, subset: {{component: y_final}}}}}}
    - id: fit_model
      type: step
      op: fit_model
      params: {param_text}
      inputs: {inputs}
    - {{id: predict_model, type: step, op: predict, inputs: [fit_model, src_X]}}
  sinks:
    l4_forecasts_v1: predict_model
    l4_model_artifacts_v1: fit_model
    l4_training_metadata_v1: auto
"""


def make_l4_yaml_no_benchmark() -> str:
    return make_l4_yaml("ridge")


def make_l4_yaml_with_strategy(strategy: str) -> str:
    return make_l4_yaml("ridge", forecast_strategy=strategy)


def make_l4_yaml_with_cv_path(family: str) -> str:
    return make_l4_yaml(family, search_algorithm="cv_path")


def make_l4_yaml_with_validation_method(method: str) -> str:
    return make_l4_yaml("ridge", search_algorithm="grid_search", validation_method=method) + "\n  leaf_config:\n    tuning_grid: {alpha: [0.1, 1.0]}\n    n_splits: 5\n"


def make_l4_yaml_training_window(rule: str) -> str:
    return make_l4_yaml("ridge", training_start_rule=rule)


def make_l4_yaml_refit(policy: str) -> str:
    return make_l4_yaml("ridge", refit_policy=policy)


def make_l4_yaml_search(search_algorithm: str) -> str:
    return make_l4_yaml("ridge", search_algorithm=search_algorithm)


def make_l4_yaml_with_combine_temporal(rule: str) -> str:
    return _ensemble_yaml(combine_op="weighted_average_forecast", combine_params={"weights_method": "dmsfe", "temporal_rule": rule}, n_inputs=2)


def make_l4_yaml_with_combine_method(method: str) -> str:
    return _ensemble_yaml(combine_op="weighted_average_forecast", combine_params={"weights_method": method, "temporal_rule": "expanding_window_per_origin"}, n_inputs=2)


def make_l4_yaml_with_combine(n_inputs: int) -> str:
    return _ensemble_yaml(combine_op="weighted_average_forecast", combine_params={"weights_method": "dmsfe"}, n_inputs=n_inputs)


def make_l4_yaml_with_combine_op(op_name: str, n_inputs: int) -> str:
    return _ensemble_yaml(combine_op=op_name, combine_params={}, n_inputs=n_inputs)


def _parse_node(raw_node: dict[str, Any]) -> Node:
    selector = raw_node.get("selector")
    return Node(
        id=raw_node["id"],
        type=raw_node["type"],
        layer_id="l4",
        op=raw_node.get("op", raw_node["type"]),
        params=raw_node.get("params", {}) or {},
        inputs=tuple(_parse_ref(ref) for ref in raw_node.get("inputs", ())),
        selector=SourceSelector(selector["layer_ref"], selector["sink_name"], selector.get("subset", {}) or {}) if selector else None,
        is_benchmark=bool(raw_node.get("is_benchmark", False)),
        status=raw_node.get("status", "operational"),
    )


def _parse_ref(raw: Any) -> NodeRef:
    if isinstance(raw, str):
        return NodeRef(raw)
    return NodeRef(raw["node_id"], raw.get("output_port", "default"))


def _validate_sinks(dag: DAG) -> list[Issue]:
    required = {"l4_forecasts_v1", "l4_model_artifacts_v1", "l4_training_metadata_v1"}
    missing = required - set(dag.sinks)
    return [_issue("l4.sinks", f"missing required sink {sink}") for sink in sorted(missing)]


def _validate_benchmark(raw: dict[str, Any]) -> list[Issue]:
    benchmarks = [node for node in raw.get("nodes", ()) if node.get("op") == "fit_model" and node.get("is_benchmark")]
    if len(benchmarks) > 1:
        return [_issue("l4.benchmark", "exactly one or zero benchmark fit_model nodes allowed")]
    return []


def _validate_fit_nodes(raw: dict[str, Any]) -> list[Issue]:
    issues = []
    leaf = raw.get("leaf_config", {}) or {}
    for node in raw.get("nodes", ()):
        if node.get("op") != "fit_model":
            continue
        params = node.get("params", {}) or {}
        family = params.get("family", "ridge")
        if MODEL_FAMILY_STATUS.get(family) == "future":
            issues.append(_issue(f"l4.{node['id']}", f"model family {family} is future and not executable"))
        elif family not in MODEL_FAMILY_STATUS:
            issues.append(_issue(f"l4.{node['id']}", f"unknown model family {family!r}"))
        strategy = params.get("forecast_strategy", "direct")
        if strategy not in {"direct", "iterated", "path_average"}:
            issues.append(_issue(f"l4.{node['id']}", "forecast_strategy must be direct, iterated, or path_average"))
        search = params.get("search_algorithm", "none")
        if search == "cv_path" and family not in {"lasso", "ridge", "elastic_net"}:
            issues.append(_issue(f"l4.{node['id']}", "cv_path search_algorithm requires lasso, ridge, or elastic_net"))
        if search == "grid_search" and "tuning_grid" not in leaf:
            issues.append(_issue(f"l4.{node['id']}", "grid_search requires leaf_config.tuning_grid"))
        if search in {"random_search", "bayesian_optimization"}:
            if "tuning_distributions" not in leaf or "tuning_budget" not in leaf:
                issues.append(_issue(f"l4.{node['id']}", f"{search} requires tuning_distributions and tuning_budget"))
        if search == "genetic_algorithm":
            for key in ("tuning_budget", "genetic_algorithm_population", "genetic_algorithm_generations"):
                if key not in leaf:
                    issues.append(_issue(f"l4.{node['id']}", f"genetic_algorithm requires {key}"))
        if search == "cv_path" and "cv_path_alphas" not in leaf:
            issues.append(_issue(f"l4.{node['id']}", "cv_path requires leaf_config.cv_path_alphas"))
        if params.get("validation_method") == "kfold" and int(leaf.get("n_splits", 5)) > 2:
            issues.append(_issue(f"l4.{node['id']}", "kfold validation is rejected for time-series data"))
        if params.get("training_start_rule") == "fixed" and "fixed_training_end_date" not in leaf:
            issues.append(_issue(f"l4.{node['id']}", "fixed training_start_rule requires fixed_training_end_date"))
        if params.get("refit_policy") == "every_n_origins" and "refit_interval" not in leaf:
            issues.append(_issue(f"l4.{node['id']}", "every_n_origins requires refit_interval"))
        if params.get("regime_wrapper") and not _has_regime_input(node):
            issues.append(_issue(f"l4.{node['id']}", "regime_wrapper requires l1_regime_metadata_v1 input"))
    return issues


def _validate_combine_nodes(raw: dict[str, Any]) -> list[Issue]:
    issues = []
    for node in raw.get("nodes", ()):
        op = node.get("op")
        params = node.get("params", {}) or {}
        if op in {"weighted_average_forecast", "median_forecast", "trimmed_mean_forecast", "bma_forecast"} and len(node.get("inputs", ())) < 2:
            issues.append(_issue(f"l4.{node['id']}", f"{op} requires 2 or more inputs"))
        if op == "bivariate_ardl_combination" and len(node.get("inputs", ())) != 2:
            issues.append(_issue(f"l4.{node['id']}", "bivariate_ardl_combination requires exactly 2 inputs"))
        temporal = params.get("temporal_rule") or params.get("combination_weights_temporal_rule")
        if temporal == "full_sample_once":
            issues.append(_issue(f"l4.{node['id']}", "full_sample_once is rejected for forecast combination temporal_rule"))
    return issues


def _validate_regime(raw: dict[str, Any], recipe_context: dict[str, Any] | None) -> list[Issue]:
    if not recipe_context or recipe_context.get("regime_definition") != "none":
        return []
    for node in raw.get("nodes", ()):
        if node.get("op") == "fit_model" and (node.get("params", {}) or {}).get("regime_wrapper"):
            return [_issue(f"l4.{node['id']}", "regime_wrapper requires L1 regime_definition != none")]
    return []


def _validate_path_average(raw: dict[str, Any], recipe_context: dict[str, Any] | None) -> list[Issue]:
    if not recipe_context:
        return []
    if recipe_context.get("target_mode") in {None, "cumulative_average"}:
        return []
    for node in raw.get("nodes", ()):
        if node.get("op") == "fit_model" and (node.get("params", {}) or {}).get("forecast_strategy") == "path_average":
            return [_issue(f"l4.{node['id']}", "path_average requires L3 target_construction.mode == cumulative_average")]
    return []


def _recipe_context(root: dict[str, Any]) -> dict[str, Any]:
    context = {"regime_definition": ((root.get("1_data", {}) or {}).get("fixed_axes", {}) or {}).get("regime_definition", "none")}
    for node in ((root.get("3_feature_engineering", {}) or {}).get("nodes", ())):
        if node.get("op") == "target_construction":
            context["target_mode"] = (node.get("params", {}) or {}).get("mode")
            break
    return context


def _has_regime_input(node: dict[str, Any]) -> bool:
    return any(ref == "src_regime" for ref in node.get("inputs", ()) if isinstance(ref, str))


def _ensemble_yaml(combine_op: str, combine_params: dict[str, Any], n_inputs: int) -> str:
    fit_predict = []
    inputs = []
    for idx in range(n_inputs):
        fit_predict.append(f"    - {{id: fit_{idx}, type: step, op: fit_model, params: {{family: ridge}}, inputs: [src_X, src_y]}}")
        fit_predict.append(f"    - {{id: predict_{idx}, type: step, op: predict, inputs: [fit_{idx}, src_X]}}")
        inputs.append(f"predict_{idx}")
    return f"""
4_forecasting_model:
  nodes:
    - {{id: src_X, type: source, selector: {{layer_ref: l3, sink_name: l3_features_v1, subset: {{component: X_final}}}}}}
    - {{id: src_y, type: source, selector: {{layer_ref: l3, sink_name: l3_features_v1, subset: {{component: y_final}}}}}}
{chr(10).join(fit_predict)}
    - id: ensemble
      type: combine
      op: {combine_op}
      params: {_yaml_inline(combine_params)}
      inputs: [{', '.join(inputs)}]
  sinks:
    l4_forecasts_v1: ensemble
    l4_model_artifacts_v1: [{', '.join(f'fit_{idx}' for idx in range(n_inputs))}]
    l4_training_metadata_v1: auto
"""


def _yaml_inline(params: dict[str, Any]) -> str:
    parts = []
    for key, value in params.items():
        if isinstance(value, str):
            rendered = value
        else:
            rendered = repr(value)
        parts.append(f"{key}: {rendered}")
    return "{" + ", ".join(parts) + "}"


def _issue(location: str, message: str) -> Issue:
    from ..validator import Issue, Severity

    return Issue("l4_contract", Severity.HARD, "layer", location, message)
