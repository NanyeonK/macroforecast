from __future__ import annotations

from pathlib import Path

from macroforecast.core import (
    DAG,
    LAYER_YAML_KEYS,
    Node,
    NodeRef,
    Recipe,
    Severity,
    SourceSelector,
    SweepCombination,
    canonical_serialize,
    collect_all_sweeps,
    ensure_cache_layout,
    execute_node,
    expand_sweeps,
    layer_hash,
    node_hash,
    normalize_to_dag_form,
    recipe_hash,
    validate_recipe,
    validate_sweepable_params,
)


def _sweep_dags() -> dict:
    return {
        "l3": DAG(
            layer_id="l3",
            nodes={
                "src": Node(
                    id="src",
                    type="source",
                    layer_id="l3",
                    op="source",
                    selector=SourceSelector(layer_ref="l2", sink_name="l2_clean_panel_v1"),
                ),
                "lag_x": Node(
                    id="lag_x",
                    type="step",
                    layer_id="l3",
                    op="lag",
                    params={"n_lag": {"sweep": [4, 8, 12]}},
                    inputs=(NodeRef("src"),),
                ),
            },
            sinks={"features_v1": "lag_x"},
        ),
        "l4": DAG(
            layer_id="l4",
            nodes={},
            layer_globals={"model_family": {"sweep": ["ridge", "xgboost", "random_forest"]}},
        ),
    }


def test_param_and_external_axis_sweep_expand_grid_and_zip() -> None:
    dags = _sweep_dags()

    specs = collect_all_sweeps(dags)
    grid_cells = expand_sweeps(dags)
    zip_cells = expand_sweeps(dags, SweepCombination(mode="zip"))

    assert len(specs) == 2
    assert len(grid_cells) == 9
    assert len(zip_cells) == 3
    assert grid_cells[0].concrete_dag["l3"].nodes["lag_x"].params["n_lag"] == 4
    assert grid_cells[-1].concrete_dag["l4"].layer_globals["model_family"] == "random_forest"
    assert grid_cells[0].cell_id


def test_node_group_sweep_switches_sink_target() -> None:
    dag = DAG(
        layer_id="l3",
        nodes={
            "src": Node(
                id="src",
                type="source",
                layer_id="l3",
                op="source",
                selector=SourceSelector(layer_ref="l2", sink_name="l2_clean_panel_v1"),
            ),
            "pipeline_a": Node(id="pipeline_a", type="step", layer_id="l3", op="lag", params={"n_lag": 1}, inputs=(NodeRef("src"),)),
            "pipeline_b": Node(id="pipeline_b", type="step", layer_id="l3", op="lag", params={"n_lag": 2}, inputs=(NodeRef("src"),)),
        },
        sinks={"pipeline_choice": "pipeline_a"},
        layer_globals={"_sweep_groups": ({"id": "pipeline_choice", "members": ["pipeline_a", "pipeline_b"]},)},
    )

    cells = expand_sweeps({"l3": dag})

    assert len(cells) == 2
    assert cells[0].concrete_dag["l3"].sinks["pipeline_choice"] == "pipeline_a"
    assert cells[1].concrete_dag["l3"].sinks["pipeline_choice"] == "pipeline_b"


def test_sweepable_param_validator_rejects_non_sweepable_param() -> None:
    dags = {
        "l3": DAG(
            layer_id="l3",
            nodes={
                "src": Node(
                    id="src",
                    type="source",
                    layer_id="l3",
                    op="source",
                    selector=SourceSelector(layer_ref="l2", sink_name="l2_clean_panel_v1"),
                ),
                "lag_x": Node(
                    id="lag_x",
                    type="step",
                    layer_id="l3",
                    op="lag",
                    params={"unknown_param": {"sweep": [1, 2]}},
                    inputs=(NodeRef("src"),),
                ),
            },
            sinks={"features_v1": "lag_x"},
        )
    }

    try:
        validate_sweepable_params(dags)
    except ValueError as exc:
        assert "not sweepable" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected non-sweepable param to fail")


def test_cache_hashes_are_stable_and_sensitive_to_param_changes(tmp_path: Path) -> None:
    dag = _sweep_dags()["l3"]
    concrete_a = expand_sweeps({"l3": dag})[0].concrete_dag["l3"]
    concrete_b = expand_sweeps({"l3": dag})[1].concrete_dag["l3"]

    assert canonical_serialize({"b": 1, "a": 2}) == canonical_serialize({"a": 2, "b": 1})
    assert node_hash(concrete_a.nodes["src"], concrete_a, {"T": 100}) == node_hash(concrete_b.nodes["src"], concrete_b, {"T": 100})
    assert node_hash(concrete_a.nodes["lag_x"], concrete_a, {"T": 100}) != node_hash(concrete_b.nodes["lag_x"], concrete_b, {"T": 100})
    assert layer_hash(concrete_a, {"T": 100}) != layer_hash(concrete_b, {"T": 100})
    assert recipe_hash({"l3": concrete_a}) == recipe_hash({"l3": concrete_a})

    ensure_cache_layout(tmp_path)
    assert (tmp_path / "nodes").is_dir()
    assert (tmp_path / "cells").is_dir()
    assert (tmp_path / "runtime_context").is_dir()


def test_execute_node_uses_source_context_and_cache(tmp_path: Path, mock_clean_panel) -> None:
    dag = DAG(
        layer_id="l3",
        nodes={
            "src": Node(
                id="src",
                type="source",
                layer_id="l3",
                op="source",
                selector=SourceSelector(layer_ref="l2", sink_name="l2_clean_panel_v1"),
            ),
            "lag_x": Node(id="lag_x", type="step", layer_id="l3", op="lag", params={"n_lag": 2}, inputs=(NodeRef("src"),)),
        },
        sinks={"features_v1": "lag_x"},
    )

    result = execute_node(dag.nodes["lag_x"], dag, {"sources": {"l2.l2_clean_panel_v1": mock_clean_panel}}, tmp_path)
    cached = execute_node(dag.nodes["lag_x"], dag, {"sources": {"l2.l2_clean_panel_v1": mock_clean_panel}}, tmp_path)

    assert result.column_names[:2] == ("INDPRO_lag1", "INDPRO_lag2")
    assert cached == result


def test_yaml_sugar_form_normalizes_to_dag() -> None:
    dag = normalize_to_dag_form(
        {
            "fixed_axes": {
                "failure_policy": "fail_fast",
                "reproducibility_mode": "seeded_reproducible",
                "compute_mode": "serial",
            },
            "leaf_config": {"random_seed": 42},
        },
        "l0",
    )

    assert dag.nodes["axis_failure_policy"].params["value"] == "fail_fast"
    assert dag.nodes["meta_aggregate"].op == "layer_meta_aggregate"


def test_recipe_yaml_dag_form_and_validation_report() -> None:
    recipe = Recipe.from_yaml(
        """
metadata:
  name: demo
0_meta:
  fixed_axes:
    failure_policy: fail_fast
3_feature_engineering:
  nodes:
    - id: src_clean
      type: source
      selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {predictors: true}}
    - id: lag_x
      type: step
      op: lag
      params: {n_lag: {sweep: [4, 8]}}
      inputs: [src_clean]
  sinks:
    features_v1: lag_x
sweep_combination:
  mode: grid
"""
    )

    dags = recipe.to_dag_form()
    report = validate_recipe(recipe)

    assert LAYER_YAML_KEYS["l4"] == "4_forecasting_model"
    assert dags["l3"].nodes["lag_x"].params["n_lag"]["sweep"] == [4, 8]
    assert len(recipe.cells) == 2
    assert not report.has_hard_errors


def test_yaml_dag_form_parses_node_group_sweeps() -> None:
    recipe = Recipe.from_yaml(
        """
3_feature_engineering:
  nodes:
    - id: src_clean
      type: source
      selector: {layer_ref: l2, sink_name: l2_clean_panel_v1}
    - {id: pipeline_a, type: step, op: lag, params: {n_lag: 1}, inputs: [src_clean]}
    - {id: pipeline_b, type: step, op: lag, params: {n_lag: 2}, inputs: [src_clean]}
  sinks:
    pipeline_choice: pipeline_a
  sweep_groups:
    - id: pipeline_choice
      members: [pipeline_a, pipeline_b]
"""
    )

    assert len(recipe.cells) == 2
    assert recipe.cells[1].concrete_dag["l3"].sinks["pipeline_choice"] == "pipeline_b"


def test_validation_report_surfaces_hard_errors() -> None:
    recipe = Recipe.from_yaml(
        """
3_feature_engineering:
  nodes:
    - {id: lag_x, type: step, op: lag, params: {n_lag: 2}, inputs: [missing]}
  sinks:
    features_v1: lag_x
"""
    )

    report = validate_recipe(recipe)

    assert report.has_hard_errors
    assert report.hard_errors
    assert all(issue.severity is Severity.HARD for issue in report.hard_errors)
