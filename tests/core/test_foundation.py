from __future__ import annotations

import pytest

from macrocast.core import (
    DAG,
    LAYER_GLOBALS,
    LAYER_SINKS,
    GatePredicate,
    LaggedPanel,
    Node,
    NodeRef,
    Panel,
    SourceContext,
    SourceSelector,
    get_layer,
    get_op,
    list_layers,
    resolve_source_selector,
    validate_dag,
)


def test_dag_rejects_node_key_id_mismatch() -> None:
    with pytest.raises(ValueError, match="node mapping keys must match"):
        DAG(
            layer_id="l3",
            nodes={
                "a": Node(id="b", type="axis", layer_id="l3", op="demo"),
            },
        )


def test_minimal_valid_dag() -> None:
    dag = DAG(
        layer_id="l3",
        nodes={
            "src": Node(
                id="src",
                type="source",
                layer_id="l3",
                op="source",
                selector=SourceSelector(layer_ref="l2", sink_name="clean_panel_v1", subset={"predictors": True}),
            ),
            "lag_x": Node(
                id="lag_x",
                type="step",
                layer_id="l3",
                op="lag",
                params={"n_lag": 2},
                inputs=(NodeRef("src"),),
            ),
        },
        sinks={"lagged_x": "lag_x"},
    )

    result = validate_dag(dag)

    assert result.valid
    assert result.node_output_types["src"] is Panel
    assert result.node_output_types["lag_x"] is LaggedPanel


def test_invalid_input_type() -> None:
    dag = DAG(
        layer_id="l3",
        nodes={
            "models": Node(
                id="models",
                type="source",
                layer_id="l3",
                op="source",
                selector=SourceSelector(layer_ref="l4", sink_name="model_artifacts_v1"),
            ),
            "lag_models": Node(
                id="lag_models",
                type="step",
                layer_id="l3",
                op="lag",
                params={"n_lag": 2},
                inputs=(NodeRef("models"),),
            ),
        },
        sinks={"bad": "lag_models"},
    )

    result = validate_dag(dag)

    assert not result.valid
    assert any("not compatible" in issue.message for issue in result.issues)


def test_gate_propagation() -> None:
    dag = DAG(
        layer_id="l1",
        nodes={
            "geo": Node(
                id="geo",
                type="axis",
                layer_id="l1",
                op="state_selection",
                params={"value": "CA"},
                gates=(GatePredicate(kind="axis_in", target="dataset", value=["fred_sd", "fred_md+fred_sd"]),),
            ),
            "geo_sink": Node(
                id="geo_sink",
                type="sink",
                layer_id="l1",
                op="sink",
                inputs=(NodeRef("geo"),),
            ),
        },
        sinks={"state_scope": "geo_sink"},
        layer_globals={"dataset": "fred_md"},
    )

    result = validate_dag(dag)

    assert "geo" in result.disabled_nodes
    assert "geo_sink" in result.disabled_nodes
    assert not result.valid
    assert any("disabled node" in issue.message for issue in result.issues)


def test_axis_starts_with_gate_propagation() -> None:
    active = DAG(
        layer_id="l1",
        nodes={
            "temporal_rule": Node(
                id="temporal_rule",
                type="axis",
                layer_id="l1",
                op="regime_estimation_temporal_rule",
                params={"value": "expanding_window_per_origin"},
                gates=(GatePredicate(kind="axis_starts_with", target="regime_definition", value="estimated_"),),
            )
        },
        sinks={"temporal_rule": "temporal_rule"},
        layer_globals={"regime_definition": "estimated_markov_switching"},
    )
    inactive = DAG(
        layer_id="l1",
        nodes=active.nodes,
        sinks=active.sinks,
        layer_globals={"regime_definition": "external_nber"},
    )

    assert validate_dag(active).valid
    inactive_result = validate_dag(inactive)
    assert "temporal_rule" in inactive_result.disabled_nodes
    assert not inactive_result.valid


def test_cross_layer_reference() -> None:
    selector = SourceSelector(
        layer_ref="l6",
        sink_name="tests_v1",
        subset={"family": "multiple_model", "name": "mcs_inclusion"},
    )
    active = SourceContext(
        active_layers=frozenset({"l6"}),
        available_sinks={"l6": frozenset({"tests_v1"})},
    )

    resolved = resolve_source_selector(selector, active)

    assert resolved is LAYER_SINKS["l6"]["tests_v1"]


def test_cross_layer_reference_requires_active_layer() -> None:
    selector = SourceSelector(layer_ref="l6", sink_name="tests_v1")
    inactive = SourceContext(active_layers=frozenset({"l5"}), available_sinks={"l5": frozenset({"evaluation_v1"})})

    with pytest.raises(ValueError, match="source layer is not active"):
        resolve_source_selector(selector, inactive)


def test_layer_and_op_registries_expose_foundation_contracts() -> None:
    layers = list_layers()

    assert layers["l3"].category == "construction"
    assert layers["l3"].expected_inputs == ("l2.clean_panel_v1", "l1.raw_panel_v1")
    assert get_layer("l7").category == "consumption"
    assert get_op("lag").output_type is LaggedPanel
    assert LAYER_GLOBALS["l6"] == ("test_scope", "dependence_correction", "overlap_handling")


def test_node_order_does_not_affect_type_resolution() -> None:
    dag = DAG(
        layer_id="l3",
        nodes={
            "lag_x": Node(
                id="lag_x",
                type="step",
                layer_id="l3",
                op="lag",
                params={"n_lag": 2},
                inputs=(NodeRef("src"),),
            ),
            "src": Node(
                id="src",
                type="source",
                layer_id="l3",
                op="source",
                selector=SourceSelector(layer_ref="l2", sink_name="clean_panel_v1"),
            ),
        },
        sinks={"lagged_x": "lag_x"},
    )

    result = validate_dag(dag)

    assert result.valid
    assert result.node_output_types["lag_x"] is LaggedPanel


def test_non_operational_status_is_hard_error() -> None:
    dag = DAG(
        layer_id="l3",
        nodes={
            "future_step": Node(
                id="future_step",
                type="step",
                layer_id="l3",
                op="lag",
                status="future",
            ),
        },
    )

    result = validate_dag(dag)

    assert not result.valid
    assert any("not executable" in issue.message for issue in result.issues)
