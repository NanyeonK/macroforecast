from __future__ import annotations

from .registry import Rule, register_op
from ..types import DataType, Factor, LaggedPanel, MappingArtifact, Panel, Series


@register_op(
    name="identity",
    layer_scope="universal",
    input_types={"default": (Panel, Series, LaggedPanel, Factor)},
    output_type=(Panel, Series, LaggedPanel, Factor),
)
def identity(input_data, params):
    return input_data


@register_op(
    name="concat",
    layer_scope="universal",
    input_types={"default": (Panel, LaggedPanel, Factor)},
    output_type=Panel,
)
def concat(inputs, params):
    raise NotImplementedError("concat runtime execution is not wired in foundation schema")


@register_op(
    name="lag",
    layer_scope="universal",
    input_types={"default": (Panel, Series)},
    output_type=LaggedPanel,
    params_schema={
        "n_lag": {"type": int, "default": 4, "sweepable": True},
        "include_contemporaneous": {"type": bool, "default": False},
    },
    hard_rules=(
        Rule("hard", lambda dag, nref: dag.node(nref.node_id).params.get("n_lag", 4) >= 1, "n_lag must be >= 1"),
    ),
    soft_rules=(
        Rule(
            "soft",
            lambda dag, nref: "T" not in dag.layer_globals or dag.node(nref.node_id).params.get("n_lag", 4) <= dag.layer_globals["T"] / 10,
            "high lag/observation ratio, overfitting risk",
        ),
    ),
)
def lag(input_data, params):
    raise NotImplementedError("lag runtime execution is not wired in foundation schema")


@register_op(
    name="layer_meta_aggregate",
    layer_scope="universal",
    input_types={"default": DataType},
    output_type=MappingArtifact,
)
def layer_meta_aggregate(inputs, params):
    return {"inputs": inputs, **params}
