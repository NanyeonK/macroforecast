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
    return input_data[0] if isinstance(input_data, list) and input_data else input_data


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
    source = input_data[0] if isinstance(input_data, list) else input_data
    n_lag = int(params.get("n_lag", 4))
    include_now = bool(params.get("include_contemporaneous", False))
    if isinstance(source, Panel):
        base_columns = source.column_names
        lag_columns = [
            f"{column}_lag{lag}"
            for column in base_columns
            for lag in range(0 if include_now else 1, n_lag + 1)
        ]
        width = len(lag_columns)
        return LaggedPanel(
            shape=(source.shape[0], width) if source.shape else None,
            column_names=tuple(lag_columns),
            n_lag=n_lag,
        )
    if isinstance(source, Series):
        lag_columns = [
            f"{source.name}_lag{lag}"
            for lag in range(0 if include_now else 1, n_lag + 1)
        ]
        return LaggedPanel(
            shape=(source.shape[0], len(lag_columns)) if source.shape else None,
            column_names=tuple(lag_columns),
            n_lag=n_lag,
        )
    raise TypeError(f"lag expected Panel or Series, got {type(source).__name__}")


@register_op(
    name="layer_meta_aggregate",
    layer_scope="universal",
    input_types={"default": DataType},
    output_type=MappingArtifact,
)
def layer_meta_aggregate(inputs, params):
    return {"inputs": inputs, **params}
