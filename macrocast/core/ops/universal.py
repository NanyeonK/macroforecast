from __future__ import annotations

from .registry import Rule, register_op
from ..types import DataType, Factor, ForecastArtifact, L4ForecastsArtifact, LaggedPanel, MappingArtifact, Panel, Series


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
    params_schema={"axis": {"type": str, "default": "column", "sweepable": False}},
    hard_rules=(
        Rule("hard", lambda dag, nref: len(dag.node(nref.node_id).inputs) >= 2, "concat requires at least 2 inputs"),
    ),
)
def concat(inputs, params):
    raise NotImplementedError("concat runtime execution is not wired in foundation schema")


@register_op(
    name="lag",
    layer_scope="universal",
    input_types={"default": (Panel, Series, Factor)},
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
    name="level",
    layer_scope=("l2", "l3"),
    input_types={"default": (Panel, Series)},
    output_type=(Panel, Series),
)
def level(input_data, params):
    raise NotImplementedError("level runtime execution is not wired in schema layer")


@register_op(
    name="log",
    layer_scope=("l2", "l3"),
    input_types={"default": (Panel, Series, LaggedPanel, Factor)},
    output_type=(Panel, Series, LaggedPanel, Factor),
)
def log(input_data, params):
    raise NotImplementedError("log runtime execution is not wired in schema layer")


@register_op(
    name="diff",
    layer_scope=("l2", "l3"),
    input_types={"default": (Panel, Series)},
    output_type=(Panel, Series),
    params_schema={"n_diff": {"type": int, "default": 1, "sweepable": True}},
    hard_rules=(Rule("hard", lambda dag, nref: dag.node(nref.node_id).params.get("n_diff", 1) >= 1, "n_diff must be >= 1"),),
)
def diff(input_data, params):
    raise NotImplementedError("diff runtime execution is not wired in schema layer")


@register_op(
    name="log_diff",
    layer_scope=("l2", "l3"),
    input_types={"default": (Panel, Series)},
    output_type=(Panel, Series),
    params_schema={"n_diff": {"type": int, "default": 1, "sweepable": True}},
    hard_rules=(Rule("hard", lambda dag, nref: dag.node(nref.node_id).params.get("n_diff", 1) >= 1, "n_diff must be >= 1"),),
)
def log_diff(input_data, params):
    raise NotImplementedError("log_diff runtime execution is not wired in schema layer")


@register_op(
    name="pct_change",
    layer_scope=("l2", "l3"),
    input_types={"default": (Panel, Series)},
    output_type=(Panel, Series),
    params_schema={"n_periods": {"type": int, "default": 1, "sweepable": True}},
    hard_rules=(Rule("hard", lambda dag, nref: dag.node(nref.node_id).params.get("n_periods", 1) >= 1, "n_periods must be >= 1"),),
)
def pct_change(input_data, params):
    raise NotImplementedError("pct_change runtime execution is not wired in schema layer")


@register_op(
    name="interact",
    layer_scope="universal",
    input_types={"default": (Panel, LaggedPanel, Factor)},
    output_type=Panel,
    hard_rules=(Rule("hard", lambda dag, nref: len(dag.node(nref.node_id).inputs) == 2, "interact requires exactly 2 inputs"),),
)
def interact(inputs, params):
    raise NotImplementedError("interact runtime execution is not wired in schema layer")


@register_op(
    name="hierarchical_pca",
    layer_scope="universal",
    input_types={"default": (Panel, LaggedPanel, Factor)},
    output_type=Factor,
    params_schema={
        "n_components_per_block": {"type": int, "default": 4, "sweepable": True},
        "n_components_top": {"type": int, "default": 4, "sweepable": True},
    },
)
def hierarchical_pca(inputs, params):
    raise NotImplementedError("hierarchical_pca runtime execution is not wired in schema layer")


@register_op(
    name="weighted_concat",
    layer_scope="universal",
    input_types={"default": (Panel, LaggedPanel, Factor)},
    output_type=Panel,
    params_schema={"weights": {"type": list, "default": [], "sweepable": False}},
    hard_rules=(
        Rule(
            "hard",
            lambda dag, nref: not dag.node(nref.node_id).params.get("weights")
            or len(dag.node(nref.node_id).params.get("weights", ())) == len(dag.node(nref.node_id).inputs),
            "weighted_concat weight count must match input count",
        ),
    ),
)
def weighted_concat(inputs, params):
    raise NotImplementedError("weighted_concat runtime execution is not wired in schema layer")


@register_op(
    name="simple_average",
    layer_scope="universal",
    input_types={"default": (Panel, LaggedPanel, Factor)},
    output_type=Panel,
    hard_rules=(Rule("hard", lambda dag, nref: len(dag.node(nref.node_id).inputs) >= 2, "simple_average requires at least 2 inputs"),),
)
def simple_average(inputs, params):
    raise NotImplementedError("simple_average runtime execution is not wired in schema layer")


@register_op(
    name="weighted_average_forecast",
    layer_scope=("l4",),
    input_types={"default": (ForecastArtifact, L4ForecastsArtifact)},
    output_type=L4ForecastsArtifact,
    params_schema={
        "weights_method": {
            "type": str,
            "default": "dmsfe",
            "sweepable": True,
            "options": ["equal", "dmsfe", "inverse_msfe", "mallows_cp", "sic_weights", "granger_ramanathan", "cv_optimized"],
        },
        "temporal_rule": {"type": str, "default": "expanding_window_per_origin", "sweepable": True},
        "dmsfe_theta": {"type": float, "default": 0.95, "sweepable": True},
        "cv_optimized_window": {"type": int, "default": 60, "sweepable": True},
        "granger_ramanathan_constraint": {"type": str, "default": "sum_to_one", "sweepable": True},
    },
    hard_rules=(
        Rule("hard", lambda dag, nref: len(dag.node(nref.node_id).inputs) >= 2, "weighted_average_forecast requires at least 2 inputs"),
        Rule(
            "hard",
            lambda dag, nref: dag.node(nref.node_id).params.get("temporal_rule", "expanding_window_per_origin") != "full_sample_once",
            "full_sample_once is rejected for forecast combination temporal_rule",
        ),
    ),
)
def weighted_average_forecast(inputs, params):
    raise NotImplementedError("Phase 1 runtime: weighted_average_forecast implementation in execution PR")


@register_op(
    name="median_forecast",
    layer_scope=("l4",),
    input_types={"default": (ForecastArtifact, L4ForecastsArtifact)},
    output_type=L4ForecastsArtifact,
    hard_rules=(Rule("hard", lambda dag, nref: len(dag.node(nref.node_id).inputs) >= 2, "median_forecast requires at least 2 inputs"),),
)
def median_forecast(inputs, params):
    raise NotImplementedError("Phase 1 runtime: median_forecast implementation in execution PR")


@register_op(
    name="trimmed_mean_forecast",
    layer_scope=("l4",),
    input_types={"default": (ForecastArtifact, L4ForecastsArtifact)},
    output_type=L4ForecastsArtifact,
    params_schema={"trim_pct": {"type": float, "default": 0.1, "sweepable": True}},
    hard_rules=(Rule("hard", lambda dag, nref: len(dag.node(nref.node_id).inputs) >= 2, "trimmed_mean_forecast requires at least 2 inputs"),),
)
def trimmed_mean_forecast(inputs, params):
    raise NotImplementedError("Phase 1 runtime: trimmed_mean_forecast implementation in execution PR")


@register_op(
    name="bma_forecast",
    layer_scope=("l4",),
    input_types={"default": (ForecastArtifact, L4ForecastsArtifact)},
    output_type=L4ForecastsArtifact,
    params_schema={"prior_method": {"type": str, "default": "uniform", "sweepable": True}},
    hard_rules=(Rule("hard", lambda dag, nref: len(dag.node(nref.node_id).inputs) >= 2, "bma_forecast requires at least 2 inputs"),),
)
def bma_forecast(inputs, params):
    raise NotImplementedError("Phase 1 runtime: bma_forecast implementation in execution PR")


@register_op(
    name="bivariate_ardl_combination",
    layer_scope=("l4",),
    input_types={"default": (ForecastArtifact, L4ForecastsArtifact)},
    output_type=L4ForecastsArtifact,
    params_schema={"combination_weights_temporal_rule": {"type": str, "default": "expanding_window_per_origin", "sweepable": True}},
    hard_rules=(Rule("hard", lambda dag, nref: len(dag.node(nref.node_id).inputs) == 2, "bivariate_ardl_combination requires exactly 2 inputs"),),
)
def bivariate_ardl_combination(inputs, params):
    raise NotImplementedError("Phase 1 runtime: bivariate_ardl_combination implementation in execution PR")


@register_op(
    name="layer_meta_aggregate",
    layer_scope="universal",
    input_types={"default": DataType},
    output_type=MappingArtifact,
)
def layer_meta_aggregate(inputs, params):
    return {"inputs": inputs, **params}
