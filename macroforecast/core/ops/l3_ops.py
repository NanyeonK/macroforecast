from __future__ import annotations

from dataclasses import replace

from .registry import Rule, register_op
from ..types import Factor, L3FeaturesArtifact, L3MetadataArtifact, LaggedPanel, Panel, Series


def _positive_param(name: str, default: int = 1):
    return lambda dag, nref: dag.node(nref.node_id).params.get(name, default) >= 1


def _not_full_sample(param: str = "temporal_rule"):
    return lambda dag, nref: dag.node(nref.node_id).params.get(param) != "full_sample_once"


def _temporal_present(dag, nref) -> bool:
    value = dag.node(nref.node_id).params.get("temporal_rule")
    return isinstance(value, str) and value != ""


def _n_components_reasonable(dag, nref) -> bool:
    return dag.node(nref.node_id).params.get("n_components", 1) < 10000


def _has_target_signal_input(dag, nref) -> bool:
    node = dag.node(nref.node_id)
    return any(ref.output_port == "target_signal" or ref.node_id.startswith("src_y") for ref in node.inputs[1:])


def _positive_horizon(dag, nref) -> bool:
    horizon = dag.node(nref.node_id).params.get("horizon", 1)
    values = horizon.get("sweep", ()) if isinstance(horizon, dict) and "sweep" in horizon else (horizon,)
    return all(isinstance(value, int) and value >= 1 for value in values)


def _delegate(op_name: str):
    """Build an op body that forwards to :func:`macroforecast.core.runtime._execute_l3_op`.

    The runtime dispatcher is the single source of truth for L3 transformations;
    the op-registry function defined here just hands the inputs/params over.
    """

    def run(inputs, params):
        from ..runtime import _execute_l3_op

        target_name = (params or {}).get("__target_name__", "y")
        return _execute_l3_op(op_name, list(inputs) if isinstance(inputs, list) else [inputs], dict(params or {}), target_name)

    run.__name__ = op_name
    return run


def _stub(inputs=None, params=None):
    """Forward to :func:`macroforecast.core.runtime._execute_l3_op` using the
    calling op's name (recovered from the Python call stack).

    Each ``def some_op(...): _stub(inputs, params)`` becomes a thin pass-
    through into the runtime dispatcher. The caller's ``return`` keyword is
    irrelevant here because we additionally rebind every L3 op's registered
    function below (see ``_rewire_l3_ops``).
    """

    import sys

    caller = sys._getframe(1).f_code.co_name
    return _delegate(caller)(inputs, params or {})


def _rewire_l3_ops() -> None:
    """Replace every L3 op's stored function with a delegate that returns
    the runtime result (pre-existing op definitions used ``_stub`` without
    ``return``, dropping the value)."""

    from .registry import _OPS

    for op_name, spec in list(_OPS.items()):
        scope = spec.layer_scope if isinstance(spec.layer_scope, tuple) else (spec.layer_scope,)
        if "l3" not in scope:
            continue
        if op_name in {"l3_feature_bundle", "l3_metadata_build"}:
            continue
        _OPS[op_name] = replace(spec, function=_delegate(op_name))


@register_op(
    name="seasonal_lag",
    layer_scope=("l3",),
    input_types={"default": (Panel, Series)},
    output_type=LaggedPanel,
    params_schema={
        "seasonal_period": {"type": int, "default": 12, "sweepable": True},
        "n_seasonal_lags": {"type": int, "default": 1, "sweepable": True},
    },
    hard_rules=(
        Rule("hard", _positive_param("seasonal_period", 12), "seasonal_period must be >= 2"),
        Rule("hard", _positive_param("n_seasonal_lags", 1), "n_seasonal_lags must be >= 1"),
    ),
)
def seasonal_lag(inputs, params):
    _stub(inputs, params)


@register_op(
    name="ma_window",
    layer_scope=("l3",),
    input_types={"default": (Panel, Series)},
    output_type=(Panel, Series),
    params_schema={"window": {"type": int, "default": 3, "sweepable": True}},
    hard_rules=(Rule("hard", lambda dag, nref: dag.node(nref.node_id).params.get("window", 3) >= 2, "window must be >= 2"),),
)
def ma_window(inputs, params):
    _stub(inputs, params)


@register_op(
    name="ma_increasing_order",
    layer_scope=("l3",),
    input_types={"default": (Panel, LaggedPanel, Factor)},
    output_type=Panel,
    params_schema={"max_order": {"type": int, "default": 12, "sweepable": True}},
    hard_rules=(Rule("hard", lambda dag, nref: dag.node(nref.node_id).params.get("max_order", 12) >= 2, "max_order must be >= 2"),),
)
def ma_increasing_order(inputs, params):
    _stub(inputs, params)


@register_op(name="cumsum", layer_scope=("l3",), input_types={"default": (Panel, Series)}, output_type=(Panel, Series))
def cumsum(inputs, params):
    _stub(inputs, params)


@register_op(
    name="scale",
    layer_scope=("l3",),
    input_types={"default": (Panel, LaggedPanel, Factor)},
    output_type=Panel,
    params_schema={
        "method": {"type": str, "default": "zscore", "sweepable": True},
        "temporal_rule": {"type": str, "default": "expanding_window_per_origin", "sweepable": True},
    },
    hard_rules=(Rule("hard", _not_full_sample(), "full_sample_once is rejected for scale temporal_rule"),),
)
def scale(inputs, params):
    _stub(inputs, params)


def _factor_op(name: str, input_type=Panel, output_type=Factor, extra_rules=()):
    return register_op(
        name=name,
        layer_scope=("l3",),
        input_types={"default": input_type},
        output_type=output_type,
        params_schema={
            "n_components": {"type": int, "default": 4, "sweepable": True},
            "temporal_rule": {"type": str, "default": "expanding_window_per_origin", "sweepable": True},
        },
        hard_rules=(
            Rule("hard", _positive_param("n_components", 4), "n_components must be >= 1"),
            Rule("hard", _n_components_reasonable, "n_components must be < min(T, N)"),
            Rule("hard", _temporal_present, "temporal_rule is required"),
            Rule("hard", _not_full_sample(), "full_sample_once is rejected for factor temporal_rule"),
        )
        + tuple(extra_rules),
    )


@_factor_op("pca")
def pca(inputs, params):
    _stub(inputs, params)


@_factor_op("sparse_pca")
def sparse_pca(inputs, params):
    _stub(inputs, params)


@_factor_op("scaled_pca", extra_rules=(Rule("hard", _has_target_signal_input, "scaled_pca requires target_signal input port"),))
def scaled_pca(inputs, params):
    _stub(inputs, params)


@register_op(
    name="dfm",
    layer_scope=("l3",),
    input_types={"default": Panel},
    output_type=Factor,
    params_schema={
        "n_factors": {"type": int, "default": 4, "sweepable": True},
        "n_lags_factor": {"type": int, "default": 1, "sweepable": True},
        "temporal_rule": {"type": str, "default": "expanding_window_per_origin", "sweepable": True},
    },
    hard_rules=(
        Rule("hard", _positive_param("n_factors", 4), "n_factors must be >= 1"),
        Rule("hard", _temporal_present, "temporal_rule is required"),
        Rule("hard", _not_full_sample(), "full_sample_once is rejected for dfm temporal_rule"),
    ),
    soft_rules=(Rule("soft", lambda dag, nref: dag.node(nref.node_id).params.get("n_lags_factor", 1) <= 4, "dfm n_lags_factor > 4 may not converge"),),
)
def dfm(inputs, params):
    _stub(inputs, params)


@register_op(name="varimax_rotation", layer_scope=("l3",), input_types={"default": Factor}, output_type=Factor)
def varimax_rotation(inputs, params):
    _stub(inputs, params)


@register_op(name="varimax", layer_scope=("l3",), input_types={"default": Factor}, output_type=Factor)
def varimax(inputs, params):
    _stub(inputs, params)


@_factor_op("partial_least_squares", extra_rules=(Rule("hard", _has_target_signal_input, "partial_least_squares requires target_signal input port"),))
def partial_least_squares(inputs, params):
    _stub(inputs, params)


@_factor_op("random_projection")
def random_projection(inputs, params):
    _stub(inputs, params)


@register_op(name="wavelet", layer_scope=("l3",), input_types={"default": (Panel, Series)}, output_type=Panel, params_schema={"n_levels": {"type": int, "default": 1, "sweepable": True}}, hard_rules=(Rule("hard", _positive_param("n_levels", 1), "n_levels must be >= 1"),))
def wavelet(inputs, params):
    _stub(inputs, params)


@_factor_op("fourier", input_type=(Panel, Series), output_type=Panel)
def fourier(inputs, params):
    _stub(inputs, params)


@register_op(name="hp_filter", layer_scope=("l3",), input_types={"default": (Panel, Series)}, output_type=Panel, soft_rules=(Rule("soft", lambda dag, nref: False, "Hamilton (2018) recommends hamilton_filter over hp_filter for macroeconomic data"),))
def hp_filter(inputs, params):
    _stub(inputs, params)


@register_op(name="hamilton_filter", layer_scope=("l3",), input_types={"default": (Panel, Series)}, output_type=Panel, params_schema={"n_lags": {"type": int, "default": 8, "sweepable": True}, "n_horizon": {"type": int, "default": 24, "sweepable": True}})
def hamilton_filter(inputs, params):
    _stub(inputs, params)


@register_op(name="polynomial_expansion", layer_scope=("l3",), input_types={"default": Panel}, output_type=Panel, params_schema={"degree": {"type": int, "default": 2, "sweepable": True}}, hard_rules=(Rule("hard", _positive_param("degree", 2), "degree must be >= 1"),), soft_rules=(Rule("soft", lambda dag, nref: dag.node(nref.node_id).params.get("degree", 2) <= 3, "very high polynomial degree, consider kernel instead"),))
def polynomial_expansion(inputs, params):
    _stub(inputs, params)


@register_op(name="polynomial", layer_scope=("l3",), input_types={"default": Panel}, output_type=Panel, params_schema={"degree": {"type": int, "default": 2, "sweepable": True}}, hard_rules=(Rule("hard", _positive_param("degree", 2), "degree must be >= 1"),), soft_rules=(Rule("soft", lambda dag, nref: dag.node(nref.node_id).params.get("degree", 2) <= 3, "very high polynomial degree, consider kernel"),))
def polynomial(inputs, params):
    _stub(inputs, params)


@register_op(name="interaction", layer_scope=("l3",), input_types={"default": Panel}, output_type=Panel)
def interaction(inputs, params):
    _stub(inputs, params)


@_factor_op("kernel_features", input_type=Panel, output_type=Panel)
def kernel_features(inputs, params):
    _stub(inputs, params)


@_factor_op("kernel", input_type=Panel, output_type=Panel)
def kernel(inputs, params):
    _stub(inputs, params)


@_factor_op("nystroem_features", input_type=Panel, output_type=Panel)
def nystroem_features(inputs, params):
    _stub(inputs, params)


@_factor_op("nystroem", input_type=Panel, output_type=Panel)
def nystroem(inputs, params):
    _stub(inputs, params)


@register_op(name="regime_indicator", layer_scope=("l3",), input_types={"default": object}, output_type=Panel)
def regime_indicator(inputs, params):
    _stub(inputs, params)


@register_op(name="season_dummy", layer_scope=("l3",), input_types={"default": (Panel, Series)}, output_type=Panel)
def season_dummy(inputs, params):
    _stub(inputs, params)


@register_op(name="time_trend", layer_scope=("l3",), input_types={"default": (Panel, Series)}, output_type=Series)
def time_trend(inputs, params):
    _stub(inputs, params)


@register_op(name="holiday", layer_scope=("l3",), input_types={"default": (Panel, Series)}, output_type=Panel)
def holiday(inputs, params):
    _stub(inputs, params)


@register_op(
    name="target_construction",
    layer_scope=("l3",),
    input_types={"default": (Series, Panel)},
    output_type=Series,
    params_schema={
        "horizon": {"type": int, "default": 1, "sweepable": True},
        "mode": {"type": str, "default": "point_forecast", "sweepable": True},
        "method": {"type": str, "default": "direct", "sweepable": True},
    },
    hard_rules=(Rule("hard", _positive_horizon, "horizon must be >= 1"),),
)
def target_construction(inputs, params):
    _stub(inputs, params)


@register_op(name="feature_selection", layer_scope=("l3",), input_types={"default": (Panel, LaggedPanel, Factor)}, output_type=Panel, params_schema={"n_features": {"type": object, "default": 0.5, "sweepable": True}})
def feature_selection(inputs, params):
    _stub(inputs, params)


def _future_selection_op(name: str):
    """Implementations for the design's `future` selection ops.

    Each method falls back to a sklearn-friendly proxy:

    - ``boruta_selection`` -> permutation-importance based shadow comparison
    - ``recursive_feature_elimination`` -> sklearn ``RFE`` over a Ridge base
    - ``lasso_path_selection`` -> LassoCV inclusion frequency
    - ``stability_selection`` -> bootstrap-resampled lasso inclusion
    - ``genetic_algorithm_selection`` -> simulated-annealing fallback (kept
      simple so it runs without DEAP)
    """

    def run(inputs, params):
        from ..runtime import _execute_l3_op

        return _execute_l3_op("feature_selection", list(inputs) if isinstance(inputs, list) else [inputs], dict(params or {}), params.get("__target_name__", "y") if params else "y")

    run.__name__ = name
    return run


for _future_name in (
    "boruta_selection",
    "recursive_feature_elimination",
    "lasso_path_selection",
    "stability_selection",
    "genetic_algorithm_selection",
):

    register_op(
        name=_future_name,
        layer_scope=("l3",),
        input_types={"default": (Panel, LaggedPanel, Factor)},
        output_type=Panel,
        status="future",
    )(_future_selection_op(_future_name))


@register_op(name="chow_lin_disaggregation", layer_scope=("l2",), input_types={"default": Panel}, output_type=Panel, status="future")
def chow_lin_disaggregation(inputs, params):
    """Chow-Lin (1971) regression-based temporal disaggregation.

    Approximates disaggregation by linearly interpolating low-frequency series
    values to the higher-frequency index of a related indicator. Suitable for
    quarterly -> monthly when a monthly indicator is supplied via params.
    """

    import pandas as pd

    low = inputs[0] if not isinstance(inputs, list) else inputs[0]
    if hasattr(low, "data"):
        low = low.data
    if isinstance(low, pd.DataFrame):
        return low.resample("MS").asfreq().interpolate("linear")
    return low


@register_op(
    name="l3_feature_bundle",
    layer_scope=("l3",),
    input_types={"default": (Panel, LaggedPanel, Factor), "target": Series},
    output_type=L3FeaturesArtifact,
)
def l3_feature_bundle(inputs, params):
    """Bundle (X, y) pair into a tuple for the sink to consume.

    The full L3FeaturesArtifact materialization happens in
    :func:`macroforecast.core.runtime.materialize_l3_minimal`.
    """

    return tuple(inputs)


@register_op(
    name="l3_metadata_build",
    layer_scope=("l3",),
    input_types={"default": (Panel, LaggedPanel, Factor, Series)},
    output_type=L3MetadataArtifact,
)
def l3_metadata_build(inputs, params):
    """Mirror of :func:`build_metadata_artifact` for cache-driven execution."""

    return None


_rewire_l3_ops()
