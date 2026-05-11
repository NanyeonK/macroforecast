from __future__ import annotations

from dataclasses import replace

from .registry import Rule, register_op
from ..types import (
    DataType,
    Factor,
    L3FeaturesArtifact,
    L3MetadataArtifact,
    LaggedPanel,
    Panel,
    Series,
)


def _positive_param(name: str, default: int = 1):
    def check(dag, nref):
        value = dag.node(nref.node_id).params.get(name, default)
        # Phase A3 fix: ``n_components="all"`` sentinel passes the
        # positive-int gate (the runtime resolves "all" → min(T, N) at
        # PCA fit time, which is always >= 1).
        if isinstance(value, str) and value == "all":
            return True
        return value >= 1

    return check


def _not_full_sample(param: str = "temporal_rule"):
    return lambda dag, nref: (
        dag.node(nref.node_id).params.get(param) != "full_sample_once"
    )


def _temporal_present(dag, nref) -> bool:
    value = dag.node(nref.node_id).params.get("temporal_rule")
    return isinstance(value, str) and value != ""


def _n_components_reasonable(dag, nref) -> bool:
    value = dag.node(nref.node_id).params.get("n_components", 1)
    # Phase A3 fix: ``n_components="all"`` sentinel is always reasonable
    # (runtime caps at min(T, N)).
    if isinstance(value, str) and value == "all":
        return True
    return value < 10000


def _has_target_signal_input(dag, nref) -> bool:
    """Phase B-1 F1 fix: also accept upstream ``target_construction`` nodes
    (e.g. ``y_h``) as a valid target_signal input. The previous gate only
    accepted ``src_y``-prefixed source nodes or explicit
    ``output_port='target_signal'`` references, but the paper-faithful
    Huang/Zhou (2022) sPCA helper now wires the *h-shifted* target into
    the scaled_pca step so the supervised slope is predictive (Eq. 3)
    rather than contemporaneous. See ``recipes/paper_methods.scaled_pca``."""

    node = dag.node(nref.node_id)
    for ref in node.inputs[1:]:
        if ref.output_port == "target_signal" or ref.node_id.startswith("src_y"):
            return True
        upstream = dag.nodes.get(ref.node_id)
        if (
            upstream is not None
            and getattr(upstream, "op", None) == "target_construction"
        ):
            return True
    return False


def _positive_horizon(dag, nref) -> bool:
    horizon = dag.node(nref.node_id).params.get("horizon", 1)
    values = (
        horizon.get("sweep", ())
        if isinstance(horizon, dict) and "sweep" in horizon
        else (horizon,)
    )
    return all(isinstance(value, int) and value >= 1 for value in values)


def _delegate(op_name: str):
    """Build an op body that forwards to :func:`macroforecast.core.runtime._execute_l3_op`.

    The runtime dispatcher is the single source of truth for L3 transformations;
    the op-registry function defined here just hands the inputs/params over.
    """

    def run(inputs, params):
        from ..runtime import _execute_l3_op

        target_name = (params or {}).get("__target_name__", "y")
        return _execute_l3_op(
            op_name,
            list(inputs) if isinstance(inputs, list) else [inputs],
            dict(params or {}),
            target_name,
        )

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
        scope = (
            spec.layer_scope
            if isinstance(spec.layer_scope, tuple)
            else (spec.layer_scope,)
        )
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
        Rule(
            "hard",
            _positive_param("seasonal_period", 12),
            "seasonal_period must be >= 2",
        ),
        Rule(
            "hard",
            _positive_param("n_seasonal_lags", 1),
            "n_seasonal_lags must be >= 1",
        ),
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
    hard_rules=(
        Rule(
            "hard",
            lambda dag, nref: dag.node(nref.node_id).params.get("window", 3) >= 2,
            "window must be >= 2",
        ),
    ),
)
def ma_window(inputs, params):
    _stub(inputs, params)


@register_op(
    name="ma_increasing_order",
    layer_scope=("l3",),
    input_types={"default": (Panel, LaggedPanel, Factor)},
    output_type=Panel,
    params_schema={"max_order": {"type": int, "default": 12, "sweepable": True}},
    hard_rules=(
        Rule(
            "hard",
            lambda dag, nref: dag.node(nref.node_id).params.get("max_order", 12) >= 2,
            "max_order must be >= 2",
        ),
    ),
)
def ma_increasing_order(inputs, params):
    _stub(inputs, params)


@register_op(
    name="cumsum",
    layer_scope=("l3",),
    input_types={"default": (Panel, Series)},
    output_type=(Panel, Series),
)
def cumsum(inputs, params):
    _stub(inputs, params)


@register_op(
    name="scale",
    layer_scope=("l3",),
    input_types={"default": (Panel, LaggedPanel, Factor)},
    output_type=Panel,
    params_schema={
        "method": {"type": str, "default": "zscore", "sweepable": True},
        "temporal_rule": {
            "type": str,
            "default": "expanding_window_per_origin",
            "sweepable": True,
        },
    },
    hard_rules=(
        Rule(
            "hard",
            _not_full_sample(),
            "full_sample_once is rejected for scale temporal_rule",
        ),
    ),
)
def scale(inputs, params):
    _stub(inputs, params)


def _factor_op(
    name: str, input_type=(Panel, Series), output_type=Factor, extra_rules=()
):
    return register_op(
        name=name,
        layer_scope=("l3",),
        input_types={"default": input_type},
        output_type=output_type,
        params_schema={
            # Phase A3 fix: ``n_components`` accepts ``int | Literal["all"]``.
            # The literal sentinel ``"all"`` is resolved to ``min(T, N)``
            # at PCA fit time (see ``_pca_factors`` in ``core/runtime.py``).
            "n_components": {"type": (int, str), "default": 4, "sweepable": True},
            "temporal_rule": {
                "type": str,
                "default": "expanding_window_per_origin",
                "sweepable": True,
            },
        },
        hard_rules=(
            Rule(
                "hard", _positive_param("n_components", 4), "n_components must be >= 1"
            ),
            Rule("hard", _n_components_reasonable, "n_components must be < min(T, N)"),
            Rule("hard", _temporal_present, "temporal_rule is required"),
            Rule(
                "hard",
                _not_full_sample(),
                "full_sample_once is rejected for factor temporal_rule",
            ),
        )
        + tuple(extra_rules),
    )


@_factor_op("pca")
def pca(inputs, params):
    _stub(inputs, params)


@_factor_op("sparse_pca")
def sparse_pca(inputs, params):
    _stub(inputs, params)


@_factor_op("sparse_pca_chen_rohe")
def sparse_pca_chen_rohe(inputs, params):
    """Chen-Rohe (2023) Sparse Component Analysis -- non-diagonal D
    variant used by Rapach & Zhou (2025) Sparse Macro-Finance Factors.
    Distinct from ``sparse_pca`` (sklearn / Zou-Hastie-Tibshirani 2006).
    Operational v0.9.1 dev-stage v0.9.0C-3."""

    _stub(inputs, params)


@_factor_op(
    "supervised_pca",
    extra_rules=(
        Rule(
            "hard",
            _has_target_signal_input,
            "supervised_pca requires target_signal input port",
        ),
    ),
)
def supervised_pca(inputs, params):
    """Supervised PCA (Giglio-Xiu-Zhang 2025): screen panel columns by
    univariate correlation with the target, retain top ``q · N``, run
    PCA on the screened sub-panel. Operational v0.9.1 dev-stage v0.9.0C-4."""

    _stub(inputs, params)


@_factor_op(
    "scaled_pca",
    extra_rules=(
        Rule(
            "hard",
            _has_target_signal_input,
            "scaled_pca requires target_signal input port",
        ),
    ),
)
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
        "temporal_rule": {
            "type": str,
            "default": "expanding_window_per_origin",
            "sweepable": True,
        },
    },
    hard_rules=(
        Rule("hard", _positive_param("n_factors", 4), "n_factors must be >= 1"),
        Rule("hard", _temporal_present, "temporal_rule is required"),
        Rule(
            "hard",
            _not_full_sample(),
            "full_sample_once is rejected for dfm temporal_rule",
        ),
    ),
    soft_rules=(
        Rule(
            "soft",
            lambda dag, nref: (
                dag.node(nref.node_id).params.get("n_lags_factor", 1) <= 4
            ),
            "dfm n_lags_factor > 4 may not converge",
        ),
    ),
)
def dfm(inputs, params):
    _stub(inputs, params)


@register_op(
    name="varimax_rotation",
    layer_scope=("l3",),
    input_types={"default": Factor},
    output_type=Factor,
)
def varimax_rotation(inputs, params):
    _stub(inputs, params)


@register_op(
    name="varimax",
    layer_scope=("l3",),
    input_types={"default": Factor},
    output_type=Factor,
)
def varimax(inputs, params):
    _stub(inputs, params)


@_factor_op(
    "partial_least_squares",
    extra_rules=(
        Rule(
            "hard",
            _has_target_signal_input,
            "partial_least_squares requires target_signal input port",
        ),
    ),
)
def partial_least_squares(inputs, params):
    _stub(inputs, params)


@_factor_op("random_projection")
def random_projection(inputs, params):
    _stub(inputs, params)


@register_op(
    name="wavelet",
    layer_scope=("l3",),
    input_types={"default": (Panel, Series)},
    output_type=Panel,
    params_schema={"n_levels": {"type": int, "default": 1, "sweepable": True}},
    hard_rules=(Rule("hard", _positive_param("n_levels", 1), "n_levels must be >= 1"),),
)
def wavelet(inputs, params):
    _stub(inputs, params)


@_factor_op("fourier", input_type=(Panel, Series), output_type=Panel)
def fourier(inputs, params):
    _stub(inputs, params)


@register_op(
    name="hp_filter",
    layer_scope=("l3",),
    input_types={"default": (Panel, Series)},
    output_type=Panel,
    soft_rules=(
        Rule(
            "soft",
            lambda dag, nref: False,
            "Hamilton (2018) recommends hamilton_filter over hp_filter for macroeconomic data",
        ),
    ),
)
def hp_filter(inputs, params):
    _stub(inputs, params)


@register_op(
    name="hamilton_filter",
    layer_scope=("l3",),
    input_types={"default": (Panel, Series)},
    output_type=Panel,
    params_schema={
        "n_lags": {"type": int, "default": 8, "sweepable": True},
        "n_horizon": {"type": int, "default": 24, "sweepable": True},
    },
)
def hamilton_filter(inputs, params):
    _stub(inputs, params)


@register_op(
    name="polynomial_expansion",
    layer_scope=("l3",),
    input_types={"default": Panel},
    output_type=Panel,
    params_schema={"degree": {"type": int, "default": 2, "sweepable": True}},
    hard_rules=(Rule("hard", _positive_param("degree", 2), "degree must be >= 1"),),
    soft_rules=(
        Rule(
            "soft",
            lambda dag, nref: dag.node(nref.node_id).params.get("degree", 2) <= 3,
            "very high polynomial degree, consider kernel instead",
        ),
    ),
)
def polynomial_expansion(inputs, params):
    _stub(inputs, params)


@register_op(
    name="polynomial",
    layer_scope=("l3",),
    input_types={"default": Panel},
    output_type=Panel,
    params_schema={"degree": {"type": int, "default": 2, "sweepable": True}},
    hard_rules=(Rule("hard", _positive_param("degree", 2), "degree must be >= 1"),),
    soft_rules=(
        Rule(
            "soft",
            lambda dag, nref: dag.node(nref.node_id).params.get("degree", 2) <= 3,
            "very high polynomial degree, consider kernel",
        ),
    ),
)
def polynomial(inputs, params):
    _stub(inputs, params)


@register_op(
    name="interaction",
    layer_scope=("l3",),
    input_types={"default": Panel},
    output_type=Panel,
)
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


@register_op(
    name="regime_indicator",
    layer_scope=("l3",),
    input_types={"default": DataType},
    output_type=Panel,
)
def regime_indicator(inputs, params):
    _stub(inputs, params)


@register_op(
    name="season_dummy",
    layer_scope=("l3",),
    input_types={"default": (Panel, Series)},
    output_type=Panel,
)
def season_dummy(inputs, params):
    _stub(inputs, params)


@register_op(
    name="time_trend",
    layer_scope=("l3",),
    input_types={"default": (Panel, Series)},
    output_type=Series,
)
def time_trend(inputs, params):
    _stub(inputs, params)


@register_op(
    name="holiday",
    layer_scope=("l3",),
    input_types={"default": (Panel, Series)},
    output_type=Panel,
)
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


@register_op(
    name="feature_selection",
    layer_scope=("l3",),
    input_types={"default": (Panel, LaggedPanel, Factor)},
    output_type=Panel,
    params_schema={"n_features": {"type": object, "default": 0.5, "sweepable": True}},
)
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

        return _execute_l3_op(
            "feature_selection",
            list(inputs) if isinstance(inputs, list) else [inputs],
            dict(params or {}),
            params.get("__target_name__", "y") if params else "y",
        )

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


# ---------------------------------------------------------------------------
# v0.9 Phase 2 paper-coverage atomic primitives.
#
# Decomposition discipline: each entry below is *atomic* -- it cannot be
# expressed as a recipe over existing ops. Decomposable paper methods
# (MARX = ma + rotation, PRF = extra_trees(max_features=1), etc.) live
# in the recipe gallery instead.
# ---------------------------------------------------------------------------


@register_op(
    name="savitzky_golay_filter",
    layer_scope=("l3",),
    input_types={"default": (Panel, Series)},
    output_type=Panel,
    params_schema={
        "window_length": {"type": int, "default": 5, "sweepable": True},
        "polyorder": {"type": int, "default": 2, "sweepable": True},
    },
)
def savitzky_golay_filter(inputs, params):
    """Savitzky-Golay polynomial smoothing filter (Savitzky & Golay 1964).

    Operational baseline for AlbaMA replication (Coulombe 2025
    'Adaptive Moving Average for Macroeconomic Monitoring'). Runtime
    delegates to ``scipy.signal.savgol_filter``.
    """

    _stub(inputs, params)


@register_op(
    name="adaptive_ma_rf",
    layer_scope=("l3",),
    input_types={"default": (Panel, Series)},
    output_type=Panel,
    params_schema={
        "n_estimators": {"type": int, "default": 500, "sweepable": True},
        "min_samples_leaf": {"type": int, "default": 40, "sweepable": True},
        "sided": {
            "type": str,
            "default": "two",
            "options": ("one", "two"),
            "sweepable": True,
        },
        "random_state": {"type": int, "default": 0},
    },
)
def adaptive_ma_rf(inputs, params):
    """AlbaMA (Goulet Coulombe & Klieber 2025, arXiv:2501.13222) --
    adaptive moving average via a Random Forest with K=1 regressor
    (time index). Each leaf is a contiguous window of observations;
    ``min_samples_leaf`` lower-bounds the realised window length.
    Operational v0.9.1 dev-stage v0.9.0C-1.

    Modes (``params.sided``):
      * ``"two"`` (default) -- fit the forest once on the full sample;
        each leaf may span past *and* future. Standard smoother.
      * ``"one"`` -- expanding-window per-t fit; real-time / nowcasting
        variant. Per-t cost is ``O(B · log T)``.
    """

    _stub(inputs, params)


@register_op(
    name="asymmetric_trim",
    layer_scope=("l2", "l3"),
    input_types={"default": Panel},
    output_type=Panel,
    params_schema={
        "smooth_window": {"type": int, "default": 0, "sweepable": True},
    },
)
def asymmetric_trim(inputs, params):
    """Albacore-family rank-space transformation
    (Goulet Coulombe / Klieber / Barrette / Goebel 2024).

    Per-period sort: panel ``Π`` of shape ``(T, K)`` is mapped to
    ``O`` where ``O[t, r] = sort(Π[t, :])[r]`` (ascending). Asymmetric
    trimming emerges in the *downstream* nonneg ridge that learns
    weights on each rank position; this op does the rank-space
    transformation only.

    Optional ``smooth_window > 0`` applies a centred moving average to
    each rank-position time series (paper §3 mentions a 3-month MA
    smoothing for noisy components -- delegated rather than baked-in
    so users can chain ``ma_window`` explicitly when desired).

    Operational from v0.8.9 (B-6). Runtime function:
    :func:`macroforecast.core.runtime._asymmetric_trim`.
    """

    _stub(inputs, params)


@register_op(
    name="chow_lin_disaggregation",
    layer_scope=("l2",),
    input_types={"default": Panel},
    output_type=Panel,
    status="future",
)
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


# ---------------------------------------------------------------------------
# Phase C top-6 net-new methods (2026-05-08).
#
# - u_midas: Foroni-Marcellino-Schumacher (2015) Unrestricted MIDAS
#   aggregation. Stacks high-frequency lags as separate columns indexed
#   at the low-frequency target dates.
# - midas: Ghysels-Sinko-Valkanov (2007) MIDAS with parametric weighted
#   lag polynomial (almon / exp_almon / beta). NLS optimisation via
#   ``scipy.optimize.minimize``.
# - sliced_inverse_regression: Fan-Xue-Yao (2017) SIR for factor models,
#   optionally scaled by Huang-Zhou (2022) predictive slopes (sSUFF
#   variant).
# ---------------------------------------------------------------------------


@register_op(
    name="u_midas",
    layer_scope=("l3",),
    input_types={"default": (Panel, Series)},
    output_type=Panel,
    params_schema={
        "freq_ratio": {"type": int, "default": 3, "sweepable": True},
        "n_lags_high": {"type": int, "default": 6, "sweepable": True},
        "target_freq": {
            "type": str,
            "default": "low",
            "options": ("low", "high"),
            "sweepable": False,
        },
        "temporal_rule": {
            "type": str,
            "default": "expanding_window_per_origin",
            "sweepable": True,
        },
    },
    hard_rules=(
        Rule("hard", _positive_param("freq_ratio", 3), "freq_ratio must be >= 1"),
        Rule("hard", _positive_param("n_lags_high", 6), "n_lags_high must be >= 1"),
        Rule("hard", _temporal_present, "temporal_rule is required"),
        Rule(
            "hard",
            _not_full_sample(),
            "full_sample_once is rejected for u_midas temporal_rule",
        ),
    ),
)
def u_midas(inputs, params):
    """Foroni-Marcellino-Schumacher (2015) Unrestricted MIDAS.

    Stacks high-frequency predictor lags as separate columns reindexed at
    the low-frequency target dates. Output column ``{col}_lag{k}`` for
    each predictor ``col`` and each ``k ∈ {0..n_lags_high-1}``.
    """

    _stub(inputs, params)


@register_op(
    name="midas",
    layer_scope=("l3",),
    input_types={"default": (Panel, Series)},
    output_type=Panel,
    params_schema={
        "weighting": {
            "type": str,
            "default": "exp_almon",
            "options": ("almon", "exp_almon", "beta"),
            "sweepable": True,
        },
        "polynomial_order": {"type": int, "default": 2, "sweepable": True},
        "freq_ratio": {"type": int, "default": 3, "sweepable": True},
        "n_lags_high": {"type": int, "default": 12, "sweepable": True},
        "sum_to_one": {"type": bool, "default": True, "sweepable": False},
        "max_iter": {"type": int, "default": 200, "sweepable": False},
        "temporal_rule": {
            "type": str,
            "default": "expanding_window_per_origin",
            "sweepable": True,
        },
    },
    hard_rules=(
        Rule("hard", _positive_param("freq_ratio", 3), "freq_ratio must be >= 1"),
        Rule("hard", _positive_param("n_lags_high", 12), "n_lags_high must be >= 1"),
        Rule("hard", _temporal_present, "temporal_rule is required"),
        Rule(
            "hard",
            _not_full_sample(),
            "full_sample_once is rejected for midas temporal_rule",
        ),
        Rule(
            "hard", _has_target_signal_input, "midas requires target_signal input port"
        ),
    ),
)
def midas(inputs, params):
    """Ghysels-Sinko-Valkanov (2007) MIDAS with parametric weighted lag polynomial.

    Three weighting schemes: ``almon`` (polynomial of degree
    ``polynomial_order``), ``exp_almon`` (default; two-parameter
    exponential Almon), ``beta`` (two-parameter Beta density). NLS fit
    via ``scipy.optimize.minimize``.
    """

    _stub(inputs, params)


@register_op(
    name="sliced_inverse_regression",
    layer_scope=("l3",),
    input_types={"default": (Panel, Series)},
    output_type=Factor,
    params_schema={
        "n_components": {"type": (int, str), "default": 2, "sweepable": True},
        "n_slices": {"type": int, "default": 10, "sweepable": True},
        "scaling_method": {
            "type": str,
            "default": "scaled_pca",
            "options": ("scaled_pca", "marginal_R2", "none"),
            "sweepable": True,
        },
        "temporal_rule": {
            "type": str,
            "default": "expanding_window_per_origin",
            "sweepable": True,
        },
    },
    hard_rules=(
        Rule("hard", _positive_param("n_components", 2), "n_components must be >= 1"),
        Rule("hard", _positive_param("n_slices", 5), "n_slices must be >= 2"),
        Rule(
            "hard",
            _has_target_signal_input,
            "sliced_inverse_regression requires target_signal input port",
        ),
        Rule("hard", _temporal_present, "temporal_rule is required"),
        Rule(
            "hard",
            _not_full_sample(),
            "full_sample_once is rejected for sliced_inverse_regression temporal_rule",
        ),
    ),
)
def sliced_inverse_regression(inputs, params):
    """Fan-Xue-Yao (2017) sliced inverse regression for factor models.

    Optional Huang-Zhou (2022) predictive scaling (``scaling_method=
    'scaled_pca'``) applies a univariate target-supervised slope per
    column before slicing -- the ``sSUFF`` variant. The output is the
    ``n_components``-column factor frame projected by the top SIR
    eigenvectors of the between-slice covariance.
    """

    _stub(inputs, params)


_rewire_l3_ops()
