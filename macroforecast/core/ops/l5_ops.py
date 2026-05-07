from __future__ import annotations

from .registry import register_op
from ..types import L1DataDefinitionArtifact, L1RegimeMetadataArtifact, L3MetadataArtifact, L4ForecastsArtifact, L4ModelArtifactsArtifact, L5EvaluationArtifact


@register_op(
    name="l5_collect_inputs",
    layer_scope=("l5",),
    input_types={"default": (L4ForecastsArtifact, L4ModelArtifactsArtifact, L1DataDefinitionArtifact, L1RegimeMetadataArtifact, L3MetadataArtifact)},
    output_type=L5EvaluationArtifact,
)
def l5_collect_inputs(inputs, params):
    """Pass through L4/L1/L3 artifacts as the L5 input bundle.

    The full evaluation pipeline lives in
    :func:`macroforecast.core.runtime.materialize_l5_minimal`, which composes
    metric_compute -> benchmark_relative -> aggregate -> slice_and_decompose ->
    rank_and_report on the concrete artifact payloads. This op simply records
    the inputs so that an explicit DAG-driven L5 still has a sink to pull from.
    """

    return {"inputs": list(inputs), **params}


@register_op(name="metric_compute", layer_scope=("l5",), input_types={"default": L5EvaluationArtifact}, output_type=L5EvaluationArtifact)
def metric_compute(inputs, params):
    return inputs[0] if isinstance(inputs, list) and inputs else inputs


@register_op(name="benchmark_relative", layer_scope=("l5",), input_types={"default": L5EvaluationArtifact}, output_type=L5EvaluationArtifact)
def benchmark_relative(inputs, params):
    return inputs[0] if isinstance(inputs, list) and inputs else inputs


@register_op(name="aggregate", layer_scope=("l5",), input_types={"default": L5EvaluationArtifact}, output_type=L5EvaluationArtifact)
def aggregate(inputs, params):
    return inputs[0] if isinstance(inputs, list) and inputs else inputs


@register_op(name="slice_and_decompose", layer_scope=("l5",), input_types={"default": L5EvaluationArtifact}, output_type=L5EvaluationArtifact)
def slice_and_decompose(inputs, params):
    return inputs[0] if isinstance(inputs, list) and inputs else inputs


@register_op(name="rank_and_report", layer_scope=("l5",), input_types={"default": L5EvaluationArtifact}, output_type=L5EvaluationArtifact)
def rank_and_report(inputs, params):
    return inputs[0] if isinstance(inputs, list) and inputs else inputs


# ---------------------------------------------------------------------------
# v0.9 Phase 2 paper-coverage atomic primitive.
# ---------------------------------------------------------------------------


@register_op(
    name="blocked_oob_reality_check",
    layer_scope=("l5",),
    input_types={"default": L5EvaluationArtifact},
    output_type=L5EvaluationArtifact,
    params_schema={
        "block_length": {"type": int, "default": 4, "sweepable": True},
        "n_bootstraps": {"type": int, "default": 1000, "sweepable": True},
        "alpha": {"type": float, "default": 0.05, "sweepable": True},
    },
)
def blocked_oob_reality_check(inputs, params):
    """Goulet Coulombe / Frenette / Klieber (2025 JAE) blocked OOB
    reality check for HNN volatility forecasts.

    Block-bootstrap variant of White (2000) reality check that respects
    serial dependence in macro residuals. Atomic L5 primitive: existing
    ``mcs_inclusion`` / ``reality_check`` ops use stationary block
    bootstrap on cross-sectional max-stat; this op runs a moving-block
    bootstrap on per-origin loss differentials vs a benchmark, which
    is the correct sizing in the small-sample serially-correlated
    macro setting where HNN is evaluated.

    **Operational from v0.8.9.** Runtime function:
    :func:`macroforecast.core.runtime._blocked_oob_reality_check_p_values`.
    Called directly by the v0.9.x Phase 1 promotion test path; users
    can also invoke the helper standalone on any per-origin (model x
    origin) loss panel.
    """

    return inputs[0] if isinstance(inputs, list) and inputs else inputs
