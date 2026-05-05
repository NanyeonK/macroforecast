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
