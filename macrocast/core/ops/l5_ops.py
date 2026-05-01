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
    raise NotImplementedError("Phase 1 runtime: L5 input collection in execution PR")


@register_op(name="metric_compute", layer_scope=("l5",), input_types={"default": L5EvaluationArtifact}, output_type=L5EvaluationArtifact)
def metric_compute(inputs, params):
    raise NotImplementedError("Phase 1 runtime: metric_compute in execution PR")


@register_op(name="benchmark_relative", layer_scope=("l5",), input_types={"default": L5EvaluationArtifact}, output_type=L5EvaluationArtifact)
def benchmark_relative(inputs, params):
    raise NotImplementedError("Phase 1 runtime: benchmark_relative in execution PR")


@register_op(name="aggregate", layer_scope=("l5",), input_types={"default": L5EvaluationArtifact}, output_type=L5EvaluationArtifact)
def aggregate(inputs, params):
    raise NotImplementedError("Phase 1 runtime: aggregate in execution PR")


@register_op(name="slice_and_decompose", layer_scope=("l5",), input_types={"default": L5EvaluationArtifact}, output_type=L5EvaluationArtifact)
def slice_and_decompose(inputs, params):
    raise NotImplementedError("Phase 1 runtime: slice_and_decompose in execution PR")


@register_op(name="rank_and_report", layer_scope=("l5",), input_types={"default": L5EvaluationArtifact}, output_type=L5EvaluationArtifact)
def rank_and_report(inputs, params):
    raise NotImplementedError("Phase 1 runtime: rank_and_report in execution PR")
