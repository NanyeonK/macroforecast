from __future__ import annotations

from .registry import register_op
from ..types import L1DataDefinitionArtifact, L1RegimeMetadataArtifact, L4ForecastsArtifact, L4ModelArtifactsArtifact, L5EvaluationArtifact, L6TestsArtifact


@register_op(
    name="l6_collect_inputs",
    layer_scope=("l6",),
    input_types={"default": (L4ForecastsArtifact, L4ModelArtifactsArtifact, L5EvaluationArtifact, L1DataDefinitionArtifact, L1RegimeMetadataArtifact)},
    output_type=L6TestsArtifact,
)
def l6_collect_inputs(inputs, params):
    raise NotImplementedError("Phase 1 runtime: L6 input collection in execution PR")


def _stub(name: str):
    def run(inputs, params):
        raise NotImplementedError(f"Phase 1 runtime: {name} in execution PR")

    return run


for _name in (
    "L6_A_equal_predictive",
    "L6_B_nested",
    "L6_C_cpa",
    "L6_D_multiple_model",
    "L6_E_density_interval",
    "L6_F_direction",
    "L6_G_residual",
    "multiple_model_test_step_m_romano_wolf",
):
    register_op(name=_name, layer_scope=("l6",), input_types={"default": L6TestsArtifact}, output_type=L6TestsArtifact)(_stub(_name))
