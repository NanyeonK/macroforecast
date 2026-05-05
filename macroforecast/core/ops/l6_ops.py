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
    """Collect L4/L5/L1 inputs for L6 statistical testing.

    The full statistical-test pipeline (DM/HLN, CW, MCS bootstrap, PT/HM,
    statsmodels-backed residual tests) lives in
    :func:`macroforecast.core.runtime.materialize_l6_runtime`. This op simply
    bundles the inputs so a generic DAG executor can materialize the sink.
    """

    return {"inputs": list(inputs) if isinstance(inputs, list) else [inputs], **params}


def _passthrough(name: str):
    def run(inputs, params):
        return inputs[0] if isinstance(inputs, list) and inputs else inputs

    run.__name__ = name
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
    register_op(name=_name, layer_scope=("l6",), input_types={"default": L6TestsArtifact}, output_type=L6TestsArtifact)(_passthrough(_name))
