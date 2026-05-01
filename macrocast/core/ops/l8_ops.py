from __future__ import annotations

from .registry import register_op
from ..types import L8ArtifactsArtifact


def _stub(name: str):
    def run(inputs, params):
        raise NotImplementedError(f"Phase 1 runtime: {name} implementation in execution PR")

    return run


for _name in ("l8_collect_inputs", "l8_export_format", "l8_saved_objects", "l8_provenance", "l8_artifact_granularity"):
    register_op(name=_name, layer_scope=("l8",), input_types={"default": object}, output_type=L8ArtifactsArtifact)(_stub(_name))
