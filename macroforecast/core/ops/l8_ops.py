from __future__ import annotations

from .registry import register_op
from ..types import DataType, L8ArtifactsArtifact


def _passthrough(name: str):
    """L8 export ops collect upstream sinks and pass them on; the actual
    file writing happens in :func:`macroforecast.core.runtime.materialize_l8_runtime`."""

    def run(inputs, params):
        return {"op": name, "inputs": list(inputs) if isinstance(inputs, list) else [inputs], "params": dict(params)}

    run.__name__ = name
    return run


for _name in ("l8_collect_inputs", "l8_export_format", "l8_saved_objects", "l8_provenance", "l8_artifact_granularity"):
    register_op(name=_name, layer_scope=("l8",), input_types={"default": DataType}, output_type=L8ArtifactsArtifact)(_passthrough(_name))
