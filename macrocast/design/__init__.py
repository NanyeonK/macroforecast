"""Stage 0 framework — the pre-execution grammar of a macrocast study.

Stage 0 fixes *how* a study is described before any later content layer
(dataset adapters, preprocessing axes, model families, evaluation axes)
is expanded. Its public surface is a small set of dataclasses plus one
canonical builder; see docs/user_guide/stage0.md for the full
contract.

Distinct from (but related to) the ``macrocast.registry.stage0`` package,
which holds the *registry layer* for the 7 Layer 0 meta axes
(axis_type, compute_mode, experiment_unit, failure_policy,
reproducibility_mode, study_mode). Framework dataclasses
defined here consume values from those meta axes.
"""

from .build import (
    build_design_frame,
    check_design_completeness,
    resolve_route_owner,
    design_summary,
)
from .errors import (
    DesignCompletenessError,
    DesignError,
    DesignNormalizationError,
    DesignRoutingError,
    DesignValidationError,
)
from .serialize import design_from_dict, design_to_dict
from .types import (
    ComparisonContract,
    FixedDesign,
    ReplicationInput,
    DesignFrame,
    VaryingDesign,
)

__all__ = [
    "build_design_frame",
    "check_design_completeness",
    "resolve_route_owner",
    "design_summary",
    "design_to_dict",
    "design_from_dict",
    "DesignError",
    "DesignNormalizationError",
    "DesignValidationError",
    "DesignCompletenessError",
    "DesignRoutingError",
    "FixedDesign",
    "VaryingDesign",
    "ComparisonContract",
    "ReplicationInput",
    "DesignFrame",
]
