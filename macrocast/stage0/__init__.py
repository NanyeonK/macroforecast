"""Stage 0 framework — the pre-execution grammar of a macrocast study.

Stage 0 fixes *how* a study is described before any later content layer
(dataset adapters, preprocessing axes, model families, evaluation axes)
is expanded. Its public surface is a small set of dataclasses plus one
canonical builder; see docs/user_guide/stage0.md for the full
contract.

Distinct from (but related to) the ``macrocast.registry.stage0`` package,
which holds the *registry layer* for the 7 Layer 0 meta axes
(axis_type, compute_mode, experiment_unit, failure_policy,
registry_type, reproducibility_mode, study_mode). Framework dataclasses
defined here consume values from those meta axes.
"""

from .build import (
    build_stage0_frame,
    check_stage0_completeness,
    resolve_route_owner,
    stage0_summary,
)
from .errors import (
    Stage0CompletenessError,
    Stage0Error,
    Stage0NormalizationError,
    Stage0RoutingError,
    Stage0ValidationError,
)
from .serialize import stage0_from_dict, stage0_to_dict
from .types import (
    ComparisonContract,
    FixedDesign,
    ReplicationInput,
    Stage0Frame,
    VaryingDesign,
)

__all__ = [
    "build_stage0_frame",
    "check_stage0_completeness",
    "resolve_route_owner",
    "stage0_summary",
    "stage0_to_dict",
    "stage0_from_dict",
    "Stage0Error",
    "Stage0NormalizationError",
    "Stage0ValidationError",
    "Stage0CompletenessError",
    "Stage0RoutingError",
    "FixedDesign",
    "VaryingDesign",
    "ComparisonContract",
    "ReplicationInput",
    "Stage0Frame",
]
