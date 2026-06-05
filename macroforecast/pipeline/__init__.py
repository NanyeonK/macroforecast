"""Comprehensive pseudo-out-of-sample forecasting pipeline."""
from __future__ import annotations

from .run import run_arms
from .spec import (
    TCODE_TARGET_MAP,
    Arm,
    CombinationContender,
    EvalSpec,
    InterpretSpec,
    PipelineSpec,
    ResolvedTarget,
    TargetSpec,
    contender_names,
    pipeline_spec,
    resolve_target,
)

__all__ = [
    "TCODE_TARGET_MAP",
    "run_arms",
    "Arm",
    "CombinationContender",
    "EvalSpec",
    "InterpretSpec",
    "PipelineSpec",
    "ResolvedTarget",
    "TargetSpec",
    "contender_names",
    "pipeline_spec",
    "resolve_target",
]
