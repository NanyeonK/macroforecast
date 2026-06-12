"""Comprehensive pseudo-out-of-sample forecasting pipeline."""
from __future__ import annotations

from .interpret import interpret_pipeline
from .evaluate import apply_combinations, evaluate
from .run import run_arms, run_pipeline
from .spec import (
    TCODE_TARGET_MAP,
    Arm,
    CombinationContender,
    EvalSpec,
    InterpretSpec,
    PipelineReport,
    PipelineSpec,
    ResolvedTarget,
    TargetSpec,
    contender_names,
    model_arms,
    pipeline_spec,
    resolve_target,
)

__all__ = [
    "TCODE_TARGET_MAP",
    "run_arms",
    "run_pipeline",
    "interpret_pipeline",
    "PipelineReport",
    "evaluate",
    "apply_combinations",
    "Arm",
    "CombinationContender",
    "EvalSpec",
    "InterpretSpec",
    "PipelineSpec",
    "ResolvedTarget",
    "TargetSpec",
    "contender_names",
    "model_arms",
    "pipeline_spec",
    "resolve_target",
]
