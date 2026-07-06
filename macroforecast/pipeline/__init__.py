"""Comprehensive pseudo-out-of-sample forecasting pipeline."""
from __future__ import annotations

from .interpret import interpret_pipeline
from .evaluate import apply_combinations, evaluate, evaluate_cross_policy
from .parallelism import auto_parallelism
from .rescore import rescore
from .run import run_arms, run_pipeline
from .spec import (
    TCODE_TARGET_MAP,
    Arm,
    CombinationContender,
    DIRECT_POLICY_GUARD_MODELS,
    EvalSpec,
    InterpretSpec,
    PipelineReport,
    PipelineSpec,
    ResolvedTarget,
    TargetSpec,
    contender_names,
    is_vintage_aware,
    model_arms,
    pipeline_spec,
    resolve_target,
)

__all__ = [
    "TCODE_TARGET_MAP",
    "auto_parallelism",
    "rescore",
    "run_arms",
    "run_pipeline",
    "interpret_pipeline",
    "PipelineReport",
    "evaluate",
    "evaluate_cross_policy",
    "apply_combinations",
    "Arm",
    "CombinationContender",
    "DIRECT_POLICY_GUARD_MODELS",
    "EvalSpec",
    "InterpretSpec",
    "PipelineSpec",
    "ResolvedTarget",
    "TargetSpec",
    "contender_names",
    "is_vintage_aware",
    "model_arms",
    "pipeline_spec",
    "resolve_target",
]
