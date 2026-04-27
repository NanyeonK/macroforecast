from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ResearchDesign = Literal[
    "single_forecast_run",
    "controlled_variation",
    "study_bundle",
    "replication_recipe",
]

ExecutionPosture = Literal[
    "single_run_recipe",
    "single_run_with_internal_sweep",
    "wrapper_bundle_plan",
    "replication_locked_plan",
]

DesignShape = Literal[
    "one_fixed_env_one_tool_surface",
    "one_fixed_env_multi_tool_surface",
    "one_fixed_env_controlled_axis_variation",
    "wrapper_managed_multi_run_bundle",
]


@dataclass(frozen=True)
class FixedDesign:
    dataset_adapter: str
    information_set: str
    sample_split: str
    benchmark: str
    evaluation_protocol: str
    forecast_task: str


@dataclass(frozen=True)
class VaryingDesign:
    model_families: tuple[str, ...] = ()
    feature_recipes: tuple[str, ...] = ()
    preprocess_variants: tuple[str, ...] = ()
    tuning_variants: tuple[str, ...] = ()
    horizons: tuple[str, ...] = ()


@dataclass(frozen=True)
class ComparisonContract:
    information_set_policy: str
    sample_split_policy: str
    benchmark_policy: str
    evaluation_policy: str


@dataclass(frozen=True)
class ReplicationInput:
    source_type: str
    source_id: str
    locked_constraints: tuple[str, ...] = ()
    override_reason: str | None = None


@dataclass(frozen=True)
class DesignFrame:
    research_design: str
    fixed_design: FixedDesign
    comparison_contract: ComparisonContract
    varying_design: VaryingDesign
    execution_posture: str
    design_shape: str
    replication_input: ReplicationInput | None = None
    experiment_unit: str | None = None
