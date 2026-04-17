from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..base import AxisDefinition, EnumRegistryEntry

RouteOwner = Literal["single_run", "wrapper", "orchestrator", "replication"]


@dataclass(frozen=True)
class ExperimentUnitEntry(EnumRegistryEntry):
    route_owner: RouteOwner
    requires_multi_target: bool
    requires_wrapper: bool
    runner: str | None = None


EXPERIMENT_UNIT_ENTRIES: tuple[ExperimentUnitEntry, ...] = (
    ExperimentUnitEntry(
        id="single_target_single_model",
        description="Single-target executable single-model run.",
        status="operational",
        priority="A",
        route_owner="single_run",
        requires_multi_target=False,
        requires_wrapper=False,
    ),
    ExperimentUnitEntry(
        id="single_target_model_grid",
        description="Single-target model-grid comparison within the single-run family.",
        status="operational",
        priority="A",
        route_owner="single_run",
        requires_multi_target=False,
        requires_wrapper=False,
    ),
    ExperimentUnitEntry(
        id="single_target_full_sweep",
        description="Single-target full sweep requiring wrapper/orchestrator ownership.",
        status="operational",
        priority="A",
        route_owner="wrapper",
        requires_multi_target=False,
        requires_wrapper=True,
    ),
    ExperimentUnitEntry(
        id="multi_target_separate_runs",
        description="Multi-target wrapper that fans out separate single-target runs.",
        status="registry_only",
        priority="A",
        route_owner="wrapper",
        requires_multi_target=True,
        requires_wrapper=True,
    ),
    ExperimentUnitEntry(
        id="multi_target_shared_design",
        description="Multi-target shared-design wrapper family.",
        status="planned",
        priority="A",
        route_owner="wrapper",
        requires_multi_target=True,
        requires_wrapper=True,
    ),
    ExperimentUnitEntry(
        id="multi_output_joint_model",
        description="Multi-target single-run joint-model family.",
        status="registry_only",
        priority="B",
        route_owner="single_run",
        requires_multi_target=True,
        requires_wrapper=False,
    ),
    ExperimentUnitEntry(
        id="hierarchical_forecasting_run",
        description="Hierarchical forecasting orchestrator family.",
        status="future",
        priority="B",
        route_owner="orchestrator",
        requires_multi_target=True,
        requires_wrapper=True,
    ),
    ExperimentUnitEntry(
        id="panel_forecasting_run",
        description="Panel forecasting orchestrator family.",
        status="future",
        priority="B",
        route_owner="orchestrator",
        requires_multi_target=True,
        requires_wrapper=True,
    ),
    ExperimentUnitEntry(
        id="state_space_run",
        description="Single-run state-space forecasting family.",
        status="future",
        priority="B",
        route_owner="single_run",
        requires_multi_target=False,
        requires_wrapper=False,
    ),
    ExperimentUnitEntry(
        id="replication_recipe",
        description="Replication-locked recipe/unit.",
        status="operational",
        priority="A",
        route_owner="replication",
        requires_multi_target=False,
        requires_wrapper=False,
        runner="macrocast.studies.replication:execute_replication",
    ),
    ExperimentUnitEntry(
        id="benchmark_suite",
        description="Wrapper-managed benchmark suite.",
        status="planned",
        priority="A",
        route_owner="wrapper",
        requires_multi_target=False,
        requires_wrapper=True,
    ),
    ExperimentUnitEntry(
        id="ablation_study",
        description="Wrapper-managed ablation study.",
        status="operational",
        priority="A",
        route_owner="wrapper",
        requires_multi_target=False,
        requires_wrapper=True,
        runner="macrocast.studies.ablation:execute_ablation",
    ),
)

AXIS_DEFINITION = AxisDefinition(
    axis_name="experiment_unit",
    layer="0_meta",
    axis_type="enum",
    default_policy="fixed",
    entries=EXPERIMENT_UNIT_ENTRIES,
    compatible_with={},
    incompatible_with={},
)

_BY_ID = {entry.id: entry for entry in EXPERIMENT_UNIT_ENTRIES}


def get_experiment_unit_entry(experiment_unit: str) -> ExperimentUnitEntry:
    return _BY_ID[experiment_unit]


def experiment_unit_options_for_wizard(study_mode: str, task: str) -> tuple[str, ...]:
    if study_mode == "replication_override_study":
        return ("replication_recipe",)
    if task == "multi_target_point_forecast":
        return (
            "multi_target_separate_runs",
            "multi_target_shared_design",
            "multi_output_joint_model",
        )
    if study_mode == "orchestrated_bundle_study":
        return ("benchmark_suite", "ablation_study")
    return (
        "single_target_single_model",
        "single_target_model_grid",
        "single_target_full_sweep",
    )


def derive_experiment_unit_default(
    *,
    study_mode: str,
    task: str,
    model_axis_mode: str = "fixed",
    feature_axis_mode: str = "fixed",
    wrapper_family: str | None = None,
) -> str:
    if study_mode == "replication_override_study":
        return "replication_recipe"
    if wrapper_family in _BY_ID:
        return wrapper_family
    if task == "multi_target_point_forecast":
        return "multi_target_separate_runs"
    if study_mode == "orchestrated_bundle_study":
        return "benchmark_suite"
    if model_axis_mode == "sweep" and feature_axis_mode == "sweep":
        return "single_target_full_sweep"
    if model_axis_mode == "sweep":
        return "single_target_model_grid"
    return "single_target_single_model"
