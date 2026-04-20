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
        description="Multi-target wrapper that fans out N single-target execute_recipe calls, one per target, each under its own artifact directory. Distinct from shared_design (same design, aggregated predictions.csv). Runner: macrocast.studies.multi_target.execute_separate_runs.",
        status="operational",
        priority="A",
        route_owner="wrapper",
        requires_multi_target=True,
        requires_wrapper=True,
        runner="macrocast.studies.multi_target:execute_separate_runs",
    ),
    ExperimentUnitEntry(
        id="multi_target_shared_design",
        description="Multi-target shared-design run: one compiled recipe evaluates all targets with the same design and produces an aggregated predictions table. Handled by execute_recipe's multi-target path.",
        status="operational",
        priority="A",
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
        status="operational",
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


def experiment_unit_options_for_wizard(research_design: str, task: str) -> tuple[str, ...]:
    """Return the experiment_unit options the wizard/UI should surface.

    Only values whose current registry status is operational are
    returned — registry_only and future entries are intentionally filtered
    so UIs do not propose non-executable units.
    """
    if research_design == "replication_override":
        candidates = ("replication_recipe",)
    elif task == "multi_target_point_forecast":
        candidates = (
            "multi_target_shared_design",
            "multi_target_separate_runs",
        )
    elif research_design == "orchestrated_bundle":
        candidates = ("benchmark_suite", "ablation_study")
    else:
        candidates = (
            "single_target_single_model",
            "single_target_model_grid",
            "single_target_full_sweep",
        )
    return tuple(cid for cid in candidates if _BY_ID[cid].status == "operational")


def derive_experiment_unit_default(
    *,
    research_design: str,
    task: str,
    model_axis_mode: str = "fixed",
    feature_axis_mode: str = "fixed",
    wrapper_family: str | None = None,
) -> str:
    if research_design == "replication_override":
        return "replication_recipe"
    if wrapper_family in _BY_ID:
        return wrapper_family
    if task == "multi_target_point_forecast":
        # shared_design is the operational multi-target path handled by
        # execute_recipe. separate_runs is registry_only (v1.1 wrapper).
        return "multi_target_shared_design"
    if research_design == "orchestrated_bundle":
        return "benchmark_suite"
    if model_axis_mode == "sweep" and feature_axis_mode == "sweep":
        return "single_target_full_sweep"
    if model_axis_mode == "sweep":
        return "single_target_model_grid"
    return "single_target_single_model"
