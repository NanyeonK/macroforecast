from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..base import AxisDefinition, EnumRegistryEntry
from ..naming import canonical_axis_value

RouteOwner = Literal["comparison_sweep", "wrapper", "orchestrator", "replication"]


@dataclass(frozen=True)
class ExperimentUnitEntry(EnumRegistryEntry):
    route_owner: RouteOwner
    requires_multi_target: bool
    requires_wrapper: bool
    runner: str | None = None


EXPERIMENT_UNIT_ENTRIES: tuple[ExperimentUnitEntry, ...] = (
    ExperimentUnitEntry(
        id="single_target_single_generator",
        description="Single-target executable single-generator run.",
        status="operational",
        priority="A",
        route_owner="comparison_sweep",
        requires_multi_target=False,
        requires_wrapper=False,
    ),
    ExperimentUnitEntry(
        id="single_target_generator_grid",
        description="Single-target controlled comparison within the comparison_sweep family; usually a generator grid.",
        status="operational",
        priority="A",
        route_owner="comparison_sweep",
        requires_multi_target=False,
        requires_wrapper=False,
    ),
    ExperimentUnitEntry(
        id="single_target_full_sweep",
        description="Single-target full sweep grammar retained for future wrapper/orchestrator ownership; no executable runner contract in the current runtime.",
        status="registry_only",
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
        route_owner="comparison_sweep",
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
        route_owner="comparison_sweep",
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
        description="Wrapper-managed benchmark suite grammar retained for future PaperReadyBundle/runtime work; no executable runner contract in the current runtime.",
        status="registry_only",
        priority="A",
        route_owner="wrapper",
        requires_multi_target=False,
        requires_wrapper=True,
    ),
    ExperimentUnitEntry(
        id="ablation_study",
        description="Ablation study grammar retained for the standalone AblationSpec runner; not yet a compiled-recipe wrapper contract.",
        status="registry_only",
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
    return _BY_ID[canonical_axis_value("experiment_unit", experiment_unit)]


def experiment_unit_options_for_wizard(task: str = "single_target") -> tuple[str, ...]:
    """Return the experiment_unit options the wizard/UI should surface.

    Only values whose current registry status is operational are
    returned — registry_only and future entries are intentionally filtered
    so UIs do not propose non-executable units.
    """
    if task == "multi_target":
        candidates = (
            "multi_target_shared_design",
            "multi_target_separate_runs",
        )
    else:
        candidates = (
            "single_target_single_generator",
            "single_target_generator_grid",
            "replication_recipe",
        )
    return tuple(cid for cid in candidates if _BY_ID[cid].status == "operational")


def derive_experiment_unit_default(
    *,
    task: str,
    model_axis_mode: str = "fixed",
    feature_axis_mode: str = "fixed",
    wrapper_family: str | None = None,
) -> str:
    if wrapper_family in _BY_ID:
        return wrapper_family
    if task == "multi_target":
        # shared_design is handled by execute_recipe; separate_runs is the
        # supported wrapper-runner fan-out path.
        return "multi_target_shared_design"
    if model_axis_mode == "sweep" and feature_axis_mode == "sweep":
        return "single_target_full_sweep"
    if model_axis_mode == "sweep" or feature_axis_mode == "sweep":
        return "single_target_generator_grid"
    return "single_target_single_generator"
