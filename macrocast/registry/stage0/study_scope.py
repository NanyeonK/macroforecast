from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..base import AxisDefinition, EnumRegistryEntry
from ..naming import canonical_axis_value

RouteOwner = Literal["comparison_sweep"]


@dataclass(frozen=True)
class StudyScopeEntry(EnumRegistryEntry):
    route_owner: RouteOwner
    requires_multi_target: bool
    compares_methods: bool


STUDY_SCOPE_ENTRIES: tuple[StudyScopeEntry, ...] = (
    StudyScopeEntry(
        id="one_target_one_method",
        description="One target evaluated with one fixed forecasting method path.",
        status="operational",
        priority="A",
        route_owner="comparison_sweep",
        requires_multi_target=False,
        compares_methods=False,
    ),
    StudyScopeEntry(
        id="one_target_compare_methods",
        description="One target evaluated across a controlled set of method alternatives.",
        status="operational",
        priority="A",
        route_owner="comparison_sweep",
        requires_multi_target=False,
        compares_methods=True,
    ),
    StudyScopeEntry(
        id="multiple_targets_one_method",
        description="Multiple targets evaluated with one fixed forecasting method path.",
        status="operational",
        priority="A",
        route_owner="comparison_sweep",
        requires_multi_target=True,
        compares_methods=False,
    ),
    StudyScopeEntry(
        id="multiple_targets_compare_methods",
        description="Multiple targets evaluated across a controlled set of method alternatives.",
        status="operational",
        priority="A",
        route_owner="comparison_sweep",
        requires_multi_target=True,
        compares_methods=True,
    ),
)

AXIS_DEFINITION = AxisDefinition(
    axis_name="study_scope",
    layer="0_meta",
    axis_type="enum",
    default_policy="fixed",
    entries=STUDY_SCOPE_ENTRIES,
    compatible_with={},
    incompatible_with={},
)

_BY_ID = {entry.id: entry for entry in STUDY_SCOPE_ENTRIES}


def get_study_scope_entry(study_scope: str) -> StudyScopeEntry:
    return _BY_ID[canonical_axis_value("study_scope", study_scope)]


def study_scope_options_for_wizard(task: str | None = None) -> tuple[str, ...]:
    """Return the four public Study Scope choices.

    Study Scope owns both target cardinality and method-comparison cardinality,
    so the wizard shows the same four choices before target details are known.
    """
    return tuple(entry.id for entry in STUDY_SCOPE_ENTRIES if entry.status == "operational")


def derive_study_scope_default(
    *,
    task: str,
    model_axis_mode: str = "fixed",
    feature_axis_mode: str = "fixed",
    wrapper_family: str | None = None,
) -> str:
    compares_methods = model_axis_mode == "sweep" or feature_axis_mode == "sweep"
    if task == "multi_target":
        return "multiple_targets_compare_methods" if compares_methods else "multiple_targets_one_method"
    return "one_target_compare_methods" if compares_methods else "one_target_one_method"
