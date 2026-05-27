"""Built-in FRED-SD source-load groups.

These groups are Layer 1 recipe conveniences. They resolve to explicit
FRED-SD state and workbook-variable lists before the raw loader runs.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

FRED_SD_GROUP_SELECTION_CONTRACT_VERSION = "fred_sd_group_selection_v1"

FRED_SD_STATE_GROUPS: dict[str, tuple[str, ...]] = {
    "census_region_northeast": ("CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"),
    "census_region_midwest": ("IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"),
    "census_region_south": (
        "DE",
        "DC",
        "FL",
        "GA",
        "MD",
        "NC",
        "SC",
        "VA",
        "WV",
        "AL",
        "KY",
        "MS",
        "TN",
        "AR",
        "LA",
        "OK",
        "TX",
    ),
    "census_region_west": ("AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "AK", "CA", "HI", "OR", "WA"),
    "census_division_new_england": ("CT", "ME", "MA", "NH", "RI", "VT"),
    "census_division_middle_atlantic": ("NJ", "NY", "PA"),
    "census_division_east_north_central": ("IL", "IN", "MI", "OH", "WI"),
    "census_division_west_north_central": ("IA", "KS", "MN", "MO", "NE", "ND", "SD"),
    "census_division_south_atlantic": ("DE", "DC", "FL", "GA", "MD", "NC", "SC", "VA", "WV"),
    "census_division_east_south_central": ("AL", "KY", "MS", "TN"),
    "census_division_west_south_central": ("AR", "LA", "OK", "TX"),
    "census_division_mountain": ("AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY"),
    "census_division_pacific": ("AK", "CA", "HI", "OR", "WA"),
    "contiguous_48_plus_dc": (
        "AL",
        "AZ",
        "AR",
        "CA",
        "CO",
        "CT",
        "DE",
        "DC",
        "FL",
        "GA",
        "ID",
        "IL",
        "IN",
        "IA",
        "KS",
        "KY",
        "LA",
        "ME",
        "MD",
        "MA",
        "MI",
        "MN",
        "MS",
        "MO",
        "MT",
        "NE",
        "NV",
        "NH",
        "NJ",
        "NM",
        "NY",
        "NC",
        "ND",
        "OH",
        "OK",
        "OR",
        "PA",
        "RI",
        "SC",
        "SD",
        "TN",
        "TX",
        "UT",
        "VT",
        "VA",
        "WA",
        "WV",
        "WI",
        "WY",
    ),
}

FRED_SD_VARIABLE_GROUPS: dict[str, tuple[str, ...]] = {
    "labor_market_core": ("ICLAIMS", "LF", "NA", "PARTRATE", "UR"),
    "employment_sector": ("CONS", "FIRE", "GOVT", "INFO", "MFG", "MFGHRS", "MINNG", "PSERV"),
    "gsp_output": (
        "NQGSP",
        "CONSTNQGSP",
        "FIRENQGSP",
        "GOVNQGSP",
        "INFONQGSP",
        "MANNQGSP",
        "NATURNQGSP",
        "PSERVNQGSP",
        "UTILNQGSP",
    ),
    "housing": ("BPPRIVSA", "RENTS", "STHPI"),
    "trade": ("EXPORTS", "IMPORTS"),
    "income": ("OTOT",),
    "direct_analog_high_confidence": (
        "CONS",
        "FIRE",
        "GOVT",
        "ICLAIMS",
        "INFO",
        "LF",
        "MFG",
        "MFGHRS",
        "MINNG",
        "NA",
        "PARTRATE",
        "PSERV",
        "UR",
    ),
    "provisional_analog_medium": ("EXPORTS", "IMPORTS", "NQGSP", "OTOT"),
    "semantic_review_outputs": ("CONSTNQGSP", "GOVNQGSP", "MANNQGSP", "PSERVNQGSP"),
    "no_reliable_analog": ("FIRENQGSP", "INFONQGSP", "NATURNQGSP", "RENTS", "UTILNQGSP"),
}


def _normalize_member_list(values: Sequence[Any], *, uppercase: bool) -> list[str]:
    members = [str(value).strip() for value in values]
    if uppercase:
        members = [value.upper() for value in members]
    if not members or any(not value for value in members):
        raise ValueError("custom FRED-SD group members must be a non-empty list of non-empty strings")
    return members


def _resolve_custom_group(
    leaf_config: Mapping[str, Any] | None,
    *,
    members_key: str,
    mapping_key: str,
    name_key: str,
    label: str,
    uppercase: bool = False,
) -> tuple[list[str], str]:
    payload = leaf_config or {}
    direct = payload.get(members_key)
    if isinstance(direct, Sequence) and not isinstance(direct, (str, bytes)):
        return _normalize_member_list(direct, uppercase=uppercase), members_key

    groups = payload.get(mapping_key)
    name = payload.get(name_key)
    if isinstance(groups, Mapping) and name in groups:
        members = groups[name]
        if isinstance(members, Sequence) and not isinstance(members, (str, bytes)):
            return _normalize_member_list(members, uppercase=uppercase), f"{mapping_key}.{name}"

    raise ValueError(
        f"{label} requires leaf_config.{members_key} or "
        f"leaf_config.{mapping_key} plus leaf_config.{name_key}"
    )


def resolve_fred_sd_state_group(
    group: str | None,
    leaf_config: Mapping[str, Any] | None = None,
) -> tuple[list[str] | None, str]:
    """Resolve a FRED-SD state group to state abbreviations.

    Returns ``(None, "all_states")`` for the all-state default.
    """

    key = str(group or "all_states")
    if key == "all_states":
        return None, "all_states"
    if key == "custom_state_group":
        return _resolve_custom_group(
            leaf_config,
            members_key="sd_state_group_members",
            mapping_key="sd_state_groups",
            name_key="sd_state_group_name",
            label="fred_sd_state_group='custom_state_group'",
            uppercase=True,
        )
    if key not in FRED_SD_STATE_GROUPS:
        allowed = sorted(("all_states", "custom_state_group", *FRED_SD_STATE_GROUPS))
        raise ValueError(f"unsupported fred_sd_state_group={key!r}; allowed values: {allowed}")
    return list(FRED_SD_STATE_GROUPS[key]), key


def resolve_fred_sd_variable_group(
    group: str | None,
    leaf_config: Mapping[str, Any] | None = None,
) -> tuple[list[str] | None, str]:
    """Resolve a FRED-SD workbook-variable group to sheet names."""

    key = str(group or "all_sd_variables")
    if key == "all_sd_variables":
        return None, "all_sd_variables"
    if key == "custom_sd_variable_group":
        return _resolve_custom_group(
            leaf_config,
            members_key="sd_variable_group_members",
            mapping_key="sd_variable_groups",
            name_key="sd_variable_group_name",
            label="fred_sd_variable_group='custom_sd_variable_group'",
        )
    if key not in FRED_SD_VARIABLE_GROUPS:
        allowed = sorted(("all_sd_variables", "custom_sd_variable_group", *FRED_SD_VARIABLE_GROUPS))
        raise ValueError(f"unsupported fred_sd_variable_group={key!r}; allowed values: {allowed}")
    return list(FRED_SD_VARIABLE_GROUPS[key]), key
