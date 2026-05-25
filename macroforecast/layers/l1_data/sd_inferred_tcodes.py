"""Opt-in FRED-SD inferred and empirical t-code maps.

FRED-SD does not publish official transformation codes. This module records
macroforecast research decisions derived from MD/QD analog review or from explicit
stationarity-audit maps. Runtime code may use these entries only when the user
explicitly opts in.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

MAP_VERSION = "sd-analog-v0.1"
VARIABLE_GLOBAL_STATIONARITY_MAP_VERSION = "sd-variable-global-stationarity-v0.1"
STATE_SERIES_STATIONARITY_OVERRIDE_VERSION = "sd-state-series-stationarity-override-v0.1"
OFFICIAL = False
SOURCE = "macrocast_inferred_from_md_qd_analogs"
VARIABLE_GLOBAL_STATIONARITY_SOURCE = "macrocast_empirical_stationarity_audit_2026_04_26_series_2026_03"
STATE_SERIES_STATIONARITY_SOURCE = "user_supplied_state_series_stationarity_audit"
VALID_TCODE_VALUES = frozenset(range(1, 8))

DEFAULT_RUNTIME_STATUSES = (
    "tentative_accept",
    "provisional_accept",
    "frequency_specific",
    "frequency_specific_provisional",
)

SD_INFERRED_TCODE_MAP: dict[str, dict[str, Any]] = {
    "BPPRIVSA": {
        "code": None,
        "code_by_frequency": {"monthly": 4, "quarterly": 5},
        "status": "frequency_specific",
        "confidence": "medium",
        "reason": "Follow same-frequency MD/QD permits analogs: MD monthly code 4, QD quarterly code 5.",
    },
    "CONS": {"code": 5, "status": "tentative_accept", "confidence": "high"},
    "CONSTNQGSP": {"code": 5, "status": "semantic_review", "confidence": "low"},
    "EXPORTS": {"code": 5, "status": "provisional_accept", "confidence": "medium"},
    "FIRE": {"code": 5, "status": "tentative_accept", "confidence": "high"},
    "FIRENQGSP": {"code": None, "status": "reject", "confidence": "low"},
    "GOVNQGSP": {"code": 5, "status": "semantic_review", "confidence": "low"},
    "GOVT": {"code": 5, "status": "tentative_accept", "confidence": "high"},
    "ICLAIMS": {"code": 5, "status": "tentative_accept", "confidence": "high"},
    "IMPORTS": {"code": 5, "status": "provisional_accept", "confidence": "medium"},
    "INFO": {"code": 5, "status": "tentative_accept", "confidence": "high"},
    "INFONQGSP": {"code": None, "status": "reject", "confidence": "low"},
    "LF": {"code": 5, "status": "tentative_accept", "confidence": "high"},
    "MANNQGSP": {"code": 5, "status": "semantic_review", "confidence": "low"},
    "MFG": {"code": 5, "status": "tentative_accept", "confidence": "high"},
    "MFGHRS": {"code": 1, "status": "tentative_accept", "confidence": "high"},
    "MINNG": {"code": 5, "status": "tentative_accept", "confidence": "high"},
    "NA": {"code": 5, "status": "tentative_accept", "confidence": "high"},
    "NATURNQGSP": {"code": None, "status": "reject", "confidence": "low"},
    "NQGSP": {"code": 5, "status": "provisional_accept", "confidence": "medium"},
    "OTOT": {"code": 5, "status": "provisional_accept", "confidence": "medium"},
    "PARTRATE": {"code": 2, "status": "tentative_accept", "confidence": "high"},
    "PSERV": {"code": 5, "status": "tentative_accept", "confidence": "high"},
    "PSERVNQGSP": {"code": 5, "status": "semantic_review", "confidence": "low"},
    "RENTS": {"code": None, "status": "reject", "confidence": "reject"},
    "STHPI": {
        "code": 5,
        "status": "frequency_specific_provisional",
        "confidence": "high",
        "source_frequency": "quarterly",
        "reason": "Use QD USSTHPI code 5 before any monthly interpolation; stationarity diagnostics remain weak.",
    },
    "UR": {"code": 2, "status": "tentative_accept", "confidence": "high"},
    "UTILNQGSP": {"code": None, "status": "reject", "confidence": "reject"},
}


VARIABLE_GLOBAL_STATIONARITY_TCODE_MAP: dict[str, dict[str, Any]] = {
    "BPPRIVSA": {"code": 2, "state_count": 51, "dominant_share": 0.921569, "distribution": "2:0.92; 4:0.02; 5:0.04; 6:0.02"},
    "CONS": {"code": 2, "state_count": 51, "dominant_share": 0.450980, "distribution": "1:0.02; 2:0.45; 5:0.08; 6:0.45"},
    "CONSTNQGSP": {"code": 6, "state_count": 49, "dominant_share": 0.612245, "distribution": "2:0.22; 5:0.16; 6:0.61"},
    "EXPORTS": {"code": 2, "state_count": 51, "dominant_share": 0.960784, "distribution": "2:0.96; 5:0.04"},
    "FIRE": {"code": 2, "state_count": 51, "dominant_share": 0.666667, "distribution": "2:0.67; 5:0.02; 6:0.31"},
    "FIRENQGSP": {"code": 6, "state_count": 51, "dominant_share": 0.392157, "distribution": "2:0.24; 5:0.37; 6:0.39"},
    "GOVNQGSP": {"code": 5, "state_count": 51, "dominant_share": 0.431373, "distribution": "2:0.39; 5:0.43; 6:0.18"},
    "GOVT": {"code": 2, "state_count": 51, "dominant_share": 0.980392, "distribution": "2:0.98; 6:0.02"},
    "ICLAIMS": {"code": 2, "state_count": 51, "dominant_share": 0.960784, "distribution": "1:0.02; 2:0.96; 5:0.02"},
    "IMPORTS": {"code": 2, "state_count": 51, "dominant_share": 0.980392, "distribution": "2:0.98; 6:0.02"},
    "INFO": {"code": 2, "state_count": 51, "dominant_share": 0.862745, "distribution": "2:0.86; 5:0.06; 6:0.08"},
    "INFONQGSP": {"code": 2, "state_count": 51, "dominant_share": 0.588235, "distribution": "2:0.59; 5:0.18; 6:0.24"},
    "LF": {"code": 2, "state_count": 51, "dominant_share": 0.882353, "distribution": "2:0.88; 5:0.06; 6:0.06"},
    "MANNQGSP": {"code": 2, "state_count": 49, "dominant_share": 0.938776, "distribution": "2:0.94; 5:0.04; 6:0.02"},
    "MFG": {"code": 2, "state_count": 51, "dominant_share": 0.666667, "distribution": "2:0.67; 5:0.12; 6:0.22"},
    "MFGHRS": {"code": 2, "state_count": 51, "dominant_share": 0.980392, "distribution": "2:0.98; 6:0.02"},
    "MINNG": {"code": 2, "state_count": 48, "dominant_share": 0.875000, "distribution": "2:0.88; 5:0.02; 6:0.10"},
    "NA": {"code": 2, "state_count": 51, "dominant_share": 1.000000, "distribution": "2:1.00"},
    "NATURNQGSP": {"code": 2, "state_count": 40, "dominant_share": 0.975000, "distribution": "2:0.97; 6:0.03"},
    "NQGSP": {"code": 5, "state_count": 51, "dominant_share": 0.666667, "distribution": "2:0.20; 5:0.67; 6:0.14"},
    "OTOT": {"code": 5, "state_count": 51, "dominant_share": 0.549020, "distribution": "2:0.39; 5:0.55; 6:0.06"},
    "PARTRATE": {"code": 2, "state_count": 51, "dominant_share": 0.980392, "distribution": "2:0.98; 6:0.02"},
    "PSERV": {"code": 2, "state_count": 51, "dominant_share": 1.000000, "distribution": "2:1.00"},
    "PSERVNQGSP": {"code": 5, "state_count": 51, "dominant_share": 0.686275, "distribution": "2:0.27; 5:0.69; 6:0.04"},
    "RENTS": {"code": 2, "state_count": 48, "dominant_share": 0.770833, "distribution": "2:0.77; 5:0.06; 6:0.17"},
    "STHPI": {"code": 6, "state_count": 51, "dominant_share": 0.901961, "distribution": "2:0.04; 5:0.06; 6:0.90"},
    "UR": {"code": 2, "state_count": 51, "dominant_share": 1.000000, "distribution": "2:1.00"},
    "UTILNQGSP": {"code": 2, "state_count": 51, "dominant_share": 0.882353, "distribution": "2:0.88; 5:0.06; 6:0.06"},
}


def normalize_sd_tcode_policy(policy: str | None) -> str:
    if policy in {None, "", "none"}:
        return "none"
    if policy in {"inferred", "inferred_v0_1", MAP_VERSION}:
        return "inferred_v0_1"
    if policy in {
        "variable_global_stationarity",
        "variable_global_stationarity_v0_1",
        VARIABLE_GLOBAL_STATIONARITY_MAP_VERSION,
    }:
        return "variable_global_stationarity_v0_1"
    if policy in {
        "state_series_stationarity",
        "state_series_stationarity_override",
        "state_series_stationarity_override_v0_1",
        STATE_SERIES_STATIONARITY_OVERRIDE_VERSION,
    }:
        return "state_series_stationarity_override_v0_1"
    raise ValueError(f"unsupported sd_tcode_policy={policy!r}")


def normalize_frequency(frequency: str | None) -> str:
    raw = str(frequency or "monthly").lower()
    if raw in {"m", "month", "monthly", "state_monthly"}:
        return "monthly"
    if raw in {"q", "quarter", "quarterly"}:
        return "quarterly"
    return raw


def sd_variable_from_column(column: object) -> str | None:
    name = str(column)
    for variable in sorted(SD_INFERRED_TCODE_MAP, key=len, reverse=True):
        if name == variable or name.startswith(f"{variable}_"):
            return variable
    return None


def resolve_sd_inferred_tcode(
    variable: str,
    *,
    frequency: str,
    allowed_statuses: Iterable[str] | None = None,
) -> int | None:
    entry = SD_INFERRED_TCODE_MAP.get(variable)
    if entry is None:
        return None
    allowed = set(DEFAULT_RUNTIME_STATUSES if allowed_statuses is None else allowed_statuses)
    if entry["status"] not in allowed:
        return None
    frequency_key = normalize_frequency(frequency)
    by_frequency = entry.get("code_by_frequency")
    if isinstance(by_frequency, Mapping):
        code = by_frequency.get(frequency_key)
    else:
        code = entry.get("code")
    return None if code is None else int(code)


def build_sd_inferred_transform_codes(
    columns: Iterable[object],
    *,
    frequency: str,
    allowed_statuses: Iterable[str] | None = None,
) -> tuple[dict[str, int], dict[str, Any]]:
    allowed = tuple(DEFAULT_RUNTIME_STATUSES if allowed_statuses is None else allowed_statuses)
    applied: dict[str, int] = {}
    skipped: dict[str, str] = {}
    variables: dict[str, dict[str, Any]] = {}
    normalized_frequency = normalize_frequency(frequency)

    for column in columns:
        variable = sd_variable_from_column(column)
        if variable is None:
            continue
        entry = SD_INFERRED_TCODE_MAP[variable]
        code = resolve_sd_inferred_tcode(variable, frequency=normalized_frequency, allowed_statuses=allowed)
        variables.setdefault(
            variable,
            {
                "status": entry["status"],
                "confidence": entry.get("confidence"),
                "code": entry.get("code"),
                "code_by_frequency": entry.get("code_by_frequency"),
            },
        )
        if code is None:
            skipped[str(column)] = str(entry["status"])
            continue
        applied[str(column)] = code

    report = {
        "map_version": MAP_VERSION,
        "official": OFFICIAL,
        "source": SOURCE,
        "policy_family": "national_analog_review",
        "decision_unit": "sd_variable",
        "frequency": normalized_frequency,
        "allowed_statuses": list(allowed),
        "applied": dict(applied),
        "skipped": skipped,
        "variables": variables,
    }
    return applied, report


def _validate_tcode_map(code_map: Mapping[str, Any], *, label: str) -> dict[str, int]:
    normalized: dict[str, int] = {}
    for key, raw_code in code_map.items():
        code = int(raw_code)
        if code not in VALID_TCODE_VALUES:
            raise ValueError(f"{label} contains unsupported t-code {code!r} for {key!r}")
        normalized[str(key)] = code
    return normalized


def build_sd_variable_global_stationarity_transform_codes(
    columns: Iterable[object],
    *,
    frequency: str,
) -> tuple[dict[str, int], dict[str, Any]]:
    applied: dict[str, int] = {}
    skipped: dict[str, str] = {}
    variables: dict[str, dict[str, Any]] = {}
    normalized_frequency = normalize_frequency(frequency)

    for column in columns:
        variable = sd_variable_from_column(column)
        if variable is None:
            continue
        entry = VARIABLE_GLOBAL_STATIONARITY_TCODE_MAP.get(variable)
        if entry is None:
            skipped[str(column)] = "missing_variable_global_stationarity_code"
            continue
        code = int(entry["code"])
        variables.setdefault(
            variable,
            {
                "code": code,
                "state_count": entry.get("state_count"),
                "dominant_share": entry.get("dominant_share"),
                "distribution": entry.get("distribution"),
            },
        )
        applied[str(column)] = code

    report = {
        "map_version": VARIABLE_GLOBAL_STATIONARITY_MAP_VERSION,
        "official": OFFICIAL,
        "source": VARIABLE_GLOBAL_STATIONARITY_SOURCE,
        "policy_family": "empirical_stationarity",
        "decision_unit": "sd_variable",
        "frequency": normalized_frequency,
        "audit_vintage": "series-2026-03.xlsx",
        "audit_date": "2026-04-26",
        "sample_start": "2005-06",
        "candidate_codes": [1, 2, 4, 5, 6],
        "selection_rule": "dominant_state_best_adf_kpss_score",
        "applied": dict(applied),
        "skipped": skipped,
        "variables": variables,
    }
    return applied, report


def build_sd_state_series_stationarity_transform_codes(
    columns: Iterable[object],
    *,
    frequency: str,
    code_map: Mapping[str, Any] | None,
    map_version: str | None = None,
    source: str | None = None,
    audit_uri: str | None = None,
) -> tuple[dict[str, int], dict[str, Any]]:
    if not isinstance(code_map, Mapping) or not code_map:
        raise ValueError("state-series FRED-SD t-code policy requires a non-empty sd_tcode_code_map")

    normalized_map = _validate_tcode_map(code_map, label="sd_tcode_code_map")
    normalized_frequency = normalize_frequency(frequency)
    selected_columns = {str(column) for column in columns}
    applied = {column: code for column, code in normalized_map.items() if column in selected_columns}
    unmatched = sorted(column for column in normalized_map if column not in selected_columns)
    skipped: dict[str, str] = {}
    variables: dict[str, dict[str, Any]] = {}

    for column in selected_columns:
        variable = sd_variable_from_column(column)
        if variable is None:
            continue
        variables.setdefault(variable, {"column_count": 0, "applied_column_count": 0})
        variables[variable]["column_count"] = int(variables[variable]["column_count"]) + 1
        if column in applied:
            variables[variable]["applied_column_count"] = int(variables[variable]["applied_column_count"]) + 1
        else:
            skipped[column] = "missing_state_series_override"

    report = {
        "map_version": map_version or STATE_SERIES_STATIONARITY_OVERRIDE_VERSION,
        "official": OFFICIAL,
        "source": source or STATE_SERIES_STATIONARITY_SOURCE,
        "policy_family": "empirical_stationarity_override",
        "decision_unit": "sd_variable_x_state",
        "frequency": normalized_frequency,
        "audit_uri": audit_uri,
        "input_code_count": len(normalized_map),
        "unmatched_code_columns": unmatched,
        "applied": dict(applied),
        "skipped": skipped,
        "variables": variables,
    }
    return applied, report


def build_sd_transform_codes_for_policy(
    columns: Iterable[object],
    *,
    policy: str,
    frequency: str,
    allowed_statuses: Iterable[str] | None = None,
    code_map: Mapping[str, Any] | None = None,
    map_version: str | None = None,
    source: str | None = None,
    audit_uri: str | None = None,
) -> tuple[dict[str, int], dict[str, Any]]:
    normalized_policy = normalize_sd_tcode_policy(policy)
    if normalized_policy == "inferred_v0_1":
        return build_sd_inferred_transform_codes(
            columns,
            frequency=frequency,
            allowed_statuses=allowed_statuses,
        )
    if normalized_policy == "variable_global_stationarity_v0_1":
        return build_sd_variable_global_stationarity_transform_codes(columns, frequency=frequency)
    if normalized_policy == "state_series_stationarity_override_v0_1":
        return build_sd_state_series_stationarity_transform_codes(
            columns,
            frequency=frequency,
            code_map=code_map,
            map_version=map_version,
            source=source,
            audit_uri=audit_uri,
        )
    raise ValueError(f"unsupported normalized sd_tcode_policy={normalized_policy!r}")
