"""Reviewed opt-in FRED-SD inferred t-code map.

FRED-SD does not publish official transformation codes. This module records
macrocast research decisions derived from MD/QD analog review. Runtime code may
use these entries only when the user explicitly opts in.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

MAP_VERSION = "sd-analog-v0.1"
OFFICIAL = False
SOURCE = "macrocast_inferred_from_md_qd_analogs"

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


def normalize_sd_tcode_policy(policy: str | None) -> str:
    if policy in {None, "", "none"}:
        return "none"
    if policy in {"inferred", "inferred_v0_1", MAP_VERSION}:
        return "inferred_v0_1"
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
        "frequency": normalized_frequency,
        "allowed_statuses": list(allowed),
        "applied": dict(applied),
        "skipped": skipped,
        "variables": variables,
    }
    return applied, report
