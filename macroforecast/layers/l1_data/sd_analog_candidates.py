"""Candidate analogs for non-official FRED-SD t-code research.

FRED-SD does not provide official transformation codes. The objects in this
module are human priors for a validation study, not runtime defaults.
"""
from __future__ import annotations

from dataclasses import dataclass

MAP_VERSION = "sd-analog-v0.1"
OFFICIAL = False
SOURCE = "macrocast_inferred_from_md_qd_analogs"
ALLOWED_DATASETS = ("fred_md", "fred_qd")
ALLOWED_CONFIDENCE = ("high", "medium", "low", "reject")
DEFAULT_CANDIDATE_CODES = (1, 2, 4, 5, 6)


@dataclass(frozen=True)
class AnalogSeries:
    dataset: str
    series: str
    reason: str


@dataclass(frozen=True)
class SdAnalogCandidate:
    sd_variable: str
    candidate_analogs: tuple[AnalogSeries, ...]
    candidate_codes: tuple[int, ...]
    prior_confidence: str
    note: str


SD_ANALOG_CANDIDATES: tuple[SdAnalogCandidate, ...] = (
    SdAnalogCandidate(
        "BPPRIVSA",
        (
            AnalogSeries("fred_md", "PERMIT", "national building permits"),
            AnalogSeries("fred_md", "HOUST", "national housing starts"),
            AnalogSeries("fred_qd", "PERMIT", "quarterly building permits"),
            AnalogSeries("fred_qd", "HOUST", "quarterly housing starts"),
        ),
        (4, 5),
        "medium",
        "Frequency-specific prior: use same-frequency permits analogs, MD monthly code 4 and QD quarterly code 5.",
    ),
    SdAnalogCandidate(
        "CONS",
        (AnalogSeries("fred_md", "USCONS", "national construction employment"), AnalogSeries("fred_qd", "USCONS", "quarterly construction employment")),
        (5,),
        "high",
        "State construction employment analog.",
    ),
    SdAnalogCandidate(
        "CONSTNQGSP",
        (AnalogSeries("fred_md", "USCONS", "national construction employment"), AnalogSeries("fred_qd", "USCONS", "quarterly construction employment")),
        (1, 5),
        "low",
        "Construction GSP is output, while available MD/QD analog is employment.",
    ),
    SdAnalogCandidate(
        "EXPORTS",
        (AnalogSeries("fred_qd", "EXPGSC1", "real exports of goods and services"),),
        (5,),
        "medium",
        "State exports analog to national exports.",
    ),
    SdAnalogCandidate(
        "FIRE",
        (AnalogSeries("fred_md", "USFIRE", "national financial activities employment"), AnalogSeries("fred_qd", "USFIRE", "quarterly financial activities employment")),
        (5,),
        "high",
        "State financial activities employment analog.",
    ),
    SdAnalogCandidate(
        "FIRENQGSP",
        (AnalogSeries("fred_md", "USFIRE", "national financial activities employment"), AnalogSeries("fred_qd", "USFIRE", "quarterly financial activities employment")),
        (1, 5),
        "low",
        "Financial sector GSP is output, while available analog is employment.",
    ),
    SdAnalogCandidate(
        "GOVNQGSP",
        (AnalogSeries("fred_md", "USGOVT", "national government employment"), AnalogSeries("fred_qd", "USGOVT", "quarterly government employment")),
        (1, 5),
        "low",
        "Government GSP is output, while available analog is employment.",
    ),
    SdAnalogCandidate(
        "GOVT",
        (AnalogSeries("fred_md", "USGOVT", "national government employment"), AnalogSeries("fred_qd", "USGOVT", "quarterly government employment")),
        (5,),
        "high",
        "State government employment analog.",
    ),
    SdAnalogCandidate(
        "ICLAIMS",
        (AnalogSeries("fred_md", "CLAIMSx", "national initial unemployment claims"), AnalogSeries("fred_qd", "CLAIMSx", "quarterly initial unemployment claims")),
        (5,),
        "high",
        "State initial claims analog.",
    ),
    SdAnalogCandidate(
        "IMPORTS",
        (AnalogSeries("fred_qd", "IMPGSC1", "real imports of goods and services"),),
        (5,),
        "medium",
        "State imports analog to national imports.",
    ),
    SdAnalogCandidate(
        "INFO",
        (AnalogSeries("fred_qd", "USINFO", "information employment"),),
        (5,),
        "high",
        "State information employment analog.",
    ),
    SdAnalogCandidate(
        "INFONQGSP",
        (AnalogSeries("fred_qd", "USINFO", "information employment"),),
        (1, 5),
        "low",
        "Information GSP is output, while available analog is employment.",
    ),
    SdAnalogCandidate(
        "LF",
        (AnalogSeries("fred_md", "CLF16OV", "civilian labor force"),),
        (5,),
        "high",
        "State labor force analog.",
    ),
    SdAnalogCandidate(
        "MANNQGSP",
        (AnalogSeries("fred_md", "MANEMP", "manufacturing employment"), AnalogSeries("fred_qd", "MANEMP", "quarterly manufacturing employment")),
        (1, 5),
        "low",
        "Manufacturing GSP is output, while available analog is employment.",
    ),
    SdAnalogCandidate(
        "MFG",
        (AnalogSeries("fred_md", "MANEMP", "manufacturing employment"), AnalogSeries("fred_qd", "MANEMP", "quarterly manufacturing employment")),
        (5,),
        "high",
        "State manufacturing employment analog.",
    ),
    SdAnalogCandidate(
        "MFGHRS",
        (AnalogSeries("fred_md", "AWHMAN", "average weekly manufacturing hours"), AnalogSeries("fred_qd", "AWHMAN", "quarterly average weekly manufacturing hours")),
        (1, 2),
        "high",
        "Hours/rate-like series; level is the official analog prior.",
    ),
    SdAnalogCandidate(
        "MINNG",
        (AnalogSeries("fred_qd", "USMINE", "mining and logging employment"),),
        (5,),
        "high",
        "State mining employment analog.",
    ),
    SdAnalogCandidate(
        "NA",
        (AnalogSeries("fred_md", "PAYEMS", "total nonfarm payroll employment"), AnalogSeries("fred_qd", "PAYEMS", "quarterly payroll employment")),
        (5,),
        "high",
        "State nonfarm employment analog.",
    ),
    SdAnalogCandidate(
        "NATURNQGSP",
        (AnalogSeries("fred_qd", "USMINE", "mining and logging employment"),),
        (1, 5),
        "low",
        "Natural resources GSP has no direct MD/QD output analog.",
    ),
    SdAnalogCandidate(
        "NQGSP",
        (AnalogSeries("fred_qd", "GDPC1", "real GDP"),),
        (5,),
        "medium",
        "State real GSP analog to national real GDP.",
    ),
    SdAnalogCandidate(
        "OTOT",
        (AnalogSeries("fred_md", "RPI", "real personal income"), AnalogSeries("fred_qd", "DPIC96", "real disposable personal income")),
        (5,),
        "medium",
        "State personal income-like series; definitions differ.",
    ),
    SdAnalogCandidate(
        "PARTRATE",
        (AnalogSeries("fred_qd", "CIVPART", "labor force participation rate"),),
        (1, 2),
        "high",
        "Rate-like series; official analog prior is first difference.",
    ),
    SdAnalogCandidate(
        "PSERV",
        (AnalogSeries("fred_qd", "USPBS", "professional and business services employment"),),
        (5,),
        "high",
        "State professional and business services employment analog.",
    ),
    SdAnalogCandidate(
        "PSERVNQGSP",
        (AnalogSeries("fred_qd", "USPBS", "professional and business services employment"), AnalogSeries("fred_qd", "USSERV", "service-providing employment")),
        (1, 5),
        "low",
        "Professional services GSP is output, while available analogs are employment.",
    ),
    SdAnalogCandidate(
        "RENTS",
        (),
        (1, 5),
        "reject",
        "No sufficiently direct MD/QD analog identified.",
    ),
    SdAnalogCandidate(
        "STHPI",
        (AnalogSeries("fred_qd", "USSTHPI", "national house price index"),),
        (5,),
        "high",
        "Quarterly source-frequency prior: use QD USSTHPI code 5 before any monthly interpolation.",
    ),
    SdAnalogCandidate(
        "UR",
        (AnalogSeries("fred_md", "UNRATE", "national unemployment rate"), AnalogSeries("fred_qd", "UNRATE", "quarterly unemployment rate")),
        (2,),
        "high",
        "State unemployment rate analog.",
    ),
    SdAnalogCandidate(
        "UTILNQGSP",
        (),
        (1, 5),
        "reject",
        "No sufficiently direct MD/QD utility-sector output analog identified.",
    ),
)


def candidates_by_variable() -> dict[str, SdAnalogCandidate]:
    return {candidate.sd_variable: candidate for candidate in SD_ANALOG_CANDIDATES}


def candidate_variables() -> tuple[str, ...]:
    return tuple(candidate.sd_variable for candidate in SD_ANALOG_CANDIDATES)
