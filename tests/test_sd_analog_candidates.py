from __future__ import annotations

from macrocast.raw.sd_analog_candidates import (
    ALLOWED_CONFIDENCE,
    ALLOWED_DATASETS,
    DEFAULT_CANDIDATE_CODES,
    OFFICIAL,
    SD_ANALOG_CANDIDATES,
    SOURCE,
    candidates_by_variable,
)
from macrocast.raw.sd_inferred_tcodes import (
    DEFAULT_RUNTIME_STATUSES,
    SD_INFERRED_TCODE_MAP,
    build_sd_inferred_transform_codes,
    resolve_sd_inferred_tcode,
)


def test_sd_analog_candidates_are_non_official_research_priors() -> None:
    assert OFFICIAL is False
    assert SOURCE == "macrocast_inferred_from_md_qd_analogs"


def test_sd_analog_candidates_cover_known_fred_sd_sheets_once() -> None:
    variables = [candidate.sd_variable for candidate in SD_ANALOG_CANDIDATES]

    assert len(variables) == 28
    assert len(set(variables)) == len(variables)
    assert "UR" in variables
    assert "NA" in variables
    assert "STHPI" in variables
    assert "UTILNQGSP" in variables


def test_sd_analog_candidates_use_valid_metadata() -> None:
    allowed_codes = set(DEFAULT_CANDIDATE_CODES)
    for candidate in SD_ANALOG_CANDIDATES:
        assert candidate.prior_confidence in ALLOWED_CONFIDENCE
        assert candidate.candidate_codes
        assert set(candidate.candidate_codes).issubset(allowed_codes)
        for analog in candidate.candidate_analogs:
            assert analog.dataset in ALLOWED_DATASETS
            assert analog.series


def test_candidates_by_variable_indexes_candidates() -> None:
    indexed = candidates_by_variable()

    assert indexed["UR"].candidate_codes == (2,)
    assert indexed["UR"].candidate_analogs[0].series == "UNRATE"
    assert indexed["RENTS"].prior_confidence == "reject"


def test_reviewed_sd_inferred_map_resolves_frequency_specific_codes() -> None:
    assert SD_INFERRED_TCODE_MAP["BPPRIVSA"]["status"] == "frequency_specific"
    assert resolve_sd_inferred_tcode("BPPRIVSA", frequency="monthly") == 4
    assert resolve_sd_inferred_tcode("BPPRIVSA", frequency="quarterly") == 5
    assert resolve_sd_inferred_tcode("STHPI", frequency="monthly") == 5
    assert resolve_sd_inferred_tcode("CONSTNQGSP", frequency="quarterly") is None
    assert resolve_sd_inferred_tcode(
        "CONSTNQGSP",
        frequency="quarterly",
        allowed_statuses=(*DEFAULT_RUNTIME_STATUSES, "semantic_review"),
    ) == 5


def test_build_sd_inferred_transform_codes_maps_state_columns() -> None:
    codes, report = build_sd_inferred_transform_codes(
        ["BPPRIVSA_CA", "UR_TX", "CONSTNQGSP_NY", "UNKNOWN_CA"],
        frequency="quarterly",
    )

    assert codes == {"BPPRIVSA_CA": 5, "UR_TX": 2}
    assert report["official"] is False
    assert report["variables"]["BPPRIVSA"]["status"] == "frequency_specific"
    assert report["skipped"]["CONSTNQGSP_NY"] == "semantic_review"
