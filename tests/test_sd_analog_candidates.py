from __future__ import annotations

from macroforecast.raw.sd_analog_candidates import (
    ALLOWED_CONFIDENCE,
    ALLOWED_DATASETS,
    DEFAULT_CANDIDATE_CODES,
    OFFICIAL,
    SD_ANALOG_CANDIDATES,
    SOURCE,
    candidates_by_variable,
)
from macroforecast.raw.sd_inferred_tcodes import (
    DEFAULT_RUNTIME_STATUSES,
    SD_INFERRED_TCODE_MAP,
    STATE_SERIES_STATIONARITY_OVERRIDE_VERSION,
    VARIABLE_GLOBAL_STATIONARITY_MAP_VERSION,
    VARIABLE_GLOBAL_STATIONARITY_TCODE_MAP,
    build_sd_inferred_transform_codes,
    build_sd_state_series_stationarity_transform_codes,
    build_sd_transform_codes_for_policy,
    build_sd_variable_global_stationarity_transform_codes,
    normalize_sd_tcode_policy,
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


def test_sd_tcode_policy_normalization_supports_empirical_modes() -> None:
    assert normalize_sd_tcode_policy("sd-analog-v0.1") == "inferred_v0_1"
    assert normalize_sd_tcode_policy(VARIABLE_GLOBAL_STATIONARITY_MAP_VERSION) == "variable_global_stationarity_v0_1"
    assert normalize_sd_tcode_policy(STATE_SERIES_STATIONARITY_OVERRIDE_VERSION) == "state_series_stationarity_override_v0_1"


def test_variable_global_stationarity_codes_map_state_columns() -> None:
    codes, report = build_sd_variable_global_stationarity_transform_codes(
        ["BPPRIVSA_CA", "UR_TX", "STHPI_NY", "UNKNOWN_CA"],
        frequency="quarterly",
    )

    assert VARIABLE_GLOBAL_STATIONARITY_TCODE_MAP["BPPRIVSA"]["code"] == 2
    assert codes == {"BPPRIVSA_CA": 2, "UR_TX": 2, "STHPI_NY": 6}
    assert report["map_version"] == VARIABLE_GLOBAL_STATIONARITY_MAP_VERSION
    assert report["decision_unit"] == "sd_variable"
    assert report["official"] is False
    assert report["variables"]["STHPI"]["dominant_share"] == 0.901961


def test_state_series_stationarity_override_requires_explicit_column_map() -> None:
    codes, report = build_sd_state_series_stationarity_transform_codes(
        ["UR_CA", "UR_TX", "BPPRIVSA_CA"],
        frequency="monthly",
        code_map={"UR_CA": 2, "UR_TX": 5, "MISSING_AK": 6},
        audit_uri="artifacts/sd_state_series_audit.csv",
    )

    assert codes == {"UR_CA": 2, "UR_TX": 5}
    assert report["map_version"] == STATE_SERIES_STATIONARITY_OVERRIDE_VERSION
    assert report["decision_unit"] == "sd_variable_x_state"
    assert report["audit_uri"] == "artifacts/sd_state_series_audit.csv"
    assert report["unmatched_code_columns"] == ["MISSING_AK"]
    assert report["skipped"]["BPPRIVSA_CA"] == "missing_state_series_override"


def test_build_sd_transform_codes_for_policy_dispatches_empirical_modes() -> None:
    codes, report = build_sd_transform_codes_for_policy(
        ["BPPRIVSA_CA"],
        policy="variable_global_stationarity_v0_1",
        frequency="monthly",
    )

    assert codes == {"BPPRIVSA_CA": 2}
    assert report["map_version"] == VARIABLE_GLOBAL_STATIONARITY_MAP_VERSION
