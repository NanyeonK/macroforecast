# FRED-SD Inferred T-Codes

FRED-SD does not publish official transformation codes equivalent to the
FRED-MD/FRED-QD `transform` rows. Macrocast must therefore treat any FRED-SD
t-code layer as a project-specific inferred layer, not as source-provided
metadata.

This page documents the research protocol for `sd-analog-v0.1`. The protocol is
not part of the default runtime until the validation report is reviewed.

## Contract

- FRED-SD raw values are source data.
- FRED-MD and FRED-QD t-codes are official source metadata.
- FRED-SD inferred t-codes are macroforecast research artifacts.
- Every inferred code must record its analog source, validation diagnostics, and
  confidence level.
- Low-confidence variables must remain untransformed unless the user explicitly
  opts in.
- Runtime manifests must never describe inferred FRED-SD codes as official.

Required manifest language:

```json
{
  "official": false,
  "source": "macrocast_inferred_from_md_qd_analogs",
  "map_version": "sd-analog-v0.1"
}
```

## Candidate Table

The candidate table lives in `macroforecast/raw/sd_analog_candidates.py`. It records
human priors before statistical validation:

- `sd_variable`: FRED-SD workbook sheet name.
- `candidate_analogs`: MD/QD national series that represent the closest concept.
- `candidate_codes`: t-codes evaluated for the SD variable.
- `prior_confidence`: initial judgment before diagnostics.
- `note`: reason for the prior and known caveats.

The table is intentionally allowed to contain weak candidates. Weak candidates
make rejection explicit.

## Validation Protocol

Run:

```bash
python tools/research/build_sd_tcode_validation.py \
  --sd-workbook /path/to/fred_sd_series.xlsx \
  --md-csv /path/to/fred_md_current.csv \
  --qd-csv /path/to/fred_qd_current.csv \
  --output-dir artifacts/sd_tcode_validation
```

The script writes:

- `sd_tcode_candidate_results.csv`
- `sd_tcode_selected_map.json`
- `sd_tcode_report.md`

For each SD variable, candidate code, and analog series, the script compares the
transformed SD state aggregate with the transformed MD/QD national analog.

Core diagnostics:

- aggregate correlation
- ADF pass rate
- KPSS pass rate
- ACF distance
- low-frequency ratio
- outlier rate
- missing rate
- number of states used
- sample overlap

## Scoring

The initial score is a screening metric, not an automatic truth label:

```text
score =
  0.35 * positive aggregate correlation
+ 0.20 * stationarity score
+ 0.20 * ACF similarity score
+ 0.15 * low-frequency score
+ 0.10 * outlier score
```

Selection must be reviewed manually when:

- the best code differs from the analog official t-code,
- multiple analogs disagree,
- overlap is short,
- stationarity diagnostics conflict,
- the prior confidence is low.

## Confidence Labels

`high`
: Direct conceptual analog, candidate code agrees with the official analog code,
  and diagnostics are stable.

`medium`
: Similar concept but definition differs, or diagnostics are acceptable but not
  decisive.

`low`
: Weak analog, conflicting diagnostics, or state-level behavior differs strongly
  from the national analog.

`reject`
: No reliable inferred t-code should be used.

## Runtime Policy

The default runtime remains:

```text
FRED-MD/QD: apply official t-codes.
FRED-SD: use source values as provided.
```

The current opt-in FRED-SD policy is:

```python
Experiment(...).use_sd_inferred_tcodes()
```

This activates `sd_tcode_policy="inferred_v0_1"`, backed by
`sd-analog-v0.1`.

This policy is a national-analog reviewed map. It applies one code to every
state column for a given FRED-SD variable. It does not estimate a separate code
for each state, and it does not claim to be the stationarity-only optimum.

Empirical policies are named separately:

- `variable_global_stationarity_v0_1`: one empirically selected code per SD
  variable, shared across states. The current map version is
  `sd-variable-global-stationarity-v0.1`, based on the 2026-04-26
  `series-2026-03.xlsx` audit.
- `state_series_stationarity_override_v0_1`: one empirically selected code per
  `SD variable x state` column. The current map version is
  `sd-state-series-stationarity-override-v0.1`; users must provide an explicit
  `sd_tcode_code_map`.

Both empirical policies write the map version, selected code, sample window,
vintage/audit basis, diagnostics source, and validation report location when
provided into the manifest.

Python API:

```python
exp.use_sd_empirical_tcodes(unit="variable_global")

exp.use_sd_empirical_tcodes(
    unit="state_series",
    code_map={"UR_CA": 2, "UR_TX": 5},
    audit_uri="artifacts/sd_state_series_audit.csv",
)
```
