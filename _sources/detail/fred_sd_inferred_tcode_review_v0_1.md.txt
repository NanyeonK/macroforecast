# FRED-SD Inferred T-Code Review v0.1

This document records the first manual review of the `sd-analog-v0.1`
validation output. FRED-SD does not provide official t-codes. All decisions here
are therefore macroforecast research judgments and must not be represented as
source-provided metadata.

Review date: 2026-04-21

Validation command:

```bash
python tools/research/build_sd_tcode_validation.py \
  --sd-workbook /tmp/fred_sd_series_2026_03_validation.xlsx \
  --md-csv /tmp/fred_md_current_validation.csv \
  --qd-csv /tmp/fred_qd_current_validation.csv \
  --output-dir /tmp/macrocast_sd_tcode_validation_check \
  --sample-start 2005-01 \
  --sample-end 2025-12
```

Generated artifacts:

- `/tmp/macrocast_sd_tcode_validation_check/sd_tcode_candidate_results.csv`
- `/tmp/macrocast_sd_tcode_validation_check/sd_tcode_selected_map.json`
- `/tmp/macrocast_sd_tcode_validation_check/sd_tcode_report.md`

Diagnostic-complete rerun after this review:

```bash
python tools/research/build_sd_tcode_validation.py \
  --sd-workbook /tmp/fred_sd_series_2026_03_validation.xlsx \
  --md-csv /tmp/fred_md_current_validation.csv \
  --qd-csv /tmp/fred_qd_current_validation.csv \
  --output-dir /tmp/macrocast_sd_tcode_validation_rigorous \
  --sample-start 2005-01 \
  --sample-end 2025-12
```

Additional generated artifacts:

- `/tmp/macrocast_sd_tcode_validation_rigorous/sd_tcode_candidate_results.csv`
- `/tmp/macrocast_sd_tcode_validation_rigorous/sd_tcode_selected_map.json`
- `/tmp/macrocast_sd_tcode_validation_rigorous/sd_tcode_report.md`

The rigorous report now shows every computed diagnostic for both best candidates
and all 66 candidate-code/analog rows. The CSV and selected-map JSON use the
same diagnostic fields, so the Markdown report is no longer only a `corr/score`
summary.

## Review Rules

The score is only a screening device. Manual review overrides the score when
the analog is conceptually weak.

Review labels:

`tentative_accept`
: Evidence is strong enough to use in an opt-in inferred map after one more
  full-report pass.

`provisional_accept`
: Evidence favors the code, but the analog is not direct enough for a default
  or high-confidence label.

`manual_review`
: Diagnostics and analog priors conflict. Do not select without targeted
  variable-level investigation.

`frequency_specific`
: A single global SD t-code would hide a real MD/QD frequency conflict. Select
  the inferred code from the same-frequency MD/QD analog used by the experiment.

`frequency_specific_provisional`
: The frequency source is clear, but diagnostics still require an explicit
  caution in the inferred-map metadata.

`semantic_review`
: Statistics are acceptable but the analog changes economic object, typically
  employment analogs used for sector output/GSP variables.

`reject`
: No reliable inferred t-code should be used in v0.1.

## Pipeline Checks

1. Parser integrity: fixed. FRED-QD current CSV contains a `factors` row before
   `transform`; parser now finds the explicit `transform` row and ignores
   `factors`.
2. Frequency/date alignment: fixed for validation. QD comparisons use quarterly
   period alignment, so quarter-start SD dates and quarter-end QD dates overlap.
3. Official status: unchanged. MD/QD t-codes are official source metadata;
   FRED-SD inferred t-codes are not official.
4. Diagnostic visibility: fixed. The rigorous report includes aggregate
   Pearson/Spearman correlations and p-values, rolling correlations, state-level
   correlation distribution, state stationarity pass rates, aggregate and analog
   stationarity flags, ACF distance, low-frequency ratios, outlier rates,
   volatility ratio, missing rate, and state coverage.

## Summary Decision Table

| SD variable | reviewed code | best analog | prior | corr | score | decision | reason |
|---|---:|---|---|---:|---:|---|---|
| `BPPRIVSA` | M:4, Q:5 | M `fred_md:PERMIT`; Q `fred_qd:PERMIT` | medium | M 0.981; Q 0.727 | M 0.708; Q 0.842 | frequency_specific | Follow the same-frequency permits analog: MD monthly permits are official code 4, QD quarterly permits are official code 5. |
| `CONS` | 5 | `fred_md:USCONS` | high | 0.967 | 0.894 | tentative_accept | Direct construction employment analog; code agrees with MD/QD. |
| `CONSTNQGSP` | 5 | `fred_qd:USCONS` | low | 0.786 | 0.768 | semantic_review | Code 5 is statistically favored, but GSP/output is compared to employment. |
| `EXPORTS` | 5 | `fred_qd:EXPGSC1` | medium | 0.427 | 0.705 | provisional_accept | Direct trade concept, but moderate correlation and QD-only analog. |
| `FIRE` | 5 | `fred_qd:USFIRE` | high | 0.913 | 0.876 | tentative_accept | Direct employment analog; MD and QD both support code 5. |
| `FIRENQGSP` | 5 | `fred_md:USFIRE` | low | 0.187 | 0.636 | reject | Output/GSP vs employment mismatch and weak correlation. |
| `GOVNQGSP` | 5 | `fred_qd:USGOVT` | low | 0.803 | 0.847 | semantic_review | Strong diagnostics but output/GSP is compared to employment. |
| `GOVT` | 5 | `fred_md:USGOVT` | high | 0.923 | 0.945 | tentative_accept | Direct government employment analog; MD/QD support code 5. |
| `ICLAIMS` | 5 | `fred_md:CLAIMSx` | high | 0.968 | 0.964 | tentative_accept | Direct claims analog; strong diagnostics after MD monthly alignment. |
| `IMPORTS` | 5 | `fred_qd:IMPGSC1` | medium | 0.591 | 0.817 | provisional_accept | Direct trade concept, but QD-only analog and moderate overlap/correlation. |
| `INFO` | 5 | `fred_qd:USINFO` | high | 0.927 | 0.931 | tentative_accept | Direct information employment analog. |
| `INFONQGSP` | 5 | `fred_qd:USINFO` | low | 0.351 | 0.709 | reject | Output/GSP vs employment mismatch and weak correlation. |
| `LF` | 5 | `fred_md:CLF16OV` | high | 0.770 | 0.877 | tentative_accept | Direct labor-force analog; code agrees with MD. |
| `MANNQGSP` | 5 | `fred_qd:MANEMP` | low | 0.698 | 0.845 | semantic_review | Manufacturing GSP/output compared to manufacturing employment. |
| `MFG` | 5 | `fred_md:MANEMP` | high | 0.983 | 0.941 | tentative_accept | Direct manufacturing employment analog. |
| `MFGHRS` | 1 | `fred_qd:AWHMAN` | high | 0.931 | 0.771 | tentative_accept | Direct hours analog. Code 1 preserves hours level; code 2 improves stationarity but loses analog semantics. |
| `MINNG` | 5 | `fred_qd:USMINE` | high | 0.865 | 0.886 | tentative_accept | Direct mining employment analog; some missingness but enough state coverage. |
| `NA` | 5 | `fred_md:PAYEMS` | high | 0.995 | 0.983 | tentative_accept | Direct total nonfarm employment analog; strongest evidence. |
| `NATURNQGSP` | 5 | `fred_qd:USMINE` | low | 0.374 | 0.694 | reject | Natural resources GSP has no direct national output analog; mining employment is too weak. |
| `NQGSP` | 5 | `fred_qd:GDPC1` | medium | 0.944 | 0.918 | provisional_accept | Strong state GSP to national GDP evidence; medium because state aggregate and national GDP are not identical objects. |
| `OTOT` | 5 | `fred_qd:DPIC96` | medium | 0.911 | 0.936 | provisional_accept | Income concept is close and diagnostics are strong; definitions still differ. |
| `PARTRATE` | 2 | `fred_qd:CIVPART` | high | 0.849 | 0.893 | tentative_accept | Direct participation-rate analog; code 2 clearly beats level. |
| `PSERV` | 5 | `fred_qd:USPBS` | high | 0.948 | 0.941 | tentative_accept | Direct professional/business services employment analog. |
| `PSERVNQGSP` | 5 | `fred_qd:USSERV` | low | 0.921 | 0.940 | semantic_review | Very strong diagnostics, but sector GSP/output is compared to service employment. |
| `RENTS` | none | none | reject | none | 0.000 | reject | No sufficiently direct MD/QD analog. |
| `STHPI` | 5 | `fred_qd:USSTHPI` | high | 0.903 | 0.694 | frequency_specific_provisional | STHPI is a quarterly/source-frequency HPI object; use QD `USSTHPI` code 5 before any monthly interpolation, with stationarity caution. |
| `UR` | 2 | `fred_md:UNRATE` | high | 0.983 | 0.977 | tentative_accept | Direct unemployment-rate analog; MD/QD both support code 2. |
| `UTILNQGSP` | none | none | reject | none | 0.000 | reject | No sufficiently direct MD/QD utility-sector output analog. |

## Targeted Diagnostic Audit

The rows below show every diagnostic field for the frequency-conflict variables
that cannot be judged from `corr/score` alone. The full 66-row table is in
`/tmp/macrocast_sd_tcode_validation_rigorous/sd_tcode_report.md`.

| sd | code | analog | analog code | score | n | pearson | pearson p | spearman | spearman p | roll mean | roll min | state med | state iqr | state pos | n state | adf rate | kpss rate | sd adf | sd kpss | analog adf | analog kpss | acf dist | sd lf | analog lf | lf gap | sd out | analog out | vol ratio | missing | states |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BPPRIVSA | 4 | fred_md:HOUST | 4 | 0.708 | 248 | 0.975 | 2.53e-162 | 0.961 | 7.12e-140 | 0.625 | -0.115 | 0.806 | 0.197 | 0.980 | 51 | 0.196 | 0.431 | no | no | no | no | 0.014 | 0.953 | 0.958 | 0.004 | 0.000 | 0.000 | 0.871 | 0.003 | 51 |
| BPPRIVSA | 4 | fred_md:PERMIT | 4 | 0.708 | 248 | 0.981 | 4.69e-178 | 0.970 | 3.42e-153 | 0.735 | 0.017 | 0.801 | 0.210 | 0.980 | 51 | 0.196 | 0.431 | no | no | no | no | 0.028 | 0.953 | 0.975 | 0.022 | 0.000 | 0.000 | 0.859 | 0.003 | 51 |
| BPPRIVSA | 4 | fred_qd:HOUST | 5 | 0.278 | 82 | 0.051 | 0.647 | -0.079 | 0.483 | 0.239 | -0.616 | 0.043 | 0.120 | 0.608 | 51 | 0.196 | 0.431 | no | no | yes | yes | 0.645 | 0.823 | 0.210 | 0.613 | 0.000 | 0.000 | 3.560 | 0.003 | 51 |
| BPPRIVSA | 4 | fred_qd:PERMIT | 5 | 0.281 | 82 | 0.033 | 0.772 | -0.117 | 0.295 | 0.304 | -0.917 | 0.024 | 0.159 | 0.549 | 51 | 0.196 | 0.431 | no | no | yes | yes | 0.600 | 0.823 | 0.261 | 0.562 | 0.000 | 0.000 | 3.926 | 0.003 | 51 |
| BPPRIVSA | 5 | fred_md:HOUST | 4 | 0.446 | 247 | 0.017 | 0.791 | 0.004 | 0.945 | 0.066 | -0.324 | -0.001 | 0.023 | 0.490 | 51 | 0.980 | 0.980 | no | yes | no | no | 1.004 | 0.043 | 0.957 | 0.915 | 0.000 | 0.000 | 0.233 | 0.008 | 51 |
| BPPRIVSA | 5 | fred_md:PERMIT | 4 | 0.449 | 247 | 0.027 | 0.677 | 0.012 | 0.851 | 0.109 | -0.304 | 0.007 | 0.015 | 0.725 | 51 | 0.980 | 0.980 | no | yes | no | no | 1.020 | 0.043 | 0.975 | 0.932 | 0.000 | 0.000 | 0.229 | 0.008 | 51 |
| BPPRIVSA | 5 | fred_qd:HOUST | 5 | 0.761 | 82 | 0.470 | 8.56e-06 | 0.370 | 6.14e-04 | 0.218 | -0.684 | 0.180 | 0.163 | 0.922 | 51 | 0.980 | 0.980 | no | yes | yes | yes | 0.130 | 0.155 | 0.210 | 0.055 | 0.000 | 0.000 | 0.381 | 0.008 | 51 |
| BPPRIVSA | 5 | fred_qd:PERMIT | 5 | 0.842 | 82 | 0.727 | 1.05e-14 | 0.663 | 1.19e-11 | 0.573 | -0.826 | 0.263 | 0.201 | 1.000 | 51 | 0.980 | 0.980 | no | yes | yes | yes | 0.177 | 0.155 | 0.261 | 0.106 | 0.000 | 0.000 | 0.420 | 0.008 | 51 |
| STHPI | 5 | fred_qd:USSTHPI | 5 | 0.694 | 81 | 0.903 | 8.44e-31 | 0.752 | 6.07e-16 | 0.652 | -0.385 | 0.804 | 0.089 | 1.000 | 51 | 0.137 | 0.373 | no | no | no | no | 0.166 | 0.598 | 0.471 | 0.127 | 0.000 | 0.000 | 0.912 | 0.012 | 51 |

## Accepted Direct Analogs

These variables have direct national analogs and diagnostics consistent with the
candidate code. They are candidates for an opt-in inferred map, pending one more
review of the generated CSV:

```text
CONS=5
FIRE=5
GOVT=5
ICLAIMS=5
INFO=5
LF=5
MFG=5
MFGHRS=1
MINNG=5
NA=5
PARTRATE=2
PSERV=5
UR=2
```

Notes:

- `MFGHRS=1` is accepted on analog semantics. Code 2 passes stationarity better,
  but the MD/QD official analog `AWHMAN` uses code 1.
- `PARTRATE=2` is accepted because code 2 dominates level and matches
  `CIVPART`.
- `ICLAIMS=5` has shorter original state coverage, but after period alignment it
  has enough monthly overlap and very strong diagnostics.

## Provisional Medium Analogs

These are plausible but should not be labeled high confidence:

```text
EXPORTS=5
IMPORTS=5
NQGSP=5
OTOT=5
```

Reasons:

- `EXPORTS` and `IMPORTS` have direct national trade analogs, but the validation
  evidence is moderate and QD-only.
- `NQGSP` has strong evidence against `GDPC1`, but state GSP aggregate and
  national GDP are not source-identical.
- `OTOT` has strong evidence against income analogs, but state `OTOT`, RPI, and
  DPIC96 definitions are not identical.

## Frequency-Specific Decisions

`BPPRIVSA`
: Use same-frequency analogs, not one global SD code.
  In a monthly experiment (`fred_md+fred_sd` or SD monthly source use), follow
  MD monthly `PERMIT`/`HOUST`, whose official code is 4. This preserves the
  level/log-permits convention and has very high monthly co-movement
  (`PERMIT` Pearson 0.981, Spearman 0.970). In a quarterly experiment
  (`fred_qd+fred_sd`), follow QD `PERMIT`, whose official code is 5. The
  quarterly code-5 row is the best quarterly match and passes the stationarity
  screen (`adf_pass_rate=0.980`, `kpss_pass_rate=0.980`). The low rolling
  minimum warns that this is not a high-confidence economic identity; it is a
  frequency-policy decision anchored to same-frequency MD/QD metadata.

`STHPI`
: Use QD `USSTHPI` code 5 as the source-frequency analog. STHPI is a house
  price index object with a direct QD analog and no MD same-concept monthly
  analog in the current candidate set. If an experiment later interpolates SD
  STHPI to monthly, the inferred transform should still be justified by the
  quarterly source-frequency QD analog before interpolation. Diagnostics support
  the analog relation (`Pearson=0.903`, `Spearman=0.752`, state median
  correlation 0.804), but stationarity is weak (`adf_pass_rate=0.137`,
  `kpss_pass_rate=0.373`, aggregate ADF/KPSS both fail). Therefore mark this as
  `frequency_specific_provisional`, not high-confidence acceptance.

## Semantic Review Group

These variables score well under code 5 but use employment analogs for GSP/output
series:

```text
CONSTNQGSP=5
GOVNQGSP=5
MANNQGSP=5
PSERVNQGSP=5
```

Interpretation:

- The statistics say code 5 often produces plausible transformed dynamics.
- The economic analog is not direct enough for a high-confidence t-code.
- These should be grouped under a separate `sector_output_inferred` rule if we
  keep them.
- They should not inherit employment confidence labels.

## Rejected For v0.1

```text
FIRENQGSP
INFONQGSP
NATURNQGSP
RENTS
UTILNQGSP
```

Reasons:

- `FIRENQGSP`, `INFONQGSP`, and `NATURNQGSP` have weak or conceptually poor
  analog evidence.
- `RENTS` has no direct MD/QD analog even though code 5 improves stationarity.
- `UTILNQGSP` has no direct MD/QD utility-sector output analog.

## Proposed v0.1 Output Map

The next selected-map artifact should separate decision status from selected
code:

```json
{
  "map_version": "sd-analog-v0.1",
  "official": false,
  "source": "macrocast_inferred_from_md_qd_analogs",
  "variables": {
    "NA": {"code": 5, "status": "tentative_accept"},
    "UR": {"code": 2, "status": "tentative_accept"},
    "NQGSP": {"code": 5, "status": "provisional_accept"},
    "BPPRIVSA": {
      "code": null,
      "code_by_frequency": {"M": 4, "Q": 5},
      "status": "frequency_specific",
      "reason": "Follow same-frequency MD/QD permits analogs."
    },
    "STHPI": {
      "code": 5,
      "status": "frequency_specific_provisional",
      "source_frequency": "Q",
      "reason": "Use QD USSTHPI code before any monthly interpolation."
    },
    "RENTS": {"code": null, "status": "reject"}
  }
}
```

Runtime must not consume `manual_review`, `semantic_review`, or `reject`
variables as if they were accepted. Runtime must also treat
`frequency_specific` entries as context-dependent rules rather than scalar
global t-codes.

## Next Actions

1. Add reviewed status metadata to the selected-map artifact, not only
   score-ranked candidates.
2. Add runtime support for opt-in frequency-specific SD inferred maps.
3. Add targeted plots for `STHPI`, including transformed aggregate, national
   analog, ACF, and rolling variance.
4. Search for better national/industry output analogs for sector GSP variables.
5. Keep default runtime unchanged: MD/QD official t-codes, SD source values.
