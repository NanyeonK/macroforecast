# FRED-SD Transform Policy

FRED-SD transformation codes are a research layer in macrocast. They are not
source metadata.

## Policy

Default runtime policy:

- FRED-MD official t-codes are applied when `tcode_policy="tcode_only"`.
- FRED-QD official t-codes are applied when `tcode_policy="tcode_only"`.
- FRED-SD inferred t-codes are not applied by default.

Opt-in runtime policy:

```python
exp = (
    mc.Experiment(
        dataset="fred_qd+fred_sd",
        target="GDPC1",
        start="1985-01",
        end="2019-12",
        horizons=[1, 2, 4],
    )
    .use_sd_inferred_tcodes()
)
```

The opt-in map is `sd-analog-v0.1`. It is stored in
`macrocast.raw.sd_inferred_tcodes.SD_INFERRED_TCODE_MAP`.

Important interpretation:

- `sd-analog-v0.1` is not a state-by-state stationarity optimizer.
- It applies one reviewed code to every state column for a given FRED-SD
  variable, for example all `UR_*` columns share code `2`.
- The reviewed code is anchored to the closest FRED-MD/FRED-QD national analog
  when that analog is economically direct, then checked against state-level
  diagnostics.
- The map therefore answers "what transformation is defensible for this SD
  variable as a cross-state panel?" not "what code maximizes stationarity for
  each individual state series?"

## Policy Choices

FRED-SD has no official `transform` row, so there are three distinct policies a
researcher could choose. They should not be conflated.

| policy | unit of decision | what it does | benefit | risk | macrocast status |
|---|---|---|---|---|---|
| National-analog transfer | SD variable | If an SD variable is a state version of a national FRED-MD/QD object, use the national official t-code analog and apply it to every state. | Anchored to official MD/QD source metadata; keeps state panels comparable. | May not maximize stationarity for every state; weak for variables without direct national analogs. | Current opt-in `sd-analog-v0.1`, `official=false`. |
| Variable-global empirical | SD variable | Search candidate codes on all states and select one code per SD variable using state/aggregate diagnostics. | Targets stationarity while keeping one transform per cross-state panel. | Can disagree with official national analog semantics; sample/vintage dependent. | Research-only, not runtime default. |
| State-variable empirical | SD variable x state | Search candidate codes independently for each state series, for example `UR_CA` and `UR_TX` may differ. | Maximizes stationarity diagnostics column by column. | Breaks cross-state comparability, can overfit small/state-specific samples, and may change by vintage. | Future explicit research override only; not supported by default runtime. |

Recommended default for package runtime remains:

1. Use official FRED-MD/QD t-codes where the source provides them.
2. Leave FRED-SD untransformed unless the user explicitly opts in.
3. When opting into FRED-SD, use the reviewed national-analog map first.
4. Treat variable-global or state-variable empirical t-codes as a separate
   research design, with manifest evidence and audit artifacts.

## Review Status

Runtime applies only these statuses by default:

| status | runtime use | meaning |
|---|---|---|
| `tentative_accept` | yes | Direct analog and sufficient diagnostics. |
| `provisional_accept` | yes | Plausible analog, lower confidence. |
| `frequency_specific` | yes | Code depends on experiment frequency. |
| `frequency_specific_provisional` | yes | Source frequency is clear, but diagnostics require caution. |
| `semantic_review` | no | Statistics may fit, but the economic object changes. |
| `manual_review` | no | No runtime use before targeted review. |
| `reject` | no | No inferred code in v0.1. |

Users can restrict runtime application:

```python
exp.use_sd_inferred_tcodes(statuses=["tentative_accept"])
```

They can also include additional statuses, but that should be treated as a
research override.

## Manifest Contract

When SD inferred t-codes are used, the run manifest includes
`data_reports.sd_inferred_tcodes`.

Expected fields:

| field | meaning |
|---|---|
| `map_version` | Inferred map version, currently `sd-analog-v0.1`. |
| `official` | Always `false`. |
| `source` | `macrocast_inferred_from_md_qd_analogs`. |
| `frequency` | Normalized experiment frequency. |
| `allowed_statuses` | Review statuses allowed for this run. |
| `applied` | Column-to-code map used by t-code preprocessing. |
| `skipped` | Column-to-review-status map for columns not used. |
| `variables` | Variable-level reviewed metadata. |

The manifest also emits a warning:

```text
FRED-SD inferred t-codes are macrocast research metadata, not official FRED-SD metadata
```

## Frequency-Specific Rules

`BPPRIVSA` must not be represented as one global code.

| experiment frequency | analog source | code | reason |
|---|---|---:|---|
| monthly | FRED-MD `PERMIT` / `HOUST` | 4 | Same-frequency official MD permits/housing transform. |
| quarterly | FRED-QD `PERMIT` / `HOUST` | 5 | Same-frequency official QD permits/housing transform. |

`STHPI` follows QD `USSTHPI` code `5`.

Reason:

- It is a house price index object.
- The reviewed same-concept analog is quarterly QD `USSTHPI`.
- Monthly interpolation should happen after the source-frequency transform
  decision; interpolation does not create a new official monthly analog.
- Stationarity diagnostics are weak, so the status is
  `frequency_specific_provisional`.

## State-Level Stationarity Audit

On 2026-04-26, macrocast checked the actual FRED-SD live by-series workbook
`series-2026-03.xlsx` against candidate codes `(1, 2, 4, 5, 6)` for every
state column, using observations from 2005-06 onward. For each
`SD variable x state`, the audit selected the candidate with the best simple
ADF/KPSS stationarity score. This is a diagnostic only; it does not define the
runtime policy.

The diagnostic confirms the user's concern: a stationarity-only state-level
choice often differs from the national-analog code. For example, many
employment panels have FRED-MD/QD analog code `5`, while state-by-state
ADF/KPSS screening often favors code `2`. That does not automatically mean
code `2` is the correct package default; it means the empirical-stationarity
policy is a different research design from the national-analog policy.

| SD variable | current opt-in code | dominant state-level stationarity code | dominant state share | state-level distribution |
|---|---|---:|---:|---|
| `BPPRIVSA` | monthly `4`, quarterly `5` | 2 | 0.92 | 2:0.92; 4:0.02; 5:0.04; 6:0.02 |
| `CONS` | 5 | 2/6 tie | 0.45 | 1:0.02; 2:0.45; 5:0.08; 6:0.45 |
| `CONSTNQGSP` | 5, not runtime-default status | 6 | 0.61 | 2:0.22; 5:0.16; 6:0.61 |
| `EXPORTS` | 5 | 2 | 0.96 | 2:0.96; 5:0.04 |
| `FIRE` | 5 | 2 | 0.67 | 2:0.67; 5:0.02; 6:0.31 |
| `FIRENQGSP` | none | 6 | 0.39 | 2:0.24; 5:0.37; 6:0.39 |
| `GOVNQGSP` | 5, not runtime-default status | 5 | 0.43 | 2:0.39; 5:0.43; 6:0.18 |
| `GOVT` | 5 | 2 | 0.98 | 2:0.98; 6:0.02 |
| `ICLAIMS` | 5 | 2 | 0.96 | 1:0.02; 2:0.96; 5:0.02 |
| `IMPORTS` | 5 | 2 | 0.98 | 2:0.98; 6:0.02 |
| `INFO` | 5 | 2 | 0.86 | 2:0.86; 5:0.06; 6:0.08 |
| `INFONQGSP` | none | 2 | 0.59 | 2:0.59; 5:0.18; 6:0.24 |
| `LF` | 5 | 2 | 0.88 | 2:0.88; 5:0.06; 6:0.06 |
| `MANNQGSP` | 5, not runtime-default status | 2 | 0.94 | 2:0.94; 5:0.04; 6:0.02 |
| `MFG` | 5 | 2 | 0.67 | 2:0.67; 5:0.12; 6:0.22 |
| `MFGHRS` | 1 | 2 | 0.98 | 2:0.98; 6:0.02 |
| `MINNG` | 5 | 2 | 0.88 | 2:0.88; 5:0.02; 6:0.10 |
| `NA` | 5 | 2 | 1.00 | 2:1.00 |
| `NATURNQGSP` | none | 2 | 0.97 | 2:0.97; 6:0.03 |
| `NQGSP` | 5 | 5 | 0.67 | 2:0.20; 5:0.67; 6:0.14 |
| `OTOT` | 5 | 5 | 0.55 | 2:0.39; 5:0.55; 6:0.06 |
| `PARTRATE` | 2 | 2 | 0.98 | 2:0.98; 6:0.02 |
| `PSERV` | 5 | 2 | 1.00 | 2:1.00 |
| `PSERVNQGSP` | 5, not runtime-default status | 5 | 0.69 | 2:0.27; 5:0.69; 6:0.04 |
| `RENTS` | none | 2 | 0.77 | 2:0.77; 5:0.06; 6:0.17 |
| `STHPI` | 5 | 6 | 0.90 | 2:0.04; 5:0.06; 6:0.90 |
| `UR` | 2 | 2 | 1.00 | 2:1.00 |
| `UTILNQGSP` | none | 2 | 0.88 | 2:0.88; 5:0.06; 6:0.06 |

Interpretation:

- The current opt-in map keeps cross-state comparability by using one
  variable-level code.
- The state-level diagnostic is useful for sensitivity analysis and future
  research modes, but it should not silently replace the national-analog map.
- A future state-variable empirical mode must write every selected column code,
  sample window, vintage, test battery, and tie-break rule into the manifest.

## Validation Protocol

The validation script is:

```bash
python tools/research/build_sd_tcode_validation.py \
  --sd-workbook /tmp/fred_sd_series_2026_03_validation.xlsx \
  --md-csv /tmp/fred_md_current_validation.csv \
  --qd-csv /tmp/fred_qd_current_validation.csv \
  --output-dir /tmp/macrocast_sd_tcode_validation_rigorous \
  --sample-start 2005-01 \
  --sample-end 2025-12
```

Generated artifacts:

- `sd_tcode_candidate_results.csv`
- `sd_tcode_selected_map.json`
- `sd_tcode_report.md`

The report shows every candidate-code/analog row. The JSON selected map merges
score-ranked diagnostics with reviewed status metadata from
`SD_INFERRED_TCODE_MAP`.

## Diagnostics

The validation output includes:

| diagnostic | purpose |
|---|---|
| `aggregate_corr` | Pearson correlation between transformed SD aggregate and transformed MD/QD analog. |
| `pearson_pvalue` | Pearson correlation p-value. |
| `spearman_corr` | Rank correlation for monotone relation. |
| `spearman_pvalue` | Spearman p-value. |
| `rolling_corr_mean` | Average rolling aggregate correlation. |
| `rolling_corr_min` | Worst rolling aggregate correlation. |
| `state_corr_median` | Median state-level correlation to analog. |
| `state_corr_iqr` | State-level correlation dispersion. |
| `state_corr_positive_share` | Share of states with positive analog correlation. |
| `adf_pass_rate` | State-level ADF stationarity pass rate. |
| `kpss_pass_rate` | State-level KPSS stationarity pass rate. |
| `sd_aggregate_adf_pass` | ADF result for SD aggregate. |
| `sd_aggregate_kpss_pass` | KPSS result for SD aggregate. |
| `analog_adf_pass` | ADF result for analog. |
| `analog_kpss_pass` | KPSS result for analog. |
| `acf_distance` | Distance between SD and analog autocorrelation profiles. |
| `low_frequency_ratio` | SD low-frequency variance share. |
| `analog_low_frequency_ratio` | Analog low-frequency variance share. |
| `low_frequency_distance` | Distance between SD and analog low-frequency ratios. |
| `outlier_rate` | SD transformed outlier share. |
| `analog_outlier_rate` | Analog transformed outlier share. |
| `volatility_ratio` | SD aggregate volatility divided by analog volatility. |
| `missing_rate` | Missing share after transform. |
| `n_states_used` | Number of state columns with enough observations. |

## Runtime Order

For composite datasets, macrocast first aligns component frequencies:

- `fred_md+fred_sd` uses monthly frequency.
- `fred_qd+fred_sd` uses quarterly frequency.

Then SD inferred t-code metadata is added to `transform_codes` only when the
user opted in. The existing `tcode_policy="tcode_only"` preprocessing path then
applies MD/QD official codes and any opted-in SD inferred codes together.

## Non-Goals

This policy does not make SD inferred t-codes official.

This policy does not apply `semantic_review`, `manual_review`, or `reject`
entries by default.

This policy does not add extra filtering, scaling, missing-value imputation, or
target-specific transformations beyond the existing experiment preprocessing
contract.
