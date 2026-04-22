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
