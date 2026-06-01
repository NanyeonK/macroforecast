# FRED Datasets

[Back to documentation home](../index.md)

`macroforecast` currently exposes three official FRED-family data sources and
two combined data loaders.

| Dataset | Loader | Native role | Recommended use |
| --- | --- | --- | --- |
| FRED-MD | `mf.data.load_fred_md()` | Monthly national macro panel | Monthly targets and monthly national controls. |
| FRED-QD | `mf.data.load_fred_qd()` | Quarterly national macro panel | Quarterly targets and quarterly national controls. |
| FRED-SD | `mf.data.load_fred_sd()` | State-level mixed-frequency panel | State-level predictors, targets, or controls. |
| FRED-MD + FRED-SD | `mf.data.load_fred_md_sd()` | Combined monthly panel by default | Monthly state analysis with national controls. |
| FRED-QD + FRED-SD | `mf.data.load_fred_qd_sd()` | Combined quarterly panel by default | Quarterly state analysis with national controls. |

Dataset-specific pages live under this FRED Datasets section.

```{toctree}
:maxdepth: 1
:caption: FRED Datasets

fred_md
fred_qd
fred_sd
fred_md_sd
fred_qd_sd
```

## Loader Boundary

`macroforecast.data` is responsible for:

- downloading or copying raw sources,
- parsing raw files,
- normalizing date-indexed pandas panels,
- storing raw-file provenance,
- recording official FRED-MD/FRED-QD t-codes,
- recording FRED-SD series metadata,
- combining FRED-MD/FRED-QD with FRED-SD when requested,
- warning when a combined panel changes a source series' native frequency.

`macroforecast.preprocessing` is responsible for:

- applying FRED-MD/FRED-QD official t-codes after data loading,
- applying user-specified FRED-SD t-codes,
- handling t-code lag rows,
- outlier handling,
- imputation,
- frame-edge handling.

FRED-SD t-codes are not official. They must be explicitly supplied or built
with `mf.preprocessing.fred_sd_transform_codes()`.

## Frequency Rule Of Thumb

Use FRED-MD for monthly national variables and FRED-QD for quarterly national
variables. Do not use FRED-MD as a default quarterly substitute or FRED-QD as a
default monthly substitute.

Allowed but discouraged cases:

| Case | Why discouraged | What happens |
| --- | --- | --- |
| `load_fred_md_sd(frequency="quarterly")` | FRED-MD is monthly; FRED-QD is the natural quarterly national source. | The package proceeds, writes a parse note, and records monthly-to-quarterly conversions. |
| `load_fred_qd_sd(frequency="monthly")` | FRED-QD is quarterly; FRED-MD is the natural monthly national source. | The package proceeds, writes a parse note, and records quarterly-to-monthly conversions. |

## Conversion Warnings

Combined loaders proceed when a source series must be converted from monthly to
quarterly or quarterly to monthly, but they do not do it silently. They emit a
`UserWarning` and store the same information in:

```python
bundle.metadata["frequency_conversion_warnings"]
```

This lets users inspect exactly which variables changed frequency and which
rule was used.

## Dataset Docs Status

The FRED dataset pages now cover:

- standalone FRED-MD, FRED-QD, and FRED-SD loaders,
- official URLs and direct current-source links where applicable,
- FRED-MD/FRED-QD official t-codes and source groups,
- FRED-SD variable, state, native-frequency, observed-span, and date-anchor
  metadata,
- FRED-MD+FRED-SD and FRED-QD+FRED-SD combination behavior,
- frequency conversion warnings and preserved combined-panel metadata.

The remaining data-documentation surface is custom data ingestion:
`load_custom_csv()`, `load_custom_parquet()`, and in-memory `custom_dataset()`
are documented in [Data Reference](../reference/data.md), but they do not yet
have a standalone dataset-guide page.

## Official URLs

| Dataset | Official page | Direct current source used by package |
| --- | --- | --- |
| FRED-MD | <https://www.stlouisfed.org/research/economists/mccracken/fred-databases> | <https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/current.csv> |
| FRED-QD | <https://www.stlouisfed.org/research/economists/mccracken/fred-databases> | <https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/quarterly/current.csv> |
| FRED-SD | <https://www.stlouisfed.org/research/economists/owyang/fred-sd> | The loader resolves the latest official "Data by Series" workbook from the FRED-SD page. |
