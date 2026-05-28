# FRED Dataset Overview

[Back to reference](index.md)

`macroforecast` currently exposes three official FRED-family data sources and
two combined data loaders.

| Dataset | Loader | Native role | Recommended use |
| --- | --- | --- | --- |
| FRED-MD | `mf.data.load_fred_md()` | Monthly national macro panel | Monthly targets and monthly national controls. |
| FRED-QD | `mf.data.load_fred_qd()` | Quarterly national macro panel | Quarterly targets and quarterly national controls. |
| FRED-SD | `mf.data.load_fred_sd()` | State-level mixed monthly/quarterly panel | State-level predictors, targets, or controls. |
| FRED-MD + FRED-SD | `mf.data.load_fred_md_sd()` | Combined monthly panel by default | Monthly state analysis with national controls. |
| FRED-QD + FRED-SD | `mf.data.load_fred_qd_sd()` | Combined quarterly panel by default | Quarterly state analysis with national controls. |

Dataset-specific pages:

- [FRED-MD](fred_md.md)
- [FRED-QD](fred_qd.md)
- [FRED-SD](fred_sd.md)

## Source Pages

- FRED-MD and FRED-QD: <https://www.stlouisfed.org/research/economists/mccracken/fred-databases>
- FRED-SD: <https://www.stlouisfed.org/research/economists/owyang/fred-sd>

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
