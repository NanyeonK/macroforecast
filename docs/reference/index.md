# Reference

Lookup pages for the public Python API. Each function page states accepted
inputs, outputs, defaults, side effects, and validation behavior.

> **Looking for design rationale?** Use [Explanation](../explanation/index.md).
> Reference pages define callable surfaces; explanation pages discuss package
> concepts and tradeoffs.

## Public API

## Workflow Order

| Stage | Page | What It Owns |
| --- | --- | --- |
| Meta | [Meta](meta.md) | Package-wide execution defaults such as seed, worker count, and error policy. |
| Data | [Data](data.md) | Canonical pandas panels, dataset metadata, loaders, target/horizon/sample specs. |
| Preprocessing | [Preprocessing](preprocessing.md) | Frequency alignment, t-code transforms, outlier handling, imputation, and frame rules. |
| Data Summary | [Data Summary](data_summary.md) | One-panel coverage, missingness, descriptive statistics, outlier checks, and stationarity checks. |
| Data Analysis | [Data Analysis](data_analysis.md) | Raw versus processed panel comparison with compact before/after metadata. |
| Evaluation | [Evaluation](evaluation.md) | Forecast metrics, benchmark-relative metrics, aggregation, ranking, and evaluation artifacts. |
| Output | [Output](output.md) | Saved artifacts, provenance, exports, and manifests. |

- [Meta](meta.md): package-wide execution settings.
- [Data](data.md): canonical panels, metadata, loaders, and run-level data specs.
- [FRED-MD](fred_md.md): monthly national macroeconomic database loader and metadata contract.
- [FRED-QD](fred_qd.md): quarterly national macroeconomic database loader and metadata contract.
- [FRED-SD](fred_sd.md): state-level database loader, monthly/quarterly frequency handling, and t-code limitations.
- [Preprocessing](preprocessing.md): frequency alignment, transforms, outlier handling, imputation, and frame-edge rules.
- [Data Summary](data_summary.md): single-panel coverage, missingness, descriptive statistics, and correlations.
- [Data Analysis](data_analysis.md): raw-vs-processed panel comparison analysis.
- [Evaluation](evaluation.md): forecast metrics, aggregation, ranking, and evaluation artifacts.
- [Public API](public_api.md): importable Python surface and compatibility boundaries.
- [Defaults](defaults.md): package-level default profiles.
- [FRED datasets](fred_datasets.md): dataset-selection overview and source notes.

## Pages

```{toctree}
:maxdepth: 1

meta
data
fred_md
fred_qd
fred_sd
preprocessing
data_summary
data_analysis
evaluation
public_api
defaults
fred_datasets
```
