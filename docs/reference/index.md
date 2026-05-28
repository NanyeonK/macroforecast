# Reference

Lookup pages for the public Python API. Each function page states accepted
inputs, outputs, defaults, side effects, and validation behavior.

> **Looking for design rationale?** Use [Explanation](../explanation/index.md).
> Reference pages define callable surfaces; explanation pages discuss package
> concepts and tradeoffs.

## Public API

- [Meta](meta.md): package-wide execution settings.
- [Data](data.md): canonical panels, metadata, loaders, and run-level data specs.
- [FRED-MD](fred_md.md): monthly national macroeconomic database loader and metadata contract.
- [FRED-QD](fred_qd.md): quarterly national macroeconomic database loader and metadata contract.
- [FRED-SD](fred_sd.md): state-level database loader, monthly/quarterly frequency handling, and t-code limitations.
- [Preprocessing](preprocessing.md): frequency alignment, transforms, outlier handling, imputation, and frame-edge rules.
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
defaults
fred_datasets
```
