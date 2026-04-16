# Raw data

## Purpose

The raw-data layer is the first data-facing package surface of macrocast.

Its job is narrow:
- normalize current versus explicit vintage requests
- construct deterministic cache paths
- record raw artifact provenance
- provide dataset-specific loaders on top of a shared contract

It does not do:
- forecasting transforms
- target construction
- model fitting
- evaluation

## Current implemented adapters

The rebuilt raw layer now includes:
- FRED-MD via `load_fred_md()`
- FRED-QD via `load_fred_qd()`
- FRED-SD via `load_fred_sd()`

It also includes a shared CSV parser:
- `parse_fred_csv()`

## Adapter status

### FRED-MD
- status: stable first-pass
- format: CSV
- supports current and explicit vintage requests
- has local ZIP-oriented fallback hook for historical behavior testing

### FRED-QD
- status: stable first-pass
- format: CSV
- supports current and explicit vintage requests

### FRED-SD
- status: provisional
- format: Excel workbook
- supports current and explicit vintage requests at the API level
- current implementation should be treated as provisional until broader live-source verification is complete

## Current executable real-time slice

- runtime now supports the first explicit-vintage real-time slice for single-target studies
- current route requires `info_set='real_time'` plus explicit `leaf_config.data_vintage`
- raw loader provenance switches from `version_mode='current'` to `version_mode='vintage'`
- this is not yet a historical rolling real-time panel engine

## Shared CSV format

FRED-MD and FRED-QD use a similar CSV layout, and the parser now accepts both of the relevant row orderings:
- official layout: header row first, transform row second
- legacy/fixture layout: transform row first, header row second

`parse_fred_csv()` returns:
- a pandas `DataFrame`
- a transformation-code mapping keyed by variable name

## `load_fred_md()`

`load_fred_md()` builds a `RawLoadResult` for the monthly macro dataset.

It supports:
- `current` access when `vintage=None`
- explicit vintage access when `vintage="YYYY-MM"`
- deterministic cache paths
- raw artifact hashing and manifest append
- `local_source=` for fixture-driven testing and local ingestion
- `local_zip_source=` for historical ZIP-style fallback testing and local extraction

Current robustness behavior:
- invalid vintage format fails early through version normalization
- file acquisition errors raise `RawDownloadError`
- parse failures raise `RawParseError`

## `load_fred_qd()`

`load_fred_qd()` mirrors the same contract for the quarterly dataset.

It returns the same type of result object and uses the same raw cache and manifest conventions.

Current robustness behavior:
- invalid vintage format fails early through version normalization
- acquisition failures raise `RawDownloadError`
- malformed downloaded content is surfaced as `RawDownloadError`

## `load_fred_sd()`

`load_fred_sd()` builds a `RawLoadResult` for the state-level workbook dataset.

It supports:
- `current` access when `vintage=None`
- explicit vintage access when `vintage="YYYY-MM"`
- deterministic cache paths
- raw artifact hashing and manifest append
- `local_source=` for fixture-driven testing and local ingestion
- optional `variables=` filtering by workbook sheet name
- optional `states=` filtering by state columns

The returned DataFrame is wide, with columns named:
- `{variable}_{state}`

Current provisional behavior:
- support tier is explicitly marked `provisional`
- workbook parsing uses `openpyxl`
- API availability does not yet imply full live-source robustness

## Result object

All dataset loaders return `RawLoadResult`.

That object contains:
- `data`: parsed pandas `DataFrame`
- `dataset_metadata`: semantic metadata such as frequency and support tier
- `artifact`: raw provenance record

## Cache and manifest behavior

Examples of deterministic paths:
- `~/.macrocast/raw/fred_md/current/raw.csv`
- `~/.macrocast/raw/fred_md/vintages/2020-01.csv`
- `~/.macrocast/raw/fred_qd/current/raw.csv`
- `~/.macrocast/raw/fred_sd/current/raw.xlsx`
- `~/.macrocast/raw/fred_sd/vintages/2020-01.xlsx`
- `~/.macrocast/raw/manifest/raw_artifacts.jsonl`

Every successful raw load appends a manifest record describing the artifact that was used.

## Current limitation

The current adapters are package surfaces, but the ingestion system is still incomplete relative to the archived codebase.

What is present now:
- stable first-pass MD/QD adapters
- provisional SD adapter
- shared parser and workbook ingestion path
- cache and manifest coupling
- explicit error wrapping for MD/QD

What still remains for later refinement:
- real remote historical ZIP fallback management for FRED-MD
- richer network retry behavior
- broader metadata enrichment
- broader live verification for FRED-SD vintage behavior
