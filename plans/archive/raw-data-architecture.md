# Macrocast Raw Data Architecture

Status: reboot-stage contract draft
Date: 2026-04-14
Scope: raw acquisition layer only

## Purpose

Rebuild the macrocast raw-data layer as a narrow, provenance-first foundation for all later package work.

This layer owns only:
- dataset-specific raw download and parsing
- vintage resolution
- raw cache layout
- raw provenance/index tracking
- conversion into pandas DataFrame plus metadata

This layer does not own:
- stationarity transforms
- outlier handling
- missing-data imputation
- sample design
- forecasting models
- evaluation or testing

## Core design rule

Separate three concerns:

1. Dataset loader
- knows how one source is downloaded and parsed
- example: FRED-MD CSV vs FRED-SD Excel workbook

2. Vintage manager
- knows how current vs explicit vintage is resolved
- knows cache naming, availability policy, and raw artifact identity

3. Raw index / provenance registry
- knows what file was fetched, from where, when, with what hash, and what data horizon it contains

No dataset loader should silently invent its own cache or vintage semantics.

## Target experiment-layer contract

Later layers should be able to say only this:
- dataset = `fred_md` or `fred_qd` or `fred_sd`
- version policy = `current` or explicit `vintage=YYYY-MM`

Then the raw-data layer should deterministically return:
- `pd.DataFrame`
- `RawDatasetMetadata`
- `RawArtifactRecord`

## Invariants

1. Raw layer returns pandas-first objects.
2. Every loaded dataset has explicit provenance metadata.
3. `current` and explicit `vintage` are different states and must never be conflated.
4. Raw cache paths are deterministic from dataset + version identity.
5. Every fetched raw file gets a manifest/index entry.
6. Parsing logic is dataset-specific; version logic is shared.
7. MD and QD are first-class supported datasets.
8. SD is provisional until its vintage support is verified against live source behavior.

## Recommended package layout

```text
macrocast/
  raw/
    __init__.py
    contracts.py
    manager.py
    cache.py
    manifest.py
    types.py
    datasets/
      __init__.py
      fred_md.py
      fred_qd.py
      fred_sd.py
      shared_csv.py
      shared_excel.py
    specs/
      fred_md.json
      fred_qd.json
      fred_sd.json
```

## Responsibility by file

### `macrocast/raw/contracts.py`
Defines public raw-layer contracts.

Objects:
- `RawVersionRequest`
- `RawDatasetMetadata`
- `RawArtifactRecord`
- `RawLoadResult`

### `macrocast/raw/types.py`
Common enums / literals.

Examples:
- `DatasetId = Literal["fred_md", "fred_qd", "fred_sd"]`
- `VersionMode = Literal["current", "vintage"]`
- `ArtifactFormat = Literal["csv", "xlsx"]`
- `SupportTier = Literal["stable", "provisional"]`

### `macrocast/raw/cache.py`
Deterministic cache path construction and local filesystem helpers.

Responsibilities:
- root cache discovery
- dataset subdir creation
- version-specific raw path resolution
- manifest path resolution

### `macrocast/raw/manifest.py`
Manifest/index read-write logic.

Responsibilities:
- append/update raw artifact entries
- load manifest table
- find artifact by dataset + version
- verify hash / file existence status

### `macrocast/raw/manager.py`
Shared version manager and orchestration layer.

Responsibilities:
- normalize current vs vintage request
- enumerate vintage candidates
- resolve download URL via dataset adapter
- call dataset adapter fetch/parse flow
- create `RawArtifactRecord`
- return `RawLoadResult`

### `macrocast/raw/datasets/fred_md.py`
Dataset-specific rules for FRED-MD.

Responsibilities:
- current URL
- vintage URL resolution / fallback policy
- parsing monthly CSV into DataFrame
- dataset-specific metadata extraction

### `macrocast/raw/datasets/fred_qd.py`
Dataset-specific rules for FRED-QD.

Responsibilities:
- current URL
- vintage URL resolution
- parsing quarterly CSV into DataFrame
- dataset-specific metadata extraction

### `macrocast/raw/datasets/fred_sd.py`
Dataset-specific rules for FRED-SD.

Responsibilities:
- current URL
- workbook parsing
- variable/state wide DataFrame build
- provisional vintage support only until verified

## Public API

## Minimal public functions

```python
from macrocast.raw import (
    load_fred_md,
    load_fred_qd,
    load_fred_sd,
    list_vintages,
    load_raw_dataset,
    read_raw_manifest,
)
```

### Dataset-facing loaders

```python
def load_fred_md(
    vintage: str | None = None,
    *,
    force: bool = False,
    cache_root: str | Path | None = None,
) -> RawLoadResult:
    ...


def load_fred_qd(
    vintage: str | None = None,
    *,
    force: bool = False,
    cache_root: str | Path | None = None,
) -> RawLoadResult:
    ...


def load_fred_sd(
    vintage: str | None = None,
    *,
    force: bool = False,
    cache_root: str | Path | None = None,
) -> RawLoadResult:
    ...
```

Return type is `RawLoadResult`, not bare DataFrame, because provenance is mandatory.

Usage:

```python
result = load_fred_md(vintage="2020-01")
df = result.data
meta = result.dataset_metadata
artifact = result.artifact
```

### Manager-facing generic loader

```python
def load_raw_dataset(
    dataset: str,
    vintage: str | None = None,
    *,
    force: bool = False,
    cache_root: str | Path | None = None,
) -> RawLoadResult:
    ...
```

### Vintage enumeration

```python
def list_vintages(
    dataset: str,
    start: str | None = None,
    end: str | None = None,
) -> list[str]:
    ...
```

Important: `list_vintages()` should mean expected candidate vintages, not guaranteed remote existence. If remote verification is needed later, add a separate function:

```python
def verify_vintage_availability(
    dataset: str,
    vintage: str,
) -> bool:
    ...
```

## Data contracts

### `RawVersionRequest`

```python
@dataclass(frozen=True)
class RawVersionRequest:
    dataset: str
    mode: Literal["current", "vintage"]
    vintage: str | None
```

Rules:
- `mode="current"` implies `vintage is None`
- `mode="vintage"` implies `vintage is not None`

### `RawDatasetMetadata`

```python
@dataclass(frozen=True)
class RawDatasetMetadata:
    dataset: str
    source_family: str
    frequency: str
    version_mode: Literal["current", "vintage"]
    vintage: str | None
    data_through: str | None
    support_tier: Literal["stable", "provisional"]
    parse_notes: tuple[str, ...] = ()
```

### `RawArtifactRecord`

```python
@dataclass(frozen=True)
class RawArtifactRecord:
    dataset: str
    version_mode: Literal["current", "vintage"]
    vintage: str | None
    source_url: str
    local_path: str
    file_format: Literal["csv", "xlsx"]
    downloaded_at: str
    file_sha256: str
    file_size_bytes: int
    cache_hit: bool
    manifest_version: str
```

### `RawLoadResult`

```python
@dataclass(frozen=True)
class RawLoadResult:
    data: pd.DataFrame
    dataset_metadata: RawDatasetMetadata
    artifact: RawArtifactRecord
```

## Cache layout

Use deterministic, source-transparent local paths.

```text
~/.macrocast/raw/
  manifest/
    raw_artifacts.jsonl
  fred_md/
    current/
      raw.csv
    vintages/
      1999-01.csv
      1999-02.csv
      ...
  fred_qd/
    current/
      raw.csv
    vintages/
      2005-01.csv
      ...
  fred_sd/
    current/
      raw.xlsx
    vintages/
      2005-01.xlsx
      ...
```

Rules:
- `current` is never stored as a fake vintage string.
- explicit vintages are stored under `vintages/<YYYY-MM>.<ext>`.
- manifest lives in one shared location, not inside each dataset directory.
- parsed DataFrame artifacts are not cached separately in v1 unless needed; the raw file is the canonical artifact.

## Manifest contract

Recommended format: JSONL
Reason:
- append-friendly
- grep-friendly
- robust for audit logs

Each entry should include at least:
- `dataset`
- `version_mode`
- `vintage`
- `source_url`
- `local_path`
- `downloaded_at`
- `file_sha256`
- `file_size_bytes`
- `cache_hit`
- `data_through`
- `parse_status`
- `support_tier`

Example entry:

```json
{"dataset":"fred_md","version_mode":"vintage","vintage":"2020-01","source_url":"https://.../2020-01.csv","local_path":"/home/user/.macrocast/raw/fred_md/vintages/2020-01.csv","downloaded_at":"2026-04-14T17:10:00+09:00","file_sha256":"...","file_size_bytes":123456,"cache_hit":false,"data_through":"2019-12","parse_status":"ok","support_tier":"stable"}
```

## Dataset policies

### FRED-MD

Support tier:
- stable

Raw format:
- CSV

Version policy:
- `current` supported
- explicit `vintage=YYYY-MM` supported

Special handling:
- historical ZIP fallback allowed if direct vintage URL fails
- monthly index normalization must be deterministic

Expected output:
- one wide monthly DataFrame
- variable columns named by FRED mnemonic
- pandas `DatetimeIndex`

### FRED-QD

Support tier:
- stable

Raw format:
- CSV

Version policy:
- `current` supported
- explicit `vintage=YYYY-MM` supported

Expected output:
- one wide quarterly DataFrame
- variable columns named by FRED mnemonic
- pandas `DatetimeIndex`

### FRED-SD

Support tier:
- provisional until live vintage verification passes

Raw format:
- Excel workbook

Version policy:
- `current` supported
- explicit `vintage=YYYY-MM` allowed in API only if marked provisional

Expected output:
- one wide DataFrame
- column naming convention fixed as `{variable}_{state}`
- pandas `DatetimeIndex`

Required warning:
- if `vintage` is requested for `fred_sd`, metadata must record provisional support until remote verification is complete

## Loader behavior rules

1. Every public loader calls the shared manager.
2. Every load returns both DataFrame and provenance metadata.
3. Parsing must happen after raw artifact registration inputs are known.
4. Hash is computed on the raw file, not the parsed DataFrame.
5. A cache hit still emits a manifest entry or refresh record if policy requires audit continuity.
6. No preprocessing is applied at raw-load time.
7. No target filtering or feature selection is applied at raw-load time in v1.

## Error semantics

Different failure classes must be explicit.

- `RawVersionFormatError`
  - bad vintage string format
- `RawVersionUnavailableError`
  - version requested but remote source unavailable
- `RawDownloadError`
  - network or HTTP failure
- `RawParseError`
  - file downloaded but could not be parsed
- `RawManifestError`
  - provenance write/read failure

Do not collapse all failures into `ValueError`.

## Implementation order

### Phase 1: contracts and cache
- create `types.py`
- create `contracts.py`
- create `cache.py`
- create `manifest.py`

### Phase 2: manager
- create `manager.py`
- create dataset registry mapping from dataset id to adapter module
- implement `list_vintages()`
- implement request normalization and cache path resolution

### Phase 3: stable datasets first
- implement `datasets/fred_md.py`
- implement `datasets/fred_qd.py`
- wire public loader exports

### Phase 4: provisional SD
- implement `datasets/fred_sd.py`
- mark support tier provisional
- add explicit tests for current-release path first

### Phase 5: test and verify
- cache-path tests
- manifest write/read tests
- MD current and vintage tests
- QD current and vintage tests
- SD current tests
- SD vintage verification tests once source behavior is confirmed

## Minimum test matrix

### Unit
- version normalization
- cache path construction
- manifest append/read
- hash calculation
- date-index normalization

### Dataset-specific
- FRED-MD current parse
- FRED-MD vintage parse
- FRED-QD current parse
- FRED-QD vintage parse
- FRED-SD current workbook parse

### Policy
- `current` does not masquerade as a vintage file
- unsupported or malformed vintage strings fail closed
- provisional support flag appears for SD vintage requests

## Immediate build recommendation

For the reboot branch, restore or rewrite only the raw-data layer first.

The first reusable subset should be:
- MD loader
- QD loader
- shared vintage manager
- cache path rules
- raw manifest/index

Do not restore forecasting, preprocessing, or docs bulk before this layer is stable.

## First concrete deliverables

1. `docs/planning/raw-data-architecture.md`
2. `docs/planning/raw-data-contracts.md`
3. `macrocast/raw/` skeleton
4. test skeleton for raw manager and MD/QD loaders

## Decision summary

The new package should not expose raw acquisition as a few ad hoc helper functions.
It should expose a small, explicit raw-data subsystem:
- loader APIs for each dataset
- one vintage manager
- one manifest/index contract
- pandas-first outputs with mandatory provenance
