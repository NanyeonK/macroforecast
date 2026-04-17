# Macrocast Raw Data Contracts

Status: reboot-stage contract draft
Date: 2026-04-14
Depends on: `docs/planning/raw-data-architecture.md`

## Goal

Define the minimum contracts the raw-data subsystem must satisfy before any forecasting, preprocessing, or evaluation code is rebuilt.

The raw subsystem must make three things explicit:
- what version was requested
- what raw artifact was used
- what pandas DataFrame was produced

## Contract registry

### 1. `RawVersionRequest`

Purpose:
- normalized request identity for current vs explicit vintage access

```python
@dataclass(frozen=True)
class RawVersionRequest:
    dataset: str
    mode: Literal["current", "vintage"]
    vintage: str | None
```

Rules:
- `dataset` must be one of `fred_md`, `fred_qd`, `fred_sd`
- `mode="current"` requires `vintage is None`
- `mode="vintage"` requires `vintage` in `YYYY-MM`
- request normalization happens before any cache or download logic

Examples:
- `RawVersionRequest(dataset="fred_md", mode="current", vintage=None)`
- `RawVersionRequest(dataset="fred_qd", mode="vintage", vintage="2020-01")`

### 2. `RawDatasetMetadata`

Purpose:
- semantic metadata about the parsed dataset, not the file artifact itself

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

Required meaning:
- `dataset`: internal dataset id
- `source_family`: source lineage, e.g. `fred-md` or `fred-sd`
- `frequency`: `monthly`, `quarterly`, or `state_monthly`
- `version_mode`: whether this came from `current` or an explicit vintage
- `data_through`: last observation date present in the parsed data
- `support_tier`: stability of this dataset/version path
- `parse_notes`: structured warnings or caveats

### 3. `RawArtifactRecord`

Purpose:
- provenance record for the raw file used to produce the DataFrame

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

Rules:
- hash is computed on raw bytes
- `source_url` must be the actual resolved URL used
- `local_path` must be deterministic from dataset + version identity
- `cache_hit=True` does not eliminate the need for provenance recording

### 4. `RawLoadResult`

Purpose:
- single return object from any public loader

```python
@dataclass(frozen=True)
class RawLoadResult:
    data: pd.DataFrame
    dataset_metadata: RawDatasetMetadata
    artifact: RawArtifactRecord
```

Rules:
- public loaders return `RawLoadResult`, not only `pd.DataFrame`
- `data` must have a pandas `DatetimeIndex`
- `data` must be raw-normalized only; no modeling transforms allowed

## Manager contract

### `normalize_version_request()`

Input:
- dataset id
- optional vintage string

Output:
- validated `RawVersionRequest`

Behavior:
- validate dataset id
- validate vintage format if supplied
- emit `mode=current` when no vintage is given
- emit `mode=vintage` when vintage is given

### `list_vintages()`

Input:
- dataset id
- optional start/end

Output:
- ordered list of candidate `YYYY-MM` strings

Behavior:
- enumerates candidate calendar vintages
- does not promise remote existence
- does not download anything

### `load_raw_dataset()`

Input:
- dataset id
- optional vintage
- force flag
- optional cache root

Output:
- `RawLoadResult`

Behavior:
1. normalize request
2. resolve cache path
3. fetch if needed
4. parse raw file with dataset adapter
5. compute artifact record
6. write/append manifest entry
7. return DataFrame + metadata + artifact

## Dataset adapter contract

Each dataset adapter must implement the same minimal internal surface.

### Required adapter functions

```python
def current_url() -> str:
    ...


def resolve_vintage_url(vintage: str) -> str:
    ...


def raw_filename(request: RawVersionRequest) -> str:
    ...


def file_format() -> Literal["csv", "xlsx"]:
    ...


def parse_raw_file(local_path: Path, request: RawVersionRequest) -> tuple[pd.DataFrame, RawDatasetMetadata]:
    ...
```

### Adapter rules

- adapter handles source-specific URL and parse logic only
- adapter does not choose cache roots or manifest paths
- adapter does not write manifests directly
- adapter may emit dataset-specific parse notes

## DataFrame output contract

All raw DataFrames must satisfy:
- pandas `DatetimeIndex`
- sorted ascending index
- unique column names
- numeric values where source semantics imply numeric data
- no package-level preprocessing side effects

Dataset-specific shape rules:
- FRED-MD: wide monthly macro panel
- FRED-QD: wide quarterly macro panel
- FRED-SD: wide panel with `{variable}_{state}` columns

## Manifest entry contract

Minimum JSONL fields:

```json
{
  "dataset": "fred_md",
  "version_mode": "vintage",
  "vintage": "2020-01",
  "source_url": "https://...",
  "local_path": "/home/user/.macrocast/raw/fred_md/vintages/2020-01.csv",
  "downloaded_at": "2026-04-14T17:10:00+09:00",
  "file_sha256": "...",
  "file_size_bytes": 123456,
  "cache_hit": false,
  "data_through": "2019-12",
  "parse_status": "ok",
  "support_tier": "stable"
}
```

Additional optional fields:
- `notes`
- `http_status`
- `resolved_from`
- `fallback_used`
- `manifest_version`

## Fail-closed rules

1. malformed vintage string -> fail before download
2. unavailable vintage -> raise explicit version-unavailable error
3. downloaded-but-unparseable artifact -> preserve raw file and raise parse error
4. manifest write failure -> fail loudly; do not silently continue as if provenance exists
5. SD vintage path -> if not verified, metadata must expose `support_tier="provisional"`

## Explicit non-goals

These do not belong in raw-data contracts:
- transformation codes application
- missing-data policy
- feature engineering
- target construction
- train/test split design
- benchmark selection

## Acceptance criteria for raw layer v1

The raw layer is considered ready only when all are true:
- MD current and vintage loads return `RawLoadResult`
- QD current and vintage loads return `RawLoadResult`
- manifest entries are written deterministically
- cache paths are deterministic and human-readable
- all public loaders return pandas-first outputs with provenance
- SD current path works
- SD vintage path is either verified or explicitly marked provisional
