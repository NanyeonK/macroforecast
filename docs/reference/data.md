# macroforecast.data

`macroforecast.data` loads or normalizes a data panel and then attaches run-level
data choices to that panel. It does not transform, impute, scale, engineer
features, fit models, or evaluate forecasts.

The public flow is:

```python
import macroforecast as mf

bundle = mf.data.load_fred_md()

data_spec = mf.data.spec(
    bundle,
    target="INDPRO",
    horizons=[1, 3, 6, 12],
    start="1960-01",
    end="2024-12",
    predictors="all",
)
```

## Canonical Panel

Every public loader returns a `DataBundle`.

```python
panel = bundle.panel
metadata = bundle.metadata
```

`DataBundle` also supports tuple unpacking:

```python
panel, metadata = mf.data.load_fred_md()
```

### Panel Contract

| Property | Required Value |
| --- | --- |
| Type | `pandas.DataFrame` |
| Index | `pandas.DatetimeIndex` |
| Index name | `"date"` |
| Sort order | ascending date order |
| Duplicate dates | not allowed |
| Columns | variable IDs |
| Values | numeric values or `NaN` |

Metadata is explicit on `DataBundle.metadata`. The panel also carries
`panel.attrs["macroforecast_metadata"]` for pandas-native handoff. FRED-MD and
FRED-QD transform codes are attached to
`panel.attrs["macroforecast_transform_codes"]`; preprocessing is responsible
for using them.

## DataBundle

```python
macroforecast.data.DataBundle(
    panel: pandas.DataFrame,
    metadata: dict,
)
```

### Output

| Field | Type | Meaning |
| --- | --- | --- |
| `panel` | `pandas.DataFrame` | Canonical date-indexed data panel. |
| `metadata` | `dict` | Source, vintage, artifact, frequency, and transform-code metadata. |

### Methods

| Method | Input | Output | Meaning |
| --- | --- | --- | --- |
| `attach(stage, values)` | `stage: str`, `values: Mapping` | `DataBundle` | Return a new bundle with one metadata stage added. |

Preprocessing outputs can use the same metadata-attachment pattern.

## DataSpec

```python
macroforecast.data.DataSpec(
    panel: pandas.DataFrame,
    metadata: dict,
    target: str | None,
    targets: tuple[str, ...],
    horizons: tuple[int, ...],
    start: str | None = None,
    end: str | None = None,
    predictors: "all" | tuple[str, ...] = "all",
)
```

`DataSpec` is the output of `spec(...)`. It keeps the canonical panel and
metadata together with the target, horizons, sample window, and predictor
selection for a run.

## load_fred_md

Load FRED-MD and return `DataBundle`.

```python
macroforecast.data.load_fred_md(
    vintage: str | None = None,
    *,
    force: bool = False,
    cache_root: str | pathlib.Path | None = None,
    local_source: str | pathlib.Path | None = None,
    local_zip_source: str | pathlib.Path | None = None,
) -> DataBundle
```

### Input

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `vintage` | <code>str &#124; None</code> | `None` | Vintage in `YYYY-MM` form. `None` loads current. |
| `force` | `bool` | `False` | Re-download or re-copy even if cache exists. |
| `cache_root` | path-like or `None` | `None` | Raw cache root. |
| `local_source` | path-like or `None` | `None` | Local CSV source instead of download. |
| `local_zip_source` | path-like or `None` | `None` | Local historical zip source for vintage files. |

### Output

Returns `DataBundle` with monthly FRED-MD panel and metadata.

## load_fred_qd

Load FRED-QD and return `DataBundle`.

```python
macroforecast.data.load_fred_qd(
    vintage: str | None = None,
    *,
    force: bool = False,
    cache_root: str | pathlib.Path | None = None,
    local_source: str | pathlib.Path | None = None,
) -> DataBundle
```

Returns a quarterly canonical panel.

## load_fred_sd

Load FRED-SD and return `DataBundle`.

```python
macroforecast.data.load_fred_sd(
    vintage: str | None = None,
    *,
    force: bool = False,
    cache_root: str | pathlib.Path | None = None,
    local_source: str | pathlib.Path | None = None,
    states: list[str] | None = None,
    variables: list[str] | None = None,
) -> DataBundle
```

### Input

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `states` | <code>list[str] &#124; None</code> | `None` | Optional state subset. |
| `variables` | <code>list[str] &#124; None</code> | `None` | Optional FRED-SD variable subset. |

FRED-SD columns are wide variable-state IDs such as `UR_CA`.

## load_custom_csv

Load a user CSV and normalize it to the canonical panel contract.

```python
macroforecast.data.load_custom_csv(
    path,
    *,
    date: str | None = None,
    columns: Iterable[str] | None = None,
    rename: Mapping[str, str] | None = None,
    dataset: str = "custom",
    frequency: str = "unknown",
    metadata: Mapping[str, object] | None = None,
) -> DataBundle
```

### Input

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `path` | path-like | required | CSV file path. |
| `date` | <code>str &#124; None</code> | `None` | Date column. If omitted, uses a DatetimeIndex or parses the first column. |
| `columns` | iterable or `None` | `None` | Columns to keep before renaming. |
| `rename` | mapping or `None` | `None` | Column rename map. |
| `dataset` | `str` | `"custom"` | Metadata dataset label. |
| `frequency` | `str` | `"unknown"` | Metadata frequency label. |
| `metadata` | mapping or `None` | `None` | User metadata to attach. |

## load_custom_parquet

Load a user Parquet file with the same normalization contract as
`load_custom_csv`.

```python
macroforecast.data.load_custom_parquet(
    path,
    *,
    date: str | None = None,
    columns: Iterable[str] | None = None,
    rename: Mapping[str, str] | None = None,
    dataset: str = "custom",
    frequency: str = "unknown",
    metadata: Mapping[str, object] | None = None,
) -> DataBundle
```

## as_panel

Normalize an existing pandas `DataFrame`.

```python
macroforecast.data.as_panel(
    frame,
    *,
    date: str | None = None,
    columns: Iterable[str] | None = None,
    rename: Mapping[str, str] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> pandas.DataFrame
```

`as_panel` returns a canonical panel. It raises if the date column is missing,
dates are duplicated, or any retained column cannot be represented as numeric
values or `NaN`.

## validate_panel

Validate the canonical panel contract.

```python
macroforecast.data.validate_panel(panel) -> None
```

Raises `TypeError` or `ValueError` when the panel is not canonical.

## panel_info

Return a compact panel summary.

```python
macroforecast.data.panel_info(bundle_or_panel) -> dict
```

Output keys include `n_rows`, `n_columns`, `start`, `end`, `columns`,
`missing_values`, and inferred `frequency`.

## metadata

Return explicit metadata from a `DataBundle`, `DataSpec`, `(panel, metadata)`
tuple, `DataFrame`, or advanced raw load result.

```python
macroforecast.data.metadata(obj) -> dict
```

## spec

Attach run-level data choices to a bundle or panel.

```python
macroforecast.data.spec(
    data,
    *,
    metadata: Mapping[str, object] | None = None,
    target: str | None = None,
    targets: Iterable[str] | None = None,
    horizons: Iterable[int] | int | None = None,
    start: str | None = None,
    end: str | None = None,
    predictors: "all" | Iterable[str] = "all",
) -> DataSpec
```

### Input

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `data` | `DataBundle`, `DataSpec`, `(panel, metadata)`, or `DataFrame` | required | Canonical data input. |
| `metadata` | mapping or `None` | `None` | Extra metadata to merge. |
| `target` | <code>str &#124; None</code> | `None` | Single target column. |
| `targets` | iterable or `None` | `None` | Multiple target columns. |
| `horizons` | iterable, int, or `None` | derived | Forecast horizons. |
| `start` | <code>str &#124; None</code> | `None` | Start date. Accepts `YYYY`, `YYYY-MM`, or `YYYY-MM-DD`. |
| `end` | <code>str &#124; None</code> | `None` | End date. Accepts `YYYY`, `YYYY-MM`, or `YYYY-MM-DD`. |
| `predictors` | <code>"all" &#124; iterable</code> | `"all"` | Predictor columns to keep. |

### Default Horizons

| Metadata frequency | Default horizons |
| --- | --- |
| `monthly` | `(1, 3, 6, 12)` |
| `quarterly` | `(1, 2, 4, 8)` |
| other or unknown | `(1,)` |

### Output

Returns `DataSpec`. Its metadata contains a `data_spec` entry with the chosen
target, targets, horizons, sample dates, predictors, and panel summary.

## Advanced Raw Results

The `_result` loaders return `RawLoadResult` envelopes:

```python
result = macroforecast.data.load_fred_md_result()
```

Use these when artifact provenance or raw transform codes are needed before
conversion to `DataBundle`. Most user workflows should use the non-result
loaders.

## Vintage Helpers

`list_vintages(dataset, start, end)` returns candidate vintage labels. The
selected vintage is passed to `load_fred_md`, `load_fred_qd`, or `load_fred_sd`
through `vintage=`.
