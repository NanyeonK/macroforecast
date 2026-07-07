# Custom Dataset

[Back to custom extensions](index.md)

This page is generated from the live callable signatures.

## Callable Reference

### custom_dataset

Qualified name: `macroforecast.data.panel.custom_dataset`

#### Signature

```python
macroforecast.data.custom_dataset(frame: pd.DataFrame, *, date: str | None = None, columns: Iterable[str] | None = None, rename: Mapping[str, str] | None = None, dataset: str = "custom", source_family: str = "custom", frequency: str = "unknown", frequency_by_column: Mapping[str, str] | None = None, transform_codes: Mapping[str, int] | None = None, metadata: Mapping[str, Any] | None = None, strict: bool = True) -> DataBundle
```

#### Description

Build a canonical custom ``DataBundle`` from an in-memory DataFrame.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `frame` | positional or keyword | `pd.DataFrame` | `required` |
| `date` | keyword only | `str \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `rename` | keyword only | `Mapping[str, str] \| None` | `None` |
| `dataset` | keyword only | `str` | `"custom"` |
| `source_family` | keyword only | `str` | `"custom"` |
| `frequency` | keyword only | `str` | `"unknown"` |
| `frequency_by_column` | keyword only | `Mapping[str, str] \| None` | `None` |
| `transform_codes` | keyword only | `Mapping[str, int] \| None` | `None` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `strict` | keyword only | `bool` | `True` |

#### Returns

`DataBundle`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data.custom_dataset(...)
```

### load_custom_csv

Qualified name: `macroforecast.data.loaders.load_custom_csv`

#### Signature

```python
macroforecast.data.load_custom_csv(path: str | Path, *, date: str | None = None, date_col: str | int | None = None, columns: Iterable[str] | None = None, series_columns: Iterable[str] | None = None, rename: Mapping[str, str] | None = None, dataset: str = "custom", frequency: str = "unknown", frequency_by_column: Mapping[str, str] | None = None, default_frequency: str | None = None, metadata: Mapping[str, Any] | None = None, transform_codes: Mapping[str, int] | None = None, cache_root: str | Path | None = None, strict: bool = True) -> DataBundle
```

#### Description

No public docstring is available.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `path` | positional or keyword | `str \| Path` | `required` |
| `date` | keyword only | `str \| None` | `None` |
| `date_col` | keyword only | `str \| int \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `series_columns` | keyword only | `Iterable[str] \| None` | `None` |
| `rename` | keyword only | `Mapping[str, str] \| None` | `None` |
| `dataset` | keyword only | `str` | `"custom"` |
| `frequency` | keyword only | `str` | `"unknown"` |
| `frequency_by_column` | keyword only | `Mapping[str, str] \| None` | `None` |
| `default_frequency` | keyword only | `str \| None` | `None` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `transform_codes` | keyword only | `Mapping[str, int] \| None` | `None` |
| `cache_root` | keyword only | `str \| Path \| None` | `None` |
| `strict` | keyword only | `bool` | `True` |

#### Returns

`DataBundle`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data.load_custom_csv(...)
```

### load_custom_parquet

Qualified name: `macroforecast.data.loaders.load_custom_parquet`

#### Signature

```python
macroforecast.data.load_custom_parquet(path: str | Path, *, date: str | None = None, date_col: str | int | None = None, columns: Iterable[str] | None = None, series_columns: Iterable[str] | None = None, rename: Mapping[str, str] | None = None, dataset: str = "custom", frequency: str = "unknown", frequency_by_column: Mapping[str, str] | None = None, default_frequency: str | None = None, metadata: Mapping[str, Any] | None = None, transform_codes: Mapping[str, int] | None = None, cache_root: str | Path | None = None, strict: bool = True) -> DataBundle
```

#### Description

No public docstring is available.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `path` | positional or keyword | `str \| Path` | `required` |
| `date` | keyword only | `str \| None` | `None` |
| `date_col` | keyword only | `str \| int \| None` | `None` |
| `columns` | keyword only | `Iterable[str] \| None` | `None` |
| `series_columns` | keyword only | `Iterable[str] \| None` | `None` |
| `rename` | keyword only | `Mapping[str, str] \| None` | `None` |
| `dataset` | keyword only | `str` | `"custom"` |
| `frequency` | keyword only | `str` | `"unknown"` |
| `frequency_by_column` | keyword only | `Mapping[str, str] \| None` | `None` |
| `default_frequency` | keyword only | `str \| None` | `None` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `transform_codes` | keyword only | `Mapping[str, int] \| None` | `None` |
| `cache_root` | keyword only | `str \| Path \| None` | `None` |
| `strict` | keyword only | `bool` | `True` |

#### Returns

`DataBundle`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data.load_custom_parquet(...)
```

### custom_vintages

Qualified name: `macroforecast.data.vintage.custom_vintages`

#### Signature

```python
macroforecast.data.custom_vintages(source: Callable[[pd.Timestamp], DataBundle | pd.DataFrame] | Mapping[Any, DataBundle | pd.DataFrame] | pd.DataFrame, *, vintage_column: str | None = None, date_column: str | None = None, vintage_id: Callable[[Any], Any] | None = None, dataset: str = "custom_vintages", frequency: str = "unknown", strict: bool = True) -> VintageSource
```

#### Description

Return a custom point-in-time source.

``source`` may be a callable ``origin_date -> DataBundle | DataFrame``, a
mapping from date-like vintage keys to snapshots, or a long ALFRED-style
DataFrame with one row per ``(date, vintage, series...)``. Every snapshot is
normalized through :func:`as_panel` / :func:`custom_dataset` and then
validated. Resolved snapshots are memoized by the stable identifier produced
by ``vintage_id`` (default: ``str(resolved_key)``). If a callable reads from
a non-deterministic source whose content can change for the same identifier,
run the forecast with runner/pipeline preprocessing caching disabled.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `source` | positional or keyword | `Callable[[pd.Timestamp], DataBundle \| pd.DataFrame] \| Mapping[Any, DataBundle \| pd.DataFrame] \| pd.DataFrame` | `required` |
| `vintage_column` | keyword only | `str \| None` | `None` |
| `date_column` | keyword only | `str \| None` | `None` |
| `vintage_id` | keyword only | `Callable[[Any], Any] \| None` | `None` |
| `dataset` | keyword only | `str` | `"custom_vintages"` |
| `frequency` | keyword only | `str` | `"unknown"` |
| `strict` | keyword only | `bool` | `True` |

#### Returns

`VintageSource`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data.custom_vintages(...)
```

### with_static_extras

Qualified name: `macroforecast.data.vintage.with_static_extras`

#### Signature

```python
macroforecast.data.with_static_extras(source: VintageSource, extra: DataBundle | pd.DataFrame, *, join: "Literal['outer', 'inner', 'left']" = "outer") -> VintageSource
```

#### Description

Join non-revised extra columns onto every resolved vintage bundle.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `source` | positional or keyword | `VintageSource` | `required` |
| `extra` | positional or keyword | `DataBundle \| pd.DataFrame` | `required` |
| `join` | keyword only | `Literal['outer', 'inner', 'left']` | `"outer"` |

#### Returns

`VintageSource`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.data.with_static_extras(...)
```
