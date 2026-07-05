# custom_dataset

[Back to custom extensions](index.md)

Use custom data functions when the panel does not come from FRED-MD, FRED-QD,
or FRED-SD. The output must still be a canonical `DataBundle`: a date-indexed
numeric panel plus metadata that later stages can read.

## Function Choices

| Function | Input | Output | Use case |
| --- | --- | --- | --- |
| `mf.data.custom_dataset(...)` | in-memory `DataFrame` | `DataBundle` | User code already loaded the data. |
| `mf.data.load_custom_csv(...)` | CSV path | `DataBundle` | File source is CSV. |
| `mf.data.load_custom_parquet(...)` | Parquet path | `DataBundle` | File source is Parquet. |

## custom_dataset

```python
mf.data.custom_dataset(
    data,
    *,
    date=None,
    columns=None,
    dataset="custom",
    frequency=None,
    transform_codes=None,
    metadata=None,
) -> mf.data.DataBundle
```

### Input

| Name | Type | Meaning |
| --- | --- | --- |
| `data` | `pandas.DataFrame` | User panel. It can contain a date column or already have a `DatetimeIndex`. |
| `date` | str or `None` | Date column to move into the index. Use `None` when the index is already dates. |
| `columns` | sequence or `None` | Optional variable subset after date handling. |
| `dataset` | str | Dataset label stored in metadata. |
| `frequency` | str or mapping or `None` | Panel frequency such as `monthly`, `quarterly`, or per-column frequency metadata. |
| `transform_codes` | mapping or `None` | Optional FRED-style transformation code metadata by column. |
| `metadata` | mapping or `None` | User metadata to merge into the bundle metadata. |

### Output

| Field | Contract |
| --- | --- |
| `bundle.panel` | Numeric `DataFrame`, sorted `DatetimeIndex` named `date`, no duplicate dates. |
| `bundle.metadata["dataset"]` | Dataset name. |
| `bundle.metadata["frequency"]` | Dataset-level or column-level frequency information when supplied. |
| `bundle.metadata["transform_codes"]` | Transform-code metadata when supplied. |
| `bundle.metadata["panel_normalization"]` | Normalization report for date/index/column conversion. |

### Flow

```python
bundle = mf.data.custom_dataset(
    frame,
    date="date",
    dataset="local_macro",
    frequency="monthly",
    transform_codes={"target": 1, "x": 1},
)

processed = mf.preprocessing.reprocess(bundle, transform="none")
```

This page stops at a processed panel. For the rest of the story -- a
`TargetSpec`, a custom model, a scored horse race against a benchmark, and a
one-line paper table -- continue to the full horse-race tutorial:
[Your Data, Your Model, One Table](../../guide/custom_data_tutorial.md).

## File Loaders

```python
mf.data.load_custom_csv(path, *, date, columns=None, dataset="custom", ...)
mf.data.load_custom_parquet(path, *, date, columns=None, dataset="custom", ...)
```

The file loaders normalize the same panel contract as `custom_dataset()`.
Use them when the file path itself should appear in the loader metadata.

## Validation

| Problem | Behavior |
| --- | --- |
| Missing date column | raises loader/normalization error. |
| Duplicate dates | raises unless permissive mode is explicitly requested by the loader. |
| Non-numeric selected variables | coerced or reported by panel normalization. |
| Missing frequency metadata | allowed, but frequency-aware downstream logic may need explicit `set_frequencies(...)`. |
