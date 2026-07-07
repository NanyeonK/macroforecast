# Custom Output

[Back to custom extensions](index.md)

This page is generated from the live callable signatures.

## Callable Reference

### select_outputs

Qualified name: `macroforecast.output.core.select_outputs`

#### Signature

```python
macroforecast.output.select_outputs(bundle: OutputBundle | Mapping[str, Any], *, objects: tuple[str, ...] | list[str]) -> OutputBundle
```

#### Description

Select named outputs from a bundle or mapping.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `bundle` | positional or keyword | `OutputBundle \| Mapping[str, Any]` | `required` |
| `objects` | keyword only | `tuple[str, ...] \| list[str]` | `required` |

#### Returns

`OutputBundle`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.output.select_outputs(...)
```

### write_artifacts

Qualified name: `macroforecast.output.core.write_artifacts`

#### Signature

```python
macroforecast.output.write_artifacts(artifacts: Mapping[str, Any] | ForecastResult | pd.DataFrame | OutputBundle, output_dir: str | Path, *, formats: tuple[ExportFormat, ...] = ('json', 'csv'), manifest_format: ManifestFormat = "json", include_provenance: bool = True, provenance_fields: tuple[str, ...] | None = None, compression: CompressionFormat = "none", layout: ArtifactLayout = "flat") -> ArtifactManifest
```

#### Description

Write forecast/package artifacts and a reproducibility manifest.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `artifacts` | positional or keyword | `Mapping[str, Any] \| ForecastResult \| pd.DataFrame \| OutputBundle` | `required` |
| `output_dir` | positional or keyword | `str \| Path` | `required` |
| `formats` | keyword only | `tuple[ExportFormat, ...]` | `("json", "csv")` |
| `manifest_format` | keyword only | `ManifestFormat` | `"json"` |
| `include_provenance` | keyword only | `bool` | `True` |
| `provenance_fields` | keyword only | `tuple[str, ...] \| None` | `None` |
| `compression` | keyword only | `CompressionFormat` | `"none"` |
| `layout` | keyword only | `ArtifactLayout` | `"flat"` |

#### Returns

`ArtifactManifest`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.output.write_artifacts(...)
```

### report_table

Qualified name: `macroforecast.reporting.core.report_table`

#### Signature

```python
macroforecast.reporting.report_table(table: Any, *, columns: Sequence[str] | None = None, rename: Mapping[str, str] | None = None, sort_by: str | Sequence[str] | None = None, ascending: bool | Sequence[bool] = True, index: bool = False, precision: int = 3, percent_columns: Sequence[str] = (), missing: str = "", caption: str | None = None, label: str | None = None, notes: Sequence[str] = (), metadata: Mapping[str, Any] | None = None) -> ReportTable
```

#### Description

Return a presentation-ready table without writing files.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `table` | positional or keyword | `Any` | `required` |
| `columns` | keyword only | `Sequence[str] \| None` | `None` |
| `rename` | keyword only | `Mapping[str, str] \| None` | `None` |
| `sort_by` | keyword only | `str \| Sequence[str] \| None` | `None` |
| `ascending` | keyword only | `bool \| Sequence[bool]` | `True` |
| `index` | keyword only | `bool` | `False` |
| `precision` | keyword only | `int` | `3` |
| `percent_columns` | keyword only | `Sequence[str]` | `()` |
| `missing` | keyword only | `str` | `""` |
| `caption` | keyword only | `str \| None` | `None` |
| `label` | keyword only | `str \| None` | `None` |
| `notes` | keyword only | `Sequence[str]` | `()` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |

#### Returns

`ReportTable`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.reporting.report_table(...)
```

### render_tables

Qualified name: `macroforecast.reporting.core.render_tables`

#### Signature

```python
macroforecast.reporting.render_tables(value: ReportBundle | Mapping[str, ReportTable | Any], *, format: RenderFormat = "latex") -> dict[str, str]
```

#### Description

Render all tables in a bundle or mapping.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `value` | positional or keyword | `ReportBundle \| Mapping[str, ReportTable \| Any]` | `required` |
| `format` | keyword only | `RenderFormat` | `"latex"` |

#### Returns

`dict[str, str]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.reporting.render_tables(...)
```
