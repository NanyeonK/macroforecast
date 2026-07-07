# macroforecast.output

[Back to reference](index.md)

Artifact manifests, output bundles, provenance collection, and table/record builders.

## Public Symbols

| Symbol | Kind | Summary |
| --- | --- | --- |
| `ArtifactManifest` | class | Manifest returned by ``write_artifacts``. |
| `ArtifactRecord` | class | One written artifact in a manifest. |
| `ArtifactLayout` | callable | No public docstring is available. |
| `CompressionFormat` | callable | No public docstring is available. |
| `OutputBundle` | class | Named output objects produced before artifact writing. |
| `anatomy_tables` | function | Return anatomy sidecar tables for output bundling or artifact writing. |
| `artifact_index` | function | Return an index table for a manifest, output bundle, or artifact mapping. |
| `bundle_outputs` | function | Build a named bundle of output tables and JSON-ready summaries. |
| `collect_provenance` | function | Collect lightweight package, Python, platform, and git provenance. |
| `forecast_shapley_tables` | function | Return oShapley/PBSV sidecar tables for output writing. |
| `forecast_table` | function | Return the standard forecast table output. |
| `interpretation_outputs` | function | Build output-ready artifacts from one or more interpretation results. |
| `interpretation_table` | function | Return a standardized interpretation output table. |
| `metadata_table` | function | Flatten metadata from a result, report, bundle, mapping, or object. |
| `metric_table` | function | Return the main metric score table from an evaluation output. |
| `model_selection_table` | function | Return model-selection trial or metadata output. |
| `model_table` | function | Return a compact table of fitted model metadata or stored model paths. |
| `name_outputs` | function | Rename output objects before writing artifacts. |
| `ranking_table` | function | Return a ranking table from an evaluation output. |
| `run_summary` | function | Return a compact JSON output summary for a study run. |
| `select_outputs` | function | Select named outputs from a bundle or mapping. |
| `test_table` | function | Return a flat table from one or more forecast test results. |
| `write_artifacts` | function | Write forecast/package artifacts and a reproducibility manifest. |

## Callable And Class Reference

### ArtifactManifest

Qualified name: `macroforecast.output.core.ArtifactManifest`

#### Signature

```python
macroforecast.output.ArtifactManifest(output_dir: str, artifacts: dict[str, str] = <factory>, records: list[ArtifactRecord] = <factory>, provenance: dict[str, Any] = <factory>, created_at: str = <factory>, metadata_schema: dict[str, Any] = <factory>) -> None
```

#### Description

Manifest returned by ``write_artifacts``.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `output_dir` | positional or keyword | `str` | `required` |
| `artifacts` | positional or keyword | `dict[str, str]` | `<factory>` |
| `records` | positional or keyword | `list[ArtifactRecord]` | `<factory>` |
| `provenance` | positional or keyword | `dict[str, Any]` | `<factory>` |
| `created_at` | positional or keyword | `str` | `<factory>` |
| `metadata_schema` | positional or keyword | `dict[str, Any]` | `<factory>` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.output.ArtifactManifest(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `output_dir` | `str` | `required` |
| `artifacts` | `dict[str, str]` | `default_factory` |
| `records` | `list[ArtifactRecord]` | `default_factory` |
| `provenance` | `dict[str, Any]` | `default_factory` |
| `created_at` | `str` | `default_factory` |
| `metadata_schema` | `dict[str, Any]` | `default_factory` |

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `to_dict` | `to_dict(self) -> dict[str, Any]` | No public docstring is available. |
| `to_frame` | `to_frame(self) -> pd.DataFrame` | Return artifact records as a table. |
| `to_json` | `to_json(self, path: str \| Path \| None = None, *, indent: int \| None = 2) -> str` | Return JSON text, and optionally write it to ``path``. |
### ArtifactRecord

Qualified name: `macroforecast.output.core.ArtifactRecord`

#### Signature

```python
macroforecast.output.ArtifactRecord(name: str, path: str, kind: str, format: str, source: str, metadata: dict[str, Any] = <factory>) -> None
```

#### Description

One written artifact in a manifest.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `name` | positional or keyword | `str` | `required` |
| `path` | positional or keyword | `str` | `required` |
| `kind` | positional or keyword | `str` | `required` |
| `format` | positional or keyword | `str` | `required` |
| `source` | positional or keyword | `str` | `required` |
| `metadata` | positional or keyword | `dict[str, Any]` | `<factory>` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.output.ArtifactRecord(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `name` | `str` | `required` |
| `path` | `str` | `required` |
| `kind` | `str` | `required` |
| `format` | `str` | `required` |
| `source` | `str` | `required` |
| `metadata` | `dict[str, Any]` | `default_factory` |

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `to_dict` | `to_dict(self) -> dict[str, Any]` | No public docstring is available. |
### ArtifactLayout

Qualified name: `typing.Literal`

#### Signature

```python
macroforecast.output.ArtifactLayout(*args, **kwargs)
```

#### Description

No public docstring is available.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `args` | var positional | `unspecified` | `required` |
| `kwargs` | var keyword | `unspecified` | `required` |

#### Returns

See the description and object-specific contract.

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.output.ArtifactLayout(...)
```
### CompressionFormat

Qualified name: `typing.Literal`

#### Signature

```python
macroforecast.output.CompressionFormat(*args, **kwargs)
```

#### Description

No public docstring is available.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `args` | var positional | `unspecified` | `required` |
| `kwargs` | var keyword | `unspecified` | `required` |

#### Returns

See the description and object-specific contract.

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.output.CompressionFormat(...)
```
### OutputBundle

Qualified name: `macroforecast.output.core.OutputBundle`

#### Signature

```python
macroforecast.output.OutputBundle(artifacts: dict[str, Any] = <factory>, metadata: dict[str, Any] = <factory>, metadata_schema: dict[str, Any] = <factory>) -> None
```

#### Description

Named output objects produced before artifact writing.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `artifacts` | positional or keyword | `dict[str, Any]` | `<factory>` |
| `metadata` | positional or keyword | `dict[str, Any]` | `<factory>` |
| `metadata_schema` | positional or keyword | `dict[str, Any]` | `<factory>` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.output.OutputBundle(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `artifacts` | `dict[str, Any]` | `default_factory` |
| `metadata` | `dict[str, Any]` | `default_factory` |
| `metadata_schema` | `dict[str, Any]` | `default_factory` |

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `select` | `select(self, objects: tuple[str, ...] \| list[str]) -> "'OutputBundle'"` | Return a bundle with selected artifact names. |
| `to_artifacts` | `to_artifacts(self) -> dict[str, Any]` | Return a shallow copy suitable for ``write_artifacts``. |
| `to_dict` | `to_dict(self) -> dict[str, Any]` | Return a JSON-ready description of the bundle. |
### anatomy_tables

Qualified name: `macroforecast.output.core.anatomy_tables`

#### Signature

```python
macroforecast.output.anatomy_tables(value: Any, *, prefix: str = "anatomy") -> dict[str, pd.DataFrame]
```

#### Description

Return anatomy sidecar tables for output bundling or artifact writing.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `value` | positional or keyword | `Any` | `required` |
| `prefix` | keyword only | `str` | `"anatomy"` |

#### Returns

`dict[str, pd.DataFrame]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.output.anatomy_tables(...)
```
### artifact_index

Qualified name: `macroforecast.output.core.artifact_index`

#### Signature

```python
macroforecast.output.artifact_index(value: Any) -> pd.DataFrame
```

#### Description

Return an index table for a manifest, output bundle, or artifact mapping.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `value` | positional or keyword | `Any` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.output.artifact_index(...)
```
### bundle_outputs

Qualified name: `macroforecast.output.core.bundle_outputs`

#### Signature

```python
macroforecast.output.bundle_outputs(*, forecasts: ForecastResult | pd.DataFrame | None = None, evaluation: Any | None = None, tests: Any | None = None, models: Any | None = None, model_selection: Any | None = None, interpretation: Mapping[str, Any] | pd.DataFrame | None = None, metadata: Mapping[str, Any] | None = None, include_summary: bool = True, extra: Mapping[str, Any] | None = None) -> OutputBundle
```

#### Description

Build a named bundle of output tables and JSON-ready summaries.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `forecasts` | keyword only | `ForecastResult \| pd.DataFrame \| None` | `None` |
| `evaluation` | keyword only | `Any \| None` | `None` |
| `tests` | keyword only | `Any \| None` | `None` |
| `models` | keyword only | `Any \| None` | `None` |
| `model_selection` | keyword only | `Any \| None` | `None` |
| `interpretation` | keyword only | `Mapping[str, Any] \| pd.DataFrame \| None` | `None` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `include_summary` | keyword only | `bool` | `True` |
| `extra` | keyword only | `Mapping[str, Any] \| None` | `None` |

#### Returns

`OutputBundle`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.output.bundle_outputs(...)
```
### collect_provenance

Qualified name: `macroforecast.output.core.collect_provenance`

#### Signature

```python
macroforecast.output.collect_provenance(*, cwd: str | Path | None = None, fields: tuple[str, ...] | None = None) -> dict[str, Any]
```

#### Description

Collect lightweight package, Python, platform, and git provenance.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `cwd` | keyword only | `str \| Path \| None` | `None` |
| `fields` | keyword only | `tuple[str, ...] \| None` | `None` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.output.collect_provenance(...)
```
### forecast_shapley_tables

Qualified name: `macroforecast.output.core.forecast_shapley_tables`

#### Signature

```python
macroforecast.output.forecast_shapley_tables(value: Any, *, prefix: str = "oshapley") -> dict[str, pd.DataFrame]
```

#### Description

Return oShapley/PBSV sidecar tables for output writing.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `value` | positional or keyword | `Any` | `required` |
| `prefix` | keyword only | `str` | `"oshapley"` |

#### Returns

`dict[str, pd.DataFrame]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.output.forecast_shapley_tables(...)
```
### forecast_table

Qualified name: `macroforecast.output.core.forecast_table`

#### Signature

```python
macroforecast.output.forecast_table(result: ForecastResult | pd.DataFrame) -> pd.DataFrame
```

#### Description

Return the standard forecast table output.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `result` | positional or keyword | `ForecastResult \| pd.DataFrame` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.output.forecast_table(...)
```
### interpretation_outputs

Qualified name: `macroforecast.output.core.interpretation_outputs`

#### Signature

```python
macroforecast.output.interpretation_outputs(interpretation: Mapping[str, Any] | pd.DataFrame | Any, *, prefix: str = "interpretation") -> OutputBundle
```

#### Description

Build output-ready artifacts from one or more interpretation results.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `interpretation` | positional or keyword | `Mapping[str, Any] \| pd.DataFrame \| Any` | `required` |
| `prefix` | keyword only | `str` | `"interpretation"` |

#### Returns

`OutputBundle`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.output.interpretation_outputs(...)
```
### interpretation_table

Qualified name: `macroforecast.output.core.interpretation_table`

#### Signature

```python
macroforecast.output.interpretation_table(value: Any) -> pd.DataFrame
```

#### Description

Return a standardized interpretation output table.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `value` | positional or keyword | `Any` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.output.interpretation_table(...)
```
### metadata_table

Qualified name: `macroforecast.output.core.metadata_table`

#### Signature

```python
macroforecast.output.metadata_table(value: Any, *, prefix: str = "") -> pd.DataFrame
```

#### Description

Flatten metadata from a result, report, bundle, mapping, or object.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `value` | positional or keyword | `Any` | `required` |
| `prefix` | keyword only | `str` | `""` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.output.metadata_table(...)
```
### metric_table

Qualified name: `macroforecast.output.core.metric_table`

#### Signature

```python
macroforecast.output.metric_table(report: Any) -> pd.DataFrame
```

#### Description

Return the main metric score table from an evaluation output.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `report` | positional or keyword | `Any` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.output.metric_table(...)
```
### model_selection_table

Qualified name: `macroforecast.output.core.model_selection_table`

#### Signature

```python
macroforecast.output.model_selection_table(model_selection: Any) -> pd.DataFrame
```

#### Description

Return model-selection trial or metadata output.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `model_selection` | positional or keyword | `Any` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.output.model_selection_table(...)
```
### model_table

Qualified name: `macroforecast.output.core.model_table`

#### Signature

```python
macroforecast.output.model_table(models: Any) -> pd.DataFrame
```

#### Description

Return a compact table of fitted model metadata or stored model paths.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `models` | positional or keyword | `Any` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.output.model_table(...)
```
### name_outputs

Qualified name: `macroforecast.output.core.name_outputs`

#### Signature

```python
macroforecast.output.name_outputs(bundle: OutputBundle | Mapping[str, Any], *, convention: "Literal['identity', 'descriptive', 'kind', 'prefixed']" = "descriptive", prefix: str | None = None) -> OutputBundle
```

#### Description

Rename output objects before writing artifacts.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `bundle` | positional or keyword | `OutputBundle \| Mapping[str, Any]` | `required` |
| `convention` | keyword only | `Literal['identity', 'descriptive', 'kind', 'prefixed']` | `"descriptive"` |
| `prefix` | keyword only | `str \| None` | `None` |

#### Returns

`OutputBundle`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.output.name_outputs(...)
```
### ranking_table

Qualified name: `macroforecast.output.core.ranking_table`

#### Signature

```python
macroforecast.output.ranking_table(report: Any) -> pd.DataFrame
```

#### Description

Return a ranking table from an evaluation output.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `report` | positional or keyword | `Any` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.output.ranking_table(...)
```
### run_summary

Qualified name: `macroforecast.output.core.run_summary`

#### Signature

```python
macroforecast.output.run_summary(result: ForecastResult | pd.DataFrame | None = None, *, evaluation: Any | None = None, tests: Any | None = None, model_selection: Any | None = None, models: Any | None = None, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]
```

#### Description

Return a compact JSON output summary for a study run.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `result` | positional or keyword | `ForecastResult \| pd.DataFrame \| None` | `None` |
| `evaluation` | keyword only | `Any \| None` | `None` |
| `tests` | keyword only | `Any \| None` | `None` |
| `model_selection` | keyword only | `Any \| None` | `None` |
| `models` | keyword only | `Any \| None` | `None` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.output.run_summary(...)
```
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
### test_table

Qualified name: `macroforecast.output.core.test_table`

#### Signature

```python
macroforecast.output.test_table(results: Any) -> pd.DataFrame
```

#### Description

Return a flat table from one or more forecast test results.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `results` | positional or keyword | `Any` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.output.test_table(...)
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
