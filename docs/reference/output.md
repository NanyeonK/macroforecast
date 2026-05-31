# macroforecast.output

[Back to reference](index.md)

`macroforecast.output` writes callable-stage artifacts and a schema-aware
provenance manifest. It replaces the old output stage with direct Python
functions. It does not decide which models, windows, metrics, or tests belong
in a study.

## write_artifacts

```python
macroforecast.output.write_artifacts(
    artifacts,
    output_dir,
    *,
    formats=("json", "csv"),
    manifest_format="json",
    include_provenance=True,
) -> ArtifactManifest
```

### Input

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `artifacts` | `ForecastResult`, `DataFrame`, or mapping | required | Object or named objects to write. |
| `output_dir` | path-like | required | Output directory. Created if missing. |
| `formats` | tuple | `("json", "csv")` | DataFrame export formats: `"json"`, `"csv"`, `"parquet"`, or `"markdown"`. |
| `manifest_format` | str | `"json"` | Manifest format: `"json"`, `"csv"`, or `"parquet"`. |
| `include_provenance` | bool | `True` | Include package, Python, platform, and git provenance. |

### Output

Returns `ArtifactManifest(output_dir, artifacts, records, provenance)`.

`ArtifactManifest.to_dict()` returns:

| Field | Meaning |
| --- | --- |
| `metadata_schema` | `{"kind": "artifact_manifest", "version": 1}`. |
| `created_at` | UTC ISO timestamp for the write operation. |
| `output_dir` | Directory where artifacts were written. |
| `artifacts` | Backward-compatible mapping from file name to path. |
| `records` | List of `ArtifactRecord` dictionaries, one per written study artifact. |
| `provenance` | Package, Python, platform, git, and package-version provenance. |

Each `ArtifactRecord` has:

| Field | Meaning |
| --- | --- |
| `name` | File name, such as `metrics.csv`. |
| `path` | Written path. |
| `kind` | Logical artifact kind: `forecast_result`, `forecast_table`, `dataframe`, or `json`. |
| `format` | Physical format: `json`, `csv`, `parquet`, or `markdown`. |
| `source` | Input mapping key that produced the file. |
| `metadata` | Shape, columns, object type, and any recorded metadata schema. |

`ForecastResult` writes:

| File | Meaning |
| --- | --- |
| `forecast_result.json` | Forecast rows and runner metadata. |
| `forecast_result_forecasts.csv` | Forecast table. |
| `manifest.json` | Artifact paths and provenance. |

If the forecast table contains runner-created `stored_model` dictionaries,
`write_artifacts()` also records those model artifacts in the manifest. It does
not copy or rewrite the model files. It records the existing paths so the
forecast table, model pickle, and model sidecar can be audited together.

Stored-model records use:

| `kind` | `format` | Path source |
| --- | --- | --- |
| `stored_model_pickle` | `pickle` | `stored_model["model_path"]` when present. |
| `stored_model_metadata` | `json` | `stored_model["metadata_path"]` when present. |

The record metadata includes the model alias, model spec, origin, horizon,
`save_error`, the original `stored_model` dictionary, and `path_exists`.

For a `DataFrame`, JSON export is not a bare list of records. It preserves
metadata:

```python
{
    "metadata_schema": {"kind": "dataframe_artifact", "version": 1},
    "shape": [n_rows, n_columns],
    "columns": [...],
    "index": [...],
    "attrs": frame.attrs,
    "data": [...]
}
```

This is important for outputs from `metrics`, `tests`, `feature_engineering`,
and `interpretation`, where `attrs["macroforecast_metadata_schema"]` records
the producing function and schema version. CSV and markdown are human-readable
views; the manifest still records the DataFrame attrs so metadata is not lost
silently.

### Custom artifacts

Custom outputs do not need a separate registry. Pass a mapping of names to
objects:

```python
mf.output.write_artifacts(
    {
        "forecast_result": result,
        "custom_diagnostic": diagnostic_table,
        "run_notes": {"design": "local robustness check", "accepted": True},
    },
    "results/my_run",
)
```

| Input object | Written as | Metadata behavior |
| --- | --- | --- |
| `DataFrame` | One file per requested `formats` entry. | `attrs` are stored in JSON and in the manifest record. |
| `ForecastResult` | Forecast JSON plus forecast CSV. | Runner metadata and stored-model sidecars are recorded. |
| Mapping/list/scalar | JSON. | Manifest records object type, mapping keys, or sequence length. |

This is the preferred path for project-local custom diagnostics,
interpretation tables, robustness notes, and manually curated metadata that do
not belong in a model, feature, or evaluation contract.

## collect_provenance

```python
macroforecast.output.collect_provenance(cwd=None) -> dict
```

Returns a dictionary containing `macroforecast_version`, Python version,
Python executable, current working directory, platform string,
git commit/branch/dirty flag, and core package versions.

## ArtifactManifest

```python
manifest.to_dict() -> dict
manifest.to_frame() -> pandas.DataFrame
manifest.to_json(path=None, indent=2) -> str
```

Use `to_frame()` when reviewing output in a notebook and `to_json()` when
persisting a manifest object produced elsewhere.

## Example

```python
result = mf.forecasting.run(panel, "ridge", features=features)
manifest = mf.output.write_artifacts(result, "results/ridge_run")

metrics = result.evaluate(metrics=["rmse", "mae"])
mf.output.write_artifacts(
    {"forecasts": result, "metrics": metrics},
    "results/ridge_run",
    formats=("json", "csv"),
)
```
