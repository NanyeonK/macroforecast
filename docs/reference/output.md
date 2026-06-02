# macroforecast.output

[Back to reference](index.md)

`macroforecast.output` writes callable-stage artifacts and a schema-aware
provenance manifest. It replaces the old output stage with direct Python
functions. It does not decide which models, windows, metrics, or tests belong
in a study.

The output API is split into two parts:

| Part | Functions | Role |
| --- | --- | --- |
| Output generation | `forecast_table`, `metric_table`, `ranking_table`, `test_table`, `model_table`, `model_selection_table`, `interpretation_table`, `metadata_table`, `run_summary`, `bundle_outputs`, `select_outputs`, `name_outputs`, `artifact_index` | Convert package objects into named pandas/JSON outputs. No files are written. |
| Artifact writing | `write_artifacts`, `collect_provenance`, `ArtifactManifest`, `ArtifactRecord` | Write selected outputs to disk and record file metadata. |

This means a workflow can first build all candidate outputs, inspect or rename
them, then write only the objects needed for a paper, replication package, or
notebook appendix.

## Output generation

### forecast_table

```python
macroforecast.output.forecast_table(result) -> pandas.DataFrame
```

| Input | Type | Meaning |
| --- | --- | --- |
| `result` | `ForecastResult` or `DataFrame` | Forecast runner output or an already materialized forecast table. |

Returns a `DataFrame` with
`attrs["macroforecast_metadata_schema"]["kind"] == "forecast_table"`.

### metric_table

```python
macroforecast.output.metric_table(report) -> pandas.DataFrame
```

| Input | Type | Meaning |
| --- | --- | --- |
| `report` | `EvaluationReport`, `ForecastResult`, or `DataFrame` | Evaluation report, forecast result to evaluate with defaults, or metric table. |

Returns the main metric score table. For `EvaluationReport`, this is
`report.scores`.

### ranking_table

```python
macroforecast.output.ranking_table(report) -> pandas.DataFrame
```

| Input | Type | Meaning |
| --- | --- | --- |
| `report` | `EvaluationReport` or `DataFrame` | Evaluation report or a ranking table. |

Returns the model ranking table. For `EvaluationReport`, this is
`report.ranking`.

### test_table

```python
macroforecast.output.test_table(results) -> pandas.DataFrame
```

| Input | Type | Meaning |
| --- | --- | --- |
| `results` | `TestResult`, mapping, sequence, or `DataFrame` | One or more forecast-comparison test outputs. |

Returns a flat raw table with `name`, `statistic`, `p_value`, `decision`,
`alternative`, `correction_policy`, `n_obs`, `metadata`, and any promoted
test metadata fields such as `statistic_type`, `p_value_status`,
`p_value_reference`, `null_hypothesis`, `source_reference`, `r_reference`, and
`r_alignment`.

Use `macroforecast.reporting.test_report_table(...)` for a compact paper table
and `macroforecast.reporting.test_provenance_table(...)` for the source
alignment appendix table.

### model_table

```python
macroforecast.output.model_table(models) -> pandas.DataFrame
```

| Input | Type | Meaning |
| --- | --- | --- |
| `models` | `ForecastResult`, `DataFrame`, `ModelFit`, mapping, sequence, or object | Fitted model metadata or forecast rows containing model columns. |

For forecast results, returns one row per model/model-spec group with forecast
counts, horizon counts, and stored-model counts. For `ModelFit` objects, returns
compact fit metadata without serializing the estimator.

### model_selection_table

```python
macroforecast.output.model_selection_table(model_selection) -> pandas.DataFrame
```

| Input | Type | Meaning |
| --- | --- | --- |
| `model_selection` | `SearchResult`, search-like object, `DataFrame`, or mapping | Model-selection result or trial table. |

Returns the model-selection trial table when available. Best parameter metadata is
stored in the DataFrame attrs.

### interpretation_table

```python
macroforecast.output.interpretation_table(value) -> pandas.DataFrame
```

| Input | Type | Meaning |
| --- | --- | --- |
| `value` | `DataFrame` or mapping | Interpretation output such as importances, SHAP summaries, attribution tables, or custom mappings. |

Returns a schema-tagged interpretation table.

### metadata_table

```python
macroforecast.output.metadata_table(value, *, prefix="") -> pandas.DataFrame
```

| Input | Type | Default | Meaning |
| --- | --- | --- | --- |
| `value` | result/report/bundle/manifest/mapping/object | required | Object whose metadata should be flattened. |
| `prefix` | str | `""` | Optional path prefix. |

Returns a long table with `path`, `value`, and `type`. Nested dictionaries use
dot paths such as `data.source`; sequences use indexed paths such as
`stages[0]`.

### run_summary

```python
macroforecast.output.run_summary(
    result=None,
    *,
    evaluation=None,
    tests=None,
    model_selection=None,
    models=None,
    metadata=None,
) -> dict
```

Returns a JSON-ready summary with `metadata_schema.kind == "run_summary"`.
It records row counts, forecast models, horizons, evaluation rows, test rows,
model-selection best-score fields, model rows, and user metadata when supplied.

### OutputBundle

```python
bundle = macroforecast.output.bundle_outputs(
    forecasts=None,
    evaluation=None,
    tests=None,
    models=None,
    model_selection=None,
    interpretation=None,
    metadata=None,
    include_summary=True,
    extra=None,
)
```

`OutputBundle` is a named in-memory artifact collection. It is the preferred
object to pass from the runner/evaluation/interpretation stages into output
writing.

| Field | Meaning |
| --- | --- |
| `artifacts` | Mapping from output name to `DataFrame`, `ForecastResult`, or JSON-ready object. |
| `metadata` | Bundle-level metadata such as artifact count and selected objects. |
| `metadata_schema` | `{"kind": "output_bundle", "version": 1}`. |

Methods:

| Method | Output |
| --- | --- |
| `to_artifacts()` | Shallow artifact mapping suitable for `write_artifacts`. |
| `select(objects)` | New `OutputBundle` with only selected artifact names. |
| `to_dict()` | JSON-ready bundle description, not the full DataFrame data. |

### select_outputs

```python
macroforecast.output.select_outputs(bundle, *, objects=("forecasts", "metrics")) -> OutputBundle
```

Selects named outputs from an `OutputBundle` or mapping. Missing names raise
`KeyError`. This is how a workflow can generate many outputs and save only the
tables required by a paper section.

### name_outputs

```python
macroforecast.output.name_outputs(
    bundle,
    *,
    convention="descriptive",
    prefix=None,
) -> OutputBundle
```

| `convention` | Name rule |
| --- | --- |
| `"identity"` | Keep the original artifact names. |
| `"descriptive"` | Use `<object_kind>_<original_name>`, optionally prefixed. |
| `"kind"` | Use the object kind only, with suffixes added for uniqueness. |
| `"prefixed"` | Use `<prefix>_<original_name>`, or `output_<original_name>` when no prefix is given. |

### artifact_index

```python
macroforecast.output.artifact_index(value) -> pandas.DataFrame
```

Accepts an `ArtifactManifest`, `OutputBundle`, or artifact mapping. Returns one
row per artifact with name, kind, object type, and metadata schema.

### Example: generate then write

```python
result = mf.forecasting.run(panel, "ridge", features=features)
report = mf.evaluation.evaluate_report(result)

bundle = mf.output.bundle_outputs(
    forecasts=result,
    evaluation=report,
    metadata={"study": "baseline"},
)
paper_outputs = mf.output.select_outputs(
    bundle,
    objects=("forecasts", "metrics", "ranking", "summary"),
)
paper_outputs = mf.output.name_outputs(
    paper_outputs,
    convention="prefixed",
    prefix="baseline",
)

manifest = mf.output.write_artifacts(paper_outputs, "results/baseline")
```

## write_artifacts

```python
macroforecast.output.write_artifacts(
    artifacts,
    output_dir,
    *,
    formats=("json", "csv"),
    manifest_format="json",
    include_provenance=True,
    provenance_fields=None,
    compression="none",
) -> ArtifactManifest
```

### Input

| Name | Type | Default | Meaning |
| --- | --- | --- | --- |
| `artifacts` | `OutputBundle`, `ForecastResult`, `DataFrame`, or mapping | required | Object or named objects to write. |
| `output_dir` | path-like | required | Output directory. Created if missing. |
| `formats` | tuple | `("json", "csv")` | DataFrame export formats: `"json"`, `"csv"`, `"parquet"`, or `"markdown"`. |
| `manifest_format` | str | `"json"` | Manifest format: `"json"`, `"csv"`, or `"parquet"`. |
| `include_provenance` | bool | `True` | Include package, Python, platform, and git provenance. |
| `provenance_fields` | tuple or `None` | `None` | Optional top-level provenance keys to keep, e.g. `("macroforecast_version", "git")`. |
| `compression` | str | `"none"` | Artifact compression: `"none"`, `"gzip"`, or `"zip"`. |

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
| `metadata` | Shape, columns, object type, metadata schema, file size, SHA-256 hash, path existence, and compression. |

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
Stored model files are not gzip-rewritten even when `compression="gzip"`;
the manifest records their existing paths and hashes.

Stored-model records use:

| `kind` | `format` | Path source |
| --- | --- | --- |
| `stored_model_pickle` | `pickle` | `stored_model["model_path"]` when present. |
| `stored_model_metadata` | `json` | `stored_model["metadata_path"]` when present. |

The record metadata includes the model alias, model spec, origin, horizon,
`save_error`, the original `stored_model` dictionary, and `path_exists`.

### Hashing and compression

Every written artifact record includes:

| Metadata field | Meaning |
| --- | --- |
| `path_exists` | Whether the file existed when the manifest record was created. |
| `size_bytes` | File size in bytes, or `None` when missing. |
| `sha256` | SHA-256 hash of the recorded file, or `None` when missing. |
| `compression` | `"none"`, `"gzip"`, or `"zip"`. |

`CompressionFormat` is the public type for supported compression choices:
`"none"`, `"gzip"`, or `"zip"`.

`compression="gzip"` rewrites each newly written study artifact as
`<name>.<ext>.gz` and records the compressed path. Existing stored-model
sidecars are not modified.

`compression="zip"` leaves individual artifacts in place and adds
`artifact_bundle.zip` to the manifest. The manifest itself is kept outside the
zip so it remains directly inspectable.

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

Use `fields=(...)` to keep only selected top-level fields:

```python
macroforecast.output.collect_provenance(
    fields=("macroforecast_version", "git", "packages")
)
```

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
