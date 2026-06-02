# Custom Output

[Back to custom extensions](index.md)

Use output custom artifacts when the object is already a result, table, note,
or diagnostic. Output functions do not run models or recompute metrics; they
write named objects and record provenance.

## write_artifacts

```python
mf.output.write_artifacts(
    artifacts,
    output_dir,
    *,
    formats=("json", "csv"),
    manifest_format="json",
    include_provenance=True,
    provenance_fields=None,
    compression="none",
) -> mf.output.ArtifactManifest
```

### Input

| Input object | Written as | Metadata behavior |
| --- | --- | --- |
| `ForecastResult` | forecast JSON plus forecast CSV | forecast metadata and stored-model sidecars recorded. |
| `DataFrame` | one file per requested format | `attrs` preserved in JSON and manifest records. |
| mapping | JSON | keys recorded in manifest metadata. |
| list/tuple | JSON | sequence length recorded. |
| scalar | JSON | object type recorded. |

### Example

```python
manifest = mf.output.write_artifacts(
    {
        "forecast_result": result,
        "scores": scores,
        "custom_test": test.to_dict(),
        "custom_interpretation": interpretation,
        "run_notes": {"design": "local robustness check", "accepted": True},
    },
    "results/custom_flow",
)
```

### Output

| Manifest field | Meaning |
| --- | --- |
| `artifacts` | Mapping from written file name to path. |
| `records` | One `ArtifactRecord` per written object or stored-model sidecar. |
| `provenance` | Macroforecast version, Python/platform, git, and package versions unless disabled. |
| `metadata_schema` | `{"kind": "artifact_manifest", "version": 1}`. |

## bundle_outputs With Custom Extra

```python
bundle = mf.output.bundle_outputs(
    forecasts=result,
    evaluation=report,
    metadata={"study": "custom"},
    extra={"custom_diagnostic": diagnostic_table},
)

manifest = mf.output.write_artifacts(bundle, "results/custom_bundle")
```

Use `extra={...}` for custom objects that should travel with the standard
forecast/evaluation bundle.

## Reporting Flow

If the custom object is a paper-facing table, format it before writing:

```python
paper_table = mf.reporting.report_table(
    diagnostic_table,
    caption="Custom diagnostic",
    label="tab:custom_diagnostic",
)

mf.output.write_artifacts(
    {"custom_diagnostic": paper_table.data},
    "results/custom_tables",
)
```
