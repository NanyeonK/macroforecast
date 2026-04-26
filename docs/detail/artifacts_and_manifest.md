# Artifacts And Manifest

Artifacts are the durable outputs of an experiment. The manifest is the audit trail.

`Experiment.run()` wraps these files with:

- `ExperimentRunResult` for one executable recipe
- `ExperimentSweepResult` for controlled variation sweeps

The facade does not replace artifacts. It reads saved files and exposes
`forecasts`, `metrics`, `comparison`, and `manifest` directly to researchers.
Use `metrics_json` and `comparison_json` when the exact artifact payload matters.

Layer 4 evaluation writes `evaluation_summary.json` for every run. It records
the selected evaluation spec, primary metric, per-horizon summary, aggregation
choices, and metric-family availability. If `report_style=markdown_table`, the
run also writes `evaluation_report.md`; if `report_style=latex_table`, it writes
`evaluation_report.tex`.

Layer 1 writes `layer1_official_frame.json` for every run. It records
`layer1_official_frame_v1`: the resolved official frame handed to Layer 2,
including source metadata, target/horizon identity, frame shape and columns,
information-set provenance, raw missing/outlier choices, missing-availability,
release-lag, variable-universe rules, official transform/T-code evidence, data
warnings, and data reports.

FRED-SD runs also write `fred_sd_series_metadata.json`. It records
`fred_sd_series_metadata_v1`: the selected state/variable panel, source sheets,
per-column observed windows, and inferred native-frequency counts for the
FRED-SD component. For composite FRED-MD/FRED-QD + FRED-SD runs, this file
describes the FRED-SD component before generic post-load column filtering.

Layer 5 writes `artifact_manifest.json` for every run. It is the stable
inventory of files that were actually materialized under the selected
`output_spec`.

Layer 6 writes statistical-test artifacts only for `saved_objects=full_bundle`
and only when at least one split test-family axis is active. The aggregate file
is `stat_tests.json`; per-test compatibility sidecars such as
`stat_test_dm_modified.json` are also written when a single test payload is
materialized. The run manifest records
`stat_test_contract=layer6_stat_test_split_v1` and the resolved
`stat_test_spec`.

Layer 7 writes importance artifacts only for `saved_objects=full_bundle` and
only when at least one split importance-family axis is active. The aggregate
file is `importance_artifacts.json`; per-method compatibility sidecars such as
`importance_minimal.json`, `importance_tree_shap.json`, and
`importance_permutation_importance.json` are written for active methods. The run
manifest records `importance_contract=layer7_importance_split_v1`, the resolved
`importance_spec`, `importance_file`, and `importance_files`.

## Output Spec

| Axis | Operational values | Effect |
|---|---|---|
| `export_format` | `json`, `csv`, `parquet`, `json+csv`, `all` | Controls structured metric/comparison sidecar formats. `predictions.csv` and `manifest.json` remain stable baseline files. |
| `saved_objects` | `predictions_only`, `predictions_and_metrics`, `full_bundle` | Controls the minimum object family saved by runtime. |
| `provenance_fields` | `none`, `minimal`, `standard`, `full` | Controls extra provenance fields such as package version, git commit, and config hash. |
| `artifact_granularity` | `aggregated` | Writes one aggregated run directory. Per-target/per-horizon/hierarchical layouts are registry-only or future. |

`saved_objects=predictions_only` writes the required manifest files, run
summary, prediction table, and forecast payload files when present.
`saved_objects=predictions_and_metrics` adds metrics, comparison, Layer 4
evaluation summary, optional evaluation report, and regime summaries.
`saved_objects=full_bundle` additionally writes data preview, feature fit-state,
tuning, statistical-test, and importance artifacts when available.

`artifact_manifest.json` uses `layer5_output_artifact_manifest_v1`:

```json
{
  "contract_version": "layer5_output_artifact_manifest_v1",
  "output_spec": {
    "export_format": "json",
    "saved_objects": "full_bundle",
    "provenance_fields": "full",
    "artifact_granularity": "aggregated"
  },
  "artifact_count": 6,
  "artifacts": [
    {
      "path": "predictions.csv",
      "artifact_type": "predictions",
      "layer": "5_output_provenance",
      "format": "csv"
    },
    {
      "path": "prediction_row_schema.json",
      "artifact_type": "prediction_row_schema",
      "layer": "5_output_provenance",
      "format": "json"
    },
    {
      "path": "layer1_official_frame.json",
      "artifact_type": "layer1_official_frame",
      "layer": "1_data_task",
      "format": "json"
    },
    {
      "path": "fred_sd_series_metadata.json",
      "artifact_type": "fred_sd_series_metadata",
      "layer": "1_data_task",
      "format": "json"
    }
  ]
}
```

The manifest records `artifact_manifest_file`, `output_artifact_contract`,
`saved_objects_effective`, `artifact_granularity_effective`, and
`artifact_count`. `prediction_row_schema.json` uses
`prediction_row_schema_v1` and is the versioned column contract for
`predictions.csv`. `layer1_official_frame.json` uses
`layer1_official_frame_v1` and is the versioned upstream-frame contract for
Layer 2 representation builders. `fred_sd_series_metadata.json` is present only
when the selected dataset includes FRED-SD.

This page should continue expanding:

- output directory layout
- forecasts artifact
- metrics artifact
- evaluation summary artifact
- comparison summary
- sweep manifest
- raw-data manifest
- custom method provenance
- default profile provenance
- package version and environment metadata

Design requirements:

- every run records resolved defaults
- every custom method records its registered name
- every sweep variant is traceable to axis values
- artifacts are readable without rerunning the experiment
- result objects point back to artifact paths
