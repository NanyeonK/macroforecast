# Artifacts And Manifest

Artifacts are durable outputs of an experiment. The manifest is the audit trail.

This page distinguishes current core L8 artifacts from legacy experiment artifacts.

## Core L8 Artifacts

`macroforecast.core.runtime.execute_minimal_forecast` writes L8 artifacts when the recipe includes `8_output`.

Baseline layout:

```text
output_directory/
  manifest.json or manifest.jsonl
  recipe.json
  summary/
    metrics_all_cells.csv
    ranking.csv
  cell_001/
    forecasts.csv
  diagnostics/
    *_diagnostic_v1.json
  tests_summary.json
  importance_summary.json
```

The exact files depend on `saved_objects`.

| Saved object | File currently written |
|---|---|
| `forecasts` | `cell_001/forecasts.csv` |
| `metrics` | `summary/metrics_all_cells.csv` |
| `ranking` | `summary/ranking.csv` |
| `tests` | `tests_summary.json` |
| `importance` | `importance_summary.json` |
| `feature_metadata` | `cell_001/feature_metadata.json` |
| `clean_panel` | `cell_001/clean_panel.csv` |
| `raw_panel` | `cell_001/raw_panel.csv` |
| `diagnostics_l1_5` through `diagnostics_l4_5` | `diagnostics/<sink>.json` |

`diagnostics_all` expands to the four diagnostic saved objects.

## L8 Manifest

The core L8 manifest records:

- recipe hash,
- package/runtime version marker,
- Python/runtime environment fields,
- dependency lockfile paths when present,
- saved objects,
- upstream sink inventory,
- exported files.

`L8ArtifactsArtifact` also exposes `output_directory`, `exported_files`, `artifact_count`, and `upstream_hashes` in memory.

## Legacy Experiment Artifacts

The older experiment engine can write files such as:

- `predictions.csv`
- `prediction_row_schema.json`
- `metrics.json`
- `comparison_summary.json`
- `evaluation_summary.json`
- `stat_test_*.json`
- `importance_*.json`
- `artifact_manifest.json`

These remain documented for backward compatibility, but new L0-L8 layer-contract recipes should use the core L8 output contract above.

## Design Requirements

- every run records resolved defaults where possible,
- every output can be read without rerunning the experiment,
- diagnostic outputs are non-blocking,
- advanced render/export formats must not be reported as materialized until the corresponding runtime writer exists.
