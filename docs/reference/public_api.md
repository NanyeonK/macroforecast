# Public Python API

[Back to reference](index.md)

The importable surface is intentionally narrow while the package is rebuilt.

## Top-Level Exports

| Symbol | Source | Description |
| --- | --- | --- |
| `configure`, `get_config`, `get_option`, `reset_config`, `use_config` | `macroforecast.meta` | Global package defaults. |
| `DataBundle`, `DataSpec`, `as_panel`, `metadata`, `panel_info`, `spec`, `validate_panel` | `macroforecast.data` | Canonical panel and metadata helpers. |
| `load_fred_md`, `load_fred_qd`, `load_fred_sd`, `load_fred_md_sd`, `load_fred_qd_sd` | `macroforecast.data` | Dataset loaders. |
| `load_custom_csv`, `load_custom_parquet`, `list_vintages`, `combine` | `macroforecast.data` | Custom loading, vintage discovery, and panel combination. |
| `preprocess`, `reprocess`, `PreprocessedData` | `macroforecast.preprocessing` | Direct pandas preprocessing. |
| `summarize_data`, `DataSummaryReport` | `macroforecast.data_summary` | Single-panel summaries. |
| `analyze_data`, `DataAnalysisReport` | `macroforecast.data_analysis` | Before/after panel analysis. |

## Submodules

| Module | Purpose |
| --- | --- |
| `macroforecast.meta` | Global defaults. |
| `macroforecast.data` | Data loading and study data specs. |
| `macroforecast.preprocessing` | Pandas preprocessing functions. |
| `macroforecast.data_summary` | Single-panel diagnostics and summaries. |
| `macroforecast.data_analysis` | Raw-versus-processed comparison. |
| `macroforecast.evaluation` | Reserved for the next callable evaluation pass. |

Removed legacy runtime modules are available for reference on the
`legacy-runtime-reference` branch.
