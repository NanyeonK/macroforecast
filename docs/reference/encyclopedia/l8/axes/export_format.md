# `export_format`

[Back to L8](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``export_format`` on sub-layer ``L8_A_export_format`` (layer ``l8``).

## Sub-layer

**L8_A_export_format**

## Axis metadata

- Default: `'json_csv'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 9 option(s)
- Future: 0 option(s)

## Options

### `all`  --  operational

Emit every supported tabular/markup export format together.

Comprehensive option emitting JSON + CSV + Parquet + LaTeX + Markdown for every applicable artifact. The HTML report is NOT included in ``all`` -- request ``export_format = html_report`` separately when a browser-renderable bundle is required. Largest disk footprint among the tabular/markup formats and covers every downstream consumer that reads structured tables in one run.

**When to use**

Comprehensive reproducibility / sharing -- single run that covers every tabular/markup audience.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`json`](#json), [`csv`](#csv), [`parquet`](#parquet), [`json_csv`](#json-csv), [`json_parquet`](#json-parquet)

_Last reviewed 2026-05-05 by macroforecast author._

### `csv`  --  operational

CSV tables for tabular artifacts (forecasts, metrics, importance).

Standard comma-separated values, UTF-8 encoded. The lowest-common-denominator format for spreadsheet / R workflows. Loses dtype information (everything becomes string on round-trip); for analytics workloads prefer ``parquet``.

**When to use**

Spreadsheet / R workflows; collaborators who avoid JSON.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`json`](#json), [`parquet`](#parquet), [`json_csv`](#json-csv), [`json_parquet`](#json-parquet), [`latex_tables`](#latex-tables)

_Last reviewed 2026-05-05 by macroforecast author._

### `html_report`  --  operational

Self-contained HTML report with embedded plots and tables.

Renders a single ``.html`` file combining tables (via pandas' ``to_html``) and base64-embedded matplotlib figures. Opens in any browser without a server; ideal for stakeholder-shareable reports without LaTeX tooling.

**When to use**

Stakeholder-shareable reports without LaTeX tooling.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`json`](#json), [`csv`](#csv), [`parquet`](#parquet), [`json_csv`](#json-csv), [`json_parquet`](#json-parquet)

_Last reviewed 2026-05-05 by macroforecast author._

### `json`  --  operational

JSON dump of every artifact (default).

Default round-trip-safe format; native Python / JS / R support; preserves nested structure (dicts of dicts of DataFrames). All numeric values rendered as floats with full precision; date-like values rendered as ISO 8601 strings.

**When to use**

Default; round-trips cleanly into Python / JS / R.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`csv`](#csv), [`parquet`](#parquet), [`json_csv`](#json-csv), [`json_parquet`](#json-parquet), [`latex_tables`](#latex-tables)

_Last reviewed 2026-05-05 by macroforecast author._

### `json_csv`  --  operational

Both JSON and CSV for every applicable artifact.

Convenience option emitting both formats. Used when downstream consumers vary -- Python users want JSON round-trip, R / Excel users want CSV. Doubles the artifact-directory size.

**When to use**

When downstream consumers vary across both Python and Excel / R.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`json`](#json), [`csv`](#csv), [`parquet`](#parquet), [`json_parquet`](#json-parquet), [`latex_tables`](#latex-tables)

_Last reviewed 2026-05-05 by macroforecast author._

### `json_parquet`  --  operational

Both JSON and Parquet for every applicable artifact.

Hybrid option for runs that combine reproducibility (JSON for the manifest / small artifacts) with analytics (Parquet for large forecast tables). Recommended for production sweeps.

**When to use**

Hybrid analytics + reproducibility setups.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'
* Apache Parquet specification (apache/parquet-format). <https://parquet.apache.org/docs/file-format/>

**Related options**: [`json`](#json), [`csv`](#csv), [`parquet`](#parquet), [`json_csv`](#json-csv), [`latex_tables`](#latex-tables)

_Last reviewed 2026-05-05 by macroforecast author._

### `latex_tables`  --  operational

LaTeX ``tabular`` snippets ready to ``\input`` into a paper.

Emits one ``.tex`` file per tabular artifact (forecasts, metrics, ranking). Booktabs-friendly column alignment and column-name escaping; uses pandas' ``to_latex`` backend.

**When to use**

Paper-draft pipelines. Selecting ``latex_tables`` on ``l8.export_format`` activates this branch of the layer's runtime.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'
* pandas DataFrame.to_latex documentation. <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_latex.html>

**Related options**: [`json`](#json), [`csv`](#csv), [`parquet`](#parquet), [`json_csv`](#json-csv), [`json_parquet`](#json-parquet)

_Last reviewed 2026-05-05 by macroforecast author._

### `markdown_report`  --  operational

Single Markdown report bundling tables and figure references.

Renders a self-contained ``.md`` document with pipe-aligned tables and embedded image references. Intended as the human-readable summary for stakeholder reports and GitHub / wiki documentation.

**When to use**

Lightweight Markdown / GitHub-rendered reports.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`json`](#json), [`csv`](#csv), [`parquet`](#parquet), [`json_csv`](#json-csv), [`json_parquet`](#json-parquet)

_Last reviewed 2026-05-05 by macroforecast author._

### `parquet`  --  operational

Apache Parquet (pyarrow); columnar binary tabular format.

Columnar binary format with full dtype preservation, automatic dictionary encoding for low-cardinality columns, and per-column compression. 5-10× smaller than CSV for typical macro panels; an order of magnitude faster to read for column-subset queries. Requires ``pyarrow`` (already a transitive dependency).

**When to use**

Large-scale analytics; preserving dtypes; cross-language workflows (Spark, DuckDB, R arrow).

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'
* Apache Parquet specification (apache/parquet-format). <https://parquet.apache.org/docs/file-format/>

**Related options**: [`json`](#json), [`csv`](#csv), [`json_csv`](#json-csv), [`json_parquet`](#json-parquet), [`latex_tables`](#latex-tables)

_Last reviewed 2026-05-05 by macroforecast author._
