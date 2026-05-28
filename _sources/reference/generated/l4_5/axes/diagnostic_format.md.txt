# `diagnostic_format`

[Back to L4.5](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``diagnostic_format`` on sub-layer ``L4_5_Z_export`` (layer ``l4_5``).

## Sub-layer

**L4_5_Z_export**

## Axis metadata

- Default: `'pdf'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 7 option(s)
- Future: 0 option(s)

## Options

### `csv`  --  operational

L4.5 export -- plain csv table -- one row per series / metric.

Tabular diagnostic outputs (per-series summaries, test p-values, coverage tables) are emitted as comma-separated rows. The fastest path into pandas / R / Excel for ad-hoc analysis.

Diagnostic artifacts land under ``manifest.diagnostics/l4_5/`` with one file per axis per format chosen here.

**When to use**

Quickest path to spreadsheet / pandas; collaborators who avoid JSON.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`html`](#html), [`json`](#json), [`latex_table`](#latex-table), [`multi`](#multi), [`pdf`](#pdf), [`png`](#png)

_Last reviewed 2026-05-05 by macroforecast author._

### `html`  --  operational

L4.5 export -- single-file html report with embedded plots and tables.

Renders a self-contained HTML document combining tables (via pandas ``to_html``) and base64-embedded matplotlib figures. Opens in any browser without server-side support; ideal for sharing with stakeholders who lack the codebase.

Diagnostic artifacts land under ``manifest.diagnostics/l4_5/`` with one file per axis per format chosen here.

**When to use**

Sharing diagnostics with collaborators who do not have the repo checked out.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`csv`](#csv), [`json`](#json), [`latex_table`](#latex-table), [`multi`](#multi), [`pdf`](#pdf), [`png`](#png)

_Last reviewed 2026-05-05 by macroforecast author._

### `json`  --  operational

L4.5 export -- machine-readable json dump of every diagnostic value.

Default format. Round-trips cleanly into Python / JS; mandatory when ``attach_to_manifest=True`` so the diagnostics participate in the manifest's hash-based replication check.

Diagnostic artifacts land under ``manifest.diagnostics/l4_5/`` with one file per axis per format chosen here.

**When to use**

Default; required when diagnostics participate in the bit-exact replication chain.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`csv`](#csv), [`html`](#html), [`latex_table`](#latex-table), [`multi`](#multi), [`pdf`](#pdf), [`png`](#png)

_Last reviewed 2026-05-05 by macroforecast author._

### `latex_table`  --  operational

L4.5 export -- latex ``tabular`` snippets ready to ``\input`` into a paper.

Emits one ``.tex`` file per tabular diagnostic, ready to be ``\input`` from a paper draft without further processing. Booktabs-friendly column alignment + automatic column-name escaping.

Diagnostic artifacts land under ``manifest.diagnostics/l4_5/`` with one file per axis per format chosen here.

**When to use**

Paper-quality export when the user is drafting a manuscript.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`csv`](#csv), [`html`](#html), [`json`](#json), [`multi`](#multi), [`pdf`](#pdf), [`png`](#png)

_Last reviewed 2026-05-05 by macroforecast author._

### `multi`  --  operational

L4.5 export -- emit json + png + (optional) pdf / html in a single run.

Comprehensive convenience option. Produces JSON for machine consumption + PNG for slides + (when ``latex_export=True``) LaTeX snippets for papers. Equivalent to setting ``diagnostic_format`` separately for each consumer.

Diagnostic artifacts land under ``manifest.diagnostics/l4_5/`` with one file per axis per format chosen here.

**When to use**

Comprehensive runs covering paper, slides and machine-readable consumers in one execution.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`csv`](#csv), [`html`](#html), [`json`](#json), [`latex_table`](#latex-table), [`pdf`](#pdf), [`png`](#png)

_Last reviewed 2026-05-05 by macroforecast author._

### `pdf`  --  operational

L4.5 export -- vector pdf figures (matplotlib backend).

Matplotlib's PDF backend produces vector graphics that scale without pixelation. Recommended for paper figures where journals require sub-pixel-precise typography.

Diagnostic artifacts land under ``manifest.diagnostics/l4_5/`` with one file per axis per format chosen here.

**When to use**

Publication-grade plots; LaTeX-rendered figures.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`csv`](#csv), [`html`](#html), [`json`](#json), [`latex_table`](#latex-table), [`multi`](#multi), [`png`](#png)

_Last reviewed 2026-05-05 by macroforecast author._

### `png`  --  operational

L4.5 export -- rasterised png figures (matplotlib backend).

Matplotlib's AGG backend produces 300dpi-by-default PNG files. Smaller than PDF for plot-heavy reports; the natural choice for slide decks and HTML embeddings.

Diagnostic artifacts land under ``manifest.diagnostics/l4_5/`` with one file per axis per format chosen here.

**When to use**

Slide / web embedding where vector formats are unnecessary.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`csv`](#csv), [`html`](#html), [`json`](#json), [`latex_table`](#latex-table), [`multi`](#multi), [`pdf`](#pdf)

_Last reviewed 2026-05-05 by macroforecast author._
