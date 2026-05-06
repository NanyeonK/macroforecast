# `figure_format`

[Back to L7](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``figure_format`` on sub-layer ``L7_B_output_shape_export`` (layer ``l7``).

## Sub-layer

**L7_B_output_shape_export**

## Axis metadata

- Default: `'pdf'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 3 option(s)
- Future: 0 option(s)

## Options

### `pdf`  --  operational

Vector PDF figures (matplotlib backend).

Vector graphics that scale without pixelation. Recommended for paper figures where journals require sub-pixel-precise typography. File sizes larger than PNG but renderable at any zoom level.

**When to use**

Publication-grade plots; LaTeX-rendered figures.

**When NOT to use**

Web embedding -- prefer PNG or SVG.

**References**

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'

**Related options**: [`png`](#png), [`svg`](#svg)

_Last reviewed 2026-05-05 by macroforecast author._

### `png`  --  operational

Raster PNG figures (matplotlib AGG backend).

300dpi-by-default raster images. Smaller than PDF for plot-heavy reports; the natural choice for slides, HTML embeddings, and Markdown documents that render through GitHub / Slack / web viewers.

**When to use**

Slide / web embedding where vector formats are unnecessary.

**References**

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'

**Related options**: [`pdf`](#pdf), [`svg`](#svg)

_Last reviewed 2026-05-05 by macroforecast author._

### `svg`  --  operational

Vector SVG figures (matplotlib SVG backend).

XML-based vector format renderable in browsers. Selectable text and zoom-without-pixelation; useful when the consumer wants to interactively inspect / edit the figure (e.g. via Inkscape) before final publication.

**When to use**

Web embedding with selectable text; pre-publication editable figures.

**References**

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'

**Related options**: [`pdf`](#pdf), [`png`](#png)

_Last reviewed 2026-05-05 by macroforecast author._
