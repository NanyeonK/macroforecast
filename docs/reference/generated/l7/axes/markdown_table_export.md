# `markdown_table_export`

[Back to L7](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``markdown_table_export`` on sub-layer ``L7_B_output_shape_export`` (layer ``l7``).

## Sub-layer

**L7_B_output_shape_export**

## Axis metadata

- Default: `False`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 2 option(s)
- Future: 0 option(s)

## Options

### `true`  --  operational

Co-emit Markdown tables alongside the JSON / CSV outputs.

Useful for README / wiki / GitHub-flavoured Markdown documents. Pipe-aligned columns; pairs naturally with the ``markdown_report`` L8 export format for end-to-end Markdown reporting.

**When to use**

Generating GitHub README / wiki pages from runs.

**References**

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'

**Related options**: [`false`](#false)

_Last reviewed 2026-05-05 by macroforecast author._

### `false`  --  operational

Skip the Markdown-table co-emission step.

Default. JSON / CSV outputs cover most consumers; Markdown is opt-in for documentation pipelines.

**When to use**

Default; Markdown is opt-in. Selecting ``false`` on ``l7.markdown_table_export`` activates this branch of the layer's runtime.

**References**

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'

**Related options**: [`true`](#true)

_Last reviewed 2026-05-05 by macroforecast author._
