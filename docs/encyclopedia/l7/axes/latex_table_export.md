# `latex_table_export`

[Back to L7](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``latex_table_export`` on sub-layer ``L7_B_output_shape_export`` (layer ``l7``).

## Sub-layer

**L7_B_output_shape_export**

## Axis metadata

- Default: `True`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 2 option(s)
- Future: 0 option(s)

## Options

### `true`  --  operational

Co-emit LaTeX ``tabular`` snippets alongside the JSON / CSV outputs.

Saves an extra round-trip through ``pandas.to_latex`` when the recipe is feeding a paper draft. Booktabs-friendly alignment and automatic column-name escaping; the resulting ``.tex`` file is ``\input``-able directly into a manuscript without further processing.

**When to use**

When the recipe is feeding a paper draft.

**References**

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'

**Related options**: [`false`](#false)

_Last reviewed 2026-05-05 by macroforecast author._

### `false`  --  operational

Skip the LaTeX-table co-emission step.

Default. Avoids the small-but-non-trivial overhead of pandas-to-LaTeX rendering on every importance table when no paper-quality output is needed.

**When to use**

Default; LaTeX adds tooling overhead that is wasted for non-paper runs.

**References**

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'

**Related options**: [`true`](#true)

_Last reviewed 2026-05-05 by macroforecast author._
