# `latex_export`

[Back to L1.5](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``latex_export`` on sub-layer ``L1_5_Z_export`` (layer ``l1_5``).

## Sub-layer

**L1_5_Z_export**

## Axis metadata

- Default: `True`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 2 option(s)
- Future: 0 option(s)

## Options

### `true`  --  operational

Co-emit LaTeX table snippets alongside the chosen diagnostic_format.

Independent of ``diagnostic_format`` -- ``latex_export`` is an opt-in extra that always co-emits ``\begin{tabular}`` snippets when the underlying diagnostic is tabular. Saves an extra round-trip through pandas-to-LaTeX when the recipe is feeding a paper draft.

**When to use**

When the recipe is feeding a paper draft.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`false`](#false)

_Last reviewed 2026-05-05 by macroforecast author._

### `false`  --  operational

Skip the LaTeX-table co-emission step.

Default. Avoids the small but non-trivial overhead of pandas-to-LaTeX rendering on every diagnostic axis when no paper-quality output is needed.

**When to use**

Default; LaTeX adds tooling overhead that is wasted for non-paper runs.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`true`](#true)

_Last reviewed 2026-05-05 by macroforecast author._
