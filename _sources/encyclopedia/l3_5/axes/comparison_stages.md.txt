# `comparison_stages`

[Back to L3.5](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``comparison_stages`` on sub-layer ``L3_5_A_comparison_axis`` (layer ``l3_5``).

## Sub-layer

**L3_5_A_comparison_axis**

## Axis metadata

- Default: `'cleaned_vs_features'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 3 option(s)
- Future: 0 option(s)

## Options

### `cleaned_vs_features`  --  operational

Compare cleaned panel vs feature-engineered panel (skip raw).

L3.5.A comparison stages ``cleaned_vs_features``.

This option configures the ``comparison_stages`` axis on the ``L3_5_A_comparison_axis`` sub-layer of L3.5; output is emitted under ``manifest.diagnostics/l3_5/L3_5_A_comparison_axis/`` alongside the other selected views.

**When to use**

Isolating the L3 contribution when L2's cleaning is well-trusted.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`raw_vs_cleaned_vs_features`](#raw-vs-cleaned-vs-features), [`features_only`](#features-only)

_Last reviewed 2026-05-05 by macroforecast author._

### `features_only`  --  operational

Inspect feature panel in isolation.

L3.5.A comparison stages ``features_only``.

This option configures the ``comparison_stages`` axis on the ``L3_5_A_comparison_axis`` sub-layer of L3.5; output is emitted under ``manifest.diagnostics/l3_5/L3_5_A_comparison_axis/`` alongside the other selected views.

**When to use**

When upstream stages are well-trusted and the focus is on the L3 output's properties.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`raw_vs_cleaned_vs_features`](#raw-vs-cleaned-vs-features), [`cleaned_vs_features`](#cleaned-vs-features)

_Last reviewed 2026-05-05 by macroforecast author._

### `raw_vs_cleaned_vs_features`  --  operational

Compare raw / cleaned / featurised panels in a 3-way view.

L3.5.A comparison stages ``raw_vs_cleaned_vs_features``.

This option configures the ``comparison_stages`` axis on the ``L3_5_A_comparison_axis`` sub-layer of L3.5; output is emitted under ``manifest.diagnostics/l3_5/L3_5_A_comparison_axis/`` alongside the other selected views.

**When to use**

Default broad audit; tracking the panel's evolution from raw FRED data through to the L3 feature matrix.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`cleaned_vs_features`](#cleaned-vs-features), [`features_only`](#features-only)

_Last reviewed 2026-05-05 by macroforecast author._
