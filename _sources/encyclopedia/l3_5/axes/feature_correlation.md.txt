# `feature_correlation`

[Back to L3.5](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``feature_correlation`` on sub-layer ``L3_5_C_feature_correlation`` (layer ``l3_5``).

## Sub-layer

**L3_5_C_feature_correlation**

## Axis metadata

- Default: `'cross_block'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 5 option(s)
- Future: 0 option(s)

## Options

### `cross_block`  --  operational

Correlations across blocks (e.g. PCA factors vs MARX features).

L3.5.C feature correlation view ``cross_block``.

This option configures the ``feature_correlation`` axis on the ``L3_5_C_feature_correlation`` sub-layer of L3.5; output is emitted under ``manifest.diagnostics/l3_5/L3_5_C_feature_correlation/`` alongside the other selected views.

**When to use**

Detecting block-level redundancy before L4; informs whether to drop a block.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`within_block`](#within-block), [`with_target`](#with-target), [`multi`](#multi), [`none`](#none)

_Last reviewed 2026-05-05 by macroforecast author._

### `multi`  --  operational

Run every feature-correlation view together.

L3.5.C feature correlation view ``multi``.

This option configures the ``feature_correlation`` axis on the ``L3_5_C_feature_correlation`` sub-layer of L3.5; output is emitted under ``manifest.diagnostics/l3_5/L3_5_C_feature_correlation/`` alongside the other selected views.

**When to use**

Default rich correlation audit.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`within_block`](#within-block), [`cross_block`](#cross-block), [`with_target`](#with-target), [`none`](#none)

_Last reviewed 2026-05-05 by macroforecast author._

### `none`  --  operational

Skip feature correlation diagnostic entirely.

L3.5.C feature correlation view ``none``.

This option configures the ``feature_correlation`` axis on the ``L3_5_C_feature_correlation`` sub-layer of L3.5; output is emitted under ``manifest.diagnostics/l3_5/L3_5_C_feature_correlation/`` alongside the other selected views.

**When to use**

Memory-constrained sweeps with very wide feature panels (n_features > 5000).

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`within_block`](#within-block), [`cross_block`](#cross-block), [`with_target`](#with-target), [`multi`](#multi)

_Last reviewed 2026-05-05 by macroforecast author._

### `with_target`  --  operational

Correlations of every feature with the target.

L3.5.C feature correlation view ``with_target``.

This option configures the ``feature_correlation`` axis on the ``L3_5_C_feature_correlation`` sub-layer of L3.5; output is emitted under ``manifest.diagnostics/l3_5/L3_5_C_feature_correlation/`` alongside the other selected views.

**When to use**

Spotting top candidate predictors; pairs naturally with the L7 ``cumulative_r2_contribution`` op for downstream interpretation.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`within_block`](#within-block), [`cross_block`](#cross-block), [`multi`](#multi), [`none`](#none)

_Last reviewed 2026-05-05 by macroforecast author._

### `within_block`  --  operational

Correlations within a feature block (e.g. lags of one series, PCA factors).

L3.5.C feature correlation view ``within_block``.

This option configures the ``feature_correlation`` axis on the ``L3_5_C_feature_correlation`` sub-layer of L3.5; output is emitted under ``manifest.diagnostics/l3_5/L3_5_C_feature_correlation/`` alongside the other selected views.

**When to use**

Detecting redundancy within a block -- high within-block correlations suggest a smaller block dimension would suffice.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`cross_block`](#cross-block), [`with_target`](#with-target), [`multi`](#multi), [`none`](#none)

_Last reviewed 2026-05-05 by macroforecast author._
