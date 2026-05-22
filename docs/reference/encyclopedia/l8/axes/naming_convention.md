# `naming_convention`

[Back to L8](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``naming_convention`` on sub-layer ``L8_D_artifact_granularity`` (layer ``l8``).

## Sub-layer

**L8_D_artifact_granularity**

## Axis metadata

- Default: `'descriptive'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 4 option(s)
- Future: 0 option(s)

## Options

### `cell_id`  --  operational

Use the cell numeric id (cell_001/, cell_002/, ...).

Default. Stable and short; sorts naturally in directory listings. Recommended for production sweeps where the exact axis-value mapping is captured separately by the manifest's ``cell_resolved_axes`` field.

**When to use**

Default; stable, short. Selecting ``cell_id`` on ``l8.naming_convention`` activates this branch of the layer's runtime.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`descriptive`](#descriptive), [`recipe_hash`](#recipe-hash), [`custom`](#custom)

_Last reviewed 2026-05-05 by macroforecast author._

### `custom`  --  operational

User-supplied template via ``leaf_config.naming_template``.

Bespoke directory layouts: the template string can interpolate any cell-resolved axis (e.g. ``{model_family}_{horizon}h_{seed}``) plus the cell's numeric id.

**When to use**

Bespoke directory layouts. Selecting ``custom`` on ``l8.naming_convention`` activates this branch of the layer's runtime.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`descriptive`](#descriptive), [`cell_id`](#cell-id)

_Last reviewed 2026-05-05 by macroforecast author._

### `descriptive`  --  operational

Use a descriptive template combining the cell's resolved axes.

Generates names like ``ridge_log_diff_h1_seed42/`` from the cell's resolved axes. Human-readable; useful when humans browse the output directory directly. Long names can hit filesystem limits for wide sweeps.

**When to use**

When humans browse the output directory.

**When NOT to use**

Wide sweeps with many axes -- names exceed filesystem limits.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`cell_id`](#cell-id), [`custom`](#custom)

_Last reviewed 2026-05-05 by macroforecast author._

### `recipe_hash`  --  operational

Use the per-cell recipe hash as the directory name.

Reproducibility-first naming: directory names are the first 8 chars of the cell's recipe hash. Deterministic across runs (a re-run with the same recipe produces the same directory names), but unreadable to humans.

**When to use**

Reproducibility-first naming; deterministic across runs.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`cell_id`](#cell-id), [`descriptive`](#descriptive)

_Last reviewed 2026-05-05 by macroforecast author._
