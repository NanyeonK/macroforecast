# `artifact_granularity`

[Back to L8](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``artifact_granularity`` on sub-layer ``L8_D_artifact_granularity`` (layer ``l8``).

## Sub-layer

**L8_D_artifact_granularity**

## Axis metadata

- Default: `'per_cell'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 5 option(s)
- Future: 0 option(s)

## Options

### `flat`  --  operational

Single flat directory; cells distinguished by filename suffix.

All cells write into one directory; cell IDs become filename suffixes. Useful for sweeps with thousands of small artifacts where per-cell directory creation is wasteful.

**When to use**

Sweeps with thousands of small artifacts.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`per_cell`](#per-cell), [`per_target`](#per-target), [`per_horizon`](#per-horizon), [`per_target_horizon`](#per-target-horizon)

_Last reviewed 2026-05-05 by macroforecast author._

### `per_cell`  --  operational

One sub-directory per sweep cell (default).

Default. Each cell gets its own ``cell_NNN/`` directory containing the cell's full set of artifacts. Isolates every cell's output for clean per-cell inspection.

**When to use**

Default; isolates each cell's artifacts.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`per_target`](#per-target), [`per_horizon`](#per-horizon), [`per_target_horizon`](#per-target-horizon), [`flat`](#flat)

_Last reviewed 2026-05-05 by macroforecast author._

### `per_horizon`  --  operational

Group cells by forecast horizon (one directory per horizon).

When the sweep varies over multiple horizons, groups by horizon. Useful for horizon-by-horizon analysis.

**When to use**

Multi-horizon sweeps. Selecting ``per_horizon`` on ``l8.artifact_granularity`` activates this branch of the layer's runtime.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`per_cell`](#per-cell), [`per_target`](#per-target), [`per_target_horizon`](#per-target-horizon), [`flat`](#flat)

_Last reviewed 2026-05-05 by macroforecast author._

### `per_target`  --  operational

Group cells by target variable (one directory per target).

When the sweep varies over multiple targets, this groups the artifacts by target rather than by cell. Useful when downstream analysis is per-target.

**When to use**

Multi-target studies where target-grouping aids browsing.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`per_cell`](#per-cell), [`per_horizon`](#per-horizon), [`per_target_horizon`](#per-target-horizon), [`flat`](#flat)

_Last reviewed 2026-05-05 by macroforecast author._

### `per_target_horizon`  --  operational

Group cells by (target, horizon) pair.

Combines ``per_target`` and ``per_horizon`` for sweeps that vary across both axes.

**When to use**

Studies sweeping across both target and horizon.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`per_cell`](#per-cell), [`per_target`](#per-target), [`per_horizon`](#per-horizon), [`flat`](#flat)

_Last reviewed 2026-05-05 by macroforecast author._
