# `compute_mode`

[Back to L0](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``compute_mode`` on sub-layer ``l0_a`` (layer ``l0``).

## Sub-layer

**l0_a**

## Axis metadata

- Default: `'serial'`
- Sweepable: False
- Status: operational
- Leaf-config keys: `parallel_unit`, `n_workers`

## Operational status summary

- Operational: 2 option(s)
- Future: 0 option(s)

## Options

### `serial`  --  operational

Run every sweep cell sequentially in the calling process.

The cell loop iterates expanded cells one at a time. No worker pool, no extra processes, no thread pool. ``parallel_unit`` is ignored.

This is the default because (a) most authoring sweeps are small enough that the cell loop dwarfs scheduling overhead, and (b) every parallel mode introduces an additional surface for scheduling-induced non-determinism that has to be ruled out explicitly. Pick ``parallel`` when wall-clock matters and the study has been validated under ``serial`` first.

**When to use**

Default. Authoring iteration, small sweeps, debugging, any case where a stack trace from a particular cell needs to be readable without thread / process noise.

**When NOT to use**

Multi-cell sweeps where each cell takes more than a minute and the machine has multiple CPU cores. Switch to ``parallel`` and record the speed-up in the manifest.

**References**

* macroforecast design Part 1, L0 §A: 'compute_mode = serial is the deterministic default; parallel modes are opt-in for wall-clock-sensitive sweeps.'

**Related options**: [`parallel`](#parallel)

**Examples**

*Default serial study*

```yaml
0_meta:
  fixed_axes:
    compute_mode: serial

```

_Last reviewed 2026-05-04 by macroforecast author._

### `parallel`  --  operational

Distribute work over multiple workers; pick the unit via parallel_unit.

Activates the parallel cell loop. The granularity is controlled by the ``parallel_unit`` conditional leaf_config key:

* ``cells`` -- one process per sweep cell (``ProcessPoolExecutor``).   Cell-level parallelism is the safest path because cells are by   construction independent.
* ``models`` -- threads over ``fit_model`` nodes inside a single   cell (issue #204). Sklearn-family estimators release the GIL; the   thread pool avoids the pickling overhead of processes.
* ``oos_dates`` -- threads over walk-forward origins inside a fit   node (issue #250). Per-origin RNG state is derived deterministically   from ``base_seed + position`` (issue #279) so thread scheduling   cannot affect the forecasts.
* ``horizons`` / ``targets`` -- map to the same fan-out when L4   produces single-horizon / single-target output per fit node.

``leaf_config.n_workers`` caps the pool size.

**When to use**

Long sweeps on multi-core machines. Validate the manifest under ``compute_mode = serial`` first to confirm the recipe is deterministic, then flip to ``parallel`` and verify ``replicate()`` still passes.

**When NOT to use**

Recipes that mutate global state (e.g., a custom L3 op that writes to a shared file or a base estimator with a non-thread-safe C extension). Audit thread / process safety before flipping the switch.

**References**

* macroforecast PR #173 -- cell-level ProcessPoolExecutor.
* macroforecast PR #204 / #250 -- sub-cell ThreadPoolExecutor.
* macroforecast PR #279 -- deterministic per-origin seed propagation.

**Related options**: [`serial`](#serial)

**Examples**

*Cell-level parallel sweep*

```yaml
0_meta:
  fixed_axes:
    compute_mode: parallel
    parallel_unit: cells
  leaf_config:
    n_workers: 4

```

*Sub-cell parallel over fit_model nodes*

```yaml
0_meta:
  fixed_axes:
    compute_mode: parallel
    parallel_unit: models

```

_Last reviewed 2026-05-04 by macroforecast author._
