# `failure_policy`

[Back to L0](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``failure_policy`` on sub-layer ``l0_a`` (layer ``l0``).

## Sub-layer

**l0_a**

## Axis metadata

- Default: `'fail_fast'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 2 option(s)
- Future: 0 option(s)

## Options

### `fail_fast`  --  operational

Stop the entire study on the first cell that errors.

When the cell-loop catches an exception in any sweep cell, ``fail_fast`` raises immediately and the manifest is **not** written. The remaining cells are skipped.

This is the default because the typical authoring failure mode is a schema or data error that affects every cell -- catching it after the first cell saves wall-clock and surfaces the problem with a single traceback rather than a wall of identical errors. For sweeps where cells *can* fail independently (e.g., one model family throws on a particular target while others succeed), use ``continue_on_failure`` instead so partial results survive.

**When to use**

Default for every authoring iteration. Pick this while the recipe is still being tuned; the first failure tells you exactly what to fix without waiting for a full sweep to finish.

**When NOT to use**

Long-running production sweeps where a transient failure on one cell (e.g., a memory hiccup on one bootstrap iteration) should not abort the whole study.

**References**

* macroforecast design Part 1, L0 §A: 'fail_fast vs continue_on_failure is the canonical execution-policy choice for any cell-loop study.'

**Related options**: [`continue_on_failure`](#continue-on-failure)

**Examples**

*Author-time recipe (default)*

```yaml
0_meta:
  fixed_axes:
    failure_policy: fail_fast

```

_Last reviewed 2026-05-04 by macroforecast author._

### `continue_on_failure`  --  operational

Record failed cells in the manifest and keep the sweep running.

Per-cell exceptions are caught by the cell loop, the cell's ``CellExecutionResult.error`` and ``traceback`` fields are populated, and the loop moves on to the next cell. The manifest's ``cells_summary`` distinguishes succeeded from failed cells; the failed-cell entries carry the captured traceback for post-hoc diagnosis.

Replication still runs end-to-end on a manifest with failed cells: ``replicate()`` re-executes every cell and verifies the failure occurs in the same place with the same exception class.

**When to use**

Production horse-race sweeps where partial coverage is more useful than no coverage. Common examples: a 50-cell model-family sweep where one optional family (xgboost without the extra) fails to import, or a long bootstrap where a single iteration trips a numerical edge case.

**When NOT to use**

Authoring iteration -- failures are usually configuration problems that affect every cell, and ``fail_fast`` shortens the feedback loop.

**References**

* macroforecast design Part 1, L0 §A: 'continue_on_failure preserves partial coverage; the manifest carries enough context to diagnose each failed cell after the run.'

**Related options**: [`fail_fast`](#fail-fast)

**Examples**

*Production sweep over many model families*

```yaml
0_meta:
  fixed_axes:
    failure_policy: continue_on_failure

```

_Last reviewed 2026-05-04 by macroforecast author._
