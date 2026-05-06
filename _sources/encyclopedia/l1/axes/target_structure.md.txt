# `target_structure`

[Back to L1](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``target_structure`` on sub-layer ``l1_b`` (layer ``l1``).

## Sub-layer

**l1_b**

## Axis metadata

- Default: `'single_target'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 2 option(s)
- Future: 0 option(s)

## Options

### `single_target`  --  operational

Forecast one target series at a time.

The recipe declares one ``target`` in L1 leaf_config. All downstream layers (feature DAG, model, evaluation) operate on that single series.

This is the dominant pattern for benchmark studies because most forecasting literature reports per-target metrics; multi-series studies typically compose multiple single-target runs in a sweep.

**When to use**

Default. Any standard forecasting benchmark.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`multi_target`](#multi-target)

**Examples**

*Forecast CPI inflation*

```yaml
1_data:
  leaf_config:
    target: CPIAUCSL

```

_Last reviewed 2026-05-04 by macroforecast author._

### `multi_target`  --  operational

Forecast multiple target series jointly within one cell.

The recipe declares ``targets: [a, b, c]`` in leaf_config. The L4 model is fit per-(target, horizon) tuple; the L5 metrics table carries one row per (model, target, horizon, origin).

Useful for vector-target methods (VAR, FAVAR, BVAR) and for studies that compute cross-target metrics (e.g., portfolio MSE).

**When to use**

VAR-style joint forecasting; cross-target evaluation; replicating papers that report joint metrics.

**When NOT to use**

Independent per-target studies -- those are usually clearer as separate sweep cells.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`single_target`](#single-target)

_Last reviewed 2026-05-04 by macroforecast author._
