# `fred_sd_frequency_policy`

[Back to L1](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``fred_sd_frequency_policy`` on sub-layer ``l1_a`` (layer ``l1``).

## Sub-layer

**l1_a**

## Axis metadata

- Default: `'report_only'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 4 option(s)
- Future: 0 option(s)

## Options

### `report_only`  --  operational

Report mixed-frequency status; do not gate.

Lifts mixed-frequency information to L1.5 diagnostics but allows the panel to proceed regardless of the frequency mix. The runtime defers alignment decisions to L2.A.

**When to use**

Exploratory work where mixed-frequency status is informative but should not block execution.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`allow_mixed_frequency`](#allow-mixed-frequency), [`reject_mixed_known_frequency`](#reject-mixed-known-frequency), [`require_single_known_frequency`](#require-single-known-frequency)

_Last reviewed 2026-05-05 by macroforecast author._

### `allow_mixed_frequency`  --  operational

Permit mixed-frequency panels; rely on L2.A alignment.

Default for FRED-SD recipes that combine monthly and quarterly state series. The downstream L2.A frequency-alignment rules render the mixed panel onto a single grid.

**When to use**

Standard FRED-SD pipelines that need both monthly and quarterly variables.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`report_only`](#report-only), [`reject_mixed_known_frequency`](#reject-mixed-known-frequency), [`require_single_known_frequency`](#require-single-known-frequency)

_Last reviewed 2026-05-05 by macroforecast author._

### `reject_mixed_known_frequency`  --  operational

Reject the panel when explicit mixed-frequency variables coexist.

Hard-rejects panels where a series is declared at one frequency and another at a different known frequency. Useful as a safety gate when the recipe author expects a single-frequency panel.

**When to use**

Defensive recipes that should fail loudly if FRED-SD upstream changes deliver mixed frequencies.

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`require_single_known_frequency`](#require-single-known-frequency)

_Last reviewed 2026-05-05 by macroforecast author._

### `require_single_known_frequency`  --  operational

Hard-require every variable to declare the same frequency.

Strictest setting -- the gate fails unless every series shares an identical declared frequency. Distinct from ``reject_mixed_known_frequency`` in that it also rejects unknown-frequency series.

**When to use**

Strictly mono-frequency studies (e.g. monthly-only).

**References**

* macroforecast design Part 1, L1: 'data definition is the recipe layer that pins source, target, geography, and horizon -- everything downstream branches off these choices.'

**Related options**: [`reject_mixed_known_frequency`](#reject-mixed-known-frequency)

_Last reviewed 2026-05-05 by macroforecast author._
