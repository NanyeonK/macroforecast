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

Log frequency mismatches in manifest; do not gate execution.

Default policy. When the FRED-SD pull contains variables with differing declared frequencies (e.g., monthly QCEW payroll series alongside quarterly income data), the runtime logs a diagnostic entry in the L1.5 manifest but allows the panel to proceed unchanged. No records are dropped and no error is raised.

Alignment of the mixed-frequency panel is deferred entirely to L2.A (frequency-alignment rules). This is the appropriate choice for exploratory or default pipelines where the recipe author has not yet decided how to handle the frequency mismatch -- the manifest diagnostic surfaces the issue without blocking execution.

**When to use**

Default for most FRED-SD recipes; exploratory work where mixed-frequency status should be visible in the manifest but should not stop execution; any pipeline that handles alignment in L2.A.

**When NOT to use**

Recipes that require strict frequency homogeneity -- use ``reject_mixed_known_frequency`` or ``require_single_known_frequency`` to gate early.

**References**

* macroforecast PR #251 (use_sd_inferred_tcodes) -- FRED-SD integration, state-level frequency-policy design.

**Related options**: [`allow_mixed_frequency`](#allow-mixed-frequency), [`reject_mixed_known_frequency`](#reject-mixed-known-frequency), [`require_single_known_frequency`](#require-single-known-frequency), [`mixed_frequency_representation`](#mixed-frequency-representation)

_Last reviewed 2026-05-16 by macroforecast author._

### `allow_mixed_frequency`  --  operational

Explicitly permit mixed frequencies; downstream layers must handle alignment.

Records an explicit recipe-author decision to accept a mixed-frequency FRED-SD panel. Unlike ``report_only``, selecting this option signals to downstream layers (L2.A frequency-alignment, L3 feature DAG) that the mixed-frequency structure is intentional and should be handled -- not silently passed through.

The actual frequency-alignment logic (e.g., temporal aggregation of monthly series to quarterly, or Kalman-filter mixed-frequency representation) is delegated to ``mixed_frequency_representation`` in L2.A. Use this option when the recipe is designed to combine monthly FRED-SD predictors (e.g., QCEW payrolls) with quarterly FRED-SD targets (e.g., GSP) and an explicit alignment strategy is configured in L2.

**When to use**

Standard FRED-SD pipelines that intentionally combine monthly and quarterly state series; recipes where ``mixed_frequency_representation`` is configured in L2.A.

**When NOT to use**

Pipelines that want to hard-reject mixed frequencies rather than align them -- use ``reject_mixed_known_frequency``.

**References**

* macroforecast PR #251 (use_sd_inferred_tcodes) -- FRED-SD integration, state-level frequency-policy design.

**Related options**: [`report_only`](#report-only), [`reject_mixed_known_frequency`](#reject-mixed-known-frequency), [`require_single_known_frequency`](#require-single-known-frequency), [`mixed_frequency_representation`](#mixed-frequency-representation)

_Last reviewed 2026-05-16 by macroforecast author._

### `reject_mixed_known_frequency`  --  operational

Hard-reject if pulled variables span more than one declared frequency.

Safety gate: raises a ``ValueError`` at L1 validation if any two variables in the FRED-SD pull carry different *known* frequency declarations (e.g., one series is declared monthly and another declared quarterly). Variables with unknown frequency (i.e., series for which FRED-SD does not declare a frequency) are tolerated -- only explicit mismatches between known frequencies trigger the error.

Useful when the recipe author expects a single-frequency panel and wants to fail loudly if FRED-SD upstream changes (new series additions, metadata corrections) introduce an unexpected frequency mix. The error message names the conflicting series and their declared frequencies.

**When to use**

Defensive recipes where frequency homogeneity is part of the study design; CI checks that should fail loudly if new FRED-SD series at a different frequency are inadvertently pulled.

**When NOT to use**

Pipelines designed to work with mixed frequencies (use ``allow_mixed_frequency``); pipelines where unknown-frequency series should also be rejected (use ``require_single_known_frequency``).

**References**

* macroforecast PR #251 (use_sd_inferred_tcodes) -- FRED-SD integration, state-level frequency-policy design.

**Related options**: [`report_only`](#report-only), [`allow_mixed_frequency`](#allow-mixed-frequency), [`require_single_known_frequency`](#require-single-known-frequency)

_Last reviewed 2026-05-16 by macroforecast author._

### `require_single_known_frequency`  --  operational

Enforce single frequency; reject if any variable has unknown or differing frequency.

Strictest setting. The L1 gate passes only if every variable in the FRED-SD pull (a) has a declared known frequency, and (b) all declared frequencies are identical. Two distinct failure modes raise ``ValueError``:

1. A variable carries frequency ``'unknown'`` in FRED-SD    metadata -- this would pass ``reject_mixed_known_frequency``    but fails here.
2. Two or more variables carry different *known* frequencies    (same condition as ``reject_mixed_known_frequency``).

This is the appropriate gate for strictly mono-frequency studies (e.g., monthly-only payroll analyses) that must also enforce that all series have a documented cadence -- no 'we don't know the frequency' series are permitted.

**When to use**

Strictly mono-frequency studies (e.g., monthly-only state-level employment analyses); pipelines that must guarantee every series has a known frequency declaration in the FRED-SD metadata.

**When NOT to use**

Pipelines that include FRED-SD series with undeclared frequencies (use ``reject_mixed_known_frequency`` to only block explicit mismatches, not unknown frequencies); mixed-frequency pipelines (use ``allow_mixed_frequency``).

**References**

* macroforecast PR #251 (use_sd_inferred_tcodes) -- FRED-SD integration, state-level frequency-policy design.

**Related options**: [`report_only`](#report-only), [`allow_mixed_frequency`](#allow-mixed-frequency), [`reject_mixed_known_frequency`](#reject-mixed-known-frequency)

_Last reviewed 2026-05-16 by macroforecast author._
