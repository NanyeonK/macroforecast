# `mixed_frequency_representation`

[Back to L2](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``mixed_frequency_representation`` on sub-layer ``l2_a`` (layer ``l2``).

## Sub-layer

**l2_a**

## Axis metadata

- Default: `'calendar_aligned_frame'`
- Sweepable: True
- Status: operational

## Operational status summary

- Operational: 5 option(s)
- Future: 0 option(s)

## Options

### `calendar_aligned_frame`  --  operational

Default: keep selected mixed-frequency columns on the experiment calendar.

When a panel mixes monthly and quarterly columns (FRED-SD by default; any custom panel that declares per-column native frequency in metadata), the default representation flattens all columns to the experiment calendar via the L2.A ``quarterly_to_monthly_rule`` / ``monthly_to_quarterly_rule`` alignment rules. The panel emerges as a single rectangular frame; downstream layers see a uniform sampling grid.

**When to use**

Default for mixed-frequency studies; pairs with the canonical L2.A alignment rules.

**References**

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'

**Related options**: [`drop_unknown_native_frequency`](#drop-unknown-native-frequency), [`drop_non_target_native_frequency`](#drop-non-target-native-frequency), [`native_frequency_block_payload`](#native-frequency-block-payload), [`mixed_frequency_model_adapter`](#mixed-frequency-model-adapter)

_Last reviewed 2026-05-04 by macroforecast author._

### `drop_unknown_native_frequency`  --  operational

Drop columns whose native frequency cannot be inferred.

Restricts the panel to columns whose native sampling rate is either declared in the L1 metadata or detectable from the FRED-SD workbook. Columns with unknown native frequency are dropped before any frequency-alignment rule fires.

**When to use**

Studies that demand strict provenance over per-column native frequency.

**References**

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'

**Related options**: [`calendar_aligned_frame`](#calendar-aligned-frame), [`drop_non_target_native_frequency`](#drop-non-target-native-frequency)

_Last reviewed 2026-05-04 by macroforecast author._

### `drop_non_target_native_frequency`  --  operational

Keep only columns whose native frequency matches the experiment frequency.

Restricts the panel to columns whose native sampling rate equals the L1 ``frequency``. For a monthly experiment the quarterly columns are dropped (and vice versa). Useful when the user wants a strict single-frequency panel without any interpolation artifacts.

**When to use**

Strict monthly-only or quarterly-only panels; single-frequency benchmarks.

**References**

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'

**Related options**: [`calendar_aligned_frame`](#calendar-aligned-frame), [`drop_unknown_native_frequency`](#drop-unknown-native-frequency)

_Last reviewed 2026-05-04 by macroforecast author._

### `native_frequency_block_payload`  --  operational

Emit per-frequency block metadata for downstream models.

Keeps the panel intact (no alignment / drop) and instead publishes a ``fred_sd_native_frequency_block_payload.json`` manifest entry that lists each column's native frequency. Models that consume mixed-frequency input directly (e.g. MIDAS, mixed-frequency factor models) can read this metadata from ``context['auxiliary_payloads']``.

**When to use**

Researcher-owned MIDAS / mixed-frequency factor model studies.

**References**

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'

**Related options**: [`mixed_frequency_model_adapter`](#mixed-frequency-model-adapter), [`calendar_aligned_frame`](#calendar-aligned-frame)

_Last reviewed 2026-05-04 by macroforecast author._

### `mixed_frequency_model_adapter`  --  operational

Block payload + a model-adapter contract for MIDAS-style fits.

Strictest option: emits the per-frequency block payload (see ``native_frequency_block_payload``) plus a model-adapter contract that the L4 model_family must honour. The adapter validates that the registered ``model_family`` either declares MIDAS-style mixed-frequency support or registers via ``mf.custom_model`` with the appropriate ``auxiliary_payloads`` consumption. Runtime writes ``fred_sd_mixed_frequency_model_adapter.json`` with the adapter contract details.

**When to use**

Built-in MIDAS families (``midas_almon``, ``midasr``) or registered custom mixed-frequency models.

**References**

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'

**Related options**: [`native_frequency_block_payload`](#native-frequency-block-payload), [`calendar_aligned_frame`](#calendar-aligned-frame)

_Last reviewed 2026-05-04 by macroforecast author._
