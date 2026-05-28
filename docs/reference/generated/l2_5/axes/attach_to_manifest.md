# `attach_to_manifest`

[Back to L2.5](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``attach_to_manifest`` on sub-layer ``L2_5_Z_export`` (layer ``l2_5``).

## Sub-layer

**L2_5_Z_export**

## Axis metadata

- Default: `True`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 2 option(s)
- Future: 0 option(s)

## Options

### `true`  --  operational

Embed diagnostic artifacts into manifest.json's diagnostics index.

Default. Diagnostic file paths and content hashes are recorded in ``manifest.diagnostics``, so ``macroforecast.replicate(manifest_path)`` validates that every diagnostic re-runs to a bit-identical artifact. Required for reproducibility-critical sweeps.

**When to use**

Default; ensures replication includes the diagnostics.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`false`](#false)

_Last reviewed 2026-05-05 by macroforecast author._

### `false`  --  operational

Keep diagnostic artifacts outside the manifest hash chain.

Files are still written to the run output directory but are not referenced by the manifest, so ``replicate()`` does not validate their hashes. Lighter-weight when the diagnostic surface is large and reproducibility is not the headline concern.

**When to use**

Long-running production sweeps where diagnostics blow up the manifest size.

**References**

* macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'

**Related options**: [`true`](#true)

_Last reviewed 2026-05-05 by macroforecast author._
