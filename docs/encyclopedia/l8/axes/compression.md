# `compression`

[Back to L8](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``compression`` on sub-layer ``L8_A_export_format`` (layer ``l8``).

## Sub-layer

**L8_A_export_format**

## Axis metadata

- Default: `'none'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 3 option(s)
- Future: 0 option(s)

## Options

### `gzip`  --  operational

Gzip-compress every output file individually.

Each ``.json`` / ``.csv`` becomes ``.json.gz`` / ``.csv.gz``. Reduces artifact size by 60-80% for typical macro panels with marginal write-time overhead. Read-side: pandas / pyarrow auto-detect the gzip extension.

**When to use**

Reducing artifact size for archival; production sweeps.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`none`](#none), [`zip`](#zip)

_Last reviewed 2026-05-05 by macroforecast author._

### `none`  --  operational

No compression (default).

Default. Files are written uncompressed -- cheapest at write time and most convenient for direct browsing / spot-checking. Recommended for development.

**When to use**

Default; cheapest write-time option.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`gzip`](#gzip), [`zip`](#zip)

_Last reviewed 2026-05-05 by macroforecast author._

### `zip`  --  operational

Zip-archive the entire run output directory.

Wraps the run output directory in a single ``.zip`` archive after writing. Convenient for transferring an entire run via email / web upload as a single file. Slightly less efficient than per-file gzip but shipping a single archive matters for some workflows.

**When to use**

Packaging the run for transfer over email / file-sharing services.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`none`](#none), [`gzip`](#gzip)

_Last reviewed 2026-05-05 by macroforecast author._
