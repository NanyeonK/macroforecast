# `manifest_format`

[Back to L8](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``manifest_format`` on sub-layer ``L8_C_provenance`` (layer ``l8``).

## Sub-layer

**L8_C_provenance**

## Axis metadata

- Default: `'json'`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 3 option(s)
- Future: 0 option(s)

## Options

### `json`  --  operational

Manifest written as a single JSON document (default).

Default. Round-trips cleanly into Python / JS / R; preserves nested structure. The natural choice for every consumer that uses ``macroforecast.replicate``.

**When to use**

Default; round-trips cleanly. Selecting ``json`` on ``l8.manifest_format`` activates this branch of the layer's runtime.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`yaml`](#yaml), [`json_lines`](#json-lines)

_Last reviewed 2026-05-05 by macroforecast author._

### `json_lines`  --  operational

Manifest written as JSONL (one cell per line).

Streaming-friendly format: each cell becomes one JSON object on its own line. Sweep manifests with thousands of cells stay parseable line-by-line without loading the entire manifest into memory.

**When to use**

Sweep manifests with thousands of cells; streaming consumers.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`json`](#json), [`yaml`](#yaml)

_Last reviewed 2026-05-05 by macroforecast author._

### `yaml`  --  operational

Manifest written as YAML.

Hand-readable alternative to JSON. Commits cleaner in git diffs; useful when manifests are expected to be reviewed by humans (paper supplementary materials, code reviews of recipe changes).

**When to use**

Hand-readable manifests; paper supplementary materials.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`json`](#json), [`json_lines`](#json-lines)

_Last reviewed 2026-05-05 by macroforecast author._
