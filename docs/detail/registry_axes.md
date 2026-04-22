# Registry Axes

The registry defines the allowed choices that recipes can use.

Current implementation:

- axis definitions live in `macrocast.registry`
- modules expose `AXIS_DEFINITION`
- registry discovery is dynamic
- each axis has a canonical layer, type, values, support status, and default policy

The registry `layer` field is now the source of truth for semantic ownership.
The physical module folder is allowed to lag during migration; for example an
axis can still live under `macrocast/registry/data/` while its canonical layer is
`3_training`.

## Status Rule

`operational` means tested and executable in the documented environment.
Optional dependency requirements must be explicit. Values that need extra
`leaf_config` must fail at compile time if the required input is missing.

## Migration Rule

Compiler compatibility is deliberately broader than registry ownership:

- old recipe paths are still read
- new registry layers define where the axis belongs conceptually
- docs and new recipes should use the canonical owner
- old paths should be deprecated only after tests and examples have migrated
