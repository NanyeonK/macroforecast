# TP-010 Registries Operationalization

This step has moved selected domains from metadata-only registry pointers toward inline registry truth.

Current canonical registry truth now lives directly in selected files under:
- `registries/meta/`
- `registries/data/`
- `registries/training/`
- `registries/evaluation/`
- `registries/output/`

For migrated domains, runtime loaders prefer inline registry payloads over legacy source indirection.
Legacy `config/*.yaml` remains present during migration, but selected registry files now carry canonical live content directly.
