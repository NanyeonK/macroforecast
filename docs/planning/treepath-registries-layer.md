# TP-003 Registries Layer

This first-pass registries layer does not yet replace existing `config/*.yaml` files.
Instead it creates a distinct package bucket for backing registries and records where transitional config responsibilities currently live.

Current initial registry buckets:
- `registries/meta/`
- `registries/data/`
- `registries/training/`
- `registries/evaluation/`
- `registries/output/`

Each registry file currently contains:
- stable registry id
- transitional source path
- role description

This lets the package distinguish:
- taxonomy = selectable enum universe
- registries = backing defaults/adapters/contracts
- config/*.yaml = transitional storage still in use during migration
