# Tree-Path Runtime Next Plan

## Current status after shared constructor extraction

Progress is now concentrated in three remaining areas:

1. Registries truth migration
- selected registries now have inline payload truth
- next step is to finish migrating remaining operational storage away from config/*.yaml as canonical source
- for migrated domains, config/*.yaml should be treated as compatibility or mirror material rather than primary truth

2. Runtime provenance / contract deepening
- compile/run/output now preserve recipe and path metadata
- tree_context now carries fixed/sweep summaries and values for compiled recipe flows
- next step is to propagate richer tree semantics consistently through more runtime/result surfaces

3. Docs synchronization and cleanup
- planning/runtime docs should stay aligned with active implementation and archive state
- simple staged testing path now exists through macrocast_start()

## Immediate engineering priority

Complete the remaining registries truth migration, then finish final checkpoint packaging.
