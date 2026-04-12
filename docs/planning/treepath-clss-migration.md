# TP-008 CLSS Migration

CLSS 2021 should not remain part of package core architecture.

Current migration state:
- `recipes/papers/clss2021.yaml` now exists as the canonical future paper-path artifact
- `macrocast.replication.clss2021` and `macrocast.replication.clss2021_runner` remain available only as migration scaffolding
- public package direction is generic tree-path package first, paper recipes second

Implication:
- new architecture work should land in taxonomy / registries / recipes / runs
- CLSS-specific helper code should not expand further unless needed strictly for migration and verification
