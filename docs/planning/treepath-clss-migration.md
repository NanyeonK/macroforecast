# TP-008 CLSS Migration

CLSS 2021 should not remain part of package core architecture.

Current migration state:
- `recipes/papers/clss2021.yaml` is the canonical paper-path artifact
- package-specific CLSS runner helper has been removed
- `macrocast.replication.clss2021` remains only as a study-specific preset/reference helper, not as runtime execution architecture

Implication:
- new architecture work should land in taxonomy / registries / recipes / runs
- CLSS should be exercised through recipe/path compilation and generic runtime flows
