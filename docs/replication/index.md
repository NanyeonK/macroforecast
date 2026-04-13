# Replication and Recipes

Replication should not organize the package.
Instead, replication studies should appear as recipes/paths through the generic package.

## Current rule

- package architecture first
- paper recipe second
- no paper-specific execution helper should define package structure
- any residual paper-specific support should be treated as migration scaffolding only

## Current CLSS status

Preferred architectural artifact:
- `recipes/papers/clss2021.yaml`

Study-specific preset/reference code may still exist,
but runtime execution should be understood through recipes/paths and generic package flows.
