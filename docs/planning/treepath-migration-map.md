# Tree-Path Migration Map

This document classifies current macrocast artifacts into the target structure required by `archive/legacy-plans/source/plan_2026_04_09_2358.md`.

## Target buckets
- taxonomy: selectable enum universe
- registries: backing defaults/adapters/contracts
- recipes: named forecasting studies/baselines/benchmarks/ablations
- runs: realized outputs from resolved paths
- legacy/migration: temporary compatibility or transitional code

## Current artifact classification

### Taxonomy
Current canonical taxonomy bundle already present under:
- `macrocast/taxonomy/0_meta/`
- `macrocast/taxonomy/1_data/`
- `macrocast/taxonomy/2_target_x/`
- `macrocast/taxonomy/3_preprocess/`
- `macrocast/taxonomy/4_training/`
- `macrocast/taxonomy/5_evaluation/`
- `macrocast/taxonomy/6_stat_tests/`
- `macrocast/taxonomy/7_importance/`
- `macrocast/taxonomy/8_output_provenance/`

### Registries (current, but still mixed)
Currently mixed under `config/` and package registry modules:
- `config/meta/*.yaml`
- `config/datasets.yaml`
- `config/targets.yaml`
- `config/data_tasks.yaml`
- `config/preprocessing.yaml`
- `config/features.yaml`
- `config/models.yaml`
- `config/evaluation.yaml`
- `config/tests.yaml`
- `config/interpretation.yaml`
- `config/output.yaml`
- `config/verification.yaml`

These should migrate toward a dedicated `registries/` tree later.

### Recipes (missing as first-class layer)
Missing today except for examples/plans.
Future target:
- `recipes/papers/`
- `recipes/baselines/`
- `recipes/benchmarks/`
- `recipes/ablations/`

### Runs (missing as first-class layer)
Current outputs are written ad hoc by runtime/output helpers.
Future target:
- `runs/<resolved_path_or_hash>/...`

### Legacy / migration scaffolding
These should not define final package architecture:
- `macrocast/replication/clss2021.py`
- `macrocast/replication/clss2021_runner.py`
- `docs/tutorials/clss2021-replication.md` (until rewritten around recipes)
- legacy `config/examples/*.yaml` files when they do not encode full path semantics

## Immediate migration implications
1. Taxonomy remains the canonical choice universe.
2. Current `config/*.yaml` stays operational for now, but is classified as transitional registry storage.
3. CLSS helpers remain allowed only as migration scaffolding.
4. New work should avoid adding more paper-specific package code.
5. Benchmark choice must move from resolved ids toward family + options.
