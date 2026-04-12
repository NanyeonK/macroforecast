# Tree-Path Package Overhaul Plan

## Objective

Restructure `macrocast` to follow `plan_2026_04_09_2358.md` as the primary package architecture.

Target state:
- one path = one fully specified forecasting study
- enumerated choice and numeric/free parameter are separated
- fixed axes and sweep axes are separated
- Python module tree and experiment tree are separated
- package structure is organized around `taxonomy/ + registries/ + recipes/ + runs/`
- paper-specific studies such as CLSS 2021 are expressed as tree paths / recipes, not as package-specific modules

## Package mode lock

- project type: Python research package
- domain mode: generic forecasting package, macro first, extensible later
- runtime objects: pandas/data.frame-first
- config truth: YAML + code + tests
- docs: explanatory only
- replication role: verification of package expressiveness, not package architecture driver
- researcher wins over convenience when they conflict

## Architecture target

### 1. Module tree remains engine-oriented
- `data/`
- `pipeline/`
- `evaluation/`
- `tests_eval/`
- `importance/`
- `cli.py`

### 2. Experiment tree becomes path-oriented
- `taxonomy/` — selectable option universe
- `registries/` — concrete adapters/defaults/contracts backing taxonomy ids
- `recipes/` — named studies / baselines / benchmarks / ablations
- `runs/` — realized outputs keyed by resolved path

### 3. Core rule
Module tree is not experiment tree.
The package code executes resolved studies; it does not hardcode papers.

## Critical redesigns required

### A. Benchmark redesign
Current package uses resolved benchmark ids too early (`ar_bic_expanding`).
Target state:
- benchmark family is selected explicitly (`ar`, `historical_mean`, `factor_model`, ...)
- benchmark options are selected explicitly (`aic`, `bic`, `expanding`, `rolling`, lag order, denominator rule)
- resolved benchmark id is derived later
- benchmark selection is symmetric to model selection

### B. Recipe layer introduction
Current package lacks a real `recipes/` layer.
Target state:
- `recipes/papers/`
- `recipes/baselines/`
- `recipes/benchmarks/`
- `recipes/ablations/`
- each recipe references taxonomy choices + numeric leaves

### C. Runs layer introduction
Current package writes outputs but does not yet frame them as resolved tree-path realizations.
Target state:
- `runs/<resolved_path_or_hash>/...`
- manifest contains path string, recipe id, fixed axes, sweep axes, resolved benchmark/model choices

### D. CLSS demotion from package helper to recipe example
Current package contains `macrocast.replication.clss2021*` helpers.
Target state:
- generic package first
- CLSS 2021 represented as one recipe/path under `recipes/papers/`
- any CLSS-specific helper retained only as migration scaffolding, then removed or reduced to compatibility wrapper

## Whole-package streams

1. SP-T0: architecture lock and migration policy
2. SP-T1: taxonomy canonicalization
3. SP-T2: registries layer introduction
4. SP-T3: recipe schema and tree-path resolver
5. SP-T4: benchmark/model symmetry redesign
6. SP-T5: execution + runs layer migration
7. SP-T6: output/provenance path-aware closure
8. SP-T7: verification migration from paper helper to recipe-based path
9. SP-T8: docs and examples migration

## Success criteria

The overhaul is complete when:
- taxonomy ids are the canonical enum universe
- registries provide backing defaults/adapters
- recipes encode named studies without hardcoded paper modules
- resolved tree path compiles into one executable package contract
- benchmark selection is symmetric with model selection
- outputs are written under path-aware runs
- CLSS 2021 is expressible as one recipe/path, not as package-specific core code
