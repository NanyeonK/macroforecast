# Tree-Structured Macroforecast Whole Plan

## Phase 0. Package mode lock
- Project type: research package
- Domain mode: macro forecasting primary, extensible to macro-finance / panel / uncertainty-aware settings
- Core runtime objects: pandas/data.frame-first
- Config source: YAML registries + recipe YAMLs
- Validation: lightweight validators + tests
- Outputs: parquet/csv/json/yaml + manifests
- Docs: explanatory only; truth lives in YAML/code/tests
- Replication role: verification and stress test of generic package expressiveness

## 1. Package objective
Rebuild macroforecast around a tree-structured forecasting taxonomy where one fully specified forecasting study is one path through explicit choice layers. All enumerated choices from `plan_2026_04_09_2358.md` must be representable by registries, resolvable into a run recipe, and eventually operable through Python/R hybrid backends.

Successful v1:
- taxonomy exists as package truth
- taxonomy loaders/validators exist
- one recipe schema resolves enum choices + numeric leaf config
- runtime can consume resolved recipes
- custom extension slots exist for all major families
- CLSS 2021 can be expressed as one tree path plus numeric leaf config

## 2. User modes
- Researcher mode: full tree control with sweep axes, paper overlays, custom plugins
- Practitioner mode: safe defaults, minimal recipe path, compact outputs
- Minimal baseline mode: canonical dataset + benchmark + one model + default metrics
- Extension/developer mode: add new registry entries or adapters without core rewrites

## 3. Scope
### In scope
- taxonomy registries
- recipe schema / resolution
- tuning registry and backends
- output recipe registry
- Python/R backend routing
- provenance and failure policy
- CLSS expression as config path

### Out of scope for this cycle
- full implementation of every single model/method immediately
- final notebook publication
- product deployment concerns

## 4. Architecture layers
1. Meta / Registry — path grammar, experiment unit, axes, reproducibility, failure policy
2. Data — dataset/task choices and sample rules
3. Preprocessing — target/X recipes and operation order
4. Design — recipe resolution, enum-vs-numeric separation, custom extension protocol
5. Feature / Model — feature builders, model wrappers, backend routing
6. Training / Forecast Execution — split rules, tuning methods, execution modes
7. Evaluation — metrics, benchmarks, aggregation/reporting
8. Statistical Testing — DM/CW/MCS/CPA and dependence corrections
9. Importance / Interpretation — importance methods, grouping, plotting
10. Output / Provenance — artifacts, manifests, export formats, failure logs

## 5. Invariants
- no-lookahead
- train-window-only fit for learned preprocessing steps
- target alignment explicit
- benchmark denominator explicit
- evaluation scale explicit
- provenance mandatory
- enum choices in taxonomy; numeric/free parameters in recipe leaves

## 6. Package truth sources
- `macrocast/taxonomy/**/*.yaml`
- `config/*.yaml` and future recipe schemas
- Python/R code
- tests
- output artifacts/manifests

## 7. Verification role of replication
- replication verifies generic tree expressiveness
- CLSS 2021 is first strict benchmark path
- replication checks benchmark denominators, preprocessing order, tuning semantics, appendix-style outputs
- replication must not dominate core architecture

## 8. End-to-end closure criteria
- baseline recipe resolves and runs
- metrics compute
- statistical tests compute
- interpretation path computes
- output bundle written with manifest
- one custom extension smoke test passes
- one CLSS-like verification recipe resolves without structural hacks
