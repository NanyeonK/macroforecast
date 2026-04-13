# CLSS 2021 Full-Paper Package Patch Plan

## Phase 0. Package mode lock

- project type: research package
- runtime objects: pandas / numpy / sklearn-style + R bridge models
- config source: yaml
- validation: lightweight validators + tests
- outputs: parquet/csv/json + manifests
- docs: minimal markdown until package path stabilizes
- replication role: verification of paper-equivalent implementation
- domain mode: macro forecasting

## Whole package objective

Question:
Can full CLSS paper be implemented equivalently through `macrocast` package surfaces rather than legacy scripts?

Successful v1 for this patch cycle:
- canonical paper contract exists in YAML
- ambiguity register exists in YAML
- full-scope CLSS runner exists in package
- runner writes forecast/eval/test/manifest artifacts with explicit AR benchmark semantics
- package can run ambiguity variants without ad hoc script duplication
- tests cover contract assembly and summary semantics

## Scope

In scope:
- canonical CLSS paper contract registry
- ambiguity register
- full runner orchestration
- explicit AR-benchmark-relative summary path
- explicit benchmark id provenance in outputs
- variant runner for ambiguous settings
- tests for runner/contract/summary behavior

Out of scope for this patch cycle:
- final notebook publication
- final docs/tutorial prose
- empirical interpretation writeup

## Invariants

- no-lookahead
- train-window-only fit
- target alignment explicit
- benchmark denominator explicit
- provenance mandatory
- package truth = yaml + code + tests

## Separate plans

### SP-1. Canonical paper contract layer
Objective:
- create machine-readable CLSS paper contract
Inputs:
- body/appendix extraction
- current package audit
Outputs:
- `config/plans/clss2021-paper-contract.yaml`
- `config/plans/clss2021-ambiguities.yaml`
Code modules affected:
- none initially
Tests required:
- registry load/shape test
Done criteria:
- all fixed vs ambiguous settings explicit

### SP-2. Full runner contract
Objective:
- package-native CLSS full-run interface
Inputs:
- paper contract yaml
- existing CLSS presets/models
Outputs:
- full runner entry point
- full run artifact bundle
YAML touched:
- paper contract yaml
- support matrix yaml maybe updated
Code modules affected:
- `macrocast/replication/clss2021_runner.py`
- `macrocast/replication/__init__.py`
Tests required:
- full-run contract assembly test
- artifact path test
Done criteria:
- full target/horizon/info-set/model grid assembled from YAML without legacy scripts

### SP-3. Benchmark-relative evaluation layer
Objective:
- explicit AR benchmark summary semantics
Inputs:
- ResultSet forecasts from runner
- AR benchmark records
Outputs:
- relative RMSFE / RMSE vs AR tables
- optional CW/DM comparison tables
Code modules affected:
- `macrocast/replication/clss2021_runner.py`
- maybe `macrocast/evaluation/*`
Tests required:
- benchmark-relative summary unit tests
Done criteria:
- no F-baseline shortcut in full paper path

### SP-4. Provenance closure
Objective:
- explicit benchmark id and contract snapshot in manifest
Code modules affected:
- `macrocast/output/manifest.py`
- `macrocast/replication/clss2021_runner.py`
Tests required:
- manifest benchmark id semantics test
Done criteria:
- manifest fields align with full paper contract

### SP-5. Ambiguity variant runner
Objective:
- deterministic comparison runs for unresolved settings
Outputs:
- variant run helper
- comparison artifact bundle
Code modules affected:
- `macrocast/replication/clss2021_runner.py` or sibling module
Tests required:
- ambiguity-run assembly test
Done criteria:
- multiple contract variants runnable from package path only

## Execution-ready issue stack

### I-CLSS-01
Create `config/plans/clss2021-paper-contract.yaml`
Must include:
- dataset/vintage
- target universe
- horizons
- info-set universe
- model universe
- benchmark definition
- target scheme usage
- hyperparameter-selection rules
- output table definitions

### I-CLSS-02
Create `config/plans/clss2021-ambiguities.yaml`
Must include:
- setting name
- baseline candidate
- alternatives to test
- reason ambiguous
- downstream tables/figures affected

### I-CLSS-03
Patch `macrocast/replication/clss2021_runner.py`
Add:
- load paper contract
- assemble full run grid
- explicit AR benchmark path
- explicit relative RMSFE/RMSE summary vs AR
- contract snapshot write

### I-CLSS-04
Patch manifest/provenance semantics
Add:
- explicit benchmark ids
- contract hash/path
- target scheme fields
- info-set/model coverage fields

### I-CLSS-05
Add ambiguity variant runner
Add:
- package-native variant run API
- comparison summaries across candidate contracts

### I-CLSS-06
Add tests
Files likely:
- `tests/replication/test_clss2021_runner.py`
- new tests for contract/ambiguity YAML
- new tests for manifest benchmark semantics

## Reverse-plan check

Dependency order is correct because:
1. cannot build full runner before canonical contract exists
2. cannot trust summary semantics before benchmark definition locked
3. cannot run ambiguity variants before baseline contract exists
4. notebook/docs should wait until package path is stable
