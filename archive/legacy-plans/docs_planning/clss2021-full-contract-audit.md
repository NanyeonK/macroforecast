# CLSS 2021 Full Contract Audit

## Objective

Question only:
Can full CLSS paper be implemented equivalently inside `macroforecast` package?

Replication is used here as verification and implementation audit.

## Paper-fixed items currently evidenced in repo materials

These items are supported by current local CLSS-specific code/docs and should be treated as baseline candidate fixed settings until contradicted by paper body/appendix extraction.

### Domain mode
- macro forecasting package
- monthly macro panel
- pseudo-OOS forecasting

### Canonical dataset
- FRED-MD monthly panel
- paper-oriented vintage policy repeatedly points to `2018-02`

### Core horizons
- `h ∈ {1, 3, 6, 9, 12, 24}`
- evidenced by `macrocast/replication/clss2021.py`
- older scripts/tutorials conflict on reduced subsets; do not treat those as paper truth

### Information-set universe
Current package preset exposes 16 labels:
- F
- F-X
- F-MARX
- F-MAF
- F-Level
- F-X-MARX
- F-X-MAF
- F-X-Level
- F-X-MARX-Level
- X
- MARX
- MAF
- X-MARX
- X-MAF
- X-Level
- X-MARX-Level

### Package preset defaults now encoded
From `macrocast/replication/clss2021.py`:
- `P_Y = 12`
- `K = 8`
- `P_MARX = 12`
- model preset universe: `RF`, `EN`, `AL`, `FM`, `KRR`, `SVR`

### Benchmark/evaluation pieces currently implemented in package
- AR benchmark model exists: `macrocast/pipeline/r_models.py::ARModel`
- Clark-West test exists: `macrocast/evaluation/cw.py`
- Diebold-Mariano test exists: `macrocast/evaluation/dm.py`
- relative MSFE and OOS R2 functions exist: `macrocast/evaluation/metrics.py`
- path-average target scheme exists in `ForecastExperiment`

## Critical repo conflicts discovered

1. `docs/tutorials/clss2021-replication.md`
- claims paper-faithful settings
- hardcodes reduced values like `N_FACTORS=4`, `N_LAGS=2`, `P_MARX=4`
- presents mixed reduced/full narrative

2. `scripts/clss2021_full_run.py`
- not full paper run
- targets reduced to 11
- horizons reduced to `[1, 3, 6, 12]`
- info sets reduced to 6
- OOS starts at `1999-01-01`
- uses `n_factors=4`, `n_lags=2`, `p_marx=4`

3. `scripts/clss2021_overnight_run.py`
- even more reduced
- `p_marx=1`
- 4 targets
- 4 horizons
- RF only

4. `macrocast/replication/clss2021.py`
- package preset defaults disagree with tutorial/scripts
- should now be treated as package truth candidate, not final paper truth

## Full package support matrix

### Preprocessing / feature construction
Supported now
- McCracken-Ng transforms via FRED loaders
- factor-type X / MARX / none
- raw-X append
- MARX append
- MAF via `factor_type="MARX"`
- level append
- path-average target scheme
- train-window-only feature fitting through `ForecastExperiment` + `FeatureBuilder`

Needs full-paper verification
- exact MARX definition vs paper appendix
- exact MAF definition vs paper appendix
- exact target transformation scale used for reported tables
- exact benchmark denominator scale used in reported relative RMSE tables

### Models
Supported now
- RF
- EN
- AL
- FM/ARDI
- KRR
- SVR
- AR benchmark

Still must verify against full paper requirements
- whether these are all models used in target paper headline tables
- whether any BT/LB variants are required by the exact paper being replicated
- whether package hyperparameter defaults match appendix search grids

### Hyperparameter-selection methods
Supported now
- K-fold CV via `CVScheme.KFOLD`
- BIC via `CVScheme.BIC`
- model-specific kwargs grids for RF/KRR/SVR

Still must verify
- exact search grids from appendix
- exact retuning cadence
- whether package current RF defaults match paper appendix
- whether EN/AL/FM package bridge settings match paper appendix

### Evaluation
Supported now
- MSFE
- MAE
- relative MSFE
- OOS R2
- DM
- CW
- MCS

Still must verify
- exact relative RMSE / RMSFE table definition used in paper
- whether full paper requires pooled summaries beyond current helper paths
- whether current package CLSS runner uses explicit AR benchmark denominator rather than F baseline

## Package-side critical issues before full run

1. CLSS helper path still reduced-audit oriented
- no full-scope runner
- no full target/model/info-set/horizon orchestration

2. CLSS helper manifest semantics need tightening
- benchmark currently not propagated as fully explicit benchmark id in helper path

3. CLSS summary helper still wrong for full paper
- currently easier to summarize vs feature-set `F`
- full paper path must summarize vs explicit AR benchmark where required

4. Legacy tutorial/scripts must not be trusted as implementation truth
- treat them as evidence of prior attempts only

## Ambiguities requiring controlled variant runs

These must be tested if paper body/appendix does not fully resolve them:
- `P_Y`, `K`, `P_MARX` values if appendix and legacy artifacts disagree
- full target list
- whether headline results use direct only, path-average only, or both
- exact benchmark denominator and relative RMSE construction
- retuning cadence / grid search frequency
- whether some figures/tables pool over model classes beyond current package presets

## Planner-form execution-ready issue stack

### I-FULL-01. Lock paper contract
Files to create/modify:
- `config/plans/clss2021-paper-contract.yaml`
- `docs/planning/clss2021-full-contract-audit.md`
Done when:
- fixed vs ambiguous settings explicitly listed
- all repo conflicts documented

### I-FULL-02. Build support matrix
Files:
- `config/plans/clss2021-support-matrix.yaml`
Done when:
- every preprocessing/model/hyperparameter/evaluation component marked full/partial/missing with evidence

### I-FULL-03. Patch package gaps for full-scope runner
Likely files:
- `macrocast/replication/clss2021_runner.py`
- `macrocast/output/manifest.py`
- `macrocast/evaluation/*`
Done when:
- explicit AR-benchmark summaries
- full artifact bundle
- explicit benchmark ids in manifest

### I-FULL-04. Add ambiguity-run driver
Files:
- `scripts/clss2021_variant_runs.py` or package-native equivalent
Done when:
- multiple candidate contracts can be compared deterministically

### I-FULL-05. Full replication runner
Files:
- package-native CLSS full runner
- tests for full-run contract assembly
Done when:
- full target/model/info-set/horizon grid executable from package path

### I-FULL-06. Only then notebook/docs
Done when:
- notebook documents verified package path, not legacy mixed-fidelity path
