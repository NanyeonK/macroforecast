# Macrocast Master Rebuild Alignment

Status: reboot-stage alignment memo
Date: 2026-04-14
Purpose: reconcile the active high-level package plan with the reboot-stage doctrines that were fixed after the reset

## Documents being aligned

This note aligns five active inputs:

1. `plans/plan_04_14_1734.md`
- the broad high-level package development plan

2. `plans/stage0-grammar-first.md`
- the doctrine that Stage 0 fixes grammar and contract before registry content

3. `plans/raw-data-architecture.md`
- the raw-data subsystem plan

4. `plans/raw-data-contracts.md`
- the raw-data contract registry

5. public-front purpose statement
- now pinned in `README.md`, `docs/index.md`, and `macrocast/__init__.py`
- "Given a standardized macro dataset adapter and a fixed forecasting recipe, compare forecasting tools under identical information set, sample split, benchmark, and evaluation protocol."

This note also respects one standing docs rule:
- `docs/` is public only
- internal planning lives under `plans/`

## 1. The package mission

The package mission is now fixed as follows:

> Given a standardized macro dataset adapter and a fixed forecasting recipe, compare forecasting tools under identical information set, sample split, benchmark, and evaluation protocol.

This sentence is the front-door purpose of the rebuilt package.

It means the package is not primarily:
- a model zoo
- a convenience wrapper around many forecasting tricks
- a paper-specific replication scaffold
- a generic experiment-config container

It is primarily:
- a fair macro-forecast benchmarking package
- built around standardized dataset adapters
- built around fixed recipe-defined comparison conditions
- designed to isolate tool differences from design inconsistency

## 2. What `plan_04_14_1734.md` already gets right

The high-level plan already contains the correct macro direction.

Its strongest points are:
- fair, reproducible, extensible benchmarking framework
- one path = one fully specified forecasting study
- enumerated choices separated from numeric parameters
- fixed axes separated from sweep axes
- module tree separated from experiment tree
- staged development rather than full-scope explosion at v1
- deliberate v1 restriction to FRED-MD and a minimal forecasting stack

These should be preserved.

## 3. What must be sharpened by the reboot doctrines

The high-level plan is directionally right, but the reboot established sharper package rules.

### 3.1 Stage 0 must be interpreted more strictly

The plan talks about experimental grammar.
The reboot doctrine sharpens that into a hard rule:

- Stage 0 does not fill registry content
- Stage 0 fixes the grammar and contract registries must obey

So the package must not begin rebuild by expanding registries or option inventories.
It must begin by fixing:
- route grammar
- fixed-vs-varying grammar
- comparison grammar
- execution-posture grammar
- downstream extension boundaries

### 3.2 Raw-data architecture is now explicit, not implicit

The old high-level plan says the package will use standardized datasets.
The reboot makes that operationally precise:

- raw-data layer is a separate subsystem
- dataset-specific loaders are distinct from shared vintage management
- provenance/index tracking is mandatory
- pandas-first outputs are mandatory
- FRED-MD and FRED-QD are stable first-class targets
- FRED-SD is provisional until verified

### 3.3 Public docs and internal plans must remain separate

The high-level plan says little about docs governance.
The reboot fixes this:

- `docs/` is public/package-facing only
- internal architecture notes must stay in `plans/`
- public docs should benchmark their information architecture against statsmodels stable docs

## 4. Reconciled package interpretation

The rebuilt package should be interpreted as:

A research-oriented macroeconomic forecasting package whose core job is to hold the comparison environment fixed while letting forecasting tools vary in explicit, structured ways.

In practice this means:
- raw dataset access is standardized
- study design is recipe-defined
- experimental grammar is fixed before registry content expansion
- model/tool comparisons happen under identical design conditions
- outputs are provenance-safe and reproducible

## 5. Reconciled version-1 scope

The high-level plan's v1 restriction is accepted, with reboot-stage precision added.

### Version 1 should do
- FRED-MD only
- revised monthly information set
- single-target point forecasting
- a small set of standard benchmarks
- a small set of initial models
- expanding-window evaluation
- a compact metric set
- reproducible artifact writing

### Version 1 should also lock
- the Stage 0 grammar contract
- the raw-data contract
- recipe identity and artifact identity
- fixed-vs-varying distinction
- comparison contract semantics

### Version 1 should not try to do
- FRED-QD execution support before MD path is stable
- FRED-SD execution support before MD path is stable
- rich importance stack before baseline forecasting path is stable
- expansive registry content build-out before grammar is fixed
- replication bundles as package-defining core logic

## 6. Revised development order

This is the aligned implementation order.

### Phase A. Package front and doctrine lock
Already started.

Required outputs:
- package purpose statement on front surfaces
- internal Stage 0 doctrine note
- internal raw-data architecture note

### Phase B. Stage 0 grammar contract
This is the immediate next major step.

Required outputs:
- Stage 0 object list
- meaning of each object
- downstream consumption rules
- allowed registry extension points
- forbidden ad hoc fields
- compatibility-mirror decision

### Phase C. Raw-data subsystem
Build the raw layer first because standardized dataset adapters are part of the package mission.

Required outputs:
- `macrocast/raw/` skeleton
- MD stable loader
- QD stable loader
- shared vintage manager
- raw cache and manifest/index
- SD provisional adapter contract

### Phase D. Minimal execution contract
Only after grammar and raw layer exist.

Required outputs:
- one recipe schema
- one run identity
- one artifact layout
- one minimal single-study execution path

### Phase E. Minimal forecasting pipeline
Required outputs:
- single-target FRED-MD point forecast path
- benchmark comparison
- compact metrics
- reproducible saved outputs

### Phase F. Registry expansion
Only after the previous phases are stable.

Required outputs:
- registries authored to fit the already-fixed grammar
- no registry-first redesign

## 7. Reconciled architecture stack

The high-level plan's layer idea remains good, but should be interpreted through the reboot rules.

Recommended stack:
- Stage 0: grammar and contract
- Layer 1: raw data
- Layer 2: preprocessing
- Layer 3: design / target-x / sample definition
- Layer 4: training / model / validation execution
- Layer 5: evaluation
- Layer 6: statistical tests
- Layer 7: importance / interpretation
- Layer 8: output / provenance

Important clarification:
- Stage 0 is not just another registry layer
- Stage 0 sits above later content layers because it defines their language

## 8. What "one path = one study" now means operationally

This sentence from the high-level plan must be interpreted in a stricter reboot-safe way.

It means:
- one study path records enumerated design identity
- numeric parameters live in recipe/config payloads, not path explosion
- one path corresponds to one complete comparison design
- multi-run bundles are orchestrated objects, not fake single paths
- wrapper-owned families should remain wrapper-owned rather than being jammed into single-run path grammar

## 9. What registries are allowed to become

After alignment, registries are allowed to be:
- controlled inventories of admissible content
- plugged into already-defined grammar slots
- versionable and auditable

Registries are not allowed to become:
- substitutes for execution grammar
- dumping grounds for route semantics
- places where local implementation exceptions become public schema
- uncontrolled YAML trees that define package behavior by accident

## 10. Docs strategy after alignment

Public docs should be developed with statsmodels stable docs as continuing IA reference.

This means:
- clear front-page purpose
- install/getting-started/user-guide/examples/api separation
- internal planning kept off public docs
- public docs should explain package surfaces, not expose reboot diary content

Docs work should happen only after the package grammar and minimal raw/execution contract are stable enough to document honestly.

### 10.1 Code-and-docs coupling rule

The rebuild must not treat code and docs as separate optional tracks.

Rule:
- when a new code surface becomes part of the rebuilt package surface, corresponding public documentation must be written in `docs/`
- that documentation should explain not only the existence of the code, but also its purpose, contract, usage pattern, and relationship to adjacent package surfaces
- internal planning belongs in `plans/`, but package behavior explanations belong in `docs/`

This means the package should not accumulate undocumented execution surfaces and defer explanation indefinitely.
The target state is: rebuilt code plus detailed public documentation that makes the code legible to users.

## 11. The single most important aligned rule

The package should scale by extending content inside a fixed language, not by letting content invent the language.

That is the point where the high-level plan and the reboot doctrines fully meet.

## 12. Immediate next action

Write `plans/stage0-grammar-contract.md`.

That document should translate the current doctrines into an executable architectural contract by defining:
- Stage 0 object names
- Stage 0 object meanings
- inheritance rules
- route ownership rules
- fixed vs varying grammar
- comparison grammar
- execution-posture grammar
- registry extension boundaries

## Final alignment summary

Keep from the original plan:
- fair benchmarking mission
- one-study-per-path idea
- staged development
- v1 restriction
- module-tree vs experiment-tree separation

Add from the reboot doctrines:
- Stage 0 grammar-first rule
- raw-data subsystem contract
- docs/public separation
- statsmodels-style public docs reference
- explicit warning against registry-first package growth

The rebuild should therefore proceed as:
- mission first
- grammar second
- raw-data third
- minimal execution fourth
- forecasting pipeline fifth
- registry expansion later
