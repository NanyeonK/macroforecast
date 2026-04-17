# Macrocast Stage 0 Grammar Contract

Status: reboot-stage contract
Date: 2026-04-14
Purpose: define exactly what Stage 0 fixes before any registry content expansion or execution-layer buildout

## 0. Contract statement

Stage 0 does not populate registry inventories.
Stage 0 fixes the execution language that later registries, recipes, and runs must obey.

In other words:
- Stage 0 fixes grammar
- later layers fill content inside that grammar

## 1. What Stage 0 fixes

Stage 0 fixes seven things.

1. `study_mode`
- what kind of study is being expressed at the highest structural level

2. `replication_input`
- whether the run is ordinary package use or an explicit replication override path

3. `design_frame`
- the fixed-vs-varying structure of the study

4. `comparison_contract`
- what fairness conditions must be held identical when comparing tools

5. `execution_posture`
- what kind of execution object should be emitted downstream

6. `registry_scope_contract`
- which later registries are allowed to provide content, and where

7. `compatibility_mirrors`
- which transitional fields may exist only as derived compatibility surfaces

These are Stage 0 objects.
They are not registry payload.

## 2. What Stage 0 does not fix

Stage 0 does not decide:
- which dataset ids exist
- which target ids exist
- which model ids exist
- which benchmark ids exist
- which preprocessing recipe ids exist
- which metrics or tests are implemented
- which package release-specific examples exist

Those belong to later content layers.

## 3. Stage 0 object definitions

### 3.1 `study_mode`

Purpose:
- classify the top structural shape of the study before later content is filled

Allowed values in v1:
- `single_path_benchmark_study`
- `controlled_variation_study`
- `orchestrated_bundle_study`
- `replication_override_study`

Meaning:
- `single_path_benchmark_study`
  - one forecasting study with one fixed comparison environment and one tool comparison surface
- `controlled_variation_study`
  - one baseline design plus explicit, limited variation dimensions
- `orchestrated_bundle_study`
  - multiple coordinated runs managed by a wrapper/orchestrator rather than a single execution path
- `replication_override_study`
  - execution starts from a locked replication source rather than ordinary package-default study construction

Rule:
- `study_mode` is structural and is fixed before later content selection

### 3.2 `replication_input`

Purpose:
- separate ordinary package use from explicit replication-driven override

Form:
- `none`
- or a structured replication source reference

Example structured fields:
- `source_type`
- `source_id`
- `locked_constraints`
- `override_reason`

Rule:
- ordinary use is the default
- replication is explicit, not the normal front door
- replication may constrain later grammar objects, but does not redefine Stage 0 itself

### 3.3 `design_frame`

Purpose:
- define how the study separates fixed design from intentionally varied design

Required subobjects:
- `fixed_design`
- `varying_design`
- `derived_design_shape`

#### `fixed_design`
Meaning:
- choices that define the common comparison environment and must stay constant for a fair tool comparison

In v1, these are expected to cover categories like:
- dataset adapter family
- information set regime
- sample split family
- benchmark family
- evaluation protocol family
- forecast task type

Rule:
- if changing an element makes the comparison environment no longer identical, it belongs in `fixed_design`

#### `varying_design`
Meaning:
- choices intentionally allowed to vary within the study

In v1, these may include categories like:
- model family
- feature recipe family
- controlled preprocessing variation
- tuning strategy family
- horizon set when the study explicitly defines horizon as a sweep axis

Rule:
- `varying_design` is allowed variation, not accidental drift
- every varying dimension must be named explicitly

#### `derived_design_shape`
Meaning:
- Stage 0's interpretation of the overall study shape after reading fixed/varying structure

Allowed values in v1:
- `one_fixed_env_one_tool_surface`
- `one_fixed_env_multi_tool_surface`
- `one_fixed_env_controlled_axis_variation`
- `wrapper_managed_multi_run_bundle`

Rule:
- downstream routing can consume `derived_design_shape`
- registries cannot redefine it

### 3.4 `comparison_contract`

Purpose:
- define what must remain aligned for results to count as a fair comparison

Required fields in v1:
- `information_set_policy`
- `sample_split_policy`
- `benchmark_policy`
- `evaluation_policy`

Meaning:
- `information_set_policy`
  - what identical information means for the study
- `sample_split_policy`
  - what identical temporal design means
- `benchmark_policy`
  - what common benchmark surface is required
- `evaluation_policy`
  - what common metric/test protocol is required

Hard rule:
- a comparison is invalid if these are not shared across compared tools unless the study is explicitly a controlled-variation design that says otherwise

### 3.5 `execution_posture`

Purpose:
- define what downstream execution object should be built

Allowed values in v1:
- `single_run_recipe`
- `single_run_with_internal_sweep`
- `wrapper_bundle_plan`
- `replication_locked_plan`

Meaning:
- `single_run_recipe`
  - downstream should emit one runnable study object
- `single_run_with_internal_sweep`
  - downstream should emit one study object containing controlled internal variation
- `wrapper_bundle_plan`
  - downstream should emit a bundle/orchestrator plan, not pretend this is one executable path
- `replication_locked_plan`
  - downstream should respect locked replication constraints

Rule:
- execution posture is fixed at Stage 0
- later content registries do not change a single run into a bundle by accident

### 3.6 `registry_scope_contract`

Purpose:
- define where later registries are allowed to contribute content

Allowed downstream content areas in v1:
- data content registries
- preprocess content registries
- feature/model content registries
- evaluation/test content registries
- output/provenance content registries

Hard rules:
- registries may provide admissible values
- registries may not introduce new top-level Stage 0 objects
- registries may not redefine route ownership
- registries may not smuggle execution semantics into content ids

### 3.7 `compatibility_mirrors`

Purpose:
- allow temporary derived fields for transition without letting them become canonical grammar

Candidate mirror in v1:
- legacy `experiment_unit`-style field only if derived from Stage 0 objects

Hard rules:
- compatibility mirrors are derived only
- mirrors cannot be canonical truth
- new behavior must not be designed around mirrors

## 4. Inheritance rules

Stage 0 values must be inherited downward.
Later layers may consume them, but may not redefine them locally.

Inheritance model:
- Stage 0 -> raw/data layer
- Stage 0 -> design layer
- Stage 0 -> execution layer
- Stage 0 -> evaluation/test layer
- Stage 0 -> output/provenance layer

Examples:
- `comparison_contract.sample_split_policy` fixed in Stage 0 cannot be silently changed in training config
- `execution_posture=wrapper_bundle_plan` cannot be overridden by a model registry entry
- `fixed_design.dataset_adapter_family` cannot be replaced ad hoc by a downstream feature registry

## 5. Route ownership rules

Route ownership must be fixed at Stage 0.

v1 ownership table:
- `single_path_benchmark_study` -> single-run path owner
- `controlled_variation_study` -> single-run path if variation stays internal and bounded; wrapper owner otherwise
- `orchestrated_bundle_study` -> wrapper/orchestrator owner
- `replication_override_study` -> replication-aware wrapper or locked single-run path depending on constraints

Rule:
- if a study requires coordination across multiple emitted runs, it is wrapper-owned
- wrapper-owned families must not be forced into fake single-path execution grammar

## 6. Fixed vs varying grammar rules

Stage 0 must enforce this distinction:

### A choice belongs in `fixed_design` if
- changing it breaks fairness comparability across tools
- it changes the comparison environment rather than the compared tool
- it defines the study spine

### A choice belongs in `varying_design` if
- it is intentionally varied inside the study
- its variation is itself part of the research question
- fairness is preserved because the comparison contract explicitly holds the environment fixed around it

### A choice does not belong in either if
- it is a numeric parameter nested under a later content object
- it is merely implementation detail
- it is transitional compatibility metadata

## 7. Comparison grammar rules

The package mission implies a strict comparison grammar.

A valid macrocast comparison must answer:
- compared tools over what fixed information set?
- compared tools over what fixed sample split?
- compared tools against what fixed benchmark?
- compared tools under what fixed evaluation protocol?

If any of these are not explicit, the study is grammatically incomplete.

Therefore Stage 0 completeness requires:
- `comparison_contract` exists
- `fixed_design` exists
- `study_mode` exists
- `execution_posture` exists

## 8. Execution posture grammar rules

Stage 0 must prevent downstream ambiguity.

Ambiguity to forbid:
- a path that looks single-run but actually means many coordinated runs
- a registry id that secretly implies bundle behavior
- replication ids that override fairness conditions without explicit declaration

Therefore:
- bundle semantics must be explicit in `study_mode` and `execution_posture`
- replication semantics must be explicit in `replication_input`
- internal sweep semantics must be explicit in `varying_design`

## 9. Registry extension boundaries

Registries later in the package are allowed to:
- enumerate admissible datasets, models, benchmarks, metrics, tests, outputs
- attach metadata to those values
- validate local content contracts

Registries are not allowed to:
- invent new Stage 0 structural fields
- change route ownership
- redefine fairness conditions
- convert fixed choices into varying choices locally
- encode bundle behavior in ordinary content ids

## 10. Stage 0 completeness test

A Stage 0 object is complete only if all of the following are answerable:

1. What kind of study is this?
2. Is this ordinary use or replication override?
3. What is fixed for fairness?
4. What is intentionally allowed to vary?
5. What comparison conditions must stay identical?
6. What downstream execution posture is allowed?
7. What content registries may fill later slots without changing the language?

If any answer is missing, Stage 0 is incomplete.

## 11. What Stage 0 should produce for later layers

Stage 0 should emit one canonical object:
- `Stage0GrammarFrame`

Recommended shape:
- `study_mode`
- `replication_input`
- `design_frame`
  - `fixed_design`
  - `varying_design`
  - `derived_design_shape`
- `comparison_contract`
- `execution_posture`
- `registry_scope_contract`
- `compatibility_mirrors`

This object becomes the language contract consumed by later layers.

## 12. Immediate consequence for implementation

Before building registries or execution code, macrocast should implement:
- a Python representation of `Stage0GrammarFrame`
- validation rules for each Stage 0 object
- completeness checks
- route ownership resolution from Stage 0 truth

Only after that should later registries be expanded.

## 13. One-sentence summary

Stage 0 fixes the structural language of a forecasting study: what is fixed, what is varied, how fairness is defined, and what execution object may exist downstream.