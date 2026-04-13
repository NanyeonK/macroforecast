# Optimizer Backend Research for CLSS-Capable Pipeline

## Installed in project venv
- optuna
- scikit-optimize (`skopt`)
- bayesian-optimization (`bayes_opt`)
- pygad
- deap
- scipy

## Design principle
Do not bind package core to one paper-specific optimizer.
Expose generic tuning methods in registry, then choose backend per method.

## Bayesian optimization candidates

### 1. optuna
Pros
- robust, maintained, flexible search API
- good logging / pruning / reproducibility controls
- easy to integrate with arbitrary Python objective functions
Cons
- not conceptually identical to MATLAB `bayesopt`
- default samplers / acquisition behavior differ
Use in package
- best default generic backend for `bayesian_optimization`
- document that CLSS 2021 replication uses Optuna-backed BO approximation, not MATLAB-internal bayesopt clone, unless exact matching later becomes necessary

### 2. scikit-optimize
Pros
- closer classic Bayesian optimization feel
- simple Gaussian-process based API
Cons
- less actively developed than Optuna
- integration / extensibility weaker
Use in package
- possible secondary backend
- useful if we want GP-style BO with simple search spaces

### 3. bayesian-optimization
Pros
- lightweight
Cons
- narrower ecosystem / less integrated trial management
Use in package
- not primary choice

## Genetic algorithm candidates

### 1. pygad
Pros
- very quick to wire for numeric hyperparameter search
- straightforward API
Cons
- more black-box / less research-standard than DEAP for custom operators
Use in package
- good default pragmatic backend for `genetic_search`

### 2. deap
Pros
- more extensible research toolkit
- custom operators / constraints easier for unusual search spaces
Cons
- more verbose implementation burden
Use in package
- strong fallback if `pygad` proves too rigid

### 3. scipy
Pros
- already dependency
- can support some custom search logic without new heavy dependency
Cons
- no dedicated GA framework out of box matching paper phrasing
Use in package
- helper utilities only, not primary GA backend

## MATLAB default comparison notes to document
Need docs for each paper overlay on:
- optimizer family match vs mismatch
- acquisition / surrogate defaults for BO
- population size / generations / selection defaults for GA
- stopping rule differences
- random seed semantics
- whether search domain is continuous / discrete / mixed

## Current recommended package strategy
- generic `bayesian_optimization` method -> Optuna backend by default
- generic `genetic_search` method -> PyGAD backend by default
- allow backend override in tuning registry for paper overlays
- store backend name + version in provenance
- docs must explicitly state when backend approximates MATLAB defaults rather than matching them exactly

## Next implementation implication
Tuning registry should have fields like:
- method_id
- family
- backend
- backend_version_capture
- search_space_type
- required_inputs
- provenance_fields
- notes_on_equivalence
