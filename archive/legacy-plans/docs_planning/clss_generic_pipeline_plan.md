# CLSS-Capable Generic Forecasting Pipeline Plan

## Confirmed paper fact
For CLSS 2021 Random Forest:
- no hyperparameter cross-validation
- no hyperparameter optimization
- 200 trees fixed
- node-size threshold fixed at 5
- feature subset per split fixed at #Z/3
- bootstrap sample size uses MATLAB default

Implication:
- package must support tuning method `none`
- paper overlay can set RF to fixed hyperparameters
- generic package must still support other search engines for other models/papers

## Correct package objective
Do NOT hardcode CLSS 2021 behavior into core package.
Do build generic forecasting pipeline with enough freedom to express CLSS 2021 exactly.

Core package should support:
- multiple tuning/search engines
- paper-specific search-space overlays
- benchmark-relative output recipes
- paper-specific appendix-style tables
- safe/light defaults for ordinary users

## Architecture direction

### 1. Tuning layer becomes explicit registry family
Add package-level tuning registry with methods:
- none
- bic
- grid_search
- genetic_search
- bayesian_optimization

For each tuning method define:
- required inputs
- optimizer backend
- search space schema
- provenance fields
- fallback/default behavior

### 2. Model layer and tuning layer are separated
Model recipe should not fully own tuning logic.
Instead use:
- model recipe
- tuning recipe
- search-space recipe
- paper overlay mapping

### 3. Output layer gains paper/output recipes
Add output recipes such as:
- appendix_b_fm_relative_rmse_table
- best_spec_table
- marginal_effect_table
- grouped_variable_importance_table
- significance_annotation_formatter

### 4. Paper overlays
Add paper-specific overlays under config/papers/
Example CLSS 2021:
- contract.yaml
- search_spaces.yaml
- tuning_map.yaml
- output_recipes.yaml
- ambiguities.yaml

### 5. Defaults remain simple
Default user path:
- grid_search or none
- compact point-forecast metrics
- small manifests

Paper path:
- explicit overlay
- full artifacts
- appendix tables
- ambiguity variants

## CLSS 2021 mapping currently implied
- AR -> bic
- FM -> bic
- AL -> genetic_search (ridge step) + grid_search (lasso step)
- EN -> grid_search
- RF -> none
- BT -> bayesian_optimization
- LB -> genetic_search

## Research tasks before implementation
Need backend comparison notes for:
- Python GA packages vs MATLAB GA defaults
- Python Bayesian optimization packages vs MATLAB bayesopt defaults
- mismatch documentation for reproducibility notes

## Execution issue stack
1. Define tuning registry and search-space registry
2. Define CLSS 2021 paper overlay YAMLs
3. Research Python backend choices for GA / Bayesian optimization
4. Define output-recipe architecture
5. Implement tuning abstraction
6. Implement CLSS overlay
7. Implement appendix output recipes
8. Document generic usage + CLSS mapping
