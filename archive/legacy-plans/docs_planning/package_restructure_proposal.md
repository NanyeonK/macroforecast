# Package Restructure Proposal

## Decision
Yes. Moderate package restructuring is more efficient than layering new CLSS-capable generic architecture onto the current flat structure.

Reason
Current layout mixes:
- runtime experiment orchestration
- model wrappers
- CV/tuning concepts
- paper-specific replication presets
- output/evaluation logic

That structure was acceptable for the current baseline, but it will become messy once we add:
- generic tuning registries
- multiple optimizer backends
- paper overlays
- appendix-style output recipes
- ambiguity variant runners

## Recommended restructure scope
Not a full rewrite. Controlled restructure.

### New top-level package subpackages
- `macrocast/tuning/`
  - generic tuning methods
  - optimizer backends
  - search-space handling
  - tuning provenance
- `macrocast/papers/`
  - paper overlay loaders and helpers
  - CLSS 2021 first
- `macrocast/output/recipes/`
  - benchmark-relative table builders
  - appendix-format builders
  - significance/annotation formatters
- keep `macrocast/pipeline/`
  - experiment runtime
  - feature building
  - core estimator abstractions
  - model wrappers only

### Config restructure
Add:
- `config/tuning.yaml`
- `config/search_spaces.yaml`
- `config/output_recipes.yaml`
- `config/papers/clss2021/contract.yaml`
- `config/papers/clss2021/tuning_map.yaml`
- `config/papers/clss2021/search_spaces.yaml`
- `config/papers/clss2021/output_recipes.yaml`
- `config/papers/clss2021/ambiguities.yaml`

## What should move
### From pipeline
Move out of `macrocast/pipeline/`:
- generic tuning/search behavior
- optimizer-specific backend logic
- paper-specific replication assembly

### Keep in pipeline
Keep in `macrocast/pipeline/`:
- `experiment.py`
- `features.py`
- `estimator.py`
- `results.py`
- model wrappers in `models.py` and `r_models.py`
- core enums in `components.py` initially, but extend or split carefully

## Why this is more efficient
Benefits
1. CLSS support becomes an overlay, not special-case sprawl.
2. Bayesian/genetic/grid/none/bic tuning can be implemented once and reused.
3. Output recipes become composable across papers.
4. Defaults can stay light while paper paths stay rich.
5. Future papers reuse same architecture.

Costs
- some import paths will change
- registry loading logic becomes broader
- tests need reorganization

Net
- worth it

## Recommended implementation order
1. Create new package directories and config directories.
2. Add registry loaders for tuning/search-space/output-recipe/paper overlays.
3. Implement tuning abstraction.
4. Wire pipeline to consume tuning recipes rather than embed selection logic ad hoc.
5. Add CLSS 2021 overlay.
6. Add appendix output recipes.
7. Update docs.

## Safety rule
Restructure incrementally.
Do not break current public surfaces all at once.
Use compatibility exports where cheap.
