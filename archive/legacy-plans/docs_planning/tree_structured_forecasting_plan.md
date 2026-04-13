# Tree-Structured Forecasting Pipeline Plan

## Core correction
Do not add `macrocast/papers/`.
Paper-specific work should be expressible through config overlays and registries, not a dedicated paper subpackage.

## Real package objective
The package should be able to express a forecasting experiment as a large structured choice tree.
A researcher effectively chooses around 100 decisions before a paper-grade forecast run is fully specified.
The package should make those decisions:
- explicit
- hierarchical
- mostly defaultable
- overrideable
- extensible with user custom methods

## Design image
Think of the package as a tree of decisions, not only a flat config.

Example high-level tree:
1. Meta / experiment unit
2. Data / task definition
3. Preprocessing
4. Forecasting / training
5. Evaluation / tests
6. Output / provenance

Each branch expands into finer choices.
Example:
- Data
  - dataset source
  - frequency
  - information set type
  - vintage policy
  - release lag policy
  - ragged edge policy
  - variable universe version
- Preprocessing
  - target recipe
  - X recipe
  - operation order
  - factor extraction
  - lag generation
- Training
  - model family
  - split method
  - tuning method
  - search budget
  - seed policy
- Evaluation
  - benchmark family
  - benchmark id
  - metrics
  - statistical tests
  - aggregation rules
- Output
  - artifact set
  - manifest fields
  - failure policy
  - export formats

## Package principle
The tree is not only documentation.
The tree must exist in machine-readable form.

That means:
- registry truth in YAML
- loader / validator truth in code
- docs explain tree to humans

## No dedicated paper package
Instead of `macrocast/papers/`, use:
- generic registries
- optional paper-specific config directories under `config/papers/...`
- or plain user configs under `config/projects/...`

The code path should remain generic.
CLSS 2021 should be runnable because the tree is expressive enough, not because there is a hardcoded paper module.

## Custom extensibility rule
Every major choice family must allow custom additions.

At minimum custom extension slots for:
- dataset loader
- target recipe
- X recipe
- feature recipe
- model wrapper
- tuning method backend
- search space recipe
- benchmark recipe
- metric
- statistical test
- interpretation method
- output recipe

Each custom extension should declare:
- registry key
- required inputs
- produced outputs
- compatibility rules
- provenance fields

## Recommended structure change
### Keep
- `macrocast/pipeline/` for runtime orchestration
- `macrocast/data/`
- `macrocast/preprocessing/`
- `macrocast/evaluation/`
- `macrocast/output/`

### Add
- `macrocast/tuning/`
  - tuning methods
  - backend adapters
  - search-space logic
- `macrocast/registry/`
  - shared registry loading / validation helpers for the choice tree
- `macrocast/output/recipes/`
  - table builders / appendix-format outputs / significance formatting

### Config additions
- `config/tuning.yaml`
- `config/search_spaces.yaml`
- `config/benchmarks.yaml` if needed as first-class family
- `config/output_recipes.yaml`
- `config/choice_tree.yaml`

## choice_tree.yaml purpose
This file should describe the high-level tree itself.
Not every default value, but the structure of decision families.

Example top-level nodes:
- meta
- data
- preprocessing
- forecasting
- evaluation
- output

Each node should list:
- child choices
- whether single / grid / conditional
- default source
- custom extension allowed or not

## CLSS implication
CLSS 2021 then becomes:
- one setting of the tree
- one set of overlay YAML values
- one set of output recipes
- one set of ambiguity variants

Not a special package module.

## Implementation sequence
1. Define machine-readable choice tree
2. Define tuning + search-space registries
3. Define output recipe registry
4. Add custom extension protocol to each family
5. Update loaders/validators to traverse the tree
6. Add CLSS config overlay to prove expressiveness
7. Update docs to visualize the tree

## Docs requirement
Docs should show the forecasting pipeline as a branching tree.
Good outputs:
- high-level tree map
- per-layer branch lists
- default vs override rules
- custom extension guide
- paper example as one path through the tree
