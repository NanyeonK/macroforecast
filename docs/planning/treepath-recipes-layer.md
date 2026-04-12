# TP-004 Recipes Layer

Recipes are the layer that binds:
- taxonomy path choices (enum ids)
- numeric/free parameters
- output preferences

This first pass adds:
- `recipes/schema/recipe_schema.yaml`
- scaffolding directories for papers/baselines/benchmarks/ablations
- one baseline recipe example
- package loaders/validators for recipes

Current limitation
- the runtime/compiler does not yet consume recipes directly
- this step prepares the package so TP-005 can treat recipe/path as a first-class compiler input
