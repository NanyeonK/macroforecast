# TP-005 Recipe-to-Compiled-Spec Integration

This step makes recipes a first-class compile input.

Current first pass:
- `compile_experiment_spec_from_recipe(recipe_path, ...)`
- recipe is validated against `recipes/schema/recipe_schema.yaml`
- recipe taxonomy path + numeric params are transformed into a legacy runtime config dict
- existing compiler/resolver path is reused after that transformation

Current limitation:
- recipe-to-runtime mapping is still narrow and transitional
- taxonomy ids are not yet fully resolved through dedicated registries
- this is a migration bridge, not the final tree-path compiler
