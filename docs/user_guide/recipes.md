# Recipes and execution contract

## Purpose

The recipe/execution layer is the first package layer above Stage 0 and raw data.

Its job is to turn:
- one Stage 0 study grammar frame
- one raw dataset identity
- one target definition or one explicit target set
- one horizon set

into a minimal, explicit execution contract.

In v1 this layer does not run forecasting models yet.
It defines the objects that a future execution engine will consume.

## Current public surface

The rebuilt package now exposes:
- `RecipeSpec`
- `RunSpec`
- `build_recipe_spec()`
- `build_run_spec()`
- `check_recipe_completeness()`
- `recipe_summary()`

## `RecipeSpec`

`RecipeSpec` is the minimal declarative unit of one forecasting study.

It stores:
- `recipe_id`
- `stage0`
- `target`
- optional `targets`
- `horizons`
- `raw_dataset`

Interpretation:
- `stage0` fixes the study language
- `raw_dataset` identifies the raw adapter family used by the study
- `target` identifies the forecast target for single-target runs
- `targets` identify the explicit target set for the first multi-target slice
- `horizons` identify the forecast horizon set

## `RunSpec`

`RunSpec` is the first execution-facing object.

It stores:
- `run_id`
- `recipe_id`
- `route_owner`
- `artifact_subdir`

Interpretation:
- `route_owner` is inherited from Stage 0 routing logic
- `run_id` provides a deterministic naming spine for saved artifacts
- `artifact_subdir` identifies the default run artifact location under `runs/`

## `build_recipe_spec()`

This function builds the canonical recipe object for v1.

```python
recipe = build_recipe_spec(
    recipe_id="fred_md_baseline",
    stage0=stage0,
    target="INDPRO",
    horizons=(1, 3, 6, 12),
    raw_dataset="fred_md",
)
```

## `check_recipe_completeness()`

This function fails closed if the recipe is missing the minimum execution contract.

The current v1 checks require:
- non-empty `recipe_id`
- either non-empty `target` or an explicit multi-target `targets` tuple
- non-empty `raw_dataset`
- at least one forecast horizon

## `build_run_spec()`

This function derives a run-facing object from a recipe.

```python
run = build_run_spec(recipe)
```

The current v1 implementation uses:
- Stage 0 route owner
- recipe id
- target or explicit target set
- horizon set

to construct a deterministic run identifier.

## `recipe_summary()`

Returns a human-readable summary suitable for logs, previews, and manifests.

## Why this layer exists before forecasting execution

macrocast should not jump directly from raw data and Stage 0 into model execution without an explicit recipe/run contract.

This intermediate layer makes the package clearer by separating:
- study language (`Stage0Frame`)
- data identity (`raw_dataset`)
- study declaration (`RecipeSpec`)
- run identity (`RunSpec`)

That separation keeps the future execution engine simpler.

## Current limitation

This layer is intentionally minimal.

What it does now:
- define the recipe object
- define the run object
- validate basic completeness
- derive deterministic run identity

What it does not yet do:
- store feature or model registries
- define artifact manifests beyond run subdir
- execute forecasting pipelines
- resolve preprocessing objects
- bind benchmark/evaluation artifacts into final run outputs

Those come in later execution layers.
