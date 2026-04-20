from __future__ import annotations

from .types import RecipeSpec, RunSpec
from ..design import resolve_route_owner


def _target_token(recipe: RecipeSpec) -> str:
    if recipe.targets:
        return "MULTI_" + "-".join(recipe.targets)
    return recipe.target


def build_run_spec(recipe: RecipeSpec) -> RunSpec:
    route_owner = resolve_route_owner(recipe.stage0)
    horizon_token = "-".join(str(h) for h in recipe.horizons)
    run_id = f"{recipe.recipe_id}__{_target_token(recipe)}__h{horizon_token}"
    return RunSpec(
        run_id=run_id,
        recipe_id=recipe.recipe_id,
        route_owner=route_owner,
        artifact_subdir=f"runs/{run_id}",
    )


def recipe_summary(recipe: RecipeSpec) -> str:
    route_owner = resolve_route_owner(recipe.stage0)
    horizons = ", ".join(str(h) for h in recipe.horizons)
    benchmark = recipe.stage0.fixed_design.benchmark
    target_part = f"targets=[{', '.join(recipe.targets)}]" if recipe.targets else f"target={recipe.target}"
    return (
        f"recipe_id={recipe.recipe_id}; {target_part}; raw_dataset={recipe.raw_dataset}; "
        f"benchmark={benchmark}; route={route_owner}; horizons=[{horizons}]"
    )
