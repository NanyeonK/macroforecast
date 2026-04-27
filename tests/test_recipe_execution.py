from __future__ import annotations
from macrocast import (
    build_recipe_spec,
    build_run_spec,
    build_design_frame,
    check_recipe_completeness,
    recipe_summary,
)


def _stage0(task: str = "single_target_point_forecast"):
    return build_design_frame(
        research_design="single_forecast_run",
        fixed_design={
            "dataset_adapter": "fred_md",
            "information_set": "revised_monthly",
            "sample_split": "expanding_window_oos",
            "benchmark": "ar_bic",
            "evaluation_protocol": "point_forecast_core",
            "forecast_task": task,
        },
        comparison_contract={
            "information_set_policy": "identical",
            "sample_split_policy": "identical",
            "benchmark_policy": "identical",
            "evaluation_policy": "identical",
        },
        varying_design={"model_families": ("ar", "ridge"), "horizons": ("h1", "h3")},
    )


def test_build_recipe_spec() -> None:
    recipe = build_recipe_spec(
        recipe_id="fred_md_baseline",
        stage0=_stage0(),
        target="INDPRO",
        horizons=(1, 3, 6, 12),
        raw_dataset="fred_md",
    )

    assert recipe.recipe_id == "fred_md_baseline"
    assert recipe.target == "INDPRO"
    assert recipe.raw_dataset == "fred_md"
    assert recipe.horizons == (1, 3, 6, 12)


def test_check_recipe_completeness() -> None:
    recipe = build_recipe_spec(
        recipe_id="fred_md_baseline",
        stage0=_stage0(),
        target="INDPRO",
        horizons=(1,),
        raw_dataset="fred_md",
    )
    check_recipe_completeness(recipe)


def test_build_run_spec() -> None:
    recipe = build_recipe_spec(
        recipe_id="fred_md_baseline",
        stage0=_stage0(),
        target="INDPRO",
        horizons=(1, 3),
        raw_dataset="fred_md",
    )
    run = build_run_spec(recipe)

    assert run.recipe_id == "fred_md_baseline"
    assert run.route_owner == "single_run"
    assert run.run_id.startswith("fred_md_baseline__INDPRO__")
    assert run.artifact_subdir == f"runs/{run.run_id}"


def test_recipe_summary_mentions_target_and_route() -> None:
    recipe = build_recipe_spec(
        recipe_id="fred_md_baseline",
        stage0=_stage0(),
        target="INDPRO",
        horizons=(1, 3),
        raw_dataset="fred_md",
    )

    summary = recipe_summary(recipe)

    assert "fred_md_baseline" in summary
    assert "INDPRO" in summary
    assert "single_run" in summary


def test_build_multi_target_recipe_spec_and_run_id() -> None:
    recipe = build_recipe_spec(
        recipe_id="fred_md_multi",
        stage0=_stage0(task="multi_target_point_forecast"),
        target="",
        targets=("INDPRO", "RPI"),
        horizons=(1, 3),
        raw_dataset="fred_md",
    )
    check_recipe_completeness(recipe)
    run = build_run_spec(recipe)

    assert recipe.targets == ("INDPRO", "RPI")
    assert run.run_id.startswith("fred_md_multi__MULTI_INDPRO-RPI__")
    assert "targets=[INDPRO, RPI]" in recipe_summary(recipe)
