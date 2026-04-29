"""Tests: sweep runner shares a single FRED cache across variants.

Phase 1 sub-task 01.7. Structural checks that (a) the study output_root
contains a ``.raw_cache_shared`` directory, and (b) per-variant dirs do
not accumulate their own ``.raw_cache`` subdirectories when
``execute_sweep`` passes ``cache_root`` through to ``execute_recipe``.
"""

from __future__ import annotations

from pathlib import Path

from macrocast import compile_sweep_plan, execute_sweep


FIXTURE_RAW = Path("tests/fixtures/fred_md_ar_sample.csv")


def _horse_race_recipe(models: list[str]) -> dict:
    return {
        "recipe_id": "sweep-cache-share",
        "path": {
            "0_meta": {"fixed_axes": {"study_scope": "one_target_compare_methods", "failure_policy": "skip_failed_cell"}},
            "1_data_task": {
                "fixed_axes": {
                    "dataset": "fred_md",
                    "information_set_type": "final_revised_data",
                    "target_structure": "single_target",
                },
                "leaf_config": {"target": "INDPRO", "horizons": [1, 3]},
            },
            "2_preprocessing": {
                "fixed_axes": {
                    "target_transform_policy": "raw_level",
                    "x_transform_policy": "raw_level",
                    "tcode_policy": "raw_only",
                    "target_missing_policy": "none",
                    "x_missing_policy": "none",
                    "target_outlier_policy": "none",
                    "x_outlier_policy": "none",
                    "scaling_policy": "none",
                    "dimensionality_reduction_policy": "none",
                    "feature_selection_policy": "none",
                    "preprocess_order": "none",
                    "preprocess_fit_scope": "not_applicable",
                    "inverse_transform_policy": "none",
                    "evaluation_scale": "raw_level",
                }
            },
            "3_training": {
                "fixed_axes": {
                    "framework": "rolling",
                    "benchmark_family": "zero_change",
                    "feature_builder": "raw_feature_panel",
                },
                "sweep_axes": {"model_family": models},
            },
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {
                "leaf_config": {
                    "manifest_mode": "full",
                    "benchmark_config": {
                        "minimum_train_size": 5,
                        "rolling_window_size": 5,
                    },
                }
            },
            "6_stat_tests": {"fixed_axes": {}},
            "7_importance": {"fixed_axes": {"importance_method": "minimal_importance"}},
        },
    }


def test_shared_cache_directory_is_created_at_study_root(tmp_path: Path) -> None:
    plan = compile_sweep_plan(_horse_race_recipe(["ridge", "lasso", "elasticnet"]))
    execute_sweep(plan=plan, output_root=tmp_path, local_raw_source=FIXTURE_RAW)

    assert (tmp_path / ".raw_cache_shared").is_dir()


def test_per_variant_directories_do_not_have_local_raw_cache(tmp_path: Path) -> None:
    plan = compile_sweep_plan(_horse_race_recipe(["ridge", "lasso"]))
    execute_sweep(plan=plan, output_root=tmp_path, local_raw_source=FIXTURE_RAW)

    for variant in plan.variants:
        variant_dir = tmp_path / "variants" / variant.variant_id
        assert variant_dir.exists()
        local_cache = variant_dir / ".raw_cache"
        assert not local_cache.exists(), (
            f"variant {variant.variant_id} should not have its own "
            f".raw_cache; cache_root plumbing may be broken"
        )


def test_sweep_produces_one_manifest_at_root_regardless_of_variant_count(
    tmp_path: Path,
) -> None:
    plan = compile_sweep_plan(_horse_race_recipe(["ridge", "lasso", "elasticnet", "bayesian_ridge"]))
    result = execute_sweep(plan=plan, output_root=tmp_path, local_raw_source=FIXTURE_RAW)

    manifests = list(tmp_path.glob("study_manifest.json"))
    assert len(manifests) == 1
    assert str(manifests[0]) == result.manifest_path
