from __future__ import annotations

import inspect
from pathlib import Path

from macrocast import build_preprocess_contract, build_recipe_spec, build_design_frame, execute_recipe


FIXTURE = Path("tests/fixtures/fred_md_ar_sample.csv")


def _stage0():
    return build_design_frame(
        fixed_design={
            "dataset_adapter": "fred_md",
            "information_set": "revised_monthly",
            "sample_split": "rolling_window_oos",
            "benchmark": "zero_change",
            "evaluation_protocol": "point_forecast_core",
            "forecast_task": "single_target",
        },
        comparison_contract={
            "information_set_policy": "identical",
            "sample_split_policy": "identical",
            "benchmark_policy": "identical",
            "evaluation_policy": "identical",
        },
        varying_design={"model_families": ("ridge",), "feature_recipes": ("raw_feature_panel",), "horizons": ("h1",)},
    )


def _recipe():
    return build_recipe_spec(
        recipe_id="fred_md_rolling_ridge_raw_feature_panel",
        stage0=_stage0(),
        target="INDPRO",
        horizons=(1, 3),
        raw_dataset="fred_md",
        benchmark_config={"minimum_train_size": 5, "rolling_window_size": 5},
        data_task_spec={"forecast_object": "point_mean"},
        training_spec={},
    )


def _preprocess():
    return build_preprocess_contract(
        target_transform_policy="raw_level",
        x_transform_policy="raw_level",
        tcode_policy="raw_only",
        target_missing_policy="none",
        x_missing_policy="none",
        target_outlier_policy="none",
        x_outlier_policy="none",
        scaling_policy="none",
        dimensionality_reduction_policy="none",
        feature_selection_policy="none",
        preprocess_order="none",
        preprocess_fit_scope="not_applicable",
        inverse_transform_policy="none",
        evaluation_scale="raw_level",
    )


def test_execute_recipe_signature_accepts_cache_root():
    sig = inspect.signature(execute_recipe)
    assert "cache_root" in sig.parameters
    assert sig.parameters["cache_root"].default is None


def test_default_cache_root_is_per_output_root(tmp_path: Path):
    recipe = _recipe()
    preprocess = _preprocess()

    execute_recipe(recipe=recipe, preprocess=preprocess, output_root=tmp_path / "run", local_raw_source=FIXTURE)

    assert (tmp_path / "run" / ".raw_cache").exists()


def test_shared_cache_root_reuses_cache_between_runs(tmp_path: Path):
    shared_cache = tmp_path / "shared_cache"
    recipe = _recipe()
    preprocess = _preprocess()

    execute_recipe(
        recipe=recipe,
        preprocess=preprocess,
        output_root=tmp_path / "run_a",
        local_raw_source=FIXTURE,
        cache_root=shared_cache,
    )
    assert shared_cache.exists()
    snapshot_a = sorted(p.name for p in shared_cache.rglob("*") if p.is_file())

    execute_recipe(
        recipe=recipe,
        preprocess=preprocess,
        output_root=tmp_path / "run_b",
        local_raw_source=FIXTURE,
        cache_root=shared_cache,
    )
    snapshot_b = sorted(p.name for p in shared_cache.rglob("*") if p.is_file())

    assert snapshot_a == snapshot_b
    assert not (tmp_path / "run_a" / ".raw_cache").exists()
    assert not (tmp_path / "run_b" / ".raw_cache").exists()


def test_distinct_cache_roots_are_independent(tmp_path: Path):
    recipe = _recipe()
    preprocess = _preprocess()
    cache_a = tmp_path / "cache_a"
    cache_b = tmp_path / "cache_b"

    execute_recipe(recipe=recipe, preprocess=preprocess, output_root=tmp_path / "run_a", local_raw_source=FIXTURE, cache_root=cache_a)
    execute_recipe(recipe=recipe, preprocess=preprocess, output_root=tmp_path / "run_b", local_raw_source=FIXTURE, cache_root=cache_b)

    assert cache_a.exists()
    assert cache_b.exists()
    assert set(p.name for p in cache_a.rglob("*") if p.is_file()) == set(p.name for p in cache_b.rglob("*") if p.is_file())


def test_execute_recipe_writes_manifest_exactly_once(tmp_path: Path):
    recipe = _recipe()
    preprocess = _preprocess()

    result = execute_recipe(
        recipe=recipe,
        preprocess=preprocess,
        output_root=tmp_path,
        local_raw_source=FIXTURE,
    )

    manifest_files = list(Path(result.artifact_dir).glob("manifest*.json"))
    assert len(manifest_files) == 1
    assert manifest_files[0].name == "manifest.json"
