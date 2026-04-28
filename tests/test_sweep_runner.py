"""Integration tests for the horse-race sweep runner (Phase 1 sub-task 01.7)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from macrocast.compiler.sweep_plan import compile_sweep_plan
from macrocast.execution.sweep_runner import execute_sweep


FIXTURE_RAW = Path("tests/fixtures/fred_md_ar_sample.csv")


def _horse_race_recipe() -> dict:
    return {
        "recipe_id": "sweep-rt-model",
        "path": {
            "0_meta": {"fixed_axes": {"research_design": "controlled_variation"}},
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
                "sweep_axes": {"model_family": ["ridge", "lasso"]},
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


def test_two_variant_sweep_end_to_end(tmp_path: Path) -> None:
    plan = compile_sweep_plan(_horse_race_recipe())
    assert plan.size == 2

    result = execute_sweep(
        plan=plan,
        output_root=tmp_path,
        local_raw_source=FIXTURE_RAW,
    )

    assert result.successful_count == 2
    assert result.failed_count == 0
    assert Path(result.manifest_path).exists()

    manifest = json.loads(Path(result.manifest_path).read_text())
    assert manifest["schema_version"] == "1.0"
    assert manifest["study_id"] == plan.study_id
    assert manifest["research_design"] == "controlled_variation"
    assert len(manifest["sweep_plan"]["variants"]) == 2
    assert all(v["status"] == "success" for v in manifest["sweep_plan"]["variants"])

    for variant in plan.variants:
        variant_dir = tmp_path / "variants" / variant.variant_id
        assert variant_dir.exists()


def test_fail_fast_raises_on_first_failure(tmp_path: Path) -> None:
    recipe = _horse_race_recipe()
    recipe["path"]["3_training"]["sweep_axes"] = {
        "model_family": ["ridge", "not_a_real_model"],
    }
    plan = compile_sweep_plan(recipe)

    with pytest.raises(Exception):
        execute_sweep(
            plan=plan,
            output_root=tmp_path,
            local_raw_source=FIXTURE_RAW,
        )


def test_fail_slow_records_failure_and_continues(tmp_path: Path) -> None:
    recipe = _horse_race_recipe()
    recipe["path"]["0_meta"]["fixed_axes"]["failure_policy"] = "skip_failed_cell"
    recipe["path"]["3_training"]["sweep_axes"] = {
        "model_family": ["ridge", "not_a_real_model"],
    }
    plan = compile_sweep_plan(recipe)

    result = execute_sweep(
        plan=plan,
        output_root=tmp_path,
        local_raw_source=FIXTURE_RAW,
    )

    assert result.successful_count == 1
    assert result.failed_count == 1

    manifest = json.loads(Path(result.manifest_path).read_text())
    statuses = {v["status"] for v in manifest["sweep_plan"]["variants"]}
    assert statuses == {"success", "failed"}
    assert manifest["summary"]["successful"] == 1
    assert manifest["summary"]["failed"] == 1


def test_sweep_reproducibility_study_id_stable(tmp_path: Path) -> None:
    plan_a = compile_sweep_plan(_horse_race_recipe())
    plan_b = compile_sweep_plan(_horse_race_recipe())

    ra = execute_sweep(plan=plan_a, output_root=tmp_path / "a", local_raw_source=FIXTURE_RAW)
    rb = execute_sweep(plan=plan_b, output_root=tmp_path / "b", local_raw_source=FIXTURE_RAW)

    assert ra.study_id == rb.study_id

    variants_a = {v.variant_id for v in plan_a.variants}
    variants_b = {v.variant_id for v in plan_b.variants}
    assert variants_a == variants_b


# --- compute_mode=parallel_by_model variant-level parallelism ---


def test_parallel_by_model_sweep_runs_variants_concurrently(tmp_path: Path) -> None:
    """When compute_mode=parallel_by_model AND model_family is swept, execute_sweep
    dispatches variants onto ThreadPoolExecutor instead of running sequentially."""
    recipe = _horse_race_recipe()
    recipe["path"]["0_meta"]["fixed_axes"]["compute_mode"] = "parallel_by_model"
    plan = compile_sweep_plan(recipe)
    assert plan.size == 2

    # Patch ThreadPoolExecutor in sweep_runner to count instantiations.
    from macrocast.execution import sweep_runner as sr
    original = sr.ThreadPoolExecutor
    calls: list[int] = []

    class _CountingExecutor(original):
        def __init__(self, *args, max_workers=None, **kwargs):
            calls.append(max_workers or 0)
            super().__init__(*args, max_workers=max_workers, **kwargs)

    sr.ThreadPoolExecutor = _CountingExecutor
    try:
        result = execute_sweep(plan=plan, output_root=tmp_path, local_raw_source=FIXTURE_RAW)
    finally:
        sr.ThreadPoolExecutor = original

    assert result.successful_count == 2
    # exactly one ThreadPoolExecutor was built by the variant-level parallel branch,
    # with max_workers capped at min(len(variants), 4) = 2
    assert calls == [2]


def test_parallel_by_model_without_model_family_sweep_stays_serial(tmp_path: Path) -> None:
    """compute_mode=parallel_by_model with only non-model-family sweep axes falls
    back to serial execution (no ThreadPoolExecutor created)."""
    recipe = _horse_race_recipe()
    recipe["path"]["0_meta"]["fixed_axes"]["compute_mode"] = "parallel_by_model"
    # swap model_family sweep out for framework so model_family is not swept
    recipe["path"]["3_training"]["sweep_axes"] = {"framework": ["expanding", "rolling"]}
    recipe["path"]["3_training"]["fixed_axes"]["model_family"] = "ridge"
    recipe["path"]["3_training"]["fixed_axes"].pop("framework", None)
    plan = compile_sweep_plan(recipe)
    assert plan.size == 2
    assert any(axis.endswith(".framework") for axis in plan.axes_swept)
    assert not any(axis.endswith(".model_family") for axis in plan.axes_swept)

    from macrocast.execution import sweep_runner as sr
    original = sr.ThreadPoolExecutor
    calls: list[int] = []

    class _CountingExecutor(original):
        def __init__(self, *args, max_workers=None, **kwargs):
            calls.append(max_workers or 0)
            super().__init__(*args, max_workers=max_workers, **kwargs)

    sr.ThreadPoolExecutor = _CountingExecutor
    try:
        result = execute_sweep(plan=plan, output_root=tmp_path, local_raw_source=FIXTURE_RAW)
    finally:
        sr.ThreadPoolExecutor = original

    assert result.successful_count == 2
    # No variant-level ThreadPoolExecutor should be created for this sweep.
    assert calls == []


def test_extract_parent_compute_mode_reads_fixed_axes() -> None:
    from macrocast.compiler.sweep_plan import compile_sweep_plan
    from macrocast.execution.sweep_runner import _extract_parent_compute_mode

    recipe = _horse_race_recipe()
    recipe["path"]["0_meta"]["fixed_axes"]["compute_mode"] = "parallel_by_model"
    plan = compile_sweep_plan(recipe)
    assert _extract_parent_compute_mode(plan) == "parallel_by_model"


def test_extract_parent_compute_mode_defaults_to_serial() -> None:
    from macrocast.compiler.sweep_plan import compile_sweep_plan
    from macrocast.execution.sweep_runner import _extract_parent_compute_mode

    plan = compile_sweep_plan(_horse_race_recipe())  # no compute_mode set
    assert _extract_parent_compute_mode(plan) == "serial"


# --- failure_policy-driven sweep behaviour (0.4) ---


def test_skip_failed_cell_records_failure_and_continues(tmp_path: Path) -> None:
    recipe = _horse_race_recipe()
    recipe["path"]["0_meta"]["fixed_axes"]["failure_policy"] = "skip_failed_cell"
    recipe["path"]["3_training"]["sweep_axes"] = {
        "model_family": ["ridge", "not_a_real_model"],
    }
    plan = compile_sweep_plan(recipe)
    result = execute_sweep(plan=plan, output_root=tmp_path, local_raw_source=FIXTURE_RAW)
    assert result.successful_count == 1
    assert result.failed_count == 1


def test_skip_failed_cell_skips_compile_invalid_l2_l3_cells(tmp_path: Path) -> None:
    recipe = _horse_race_recipe()
    recipe["path"]["0_meta"]["fixed_axes"]["failure_policy"] = "skip_failed_cell"
    recipe["path"]["3_training"]["sweep_axes"] = {
        "model_family": ["ridge", "ar"],
    }
    plan = compile_sweep_plan(recipe)

    result = execute_sweep(plan=plan, output_root=tmp_path, local_raw_source=FIXTURE_RAW)

    assert result.successful_count == 1
    assert result.failed_count == 0
    assert result.skipped_count == 1

    manifest = json.loads(Path(result.manifest_path).read_text())
    assert manifest["summary"]["successful"] == 1
    assert manifest["summary"]["failed"] == 0
    assert manifest["summary"]["skipped"] == 1
    assert manifest["summary"]["invalid_cells"] == 1
    skipped = next(v for v in manifest["sweep_plan"]["variants"] if v["status"] == "skipped")
    assert skipped["compiler_status"] == "blocked_by_incompatibility"
    assert skipped["layer3_capability_cell"]["model_family"] == "ar"
    assert skipped["layer3_capability_cell"]["feature_runtime"] == "raw_feature_panel"
    assert any("model_family='ar'" in reason for reason in skipped["compiler_blocked_reasons"])
    assert (tmp_path / skipped["artifact_dir"] / "compiler_manifest.json").exists()


def test_warn_only_emits_warning_and_continues(tmp_path: Path) -> None:
    import warnings as warnings_mod
    recipe = _horse_race_recipe()
    recipe["path"]["0_meta"]["fixed_axes"]["failure_policy"] = "warn_only"
    recipe["path"]["3_training"]["sweep_axes"] = {
        "model_family": ["ridge", "not_a_real_model"],
    }
    plan = compile_sweep_plan(recipe)
    with warnings_mod.catch_warnings(record=True) as caught:
        warnings_mod.simplefilter("always")
        result = execute_sweep(plan=plan, output_root=tmp_path, local_raw_source=FIXTURE_RAW)
    assert result.successful_count == 1
    assert result.failed_count == 1
    assert any(
        issubclass(w.category, RuntimeWarning) and "variant" in str(w.message) and "failed" in str(w.message)
        for w in caught
    ), f"expected a RuntimeWarning about a failed variant, got {[str(w.message) for w in caught]}"


def test_fail_fast_is_the_default_failure_policy(tmp_path: Path) -> None:
    recipe = _horse_race_recipe()
    # no failure_policy set => defaults to fail_fast
    recipe["path"]["3_training"]["sweep_axes"] = {
        "model_family": ["ridge", "not_a_real_model"],
    }
    plan = compile_sweep_plan(recipe)
    with pytest.raises(Exception):
        execute_sweep(plan=plan, output_root=tmp_path, local_raw_source=FIXTURE_RAW)


def test_extract_parent_failure_policy_reads_fixed_axes() -> None:
    from macrocast.execution.sweep_runner import _extract_parent_failure_policy
    recipe = _horse_race_recipe()
    recipe["path"]["0_meta"]["fixed_axes"]["failure_policy"] = "save_partial_results"
    plan = compile_sweep_plan(recipe)
    assert _extract_parent_failure_policy(plan) == "save_partial_results"


def test_extract_parent_failure_policy_defaults_to_fail_fast() -> None:
    from macrocast.execution.sweep_runner import _extract_parent_failure_policy
    plan = compile_sweep_plan(_horse_race_recipe())
    assert _extract_parent_failure_policy(plan) == "fail_fast"
