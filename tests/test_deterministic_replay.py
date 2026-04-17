from __future__ import annotations

import hashlib
import json
from pathlib import Path

from macrocast import build_preprocess_contract, build_recipe_spec, build_stage0_frame, execute_recipe


def _stage0():
    return build_stage0_frame(
        study_mode="single_path_benchmark_study",
        fixed_design={
            "dataset_adapter": "fred_md",
            "information_set": "revised_monthly",
            "sample_split": "rolling_window_oos",
            "benchmark": "zero_change",
            "evaluation_protocol": "point_forecast_core",
            "forecast_task": "single_target_point_forecast",
        },
        comparison_contract={
            "information_set_policy": "identical",
            "sample_split_policy": "identical",
            "benchmark_policy": "identical",
            "evaluation_policy": "identical",
        },
        varying_design={"model_families": ("randomforest",), "feature_recipes": ("raw_feature_panel",), "horizons": ("h1",)},
    )


def _recipe():
    return build_recipe_spec(
        recipe_id="fred_md_rolling_randomforest_raw_feature_panel",
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize_metrics_text(path: Path) -> str:
    # manifest.json contains per-run run_id + git_commit; normalise to shared keys only.
    data = json.loads(path.read_text())
    return json.dumps(data, sort_keys=True)


FIXTURE = Path("tests/fixtures/fred_md_ar_sample.csv")


def test_identical_recipe_yields_byte_identical_predictions(tmp_path: Path):
    recipe = _recipe()
    preprocess = _preprocess()
    provenance = {"compiler": {"reproducibility_spec": {"reproducibility_mode": "seeded_reproducible", "seed": 42}}}

    r1 = execute_recipe(recipe=recipe, preprocess=preprocess, output_root=tmp_path / "r1", local_raw_source=FIXTURE, provenance_payload=provenance)
    r2 = execute_recipe(recipe=recipe, preprocess=preprocess, output_root=tmp_path / "r2", local_raw_source=FIXTURE, provenance_payload=provenance)

    predictions_1 = Path(r1.artifact_dir) / "predictions.csv"
    predictions_2 = Path(r2.artifact_dir) / "predictions.csv"
    assert _sha256(predictions_1) == _sha256(predictions_2)

    metrics_1 = _normalize_metrics_text(Path(r1.artifact_dir) / "metrics.json")
    metrics_2 = _normalize_metrics_text(Path(r2.artifact_dir) / "metrics.json")
    assert metrics_1 == metrics_2


def test_strict_reproducible_distinct_variants_produce_distinct_predictions(tmp_path: Path):
    recipe = _recipe()
    preprocess = _preprocess()

    def _provenance(variant_id: str) -> dict:
        return {
            "variant_id": variant_id,
            "compiler": {"reproducibility_spec": {"reproducibility_mode": "strict_reproducible"}},
        }

    r_a = execute_recipe(recipe=recipe, preprocess=preprocess, output_root=tmp_path / "A", local_raw_source=FIXTURE, provenance_payload=_provenance("variant-A"))
    r_b = execute_recipe(recipe=recipe, preprocess=preprocess, output_root=tmp_path / "B", local_raw_source=FIXTURE, provenance_payload=_provenance("variant-B"))

    assert _sha256(Path(r_a.artifact_dir) / "predictions.csv") != _sha256(Path(r_b.artifact_dir) / "predictions.csv")


def test_strict_reproducible_identical_variant_is_reproducible(tmp_path: Path):
    recipe = _recipe()
    preprocess = _preprocess()
    provenance = {
        "variant_id": "same",
        "compiler": {"reproducibility_spec": {"reproducibility_mode": "strict_reproducible"}},
    }

    r1 = execute_recipe(recipe=recipe, preprocess=preprocess, output_root=tmp_path / "1", local_raw_source=FIXTURE, provenance_payload=provenance)
    r2 = execute_recipe(recipe=recipe, preprocess=preprocess, output_root=tmp_path / "2", local_raw_source=FIXTURE, provenance_payload=provenance)

    assert _sha256(Path(r1.artifact_dir) / "predictions.csv") == _sha256(Path(r2.artifact_dir) / "predictions.csv")
