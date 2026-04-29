"""Tests for macrocast.decomposition.run_decomposition."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from macrocast.decomposition import (
    DecompositionPlan,
    DecompositionResult,
    COMPONENT_NAMES,
    run_decomposition,
)


def _synthetic_manifest(
    tmp_path: Path,
    *,
    scaling_variance: float,
    model_variance: float,
    noise: float,
    seed: int = 0,
) -> Path:
    """Build a minimal study_manifest.json where preprocessing (scaling_policy)
    dominates when scaling_variance >> model_variance + noise.
    """
    rng = np.random.default_rng(seed)
    variants = []
    scaling_vals = ["none", "standard", "robust"]
    model_vals = ["ridge", "lasso"]

    scaling_effect = {"none": 0.0, "standard": 1.0 * scaling_variance, "robust": 2.0 * scaling_variance}
    model_effect = {"ridge": 0.0, "lasso": 1.0 * model_variance}

    for s in scaling_vals:
        for m in model_vals:
            metric = scaling_effect[s] + model_effect[m] + rng.normal(0, noise)
            vid = f"v-{s}-{m}"
            variants.append(
                {
                    "variant_id": vid,
                    "axis_values": {
                        "2_preprocessing.scaling_policy": s,
                        "3_training.model_family": m,
                    },
                    "status": "success",
                    "artifact_dir": f"variants/{vid}/art",
                    "metrics_summary": {
                        "metrics_by_horizon": {
                            "h1": {"msfe": float(metric)},
                        }
                    },
                    "runtime_seconds": 0.01,
                }
            )

    manifest = {
        "schema_version": "1.0",
        "study_id": "synth-study-xyz",
        "study_scope": "one_target_compare_methods",
        "parent_recipe_id": "synth-parent",
        "parent_recipe_dict": {},
        "axes_swept": ["2_preprocessing.scaling_policy", "3_training.model_family"],
        "variants": variants,
    }
    path = tmp_path / "study_manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_preprocessing_dominant_sweep_yields_high_preprocessing_share(tmp_path: Path):
    manifest = _synthetic_manifest(tmp_path, scaling_variance=5.0, model_variance=0.1, noise=0.01)
    plan = DecompositionPlan(
        study_manifest_path=str(manifest),
        components_to_decompose=COMPONENT_NAMES,
    )
    result = run_decomposition(plan, output_dir=tmp_path)
    assert result.per_component_shares["preprocessing"] > 0.7
    assert result.per_component_shares["nonlinearity"] < 0.2


def test_empty_components_is_clean_noop(tmp_path: Path):
    manifest = _synthetic_manifest(tmp_path, scaling_variance=1.0, model_variance=1.0, noise=0.01)
    plan = DecompositionPlan(
        study_manifest_path=str(manifest),
        components_to_decompose=(),
    )
    result = run_decomposition(plan, output_dir=tmp_path)
    assert result.per_component_shares == {}
    df = pd.read_parquet(result.result_parquet_path)
    assert len(df) == 0


def test_legacy_feature_builder_component_alias_normalizes_to_feature_representation(tmp_path: Path):
    variants = []
    for feature, value in [("target_lag_features", 1.0), ("raw_feature_panel", 3.0)]:
        variants.append(
            {
                "variant_id": f"v-{feature}",
                "axis_values": {"2_preprocessing.feature_builder": feature},
                "status": "success",
                "artifact_dir": "",
                "metrics_summary": {"metrics_by_horizon": {"h1": {"msfe": value}}},
            }
        )
    manifest = {
        "schema_version": "1.0",
        "study_id": "feature-representation-alias",
        "study_scope": "one_target_compare_methods",
        "parent_recipe_id": "p",
        "parent_recipe_dict": {},
        "axes_swept": ["2_preprocessing.feature_builder"],
        "variants": variants,
    }
    path = tmp_path / "study_manifest.json"
    path.write_text(json.dumps(manifest))

    result = run_decomposition(
        DecompositionPlan(
            study_manifest_path=str(path),
            components_to_decompose=("feature_builder",),
        ),
        output_dir=tmp_path,
    )

    assert result.plan.components_to_decompose == ("feature_representation",)
    assert result.per_component_shares["feature_representation"] > 0.0
    assert result.per_axis_rows[0]["component"] == "feature_representation"


def test_parquet_schema_columns_match(tmp_path: Path):
    manifest = _synthetic_manifest(tmp_path, scaling_variance=1.0, model_variance=0.5, noise=0.01)
    plan = DecompositionPlan(
        study_manifest_path=str(manifest),
        components_to_decompose=("preprocessing", "nonlinearity"),
    )
    result = run_decomposition(plan, output_dir=tmp_path)
    df = pd.read_parquet(result.result_parquet_path)
    from macrocast.decomposition.schema import expected_columns

    assert list(df.columns) == list(expected_columns())


def test_manual_anova_matches_engine(tmp_path: Path):
    manifest_path = tmp_path / "manual.json"
    variants = []
    for s, v in [("none", 1.0), ("none", 1.1), ("standard", 3.0), ("standard", 3.1)]:
        vid = f"v-{s}-{v}"
        variants.append(
            {
                "variant_id": vid,
                "axis_values": {"2_preprocessing.scaling_policy": s},
                "status": "success",
                "artifact_dir": "",
                "metrics_summary": {"metrics_by_horizon": {"h1": {"msfe": v}}},
            }
        )
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "study_id": "manual",
                "execution_route": "comparison_sweep",
                "parent_recipe_id": "p",
                "parent_recipe_dict": {},
                "axes_swept": ["2_preprocessing.scaling_policy"],
                "variants": variants,
            }
        )
    )
    plan = DecompositionPlan(
        study_manifest_path=str(manifest_path),
        components_to_decompose=("preprocessing",),
    )
    result = run_decomposition(plan, output_dir=tmp_path)

    y = np.array([1.0, 1.1, 3.0, 3.1])
    grand = y.mean()
    ss_total = ((y - grand) ** 2).sum()
    means = {"none": (1.0 + 1.1) / 2, "standard": (3.0 + 3.1) / 2}
    ss_between = sum(2 * (m - grand) ** 2 for m in means.values())
    expected_share = ss_between / ss_total

    assert result.per_component_shares["preprocessing"] == pytest.approx(expected_share, abs=1e-9)
