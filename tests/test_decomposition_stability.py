"""run_decomposition produces byte-identical parquet across two runs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from macrocast.decomposition import DecompositionPlan, run_decomposition


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build(tmp_path: Path) -> Path:
    rng = np.random.default_rng(11)
    variants = []
    for s in ["none", "standard"]:
        for m in ["ridge", "lasso"]:
            variants.append(
                {
                    "variant_id": f"v-{s}-{m}",
                    "axis_values": {
                        "2_preprocessing.scaling_policy": s,
                        "3_training.model_family": m,
                    },
                    "status": "success",
                    "artifact_dir": "",
                    "metrics_summary": {
                        "metrics_by_horizon": {
                            "h1": {"msfe": float(rng.normal())},
                        }
                    },
                }
            )
    path = tmp_path / "study_manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "study_id": "stability-xyz",
                "execution_route": "comparison_sweep",
                "parent_recipe_id": "p",
                "parent_recipe_dict": {},
                "axes_swept": ["2_preprocessing.scaling_policy", "3_training.model_family"],
                "variants": variants,
            }
        )
    )
    return path


def test_two_runs_byte_identical(tmp_path: Path):
    manifest = _build(tmp_path)
    plan = DecompositionPlan(
        study_manifest_path=str(manifest),
        components_to_decompose=("preprocessing", "nonlinearity"),
    )
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    out1.mkdir()
    out2.mkdir()
    r1 = run_decomposition(plan, output_dir=out1)
    r2 = run_decomposition(plan, output_dir=out2)
    assert _sha(Path(r1.result_parquet_path)) == _sha(Path(r2.result_parquet_path))


def test_per_component_shares_determinism(tmp_path: Path):
    manifest = _build(tmp_path)
    plan = DecompositionPlan(
        study_manifest_path=str(manifest),
        components_to_decompose=("preprocessing", "nonlinearity"),
    )
    r1 = run_decomposition(plan, output_dir=tmp_path / "a")
    r2 = run_decomposition(plan, output_dir=tmp_path / "b")
    for c in r1.per_component_shares:
        assert abs(r1.per_component_shares[c] - r2.per_component_shares[c]) < 1e-12
