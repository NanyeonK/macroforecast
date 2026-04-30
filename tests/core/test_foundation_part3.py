from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from macrocast.core import Recipe
from macrocast.core.layer_specs import AxisSpec, LAYER_SPEC_CHECKLIST, Option, PHASE1_IMPLEMENTATION_ORDER
from macrocast.core.manifest import (
    MANIFEST_SCHEMA_VERSION,
    CellSummary,
    LayerExecutionRecord,
    Manifest,
    capture_dependency_manifest,
    capture_runtime_environment,
    cell_summary_from_cell,
    compare_environments,
    install_dependencies,
    replicate,
)


RECIPE_YAML = """
metadata:
  name: manifest_demo
  author: yeonchan
0_meta:
  fixed_axes:
    failure_policy: fail_fast
3_feature_engineering:
  nodes:
    - id: src_clean
      type: source
      selector: {layer_ref: l2, sink_name: clean_panel_v1}
    - id: lag_x
      type: step
      op: lag
      params: {n_lag: {sweep: [4, 8]}}
      inputs: [src_clean]
  sinks:
    features_v1: lag_x
"""


def test_manifest_initialize_write_load_and_replicate(tmp_path: Path) -> None:
    recipe = Recipe.from_yaml(RECIPE_YAML)
    manifest = Manifest.initialize(recipe, RECIPE_YAML, tmp_path)
    cell = recipe.cells[0]
    summary = cell_summary_from_cell(cell)
    record = LayerExecutionRecord(
        layer_id="l3",
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
        duration_seconds=0.0,
        status="completed",
        nodes_executed=2,
        nodes_cache_hit=1,
        nodes_cache_miss=1,
        produced_sinks=("features_v1",),
    )
    manifest = manifest.with_cell_summary(summary).with_layer_record(record)
    manifest.write_to_disk(tmp_path)

    loaded = Manifest.load(tmp_path / "manifest.json")
    replication = replicate(tmp_path / "manifest.json")

    assert loaded.schema_version == MANIFEST_SCHEMA_VERSION
    assert loaded.recipe_hash == manifest.recipe_hash
    assert (tmp_path / "recipe.yaml").exists()
    assert (tmp_path / "cells" / summary.cell_id / "cell_manifest.json").exists()
    assert replication.matching_recipe_hash


def test_runtime_environment_and_dependency_manifest(tmp_path: Path) -> None:
    (tmp_path / "uv.lock").write_text("locked", encoding="utf-8")

    env = capture_runtime_environment(tmp_path)
    deps = capture_dependency_manifest(tmp_path)
    diff = compare_environments(env, env)

    assert env.python_version
    assert deps.python_lockfile_content == "locked"
    assert not diff.has_critical_diff


def test_install_dependencies_restores_embedded_lockfiles(tmp_path: Path) -> None:
    (tmp_path / "uv.lock").write_text("locked", encoding="utf-8")
    deps = capture_dependency_manifest(tmp_path)
    restore_dir = tmp_path / "restore"

    written = install_dependencies(deps, restore_dir, run=False)

    assert written
    assert (restore_dir / "lockfiles" / "uv.lock").read_text(encoding="utf-8") == "locked"


def test_layer_spec_dataclasses_and_checklist() -> None:
    option = Option(value="ridge", label="Ridge", description="Ridge regression")
    axis = AxisSpec(name="model_family", options=(option,), default="ridge")

    assert axis.sweepable is True
    assert LAYER_SPEC_CHECKLIST[0] == "Layer ID, name, category"
    assert PHASE1_IMPLEMENTATION_ORDER[:3] == ("l0", "l1", "l2")


def test_test_scaffolding_fixtures(mock_fred_md, mock_clean_panel, mock_model_artifact_xgboost) -> None:
    assert mock_fred_md.shape == (60, 5)
    assert mock_clean_panel.column_names[0] == "INDPRO"
    assert mock_model_artifact_xgboost.family == "xgboost"


def test_cell_summary_schema() -> None:
    summary = CellSummary(
        cell_id="cell_001",
        sweep_values={"l3.lag_x.n_lag": 4},
        cell_hash="abc123",
        status="completed",
        output_subdirectory="cells/cell_001",
    )

    assert summary.status == "completed"
    assert summary.output_subdirectory.endswith("cell_001")
