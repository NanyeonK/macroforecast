"""Smoke-test every example recipe shipped under ``examples/recipes/``.

Two layers of safety:

1. **Parse + classify**: every ``.yaml`` must load via ``yaml.safe_load``,
   produce a dict, and use the canonical NEW 12-layer schema (top-level
   keys in ``{0_meta, 1_data, ..., 8_output}`` plus the four diagnostic
   ``{1,2,3,4}_5_*`` keys).  Catches the v0.0.0 reset's old schema (the
   ``recipe_id: ... / path: { 1_data_task: ... }`` wrapper) regressing
   into ``examples/`` again.

2. **End-to-end execution**: every recipe explicitly listed in
   ``_END_TO_END_RUNNABLE`` must execute via ``macroforecast.run`` without
   raising. This guards against regressions like the
   "single_target requires leaf_config.target string" gate that took
   the CLAUDE.md Quick start example down.

Recipes that intentionally exercise optional extras (xgboost / lightgbm /
catboost / shap / torch) live behind ``importorskip`` so a fresh wheel
install passes the smoke gate without every extra installed.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
RECIPES_DIR = REPO_ROOT / "examples" / "recipes"

_NEW_LAYER_KEYS = {
    # Main 9 layers
    "0_meta", "1_data", "2_preprocessing", "3_feature_engineering",
    "4_forecasting_model", "5_evaluation", "6_statistical_tests",
    "7_interpretation", "8_output",
    # Diagnostic layers (canonical recipe keys -- match the runtime's
    # ``root.get(...)`` calls in macroforecast.core.layers.l{1,2,3,4}_5).
    "1_5_data_summary",
    "2_5_pre_post_preprocessing",
    "3_5_feature_diagnostics",
    "4_5_generator_diagnostics",
}
_OLD_LAYER_KEYS = {
    # Pre-v0.0.0 8-layer schema -- kept here so the smoke test rejects
    # any regression that re-introduces it under examples/.
    "1_data_task", "2_preprocessing_task", "3_training",
    "4_evaluation", "5_horse_race", "6_decomposition",
    "7_export", "8_audit", "5_output_provenance",
}


def _all_recipe_paths() -> list[Path]:
    """Recipes the smoke test enforces. ``archive_v0/`` is the historical
    pre-v0.0.0 schema corner -- the README there warns users not to run
    those files; we skip them here so the smoke gate stays clean."""

    archive = RECIPES_DIR / "archive_v0"
    return sorted(
        p for p in RECIPES_DIR.rglob("*.yaml")
        if archive not in p.parents
    )


# ---------------------------------------------------------------------------
# Layer 1: parse + schema classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("recipe_path", _all_recipe_paths(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_recipe_parses_as_yaml_mapping(recipe_path: Path) -> None:
    """Every example yaml must parse to a plain dict (mapping)."""

    with recipe_path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    assert isinstance(data, dict), (
        f"{recipe_path.relative_to(REPO_ROOT)}: expected top-level mapping, "
        f"got {type(data).__name__}."
    )


@pytest.mark.parametrize("recipe_path", _all_recipe_paths(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_recipe_uses_new_layer_schema(recipe_path: Path) -> None:
    """Reject the v0.0.0-restart 8-layer schema (``recipe_id`` + ``path``
    + ``1_data_task`` etc.). Every example must use the canonical
    12-layer keys so it matches the runtime executor."""

    with recipe_path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    assert isinstance(data, dict)
    keys = set(data.keys())
    old_match = keys & _OLD_LAYER_KEYS
    new_match = keys & _NEW_LAYER_KEYS
    assert not old_match, (
        f"{recipe_path.relative_to(REPO_ROOT)}: uses pre-v0.0.0 schema "
        f"keys {sorted(old_match)}. Rewrite to the 12-layer schema "
        f"({sorted(_NEW_LAYER_KEYS)[:5]}...) or move to ``archive/``."
    )
    assert new_match, (
        f"{recipe_path.relative_to(REPO_ROOT)}: no canonical layer keys "
        f"found (got {sorted(keys)}). Recipe must declare at least one "
        f"of {sorted(_NEW_LAYER_KEYS)[:5]}..."
    )


# ---------------------------------------------------------------------------
# Layer 2: end-to-end execution for the curated runnable subset
# ---------------------------------------------------------------------------

# Recipes that should run end-to-end on a stock install. Add new entries
# here whenever a new self-contained example lands; partial-layer fixtures
# (e.g. l3_minimal_lag_only.yaml that has no L1/L2 preamble) stay out.
_END_TO_END_RUNNABLE = [
    "l4_minimal_ridge.yaml",
    "l4_bagging.yaml",
    "l4_ensemble_ridge_xgb_vs_ar1.yaml",
    "l4_quantile_regression_forest.yaml",
    "l6_standard.yaml",
    "l6_full_replication.yaml",
    "goulet_coulombe_2021_replication.yaml",
]


@pytest.mark.parametrize("name", _END_TO_END_RUNNABLE)
def test_recipe_runs_end_to_end(name: str, tmp_path: Path) -> None:
    """The CLAUDE.md Quick start uses ``macroforecast.run('examples/recipes/...')``.
    These recipes must execute successfully without external data."""

    import macroforecast

    recipe_path = RECIPES_DIR / name
    assert recipe_path.exists(), f"{name} not found"
    result = macroforecast.run(recipe_path, output_directory=tmp_path / name)
    assert result.cells, f"{name}: no cells executed"
    assert all(cell.error is None for cell in result.cells), (
        f"{name}: cell errors: "
        + "; ".join(f"{c.cell_id}: {c.error}" for c in result.cells if c.error)
    )
