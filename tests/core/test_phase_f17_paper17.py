"""test_phase_f17_paper17.py — F-17 audit gap-fix smoke tests.

Closes audit-paper-17.md F11 LOW (R1):
  "add tutorial-style smoke test with attach_eval_blocks=True."

Paper: Coulombe / Surprenant / Leroux / Stevanovic (2022 JAE)
       "How is Machine Learning Useful for Macroeconomic Forecasting?"
       §2.3 Eq. (10) treatment-effect regression (α_F).

The DM-vs-benchmark statistic array is the observable proxy for Eq. (10)
execution in a minimal 1-cell smoke test.
"""

from __future__ import annotations

import numpy as np
import pytest

import macroforecast
from macroforecast.recipes.paper_methods import ml_useful_macro_horse_race


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


def _minimal_panel_for_paper17(n_obs: int = 80, n_predictors: int = 8, seed: int = 42):
    """Minimal synthetic panel for paper-17 smoke tests.

    Returns a dict suitable for the ``panel`` kwarg of paper-17 helpers.
    Deterministic: uses a fixed RNG seed.
    """
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(n_obs).tolist()
    X = {f"x{i}": rng.standard_normal(n_obs).tolist() for i in range(n_predictors)}
    return {"y": y, **X}


# ---------------------------------------------------------------------------
# Test 1: recipe structure with attach_eval_blocks=True
# ---------------------------------------------------------------------------


def test_paper17_attach_eval_blocks_recipe_structure(tmp_path):
    """Verify that recipe dicts emitted with attach_eval_blocks=True include
    both evaluation blocks and that L6 sub_layers are correctly keyed
    (L6_A_equal_predictive and L6_D_multiple_model).

    Closes audit-paper-17.md F11 LOW / R1 (structure check).
    """
    panel = _minimal_panel_for_paper17()
    grid = ml_useful_macro_horse_race(
        attach_eval_blocks=True,
        panel=panel,
        horizons=(1,),
        cv_schemes=("kfold",),
        cases=("ridge",),
        data_richness="H_minus",
    )

    assert len(grid) >= 1, "grid must contain at least one recipe"

    recipe = next(iter(grid.values()))

    # L5 evaluation block must be present
    assert "5_evaluation" in recipe, (
        "attach_eval_blocks=True must inject '5_evaluation' key"
    )

    # L6 statistical tests block must be present and enabled
    assert "6_statistical_tests" in recipe, (
        "attach_eval_blocks=True must inject '6_statistical_tests' key"
    )
    l6 = recipe["6_statistical_tests"]
    assert l6.get("enabled") is True, "L6 enabled must be True"

    # Both required sub-layers must exist
    sub_layers = l6.get("sub_layers", {})
    assert "L6_A_equal_predictive" in sub_layers, (
        "L6_A_equal_predictive sub-layer must be present"
    )
    assert "L6_D_multiple_model" in sub_layers, (
        "L6_D_multiple_model sub-layer must be present"
    )


# ---------------------------------------------------------------------------
# Test 2: end-to-end run with attach_eval_blocks=True
# ---------------------------------------------------------------------------


def test_paper17_attach_eval_blocks_true_runs_and_has_l6_artefacts(tmp_path):
    """Coulombe et al. (2022 JAE) §2.3 Eq. (10) α_F treatment-effect
    regression smoke test. Closes audit-paper-17.md F11 LOW (R1).

    Verifies:
      (1) ml_useful_macro_horse_race(attach_eval_blocks=True) builds
          recipes that include L5 + L6 blocks;
      (2) macroforecast.run executes one such recipe without error on a
          minimal synthetic panel;
      (3) the result contains at least one cell;
      (4) no cell result is None (no silent failure).
    """
    panel = _minimal_panel_for_paper17()
    grid = ml_useful_macro_horse_race(
        attach_eval_blocks=True,
        panel=panel,
        horizons=(1,),
        cv_schemes=("kfold",),
        cases=("ridge",),
        data_richness="H_minus",
    )

    assert len(grid) >= 1, "grid must contain at least one recipe"

    # Step 1: verify recipe has L5 + L6 keys
    recipe = next(iter(grid.values()))
    assert "5_evaluation" in recipe
    assert "6_statistical_tests" in recipe
    assert recipe["6_statistical_tests"].get("enabled") is True

    # Step 2: run the recipe end-to-end
    result = macroforecast.run(
        recipe,
        output_directory=str(tmp_path / "paper17_eval"),
    )

    # Step 3: result must have cells
    assert hasattr(result, "cells"), "result must expose .cells attribute"
    assert len(result.cells) >= 1, "result.cells must be non-empty"

    # Step 4: no cell result is None
    for cell in result.cells:
        assert cell is not None, "cell must not be None (silent failure)"
