"""Phase B-3 paper-3 (Goulet Coulombe 2024 "Slow-Growing Trees")
procedure tests.

Round 1 audit identified four findings on this paper:

* **F4 (HIGH helper-vs-core mismatch)** -- ``slow_growing_tree`` exposed
  ``eta_depth_step`` with default ``0.0``, silently overriding the
  class default of ``0.01``. Paper p.87 specifies "starting at η=0.1
  and increasing it by 0.01 with depth, until an imposed plateau of
  0.5". The helper used to disable this rule.
* **F4b (HIGH)** -- ``eta_max_plateau`` (paper p.87 plateau, default
  0.5) and ``mtry_frac`` (paper p.88 §2.3 "mtry = 0.75 is used
  throughout", default 0.75) were NOT exposed by the helper.
* **F5 (PARTIAL Figure 2 grid)** -- ``slow_growing_tree_grid`` returned
  a 4×3 Cartesian product. Paper §3 p.90 specifies a 3-line set
  ``[(η=0.5, H̄=0.25), (η=0.1, H̄=0.25), (η=0.01, H̄=0.05)]`` -- NOT a
  Cartesian product. The Cartesian helper grid both omitted the
  paper's η=0.01 line and silently included η=0 (which routes to
  sklearn ``DecisionTreeRegressor``, bypassing the SGT mechanism).
* **F6 (LOW silent CART fallback)** -- the L4 dispatch for
  ``family="decision_tree"`` with ``split_shrinkage == 0.0`` returns a
  plain sklearn ``DecisionTreeRegressor``. The paper-anchored helper
  ``slow_growing_tree`` should warn the user when η=0 silently
  bypasses the SGT mechanism.

The five tests below guard the helper fixes against regression.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# F4 -- helper default eta_depth_step matches paper p.87
# ----------------------------------------------------------------------


def test_sgt_helper_eta_depth_step_default_is_paper_value():
    """Phase B-3 F4: ``slow_growing_tree()`` with no ``eta_depth_step``
    arg must emit a recipe whose ``fit_params.eta_depth_step == 0.01``
    (paper p.87 rule-of-thumb), NOT ``0.0`` which silently disables the
    depth-step rule."""

    from macroforecast.recipes.paper_methods import slow_growing_tree

    recipe = slow_growing_tree()
    fit = next(
        n for n in recipe["4_forecasting_model"]["nodes"] if n.get("op") == "fit_model"
    )
    assert fit["params"]["eta_depth_step"] == 0.01


# ----------------------------------------------------------------------
# F4b -- helper exposes mtry_frac and eta_max_plateau with paper defaults
# ----------------------------------------------------------------------


def test_sgt_helper_exposes_mtry_frac_and_eta_max_plateau():
    """Phase B-3 F4b: the ``slow_growing_tree`` signature must surface
    both ``mtry_frac`` (paper p.88 §2.3, default 0.75) and
    ``eta_max_plateau`` (paper p.87 plateau, default 0.5) as
    first-class kwargs whose default values flow into the recipe's
    ``fit_params`` dict."""

    import inspect

    from macroforecast.recipes.paper_methods import slow_growing_tree

    sig = inspect.signature(slow_growing_tree)
    assert "mtry_frac" in sig.parameters, (
        "slow_growing_tree must expose ``mtry_frac`` (paper p.88)"
    )
    assert "eta_max_plateau" in sig.parameters, (
        "slow_growing_tree must expose ``eta_max_plateau`` (paper p.87)"
    )
    assert sig.parameters["mtry_frac"].default == 0.75
    assert sig.parameters["eta_max_plateau"].default == 0.5

    # And the defaults must flow into the recipe's fit_params dict.
    recipe = slow_growing_tree()
    fit = next(
        n for n in recipe["4_forecasting_model"]["nodes"] if n.get("op") == "fit_model"
    )
    assert fit["params"]["mtry_frac"] == 0.75
    assert fit["params"]["eta_max_plateau"] == 0.5


# ----------------------------------------------------------------------
# F5 -- grid matches paper Figure 2 / §3 p.90 exactly
# ----------------------------------------------------------------------


def test_sgt_grid_matches_paper_figure_2():
    """Phase B-3 F5: ``slow_growing_tree_grid()`` must return exactly
    the three (η, H̄) pairs from Goulet Coulombe (2024) §3 p.90 / Figure
    2 SGT row -- NOT a Cartesian product. Specifically:

    * ``(η = 0.5,  H̄ = 0.25)``
    * ``(η = 0.1,  H̄ = 0.25)``
    * ``(η = 0.01, H̄ = 0.05)``
    """

    from macroforecast.recipes.paper_methods import slow_growing_tree_grid

    grid = slow_growing_tree_grid()
    assert len(grid) == 3, f"paper §3 p.90 specifies 3 SGT cells, got {len(grid)}"

    expected_pairs = {(0.5, 0.25), (0.1, 0.25), (0.01, 0.05)}
    actual_pairs: set[tuple[float, float]] = set()
    for key, recipe in grid.items():
        fit = next(
            n
            for n in recipe["4_forecasting_model"]["nodes"]
            if n.get("op") == "fit_model"
        )
        eta = float(fit["params"]["split_shrinkage"])
        h_bar = float(fit["params"]["herfindahl_threshold"])
        actual_pairs.add((eta, h_bar))
        # And the paper p.87-88 rule-of-thumb defaults must flow through.
        assert fit["params"]["eta_depth_step"] == 0.01
        assert fit["params"]["eta_max_plateau"] == 0.5
        assert fit["params"]["mtry_frac"] == 0.75

    assert actual_pairs == expected_pairs, (
        f"grid (η, H̄) pairs must match paper §3 p.90 exactly; "
        f"expected {expected_pairs}, got {actual_pairs}"
    )

    # No η=0 line (which would bypass the SGT mechanism).
    assert all(eta > 0.0 for eta, _ in actual_pairs)


# ----------------------------------------------------------------------
# F6 -- η=0 helper call warns about CART fallback
# ----------------------------------------------------------------------


def test_sgt_eta_zero_warns_about_cart_fallback():
    """Phase B-3 F6: ``slow_growing_tree(split_shrinkage=0.0)`` must
    emit a ``UserWarning`` warning the user that η=0 routes to plain
    sklearn ``DecisionTreeRegressor`` (CART) and the SGT mechanism is
    bypassed. The warning is informational; the recipe still builds."""

    from macroforecast.recipes.paper_methods import slow_growing_tree

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        recipe = slow_growing_tree(split_shrinkage=0.0)

    # Recipe still built.
    assert recipe is not None
    fit = next(
        n for n in recipe["4_forecasting_model"]["nodes"] if n.get("op") == "fit_model"
    )
    assert fit["params"]["split_shrinkage"] == 0.0

    # And the user got a warning mentioning the CART fallback.
    sgt_warnings = [
        w
        for w in caught
        if issubclass(w.category, UserWarning)
        and ("CART" in str(w.message) or "DecisionTreeRegressor" in str(w.message))
    ]
    assert sgt_warnings, (
        "slow_growing_tree(split_shrinkage=0.0) must emit a UserWarning "
        "about the CART/DecisionTreeRegressor fallback bypassing SGT"
    )
    # The warning text must mention SGT-bypass somehow (CART or
    # DecisionTreeRegressor or the SGT mechanism).
    msg = str(sgt_warnings[0].message)
    assert any(token in msg for token in ("CART", "DecisionTreeRegressor", "SGT"))


# ----------------------------------------------------------------------
# F4 procedural -- the depth-step rule actually changes per-level η
# ----------------------------------------------------------------------


def test_sgt_depth_ramp_actually_changes_per_level_eta():
    """Phase B-3 F4 procedural: with ``eta_depth_step=0.01``, the
    SGT class must produce a measurably different fitted tree than
    with ``eta_depth_step=0.0`` on the same DGP. (At depth 0 both are
    η=0.1, but at deeper splits the depth-step rule pushes η toward
    the plateau at 0.5; the soft-weighting at split nodes records that
    per-node η in the tree representation.)

    The strongest check is to (1) confirm the two fitted trees have
    different per-split η values, AND (2) confirm their predictions
    differ on the same in-sample design matrix."""

    from macroforecast.core.runtime import _SlowGrowingTree

    rng = np.random.default_rng(0)
    n, K = 200, 4
    X = pd.DataFrame(rng.standard_normal((n, K)), columns=[f"x{j}" for j in range(K)])
    y = pd.Series(0.7 * X["x0"] + 0.3 * X["x1"] ** 2 + 0.4 * rng.standard_normal(n))

    sgt_no_step = _SlowGrowingTree(
        eta=0.1,
        herfindahl_threshold=0.05,
        eta_depth_step=0.0,
        eta_max_plateau=0.5,
        max_depth=4,
        random_state=0,
    ).fit(X, y)
    sgt_with_step = _SlowGrowingTree(
        eta=0.1,
        herfindahl_threshold=0.05,
        eta_depth_step=0.01,
        eta_max_plateau=0.5,
        max_depth=4,
        random_state=0,
    ).fit(X, y)

    # 1) Inspect the tree representation directly. Each split node
    #    records its per-node η at index 5 (see ``_SlowGrowingTree``
    #    docstring + ``_build``). With ``eta_depth_step=0.01`` and
    #    ``max_depth=4`` we expect at least one split node whose
    #    eta_l differs from the constant-η reference tree.
    def _split_etas(tree: _SlowGrowingTree) -> list[float]:
        out: list[float] = []
        for node in tree._nodes:
            if node and node[0] == "split":
                out.append(float(node[5]))
        return out

    etas_no_step = _split_etas(sgt_no_step)
    etas_with_step = _split_etas(sgt_with_step)
    assert etas_no_step, "constant-η tree must have at least one split"
    assert etas_with_step, "depth-step tree must have at least one split"

    # The constant-η tree's per-node η values must all equal 0.1
    # (clipped to [1e-6, plateau=0.5]). The depth-step tree must have
    # at least one node with η > 0.1 (since depth >= 1 pushes
    # η = 0.1 + 0.01 * depth above 0.1).
    assert all(abs(e - 0.1) < 1e-9 for e in etas_no_step), (
        f"with eta_depth_step=0.0 every split must use η=0.1 exactly, got {etas_no_step}"
    )
    assert any(e > 0.1 + 1e-9 for e in etas_with_step), (
        f"with eta_depth_step=0.01 at least one deeper split must use η > 0.1, "
        f"got {etas_with_step}"
    )

    # 2) Predictions must differ at least at some point. (If they
    #    were identical, the depth-step rule would have no observable
    #    effect on the fitted tree.)
    pred_no_step = sgt_no_step.predict(X)
    pred_with_step = sgt_with_step.predict(X)
    assert not np.allclose(pred_no_step, pred_with_step), (
        "SGT with eta_depth_step=0.01 must produce different predictions "
        "than with eta_depth_step=0.0 on the same DGP"
    )
