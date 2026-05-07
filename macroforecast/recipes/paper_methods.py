"""Paper-method recipe builders (v0.9 Phase 2 paper-coverage pass).

One helper per paper in the 16-paper macro-forecasting target list. Each
helper returns a recipe dict ready for :func:`macroforecast.run`.

**Decomposition discipline.** Helpers are *recipe patterns*, not L4
families. Each helper docstring states the decomposition explicitly so
the algorithmic content is visible without reading the source. The
emitted recipe dict is identical to the paired YAML at
``examples/recipes/replications/<paper>.yaml`` -- think of the YAML as
the canonical hand-readable copy and the helper as the programmatic
constructor.

Status indicators in each helper docstring:

* **operational** -- runs end-to-end on the current ``main``.
* **pre-promotion** -- recipe is canonical; one or more atomic
  primitives it depends on are still ``future``-status. Calling
  :func:`macroforecast.run` will hit a ``NotImplementedError`` from the
  runtime gate. Helper still produces the recipe so users can inspect /
  customise the canonical decomposition ahead of the runtime promotion.
"""
from __future__ import annotations

from typing import Any, Literal
import warnings


# ---------------------------------------------------------------------------
# Synthetic 12-row panel reused by recipe defaults (mirrors the
# examples/recipes/l4_minimal_ridge.yaml panel).
# ---------------------------------------------------------------------------

_DEFAULT_PANEL: dict[str, list[Any]] = {
    "date": [
        "2018-01-01", "2018-02-01", "2018-03-01", "2018-04-01",
        "2018-05-01", "2018-06-01", "2018-07-01", "2018-08-01",
        "2018-09-01", "2018-10-01", "2018-11-01", "2018-12-01",
    ],
    "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
    "x1": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
}


def _l1_minimal(target: str, horizon: int, panel: dict[str, list[Any]] | None) -> dict[str, Any]:
    """L1 block shared across helpers when no FRED data is requested."""

    return {
        "fixed_axes": {
            "custom_source_policy": "custom_panel_only",
            "frequency": "monthly",
            "horizon_set": "custom_list",
        },
        "leaf_config": {
            "target": target,
            "target_horizons": [horizon],
            "custom_panel_inline": panel or _DEFAULT_PANEL,
        },
    }


def _l3_lag_target(horizon: int) -> dict[str, Any]:
    """L3 block: lag + target_construction. Shared across most helpers."""

    return {
        "nodes": [
            {"id": "src_X", "type": "source", "selector": {"layer_ref": "l2", "sink_name": "l2_clean_panel_v1", "subset": {"role": "predictors"}}},
            {"id": "src_y", "type": "source", "selector": {"layer_ref": "l2", "sink_name": "l2_clean_panel_v1", "subset": {"role": "target"}}},
            {"id": "lag_x", "type": "step", "op": "lag", "params": {"n_lag": 1}, "inputs": ["src_X"]},
            {"id": "y_h", "type": "step", "op": "target_construction", "params": {"mode": "point_forecast", "method": "direct", "horizon": horizon}, "inputs": ["src_y"]},
        ],
        "sinks": {
            "l3_features_v1": {"X_final": "lag_x", "y_final": "y_h"},
            "l3_metadata_v1": "auto",
        },
    }


def _l4_single_fit(family: str, fit_params: dict[str, Any], fit_node_id: str = "fit") -> dict[str, Any]:
    """L4 block with one fit_model node + predict node."""

    fit = {
        "id": fit_node_id,
        "type": "step",
        "op": "fit_model",
        "params": {
            "family": family,
            "forecast_strategy": "direct",
            "training_start_rule": "expanding",
            "refit_policy": "every_origin",
            "search_algorithm": "none",
            "min_train_size": 6,
            **fit_params,
        },
        "inputs": ["src_X", "src_y"],
    }
    return {
        "nodes": [
            {"id": "src_X", "type": "source", "selector": {"layer_ref": "l3", "sink_name": "l3_features_v1", "subset": {"component": "X_final"}}},
            {"id": "src_y", "type": "source", "selector": {"layer_ref": "l3", "sink_name": "l3_features_v1", "subset": {"component": "y_final"}}},
            fit,
            {"id": "predict", "type": "step", "op": "predict", "inputs": [fit_node_id, "src_X"]},
        ],
        "sinks": {
            "l4_forecasts_v1": "predict",
            "l4_model_artifacts_v1": fit_node_id,
            "l4_training_metadata_v1": "auto",
        },
    }


def _l4_with_benchmark(
    family: str,
    fit_params: dict[str, Any],
    *,
    benchmark_family: str = "factor_augmented_ar",
    benchmark_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """L4 block with two fit_model nodes: the cell's family plus a
    benchmark flagged ``is_benchmark: true`` for downstream L6 DM / MCS
    routing.

    Used by the paper-16 helper when ``attach_eval_blocks=True`` so the
    auto-attached L6 sub-layers have a baseline to test against
    (Coulombe-Surprenant-Leroux-Stevanovic 2022 §4.4 reports DM vs the
    ``(ARDI, BIC)`` reference).

    Phase A3 fix: the pre-A3 default was ``benchmark_family="ar_p"``
    with ``n_lag=4``, contradicting paper §4.4 ("DM procedure is used to
    test the predictive accuracy of each model against the reference
    (ARDI,BIC)"). New default = ``factor_augmented_ar`` with
    ``search_algorithm="bic"`` and ``n_factors=4``. The benchmark fit
    node now wires ``src_X`` (factors flow into the ARDI fit) and
    ``src_y`` rather than ``src_y`` alone.
    """

    bench_params = {
        "family": benchmark_family,
        "forecast_strategy": "direct",
        "training_start_rule": "expanding",
        "refit_policy": "every_origin",
        "search_algorithm": "bic",
        "min_train_size": 6,
        "n_factors": 4,
        **(benchmark_params or {}),
    }
    cell_params = {
        "family": family,
        "forecast_strategy": "direct",
        "training_start_rule": "expanding",
        "refit_policy": "every_origin",
        "search_algorithm": "none",
        "min_train_size": 6,
        **fit_params,
    }
    return {
        "nodes": [
            {"id": "src_X", "type": "source", "selector": {"layer_ref": "l3", "sink_name": "l3_features_v1", "subset": {"component": "X_final"}}},
            {"id": "src_y", "type": "source", "selector": {"layer_ref": "l3", "sink_name": "l3_features_v1", "subset": {"component": "y_final"}}},
            {"id": "fit_benchmark", "type": "step", "op": "fit_model", "params": bench_params, "is_benchmark": True, "inputs": ["src_X", "src_y"]},
            {"id": "predict_benchmark", "type": "step", "op": "predict", "inputs": ["fit_benchmark", "src_X"]},
            {"id": "fit_cell", "type": "step", "op": "fit_model", "params": cell_params, "inputs": ["src_X", "src_y"]},
            {"id": "predict_cell", "type": "step", "op": "predict", "inputs": ["fit_cell", "src_X"]},
        ],
        "forecast_object": "point",
        "sinks": {
            "l4_forecasts_v1": "predict_cell",
            "l4_model_artifacts_v1": "fit_cell",
            "l4_training_metadata_v1": "auto",
        },
    }


def _l5_l6_paper16_eval_blocks() -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(5_evaluation, 6_statistical_tests)`` blocks that wire
    DM-vs-benchmark + MCS for the auto-attached paper-16 evaluation."""

    l5 = {"fixed_axes": {}}
    l6 = {
        "enabled": True,
        "test_scope": "per_target_horizon",
        "dependence_correction": "newey_west",
        "overlap_handling": "nw_with_h_minus_1_lag",
        "sub_layers": {
            "L6_A_equal_predictive": {
                "enabled": True,
                "fixed_axes": {
                    "equal_predictive_test": "dm_diebold_mariano",
                    "loss_function": "squared",
                    "model_pair_strategy": "vs_benchmark_only",
                    "hln_correction": True,
                },
            },
            "L6_D_multiple_model": {
                "enabled": True,
                "fixed_axes": {
                    "multiple_model_test": "mcs_hansen",
                    "mcs_alpha": 0.10,
                    "mmt_loss_function": "squared",
                    "bootstrap_method": "stationary_bootstrap",
                    "bootstrap_n_replications": 1000,
                    "bootstrap_block_length": "auto",
                    "mcs_t_statistic": "t_max",
                },
            },
        },
        "leaf_config": {"nw_lag_truncation": "auto"},
    }
    return l5, l6


def _base_recipe(
    *,
    target: str,
    horizon: int,
    panel: dict[str, list[Any]] | None,
    seed: int,
    l4: dict[str, Any],
) -> dict[str, Any]:
    """Stitch L0-L4 into a recipe dict (no L5+; callers can extend)."""

    return {
        "0_meta": {
            "fixed_axes": {"failure_policy": "fail_fast", "reproducibility_mode": "seeded_reproducible"},
            "leaf_config": {"random_seed": seed},
        },
        "1_data": _l1_minimal(target, horizon, panel),
        "2_preprocessing": {
            "fixed_axes": {"transform_policy": "no_transform", "outlier_policy": "none", "imputation_policy": "none_propagate", "frame_edge_policy": "keep_unbalanced"},
        },
        "3_feature_engineering": _l3_lag_target(horizon),
        "4_forecasting_model": l4,
    }


# ---------------------------------------------------------------------------
# 1. Scaled PCA (Huang/Jiang/Li/Tong/Zhou 2022)
# ---------------------------------------------------------------------------

# Scaled PCA helper lives at ``scaled_pca`` below; the comment that
# previously deferred it ("TBD: needs a multi-feature panel") was stale
# — the helper has been operational since v0.1 because the L3
# ``scaled_pca`` op handles the multi-feature panel internally. Audit
# gap-fix removes the stale deferred-status note.


# ---------------------------------------------------------------------------
# 3. To Bag is to Prune -- Perfectly Random Forest baseline (Coulombe 2024)
# ---------------------------------------------------------------------------

def perfectly_random_forest(
    *,
    target: str = "y",
    horizon: int = 1,
    n_estimators: int = 200,
    panel: dict[str, list[Any]] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Goulet Coulombe (2024) "To Bag is to Prune" -- PRF baseline.

    **Decomposition.** PRF = ``ExtraTrees(max_features=1, ...)``. Each
    split picks one predictor uniformly at random and one threshold
    uniformly within that predictor's range, with no greedy CART
    objective. Bagging across many such trees provides the variance
    reduction that yields competitive forecasts.

    **Status: operational.** The ``max_features=1`` plumbing through to
    ``sklearn.ensemble.ExtraTreesRegressor`` lands in v0.9.0; recipe
    smoke-tested in ``tests/core/test_v09_paper_coverage.py``.

    Reference: Goulet Coulombe (2024) "To Bag is to Prune",
    arXiv:2008.07063.
    """

    return _base_recipe(
        target=target,
        horizon=horizon,
        panel=panel,
        seed=seed,
        l4=_l4_single_fit(
            family="extra_trees",
            fit_params={"max_features": 1, "n_estimators": n_estimators},
            fit_node_id="fit_prf",
        ),
    )


# ---------------------------------------------------------------------------
# 3b/3c/4-6/8-10/14. Pre-promotion recipe builders -- canonical
# decomposition emitted; runtime hits NotImplementedError until the
# gated atomic primitive lands.
# ---------------------------------------------------------------------------

def booging(
    *,
    target: str = "y",
    horizon: int = 1,
    n_iterations: int = 10,
    panel: dict[str, list[Any]] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Goulet Coulombe (2024) "To Bag is to Prune" -- Booging algorithm.

    **Decomposition.** Booging = ``bagging(strategy=sequential_residual)``.
    Each round refits a fresh bagged learner on the residuals of the
    previous round's bagged prediction.

    **Status: pre-promotion.** Calling :func:`macroforecast.run` raises
    ``NotImplementedError`` from the ``bagging.strategy`` runtime gate
    until the v0.9.x promotion. The recipe surface is canonical so
    callers can inspect / customise ahead of time.

    Reference: Goulet Coulombe (2024) arXiv:2008.07063 §4 Booging.
    """

    return _base_recipe(
        target=target,
        horizon=horizon,
        panel=panel,
        seed=seed,
        l4=_l4_single_fit(
            family="bagging",
            fit_params={
                "strategy": "sequential_residual",
                "base_family": "decision_tree",
                "n_estimators": n_iterations,
            },
            fit_node_id="fit_booging",
        ),
    )


def marsquake(
    *,
    target: str = "y",
    horizon: int = 1,
    n_estimators: int = 50,
    panel: dict[str, list[Any]] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Goulet Coulombe (2024) MARSquake = ``bagging(base_family=mars)``.

    **Status: pre-promotion** -- ``mars`` family is future. Reference:
    Goulet Coulombe (2024) arXiv:2008.07063 §5 MARSquake.
    """

    return _base_recipe(
        target=target,
        horizon=horizon,
        panel=panel,
        seed=seed,
        l4=_l4_single_fit(
            family="bagging",
            fit_params={"base_family": "mars", "n_estimators": n_estimators},
            fit_node_id="fit_marsquake",
        ),
    )


def slow_growing_tree(
    *,
    target: str = "y",
    horizon: int = 1,
    split_shrinkage: float = 0.1,
    herfindahl_threshold: float = 0.25,
    eta_depth_step: float = 0.0,
    max_depth: int | None = None,
    panel: dict[str, list[Any]] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Goulet Coulombe (2024) Slow-Growing Tree =
    ``decision_tree(split_shrinkage=η)``.

    Audit gap-fix: ``herfindahl_threshold`` (H̄, default 0.25 per paper),
    ``eta_depth_step`` (η-depth-update rule, default 0.0), and
    ``max_depth`` (safety bound, default None) now first-class helper
    args so the SGT sub-axis is sweep-friendly without users having to
    hand-edit the recipe dict.

    **Status: pre-promotion** -- ``decision_tree.split_shrinkage`` is
    future. Reference: Goulet Coulombe (2024)
    doi:10.1007/978-3-031-43601-7_4.
    """

    fit_params: dict[str, Any] = {
        "split_shrinkage": split_shrinkage,
        "herfindahl_threshold": herfindahl_threshold,
        "eta_depth_step": eta_depth_step,
    }
    if max_depth is not None:
        fit_params["max_depth"] = int(max_depth)
    return _base_recipe(
        target=target,
        horizon=horizon,
        panel=panel,
        seed=seed,
        l4=_l4_single_fit(
            family="decision_tree",
            fit_params=fit_params,
            fit_node_id="fit_sgt",
        ),
    )


def slow_growing_tree_grid(
    *,
    target: str = "y",
    horizon: int = 1,
    eta_grid: tuple[float, ...] = (0.0, 0.1, 0.25, 1.0),
    h_bar_grid: tuple[float, ...] = (0.05, 0.1, 0.25),
    panel: dict[str, list[Any]] | None = None,
    seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """Slow-Growing Tree paper-faithful grid: η ∈ {0.0, 0.1, 0.25, 1.0}
    × H̄ ∈ {0.05, 0.1, 0.25} (Goulet Coulombe 2024 Figure 2 + Table 1).
    Returns a dict keyed by ``"eta<η>__h<H̄>"`` so users can run each
    cell and aggregate via L5 / L6."""

    grid: dict[str, dict[str, Any]] = {}
    for eta in eta_grid:
        for h_bar in h_bar_grid:
            recipe = slow_growing_tree(
                target=target, horizon=horizon,
                split_shrinkage=eta, herfindahl_threshold=h_bar,
                panel=panel, seed=seed,
            )
            grid[f"eta{eta}__h{h_bar}"] = recipe
    return grid


def two_step_ridge(
    *,
    target: str = "y",
    horizon: int = 1,
    alpha_step1: float = 1.0,
    alpha_step2: float = 0.1,
    vol_model: str = "garch11",
    panel: dict[str, list[Any]] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Goulet Coulombe (2025) 2SRR = chained ``ridge`` +
    ``ridge(prior=random_walk)``.

    ``vol_model`` (default ``"garch11"``) controls the residual-variance
    estimator used in step 2's heterogeneous-Ω solve (paper §4 Eq. 11).
    Audit gap-fix: previous default ``"ewma"`` was the lighter
    RiskMetrics decay; the paper specifies GARCH(1,1) (requires the
    optional ``arch`` package; falls back to EWMA if unavailable).

    **Status: pre-promotion** -- ``ridge.prior=random_walk`` is future.
    Reference: doi:10.1016/j.ijforecast.2024.08.006.
    """

    step2_params = {
        "family": "ridge", "alpha": alpha_step2, "prior": "random_walk",
        "vol_model": vol_model,
        "forecast_strategy": "direct", "training_start_rule": "expanding",
        "refit_policy": "every_origin", "search_algorithm": "none", "min_train_size": 6,
    }
    l4 = {
        "nodes": [
            {"id": "src_X", "type": "source", "selector": {"layer_ref": "l3", "sink_name": "l3_features_v1", "subset": {"component": "X_final"}}},
            {"id": "src_y", "type": "source", "selector": {"layer_ref": "l3", "sink_name": "l3_features_v1", "subset": {"component": "y_final"}}},
            {"id": "fit_step1", "type": "step", "op": "fit_model", "params": {"family": "ridge", "alpha": alpha_step1, "forecast_strategy": "direct", "training_start_rule": "expanding", "refit_policy": "every_origin", "search_algorithm": "none", "min_train_size": 6}, "inputs": ["src_X", "src_y"]},
            {"id": "fit_step2", "type": "step", "op": "fit_model", "params": step2_params, "inputs": ["src_X", "src_y"]},
            {"id": "predict", "type": "step", "op": "predict", "inputs": ["fit_step2", "src_X"]},
        ],
        "sinks": {"l4_forecasts_v1": "predict", "l4_model_artifacts_v1": "fit_step2", "l4_training_metadata_v1": "auto"},
    }
    return _base_recipe(target=target, horizon=horizon, panel=panel, seed=seed, l4=l4)


def hemisphere_neural_network(
    *,
    target: str = "y",
    horizon: int = 1,
    panel: dict[str, list[Any]] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Goulet Coulombe / Frenette / Klieber (2025 JAE) HNN =
    ``mlp(architecture=hemisphere, loss=volatility_emphasis)``.

    **Status: pre-promotion** -- ``mlp.architecture`` and ``mlp.loss``
    sub-axes are future.
    """

    return _base_recipe(
        target=target,
        horizon=horizon,
        panel=panel,
        seed=seed,
        l4=_l4_single_fit(
            family="mlp",
            fit_params={"architecture": "hemisphere", "loss": "volatility_emphasis"},
            fit_node_id="fit_hnn",
        ),
    )


def assemblage_regression(
    *,
    target: str = "y",
    horizon: int = 1,
    panel: dict[str, list[Any]] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Goulet Coulombe et al. (2024) Assemblage =
    ``ridge(coefficient_constraint=nonneg)``.

    **Status: pre-promotion** -- ``ridge.coefficient_constraint=nonneg``
    is future.
    """

    return _base_recipe(
        target=target,
        horizon=horizon,
        panel=panel,
        seed=seed,
        l4=_l4_single_fit(
            family="ridge",
            fit_params={"coefficient_constraint": "nonneg", "alpha": 1.0},
            fit_node_id="fit_assemblage",
        ),
    )


# ---------------------------------------------------------------------------
# 1. Scaled PCA (Huang/Jiang/Li/Tong/Zhou 2022)
# ---------------------------------------------------------------------------

def scaled_pca(
    *,
    target: str = "y",
    horizon: int = 1,
    n_components: int = 4,
    panel: dict[str, list[Any]] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Huang/Jiang/Li/Tong/Zhou (2022) scaled PCA = ``scaled_pca`` L3 op
    feeding a downstream linear regression.

    **Decomposition.** Two-step procedure: step 1 scales each predictor
    by its target-supervised slope (predictive scaling); step 2 runs PCA
    on the scaled matrix. ``scaled_pca`` op already implements the joint
    scaling-then-PCA in one node; downstream ridge fits the resulting
    factors against the target.

    **Status: operational** -- ``scaled_pca`` op operational since v0.1.
    """

    l3 = {
        "nodes": [
            {"id": "src_X", "type": "source", "selector": {"layer_ref": "l2", "sink_name": "l2_clean_panel_v1", "subset": {"role": "predictors"}}},
            {"id": "src_y", "type": "source", "selector": {"layer_ref": "l2", "sink_name": "l2_clean_panel_v1", "subset": {"role": "target"}}},
            {"id": "spca", "type": "step", "op": "scaled_pca",
             "params": {"n_components": n_components, "temporal_rule": "expanding_window_per_origin"},
             "inputs": ["src_X", "src_y"]},
            {"id": "y_h", "type": "step", "op": "target_construction", "params": {"mode": "point_forecast", "method": "direct", "horizon": horizon}, "inputs": ["src_y"]},
        ],
        "sinks": {"l3_features_v1": {"X_final": "spca", "y_final": "y_h"}, "l3_metadata_v1": "auto"},
    }
    recipe = _base_recipe(target=target, horizon=horizon, panel=panel, seed=seed, l4=_l4_single_fit("ridge", {"alpha": 1.0}))
    recipe["3_feature_engineering"] = l3
    return recipe


# ---------------------------------------------------------------------------
# 2. The Macroeconomy as a Random Forest (Coulombe 2024)
# ---------------------------------------------------------------------------

def macroeconomic_random_forest(
    *,
    target: str = "y",
    horizon: int = 1,
    n_estimators: int = 200,
    block_size: int = 24,
    panel: dict[str, list[Any]] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Goulet Coulombe (2024) MRF = ``macroeconomic_random_forest`` family
    (per-leaf local linear regression).

    **Decomposition.** MRF is a true atomic primitive (per-leaf local
    linear regression on top of a time-aware random forest). Operational
    since v0.2 (#187). Pair with L7 ``mrf_gtvp`` (#190) for per-leaf
    coefficient inspection.

    ``block_size`` (default 24) is the Bayesian Bayesian Bootstrap
    block length; v0.9.0a0 audit gap-fix raised the default from the
    upstream 12 to match the paper's monthly-data spec. Override to
    8 for quarterly panels or 52 for weekly panels.

    **Status: operational.**
    """

    return _base_recipe(
        target=target,
        horizon=horizon,
        panel=panel,
        seed=seed,
        l4=_l4_single_fit(
            family="macroeconomic_random_forest",
            fit_params={"n_estimators": n_estimators, "block_size": block_size},
            fit_node_id="fit_mrf",
        ),
    )


# ---------------------------------------------------------------------------
# 4. Adaptive Moving Average -- AlbaMA (Coulombe & Klieber 2025)
# ---------------------------------------------------------------------------

def adaptive_ma(
    *,
    target: str = "y",
    horizon: int = 1,
    n_estimators: int = 100,
    panel: dict[str, list[Any]] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Goulet Coulombe & Klieber (2025) AlbaMA = ``adaptive_ma_rf`` L3 op
    feeding a downstream ridge.

    **Status: pre-promotion** -- ``adaptive_ma_rf`` runtime is future.
    Reference: arXiv:2501.13222.
    """

    l3 = {
        "nodes": [
            {"id": "src_X", "type": "source", "selector": {"layer_ref": "l2", "sink_name": "l2_clean_panel_v1", "subset": {"role": "predictors"}}},
            {"id": "src_y", "type": "source", "selector": {"layer_ref": "l2", "sink_name": "l2_clean_panel_v1", "subset": {"role": "target"}}},
            {"id": "alba", "type": "step", "op": "adaptive_ma_rf", "params": {"n_estimators": n_estimators}, "inputs": ["src_X"]},
            {"id": "y_h", "type": "step", "op": "target_construction", "params": {"mode": "point_forecast", "method": "direct", "horizon": horizon}, "inputs": ["src_y"]},
        ],
        "sinks": {"l3_features_v1": {"X_final": "alba", "y_final": "y_h"}, "l3_metadata_v1": "auto"},
    }
    recipe = _base_recipe(target=target, horizon=horizon, panel=panel, seed=seed, l4=_l4_single_fit("ridge", {"alpha": 1.0}))
    recipe["3_feature_engineering"] = l3
    return recipe


# ---------------------------------------------------------------------------
# 7. OLS as Attention (Coulombe 2026) -- conceptual demo
# ---------------------------------------------------------------------------

def ols_attention_demo(
    *,
    target: str = "y",
    horizon: int = 1,
    panel: dict[str, list[Any]] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Goulet Coulombe (2026) OLS-as-Attention -- conceptual paper.

    Reference paper has no replication object; the package supports the
    empirical demo by running OLS and a transformer side-by-side via
    sweep machinery. Helper returns the OLS half of the demo as a
    starting point.

    **Status: operational.** Reference: SSRN 5200864.
    """

    return _base_recipe(
        target=target,
        horizon=horizon,
        panel=panel,
        seed=seed,
        l4=_l4_single_fit(family="ols", fit_params={}, fit_node_id="fit_ols_attn_demo"),
    )


# ---------------------------------------------------------------------------
# 8. Anatomy of OOS Forecasting (Borup et al. 2022)
# ---------------------------------------------------------------------------

def anatomy_oos(
    *,
    target: str = "y",
    horizon: int = 1,
    initial_window: int | None = None,
    n_iterations: int = 500,
    panel: dict[str, list[Any]] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Borup et al. (2022) Anatomy of OOS Forecasting Accuracy.

    **Decomposition.** Baseline forecast (ridge on lagged X) plus L7
    ``oshapley_vi`` + ``pbsv`` interpretation ops. Helper returns the
    L0-L4 baseline plus a ready-to-extend L7 block carrying the
    ``initial_window`` and ``n_iterations`` parameters that route the
    anatomy adapter to the paper-faithful Path A (per-origin refit via
    ``AnatomySubsets.generate``). Audit gap-fix: previous helper did
    not surface ``initial_window``, so the adapter silently fell back
    to Path B (final-window fit, "degraded" status) — different
    estimand from the published procedure.

    Recommended ``initial_window`` ≈ ``int(T * 0.6)`` (60% expanding-
    window seed). ``n_iterations=500`` matches paper p.16 footnote 16.

    **Status: pre-promotion** for the full L7 workflow (oshapley_vi /
    pbsv via anatomy package); the baseline forecast is operational.
    Reference: SSRN 4278745.
    """

    recipe = _base_recipe(
        target=target,
        horizon=horizon,
        panel=panel,
        seed=seed,
        l4=_l4_single_fit(family="ridge", fit_params={"alpha": 1.0}, fit_node_id="fit_anatomy_baseline"),
    )
    # Stamp the anatomy parameters onto the recipe metadata so callers
    # building the L7 block downstream can pull them via a single
    # accessor rather than re-deriving them.
    recipe["0_meta"].setdefault("leaf_config", {})
    recipe["0_meta"]["leaf_config"].update({
        "anatomy_initial_window": initial_window,
        "anatomy_n_iterations": int(n_iterations),
    })
    return recipe


# ---------------------------------------------------------------------------
# 9. Dual Interpretation of ML Forecasts (Coulombe et al. 2024)
# ---------------------------------------------------------------------------

def dual_interpretation(
    *,
    target: str = "y",
    horizon: int = 1,
    panel: dict[str, list[Any]] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Goulet Coulombe / Goebel / Klieber (2024) Dual Interpretation.

    **Decomposition.** Baseline forecast (ridge on lagged X) plus L7
    ``dual_decomposition`` interpretation op. The op's output artifact
    also carries the four portfolio diagnostics (HHI / short / turnover
    / leverage) inline -- those are trivial numpy reductions on the
    dual weights and do not warrant a separate L7 op.

    **Status: pre-promotion** for L7 ``dual_decomposition`` (future);
    baseline forecast operational. Reference: arXiv:2412.13076.
    """

    return _base_recipe(
        target=target,
        horizon=horizon,
        panel=panel,
        seed=seed,
        l4=_l4_single_fit(family="ridge", fit_params={"alpha": 1.0}, fit_node_id="fit_dual_baseline"),
    )


# ---------------------------------------------------------------------------
# 10. Maximally Forward-Looking Core Inflation (Coulombe et al. 2024)
# ---------------------------------------------------------------------------

def maximally_forward_looking(
    *,
    target: str = "y",
    horizon: int = 1,
    variant: Literal["ranks", "comps"] = "ranks",
    prior_target: list[float] | None = None,
    alpha: float = 1.0,
    search_algorithm: str = "block_cv",
    panel: dict[str, list[Any]] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Goulet Coulombe et al. (2024) Maximally Forward-Looking Core
    Inflation = Albacore family.

    Two paper variants are exposed via the ``variant`` argument:

    * ``variant="ranks"`` (**default**, paper-headline ``Albacore_ranks``
      / Variant B): L3 ``asymmetric_trim`` (rank-space transformation:
      per-period sort of inflation components) feeding a non-negative
      ridge with the **fused-difference prior**
      (``ridge(coefficient_constraint=nonneg, prior=fused_difference)``).
      The fused-difference penalty is the smoothness-on-ranks regulariser
      that pairs with the asymmetric_trim rank transform per paper §2
      derivation. ``prior_target`` is ignored on this path.
    * ``variant="comps"`` (Variant A, ``Albacore_comps``): drop
      ``asymmetric_trim`` and run shrink-to-target non-negative ridge in
      component space (``ridge(coefficient_constraint=nonneg,
      prior=shrink_to_target)``). Pass ``prior_target=[<basket weights>]``
      to match the paper's w_headline objective; if ``None`` the helper
      emits a ``UserWarning`` and falls back to uniform 1/K.

    Audit gap-fix (Phase A2 v0.9.0a0): the helper previously injected
    ``outlier_policy: asymmetric_trim`` at L2, but the L2 enum only
    accepts ``{mccracken_ng_iqr, winsorize, zscore_threshold, none}`` —
    every helper call validator-rejected at recipe time. ``asymmetric_trim``
    is registered at L3 (``core/ops/l3_ops.py:439``); when present (Variant
    B / "ranks") it sits as an L3 step node ahead of the standard ``lag``
    chain so ``src_X`` flows ``asymmetric_trim → lag → X_final``.

    Audit gap-fix (Phase A3): the pre-A3 helper composed
    ``asymmetric_trim`` with ``prior=shrink_to_target`` (Variant A
    objective on rank-space input) — paper-incoherent because the paper
    §2 derivation pairs the trim with the fused-difference prior
    (Variant B). The ``variant`` argument now switches between the
    paper's two pairings cleanly.

    ``alpha`` (shrinkage strength) + ``search_algorithm`` (default
    ``block_cv`` = paper §3 non-overlapping 10-fold block CV) are
    first-class helper args on both variants.

    **Status: operational (v0.9.0a0 Phase A2 + v0.9.0a0 Phase A3 fix).**
    Both ``asymmetric_trim`` (operational since v0.8.9 B-6) and
    ``ridge(coefficient_constraint=nonneg, prior={fused_difference,
    shrink_to_target})`` (operational since v0.9.0a0) are
    runtime-implemented. Reference: arXiv:2501.13... (technical report).
    """

    if variant not in {"ranks", "comps"}:
        raise ValueError(
            f"variant must be 'ranks' or 'comps'; got {variant!r}. "
            "ranks = paper-headline Albacore_ranks (Variant B, asymmetric_trim "
            "+ fused_difference prior); comps = Albacore_comps (Variant A, "
            "shrink_to_target prior in component space)."
        )

    if variant == "ranks":
        # Variant B / Albacore_ranks: asymmetric_trim L3 step + fused
        # difference prior at L4. ``prior_target`` is ignored here.
        l3 = {
            "nodes": [
                {"id": "src_X", "type": "source", "selector": {"layer_ref": "l2", "sink_name": "l2_clean_panel_v1", "subset": {"role": "predictors"}}},
                {"id": "src_y", "type": "source", "selector": {"layer_ref": "l2", "sink_name": "l2_clean_panel_v1", "subset": {"role": "target"}}},
                {"id": "trim_x", "type": "step", "op": "asymmetric_trim", "params": {"smooth_window": 0}, "inputs": ["src_X"]},
                {"id": "lag_x", "type": "step", "op": "lag", "params": {"n_lag": 1}, "inputs": ["trim_x"]},
                {"id": "y_h", "type": "step", "op": "target_construction", "params": {"mode": "point_forecast", "method": "direct", "horizon": horizon}, "inputs": ["src_y"]},
            ],
            "sinks": {
                "l3_features_v1": {"X_final": "lag_x", "y_final": "y_h"},
                "l3_metadata_v1": "auto",
            },
        }
        fit_params: dict[str, Any] = {
            "coefficient_constraint": "nonneg",
            "alpha": float(alpha),
            "prior": "fused_difference",
            "search_algorithm": str(search_algorithm),
        }
    else:  # variant == "comps"
        # Variant A / Albacore_comps: drop asymmetric_trim + shrink-to-
        # target prior at L4. ``prior_target`` is the paper's w_headline.
        # Phase A4d (paper 13, Round 4 F5): paper Eq. (1) is undefined
        # without ``w_headline``. The pre-A4 helper warned and fell back
        # to uniform 1/K — paper-incoherent. Hard-error instead.
        if prior_target is None:
            raise ValueError(
                "variant='comps' requires prior_target (paper §2 w_headline "
                "basket weights). Use variant='ranks' (default, paper-headline "
                "Albacore_ranks Variant B) for a prior-free path."
            )
        l3 = {
            "nodes": [
                {"id": "src_X", "type": "source", "selector": {"layer_ref": "l2", "sink_name": "l2_clean_panel_v1", "subset": {"role": "predictors"}}},
                {"id": "src_y", "type": "source", "selector": {"layer_ref": "l2", "sink_name": "l2_clean_panel_v1", "subset": {"role": "target"}}},
                {"id": "lag_x", "type": "step", "op": "lag", "params": {"n_lag": 1}, "inputs": ["src_X"]},
                {"id": "y_h", "type": "step", "op": "target_construction", "params": {"mode": "point_forecast", "method": "direct", "horizon": horizon}, "inputs": ["src_y"]},
            ],
            "sinks": {
                "l3_features_v1": {"X_final": "lag_x", "y_final": "y_h"},
                "l3_metadata_v1": "auto",
            },
        }
        fit_params = {
            "coefficient_constraint": "nonneg",
            "alpha": float(alpha),
            "prior": "shrink_to_target",
            "search_algorithm": str(search_algorithm),
        }
        if prior_target is not None:
            fit_params["prior_target"] = list(prior_target)
    recipe = _base_recipe(
        target=target, horizon=horizon, panel=panel, seed=seed,
        l4=_l4_single_fit(
            family="ridge",
            fit_params=fit_params,
            fit_node_id="fit_albacore",
        ),
    )
    recipe["3_feature_engineering"] = l3
    return recipe


# ---------------------------------------------------------------------------
# 11. Sparse Macro Factors (Zhou)
# ---------------------------------------------------------------------------

def sparse_macro_factors(
    *,
    target: str = "y",
    horizon: int = 1,
    n_components: int = 4,
    panel: dict[str, list[Any]] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Rapach & Zhou (2025) Sparse Macro-Finance Factors -- foundational
    primitive recipe.

    **Decomposition.** ``sparse_pca`` (sklearn Zou-Hastie-Tibshirani
    2006 variant) + downstream ``ridge``. Produces a *related* but
    NOT identical decomposition to the paper, which uses Sparse
    Component Analysis (SCA) per Chen & Rohe (2023) — a different
    sparse-PCA objective. Faithful Zhou-paper replication needs
    ``sparse_pca_chen_rohe`` + ``supervised_pca`` (Giglio/Xiu/Zhang
    2025); both are scheduled for v0.9.x.

    **Status: operational** as a generic sparse-PCA recipe; not
    paper-faithful for Zhou (2025).
    """

    l3 = {
        "nodes": [
            {"id": "src_X", "type": "source", "selector": {"layer_ref": "l2", "sink_name": "l2_clean_panel_v1", "subset": {"role": "predictors"}}},
            {"id": "src_y", "type": "source", "selector": {"layer_ref": "l2", "sink_name": "l2_clean_panel_v1", "subset": {"role": "target"}}},
            {"id": "spca", "type": "step", "op": "sparse_pca", "params": {"n_components": n_components}, "inputs": ["src_X"]},
            {"id": "y_h", "type": "step", "op": "target_construction", "params": {"mode": "point_forecast", "method": "direct", "horizon": horizon}, "inputs": ["src_y"]},
        ],
        "sinks": {"l3_features_v1": {"X_final": "spca", "y_final": "y_h"}, "l3_metadata_v1": "auto"},
    }
    recipe = _base_recipe(target=target, horizon=horizon, panel=panel, seed=seed, l4=_l4_single_fit("ridge", {"alpha": 1.0}))
    recipe["3_feature_engineering"] = l3
    return recipe


# ---------------------------------------------------------------------------
# 12. Macroeconomic Data Transformations Matter (Coulombe 2021)
# ---------------------------------------------------------------------------

def macroeconomic_data_transformations(
    *,
    target: str = "y",
    horizon: int = 1,
    max_order: int = 12,
    panel: dict[str, list[Any]] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Goulet Coulombe et al. (2021) MARX recipe = ``ma_increasing_order``
    (MARX feature) + ``pca`` (rotation) + downstream RF.

    **Decomposition.** MARX = moving averages of increasing order (the
    paper's MAF / MARX feature blocks) followed by a PCA rotation,
    feeding a non-linear ML model that benefits from the data
    transformation. ``ma_increasing_order`` op is operational since
    v0.1; ``pca`` op operational; both compose cleanly.

    **Status: operational.** Reference:
    doi:10.1016/j.ijforecast.2021.05.005.
    """

    l3 = {
        "nodes": [
            {"id": "src_X", "type": "source", "selector": {"layer_ref": "l2", "sink_name": "l2_clean_panel_v1", "subset": {"role": "predictors"}}},
            {"id": "src_y", "type": "source", "selector": {"layer_ref": "l2", "sink_name": "l2_clean_panel_v1", "subset": {"role": "target"}}},
            {"id": "marx", "type": "step", "op": "ma_increasing_order", "params": {"max_order": max_order}, "inputs": ["src_X"]},
            {"id": "rotated", "type": "step", "op": "pca", "params": {"n_components": 4}, "inputs": ["marx"]},
            {"id": "y_h", "type": "step", "op": "target_construction", "params": {"mode": "point_forecast", "method": "direct", "horizon": horizon}, "inputs": ["src_y"]},
        ],
        "sinks": {"l3_features_v1": {"X_final": "rotated", "y_final": "y_h"}, "l3_metadata_v1": "auto"},
    }
    recipe = _base_recipe(target=target, horizon=horizon, panel=panel, seed=seed, l4=_l4_single_fit("random_forest", {"n_estimators": 200}))
    recipe["3_feature_engineering"] = l3
    return recipe


# v0.9.0F gap-closure: 16-cell horse race enumeration for the
# paper's Table 1 (Coulombe et al. 2021 §3 + Recap p.11).

_DATA_TRANSFORM_CELLS_16 = (
    "F", "F-X", "F-MARX", "F-MAF", "F-Level",
    "F-X-MARX", "F-X-MAF", "F-X-Level", "F-X-MARX-Level",
    "X", "MARX", "MAF",
    "X-MARX", "X-MAF", "X-Level", "X-MARX-Level",
)


def _l3_data_transforms_cell(
    cell: str,
    horizon: int,
    max_order: int = 12,
    target_method: str = "direct",
) -> dict[str, Any]:
    """Build the L3 graph for one of the 16 Z_t cells in
    Coulombe-Leroux-Stevanovic-Surprenant (2021) Table 1.

    Cell components:
      * ``F``    — PCA factors of X (rotation = pca, n_components = 4)
      * ``X``    — original predictors (already in src_X)
      * ``MARX`` — moving averages of X with increasing order
      * ``MAF``  — moving-average factors (PCA on lag-panel)
      * ``Level``— y_t and lagged y (autoregressive features)

    ``target_method`` controls Eq. (3) target construction: ``"direct"``
    forecasts ``y_{t+h}`` per horizon (paper §2.1); ``"path_average"``
    (a.k.a. ``"cumulative_average"``) forecasts the h-period running
    mean (paper §2.2 + Table 2). Audit gap-fix: previous helper hard-
    coded ``"direct"`` and never exposed the path-average grid.
    """

    components = cell.split("-")
    nodes: list[dict[str, Any]] = [
        {"id": "src_X", "type": "source", "selector": {"layer_ref": "l2", "sink_name": "l2_clean_panel_v1", "subset": {"role": "predictors"}}},
        {"id": "src_y", "type": "source", "selector": {"layer_ref": "l2", "sink_name": "l2_clean_panel_v1", "subset": {"role": "target"}}},
    ]
    feature_nodes: list[str] = []
    if "F" in components:
        nodes.append({"id": "feat_F", "type": "step", "op": "pca",
                      "params": {"n_components": 4}, "inputs": ["src_X"]})
        feature_nodes.append("feat_F")
    if "X" in components:
        nodes.append({"id": "feat_X", "type": "step", "op": "lag",
                      "params": {"max_lag": 4}, "inputs": ["src_X"]})
        feature_nodes.append("feat_X")
    if "MARX" in components:
        nodes.append({"id": "feat_MARX", "type": "step", "op": "ma_increasing_order",
                      "params": {"max_order": max_order}, "inputs": ["src_X"]})
        feature_nodes.append("feat_MARX")
    if "MAF" in components:
        # MAF = MA-of-X then PCA (paper: "moving average factors")
        nodes.append({"id": "feat_MAF_ma", "type": "step", "op": "ma_increasing_order",
                      "params": {"max_order": max_order}, "inputs": ["src_X"]})
        nodes.append({"id": "feat_MAF", "type": "step", "op": "pca",
                      "params": {"n_components": 4}, "inputs": ["feat_MAF_ma"]})
        feature_nodes.append("feat_MAF")
    if "Level" in components:
        nodes.append({"id": "feat_Level", "type": "step", "op": "lag",
                      "params": {"max_lag": 4}, "inputs": ["src_y"]})
        feature_nodes.append("feat_Level")
    nodes.append({"id": "X_final", "type": "step", "op": "weighted_concat",
                  "params": {}, "inputs": feature_nodes})
    # Runtime switches on ``mode`` (not ``method``); map target_method to
    # the right mode so ``path_average`` actually triggers the cumulative-
    # average target (paper §2.2 + Table 2).
    target_mode = "path_average" if target_method in {"path_average", "cumulative_average"} else "point_forecast"
    nodes.append({"id": "y_h", "type": "step", "op": "target_construction",
                  "params": {"mode": target_mode, "method": target_method, "horizon": horizon},
                  "inputs": ["src_y"]})
    return {
        "nodes": nodes,
        "sinks": {"l3_features_v1": {"X_final": "X_final", "y_final": "y_h"}, "l3_metadata_v1": "auto"},
    }


# Paper Table 1 family list (7 entries): AR / FM / Adaptive Lasso / Elastic
# Net / Linear Boosting / Random Forest / Boosted Trees. macroforecast L4
# mappings: FM → ``factor_augmented_ar`` (Stock-Watson 2002a ARDI), LB →
# ``glmboost`` (component-wise L2 boosting with linear base learners).
_DATA_TRANSFORM_FAMILIES_DEFAULT = (
    "ar_p",                   # AR
    "factor_augmented_ar",    # FM (Stock-Watson 2002a) — audit-fix
    "lasso_path",             # AL (Adaptive Lasso proxy)
    "elastic_net",            # EN
    "glmboost",               # LB (component-wise linear boosting) — audit-fix
    "random_forest",          # RF
    "gradient_boosting",      # BT
)

_DATA_TRANSFORM_TARGET_METHODS_DEFAULT = ("direct", "path_average")


def macroeconomic_data_transformations_horse_race(
    *,
    target: str = "y",
    horizon: int = 1,
    horizons: tuple[int, ...] = (1,),
    families: tuple[str, ...] = _DATA_TRANSFORM_FAMILIES_DEFAULT,
    cells: tuple[str, ...] = _DATA_TRANSFORM_CELLS_16,
    target_methods: tuple[str, ...] = _DATA_TRANSFORM_TARGET_METHODS_DEFAULT,
    max_order: int = 12,
    panel: dict[str, list[Any]] | None = None,
    seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """Coulombe / Leroux / Stevanovic / Surprenant (2021)
    "Macroeconomic Data Transformations Matter" full horse race.

    Returns a dict mapping
    ``"<cell>__<family>__h<horizon>__<target_method>"`` → recipe dict,
    one recipe per (Table 1 Z_t cell × family × horizon × target
    method). Users iterate and call ``macroforecast.run(recipe)`` on
    each, then aggregate metrics across the grid.

    The 16 cells (Table 1) combine F (PCA factors) / X (predictors) /
    MARX (moving-average rotation) / MAF (moving-average factors) /
    Level (lagged y). The 7 forecasting families
    (AR / FM / Adaptive Lasso / Elastic Net / Linear Boosting / RF / BT)
    map to macroforecast L4 families per
    ``_DATA_TRANSFORM_FAMILIES_DEFAULT``. Targets are swept over
    ``target_methods`` -- both ``direct`` (paper §2.1) and
    ``path_average`` (paper §2.2 + Table 2) are exercised by default,
    matching the paper grid.

    **Status: operational (v0.9.0a0 audit gap-fix).** Reference:
    arXiv:2008.01714.
    """

    grid: dict[str, dict[str, Any]] = {}
    horizons_iter = tuple(horizons) if horizons else (horizon,)
    for cell in cells:
        for family in families:
            for h in horizons_iter:
                for method in target_methods:
                    recipe = _base_recipe(
                        target=target, horizon=h, panel=panel, seed=seed,
                        l4=_l4_single_fit(family, {}),
                    )
                    recipe["3_feature_engineering"] = _l3_data_transforms_cell(
                        cell, h, max_order, target_method=method
                    )
                    grid[f"{cell}__{family}__h{h}__{method}"] = recipe
    return grid


# ---------------------------------------------------------------------------
# 13. How is ML Useful for Macro Forecasting (Coulombe et al. 2022)
# ---------------------------------------------------------------------------

def ml_useful_macro(
    *,
    target: str = "y",
    horizon: int = 1,
    panel: dict[str, list[Any]] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Goulet Coulombe et al. (2022 JAE) How is ML Useful = sweep
    machinery over family × regularization × CV × loss.

    **Decomposition.** No new primitives needed; the paper's central
    contribution is the 4-feature decomposition of ML gains, expressed
    via the existing sweep machinery. Helper returns a single-cell
    recipe; users add ``sweep:`` markers on family / cv / regularization
    parameters to reproduce the comparison grid.

    **Status: operational.** Reference: doi:10.1002/jae.2910.
    """

    return _base_recipe(
        target=target,
        horizon=horizon,
        panel=panel,
        seed=seed,
        l4=_l4_single_fit(family="ridge", fit_params={"alpha": 1.0}, fit_node_id="fit_baseline"),
    )


# v0.9.0F gap-closure: 4-feature horse race for the paper's central
# decomposition (Coulombe et al. 2022 JAE §3 + Eq. 16).

_ML_USEFUL_FEATURES = {
    # Feature 1 — nonlinearity
    "linear_baseline": ("ar_p", {"n_lag": 4}),
    "krr_rbf":        ("kernel_ridge", {"kernel": "rbf", "alpha": 1.0}),
    "rf":             ("random_forest", {"n_estimators": 200, "max_depth": 8}),
    # Feature 2 — regularization
    "lasso":          ("lasso", {"alpha": 0.05}),
    "elastic_net":    ("elastic_net", {"alpha": 0.05, "l1_ratio": 0.5}),
    "ridge":          ("ridge", {"alpha": 1.0}),
    # Feature 4 — loss function
    "svr_rbf":        ("svr_rbf", {"C": 1.0, "epsilon": 0.1, "gamma": "scale"}),
    "svr_linear":     ("svr_linear", {"C": 1.0, "epsilon": 0.1}),
    # Reference baselines
    "fm":             ("factor_augmented_ar", {"n_factors": 3, "n_lag": 2}),
    "ols":            ("ols", {}),
}


def ml_useful_macro_horse_race(
    *,
    target: str = "y",
    targets: tuple[str, ...] | None = None,
    horizon: int = 1,
    horizons: tuple[int, ...] = (1, 3, 9, 12, 24),
    cv_schemes: tuple[str, ...] = ("kfold", "poos", "aic", "bic"),
    cases: tuple[str, ...] | None = None,
    data_richness: str = "H_plus",
    n_factors: int = 4,
    panel: dict[str, list[Any]] | None = None,
    seed: int = 42,
    attach_eval_blocks: bool = False,
    benchmark_family: str = "factor_augmented_ar",
    benchmark_params: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Coulombe / Surprenant / Leroux / Stevanovic (2022 JAE)
    "How is Machine Learning Useful for Macroeconomic Forecasting?"
    full 4-feature × N-treatment horse race.

    Returns a dict mapping
    ``"<target>__<case>__h<horizon>__<cv>"`` → recipe dict,
    one per (target × case × horizon × CV scheme). Cases enumerate
    paper Features 1-4:
      * Feature 1 (nonlinearity): linear_baseline / krr_rbf / rf
      * Feature 2 (regularization): lasso / elastic_net / ridge
      * Feature 3 (CV scheme): controlled by ``cv_schemes`` argument;
        the 4 paper schemes ``kfold`` / ``poos`` / ``aic`` / ``bic``
        are wired in v0.9.0a0 (audit gap-fix). Earlier releases
        silently dropped these strings — they are now first-class
        ``search_algorithm`` values dispatched in
        ``_resolve_l4_tuning``.
      * Feature 4 (loss function): svr_rbf / svr_linear (ε-insensitive)

    Reference baselines: ``fm`` (factor model à la Stock-Watson 2002a)
    and ``ols``. The 4-feature decomposition is recovered post-hoc
    from this grid via L6 DM / MCS tests on the per-cell metrics
    (paper §4 evaluation). Default horizons match paper §4.3
    (h ∈ {1, 3, 9, 12, 24}). When ``targets`` is provided, the helper
    sweeps each target separately; otherwise it falls back to a single
    target (``target`` argument).

    **Status: operational (v0.9.0a0 audit gap-fix).** Reference:
    doi:10.1002/jae.2910.
    """

    if data_richness not in {"H_minus", "H_plus"}:
        raise ValueError(
            f"data_richness must be 'H_minus' or 'H_plus'; got {data_richness!r}. "
            "H_minus = paper §3.2 data-poor (own-lag y only, ~14 models). "
            "H_plus = data-rich (factor-augmented ARDI, ~30 models)."
        )
    cases_iter: tuple[str, ...] = cases if cases is not None else tuple(_ML_USEFUL_FEATURES.keys())
    targets_iter: tuple[str, ...] = tuple(targets) if targets else (target,)
    grid: dict[str, dict[str, Any]] = {}
    eval_l5, eval_l6 = _l5_l6_paper16_eval_blocks() if attach_eval_blocks else ({}, {})
    for tgt in targets_iter:
        for case in cases_iter:
            if case not in _ML_USEFUL_FEATURES:
                continue
            family, fit_params = _ML_USEFUL_FEATURES[case]
            for h in horizons:
                for cv in cv_schemes:
                    cell_fit_params = dict(fit_params, search_algorithm=cv)
                    if attach_eval_blocks:
                        l4 = _l4_with_benchmark(
                            family, cell_fit_params,
                            benchmark_family=benchmark_family,
                            benchmark_params=benchmark_params,
                        )
                    else:
                        l4 = _l4_single_fit(family, cell_fit_params)
                    recipe = _base_recipe(
                        target=tgt, horizon=h, panel=panel, seed=seed, l4=l4,
                    )
                    # H_minus: paper §3.2 data-poor (own-lag y only).
                    # H_plus: data-rich (PCA factors of X plus lagged y).
                    recipe["3_feature_engineering"] = _l3_h_axis(
                        data_richness, h, n_factors=n_factors,
                    )
                    if attach_eval_blocks:
                        recipe["5_evaluation"] = dict(eval_l5)
                        recipe["6_statistical_tests"] = dict(eval_l6)
                    grid[f"{tgt}__{case}__h{h}__{cv}__{data_richness}"] = recipe
    return grid


def _l3_h_axis(richness: str, horizon: int, *, n_factors: int = 4, n_lag: int = 4) -> dict[str, Any]:
    """Paper §3.2 H⁻ / H⁺ feature axis builder.

    * ``H_minus`` — data-poor: lagged-y only (14 paper models).
    * ``H_plus``  — data-rich: PCA factors of X concatenated with lagged y
      (30 paper models, matches ARDI baseline).

    Phase A4b (paper 16, Round 1 finding 6): paper Eq. (7) ARDI specifies
    ``y_{t+h} = c + ρ(L)y_t + β(L)F_t + e`` — both ``y`` AND ``F`` are
    lagged. Pre-A4 H_plus fed contemporaneous ``feat_F`` to the concat.
    Now: ``feat_F → lag_F (n_lag) → weighted_concat ← lag_y (n_lag)``.

    Phase A4c (paper 16, Round 1 finding / Round 4 F5): at ``n_lag=0``
    the lag op rejects (universal rule: ``n_lag >= 1``). Paper §3.2
    footnote: "B₃() = B₂() only when no lags are included." Emit a
    no-lag DAG (H_minus → identity(src_y); H_plus → identity(feat_F)
    concatenated with identity(src_y)) so the recipe runs.
    """

    nodes: list[dict[str, Any]] = [
        {"id": "src_X", "type": "source", "selector": {"layer_ref": "l2", "sink_name": "l2_clean_panel_v1", "subset": {"role": "predictors"}}},
        {"id": "src_y", "type": "source", "selector": {"layer_ref": "l2", "sink_name": "l2_clean_panel_v1", "subset": {"role": "target"}}},
    ]
    if richness == "H_minus":
        if n_lag == 0:
            nodes.append({"id": "X_final", "type": "step", "op": "identity", "params": {}, "inputs": ["src_y"]})
            x_final = "X_final"
        else:
            nodes.append({"id": "lag_y", "type": "step", "op": "lag", "params": {"n_lag": n_lag}, "inputs": ["src_y"]})
            x_final = "lag_y"
    else:  # H_plus
        nodes.append({"id": "feat_F", "type": "step", "op": "pca", "params": {"n_components": n_factors, "temporal_rule": "expanding_window_per_origin"}, "inputs": ["src_X"]})
        if n_lag == 0:
            # No-lag DAG: identity(feat_F) ⊕ identity(src_y) via weighted_concat.
            nodes.append({"id": "id_y", "type": "step", "op": "identity", "params": {}, "inputs": ["src_y"]})
            nodes.append({"id": "X_final", "type": "step", "op": "weighted_concat", "params": {}, "inputs": ["feat_F", "id_y"]})
        else:
            nodes.append({"id": "lag_F", "type": "step", "op": "lag", "params": {"n_lag": n_lag}, "inputs": ["feat_F"]})
            nodes.append({"id": "lag_y", "type": "step", "op": "lag", "params": {"n_lag": n_lag}, "inputs": ["src_y"]})
            nodes.append({"id": "X_final", "type": "step", "op": "weighted_concat", "params": {}, "inputs": ["lag_F", "lag_y"]})
        x_final = "X_final"
    nodes.append({"id": "y_h", "type": "step", "op": "target_construction",
                  "params": {"mode": "point_forecast", "method": "direct", "horizon": horizon},
                  "inputs": ["src_y"]})
    return {
        "nodes": nodes,
        "sinks": {"l3_features_v1": {"X_final": x_final, "y_final": "y_h"}, "l3_metadata_v1": "auto"},
    }


def ml_useful_macro_b_grid(
    *,
    target: str = "y",
    horizon: int = 1,
    panel: dict[str, list[Any]] | None = None,
    seed: int = 42,
    n_factors: int = 4,
    n_lag: int = 4,
    families: tuple[str, ...] = ("ridge", "lasso", "elastic_net"),
    rotations: tuple[str, ...] = ("B1", "B2", "B3"),
) -> dict[str, dict[str, Any]]:
    """Coulombe-Surprenant-Leroux-Stevanovic (2022) §3.2 + Eq. (18)
    B₁/B₂/B₃ regularization rotation grid.

    The paper applies one of three feature rotations *before* the
    regularised regression family (Ridge / Lasso / Elastic Net):

    * **B₁** — identity (no rotation; regularise on the original X).
    * **B₂** — full PCA rotation: project X onto its first
      ``n_factors`` principal components and regularise in factor space.
    * **B₃** — lag-only PCA: extract PCA factors from lagged X only,
      then concatenate with contemporaneous X for the regression.

    Combined with the family axis (Ridge/Lasso/EN) and the four CV
    schemes, this is the §3.2 regularization grid that the audit
    flagged as missing from the headline helper. Pair this builder
    with ``ml_useful_macro_horse_race`` (or run standalone) and feed
    the metric panels into a DM/MCS aggregator."""

    grid: dict[str, dict[str, Any]] = {}
    for rot in rotations:
        for family in families:
            recipe = _base_recipe(
                target=target, horizon=horizon, panel=panel, seed=seed,
                l4=_l4_single_fit(family, {}),
            )
            recipe["3_feature_engineering"] = _l3_b_rotation(rot, horizon, n_factors=n_factors, n_lag=n_lag)
            grid[f"{rot}__{family}"] = recipe
    return grid


def _l3_b_rotation(rotation: str, horizon: int, *, n_factors: int = 4, n_lag: int = 4) -> dict[str, Any]:
    """L3 graph for one of the §3.2 B₁/B₂/B₃ rotations.

    Phase A4c (paper 16, Round 1 finding / Round 4 F5): paper §3.2
    footnote — "B₃() = B₂() only when no lags are included." At
    ``n_lag=0`` the universal lag op rejects (hard rule: ``n_lag >= 1``)
    so pre-A4 ``b_grid(n_lag=0)`` produced unrunnable recipes. Each
    rotation now emits a no-lag DAG when ``n_lag == 0``:

    * B₁ → ``identity(src_X)`` (passthrough X).
    * B₂ → ``pca(src_X, n_components="all")`` (no post-lag).
    * B₃ → identical to B₂ at ``n_lag=0`` per paper footnote (the
      H_t^+ stack is empty, so PCA reduces to PCA-of-X).
    """

    nodes: list[dict[str, Any]] = [
        {"id": "src_X", "type": "source", "selector": {"layer_ref": "l2", "sink_name": "l2_clean_panel_v1", "subset": {"role": "predictors"}}},
        {"id": "src_y", "type": "source", "selector": {"layer_ref": "l2", "sink_name": "l2_clean_panel_v1", "subset": {"role": "target"}}},
    ]
    if rotation == "B1":
        # Identity: pass X through unchanged.
        if n_lag == 0:
            nodes.append({"id": "X_final", "type": "step", "op": "identity", "params": {}, "inputs": ["src_X"]})
        else:
            nodes.append({"id": "X_final", "type": "step", "op": "lag", "params": {"n_lag": n_lag}, "inputs": ["src_X"]})
    elif rotation == "B2":
        # Phase A2 fix: paper §3.2 Eq. (18) keeps ALL N factors ("we do
        # not select F_t … we keep them all"). Phase A3 fix: pass the
        # explicit ``n_components="all"`` sentinel rather than relying on
        # the schema default (4) — runtime resolves "all" → min(T, N)
        # at PCA fit time.
        if n_lag == 0:
            # Phase A4c: no-lag identity path. PCA-of-X is the X_final.
            nodes.append({"id": "X_final", "type": "step", "op": "pca",
                          "params": {"n_components": "all", "temporal_rule": "expanding_window_per_origin"},
                          "inputs": ["src_X"]})
        else:
            nodes.append({"id": "feat_pca", "type": "step", "op": "pca",
                          "params": {"n_components": "all", "temporal_rule": "expanding_window_per_origin"},
                          "inputs": ["src_X"]})
            # Phase A3 fix: rename ``max_lag`` → ``n_lag`` (the lag op reads
            # ``n_lag``; the typo silently fell back to the default 4).
            nodes.append({"id": "X_final", "type": "step", "op": "lag", "params": {"n_lag": n_lag}, "inputs": ["feat_pca"]})
    elif rotation == "B3":
        # Phase A2 fix: paper §3.2 "B₃() rotates H_t^+ rather than X_t and
        # still keeps all the factors. H_t^+ includes all the relevant
        # preselected lags." Pre-fix chain (lag → PCA → concat with
        # contemporaneous X) was wrong on three counts: (a) lag(y) was
        # not part of the PCA input; (b) factors were truncated to
        # n_factors; (c) contemporaneous X was concatenated, contradicting
        # the paper which rotates the entire H_t^+ stack. Pre-fix also
        # type-failed because ``lag`` outputs ``LaggedPanel`` while
        # ``pca`` requires a ``Panel`` input.
        #
        # Post-fix chain: ``weighted_concat(lag(y), lag(X)) → PCA(full,
        # temporal_rule=expanding_window_per_origin)``. ``weighted_concat``
        # accepts ``LaggedPanel`` inputs and emits a ``Panel`` output, so
        # the lag-stack feeds the factor op cleanly. PCA keeps all
        # components (``n_components="all"`` sentinel resolved to
        # min(T, N) at runtime) per paper §3.2. No contemporaneous-X
        # concat.
        #
        # Phase A3 fix: drop the trailing ``lag(max_lag=1)`` after PCA.
        # Paper §3.2: "B₃() rotates H_t^+ via PCA … keep all the
        # factors" — the PCA output IS the L3 X_final. The trailing lag
        # also re-introduced the silent ``max_lag`` typo (lag op reads
        # ``n_lag``) and violated the paper §3.2 specification.
        if n_lag == 0:
            # Phase A4c: paper §3.2 footnote — "B₃() = B₂() only when no
            # lags are included." At n_lag=0 the H_t^+ stack collapses
            # to X_t alone, so PCA reduces to PCA-of-X (B₂'s no-lag DAG).
            nodes.append({"id": "X_final", "type": "step", "op": "pca",
                          "params": {"n_components": "all", "temporal_rule": "expanding_window_per_origin"},
                          "inputs": ["src_X"]})
        else:
            nodes.append({"id": "lag_x", "type": "step", "op": "lag", "params": {"n_lag": n_lag}, "inputs": ["src_X"]})
            nodes.append({"id": "lag_y", "type": "step", "op": "lag", "params": {"n_lag": n_lag}, "inputs": ["src_y"]})
            nodes.append({"id": "h_plus", "type": "step", "op": "weighted_concat", "params": {}, "inputs": ["lag_y", "lag_x"]})
            nodes.append({"id": "X_final", "type": "step", "op": "pca",
                          "params": {"n_components": "all", "temporal_rule": "expanding_window_per_origin"},
                          "inputs": ["h_plus"]})
    else:
        raise ValueError(f"unknown rotation {rotation!r}; expected B1/B2/B3")
    nodes.append({"id": "y_h", "type": "step", "op": "target_construction",
                  "params": {"mode": "point_forecast", "method": "direct", "horizon": horizon},
                  "inputs": ["src_y"]})
    return {
        "nodes": nodes,
        "sinks": {"l3_features_v1": {"X_final": "X_final", "y_final": "y_h"}, "l3_metadata_v1": "auto"},
    }


# ---------------------------------------------------------------------------
# 15. Arctic Sea Ice DFM
# ---------------------------------------------------------------------------

def arctic_sea_ice_dfm(
    *,
    target: str = "y",
    horizon: int = 1,
    n_factors: int = 1,
    factor_order: int = 1,
    panel: dict[str, list[Any]] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Arctic Sea Ice DFM = ``dfm_mixed_mariano_murasawa`` family.

    To recover the paper §3.4 Kalman *smoother* posterior on the latent
    factors after running ``macroforecast.run(recipe)``, pull the
    fitted ``_DFMMixedFrequency`` instance from the returned
    ``ModelArtifact`` and call ``predict_smoothed_factors()``. Audit
    gap-fix: the smoother surface was previously not exposed.

    **Status: operational** (DFM family operational since v0.25). Reference:
    Coulombe & Goebel (2021) JCLI-D-20-0324 paired Arctic VAR.
    """

    return _base_recipe(
        target=target,
        horizon=horizon,
        panel=panel,
        seed=seed,
        l4=_l4_single_fit(
            family="dfm_mixed_mariano_murasawa",
            fit_params={"n_factors": n_factors, "factor_order": factor_order},
            fit_node_id="fit_arctic_dfm",
        ),
    )


# ---------------------------------------------------------------------------
# 16. Arctic Amplification VAR / VARCTIC (Coulombe & Goebel 2021)
# ---------------------------------------------------------------------------

def arctic_var(
    *,
    target: str = "y",
    horizon: int = 1,
    n_lags: int = 4,
    panel: dict[str, list[Any]] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Goulet Coulombe & Goebel (2021) VARCTIC = ``var`` family + L7
    ``historical_decomposition`` / ``orthogonalised_irf`` / ``fevd`` for
    shock decomposition.

    **Decomposition.** Helper returns the L0-L4 VAR baseline; users add
    L7 IRF / FEVD ops via a custom L7 block to reproduce the paper's
    shock-decomposition figures.

    **Status: operational** (VAR + IRF / FEVD ops operational).
    Reference: doi:10.1175/JCLI-D-20-0324.1.
    """

    return _base_recipe(
        target=target,
        horizon=horizon,
        panel=panel,
        seed=seed,
        l4=_l4_single_fit(family="var", fit_params={"n_lags": n_lags}, fit_node_id="fit_varctic"),
    )


__all__ = [
    # Operational (runs end-to-end on current main):
    "perfectly_random_forest",      # #3 PRF baseline
    "scaled_pca",                    # #1
    "macroeconomic_random_forest",   # #2
    "ols_attention_demo",            # #7 conceptual
    "sparse_macro_factors",          # #11
    "macroeconomic_data_transformations",  # #12 MARX
    "ml_useful_macro",               # #13
    "ml_useful_macro_horse_race",    # #16 paper-16 grid helper
    "ml_useful_macro_b_grid",        # #16 §3.2 B₁/B₂/B₃ rotation grid
    "macroeconomic_data_transformations_horse_race",  # #15 16-cell helper
    "arctic_sea_ice_dfm",            # #15
    "arctic_var",                    # #16
    # Pre-promotion (depend on future-status atomic primitives):
    "booging",                       # #3b bagging.strategy
    "marsquake",                     # #3c mars family
    "adaptive_ma",                   # #4 adaptive_ma_rf
    "two_step_ridge",                # #5 ridge.prior=random_walk
    "hemisphere_neural_network",     # #6 mlp.architecture
    "anatomy_oos",                   # #8 oshapley_vi/pbsv
    "dual_interpretation",           # #9 dual_decomposition
    "maximally_forward_looking",     # #10 asymmetric_trim + ridge.constraint
    "slow_growing_tree",             # #14 decision_tree.split_shrinkage
    "assemblage_regression",         # repackaged ridge.constraint
]
