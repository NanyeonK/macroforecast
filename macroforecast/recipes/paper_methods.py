"""Paper-method recipe builders (v0.9 Phase 2 paper-coverage pass).

One helper per paper in the 17-paper macro-forecasting target list. Each
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
        "2018-01-01",
        "2018-02-01",
        "2018-03-01",
        "2018-04-01",
        "2018-05-01",
        "2018-06-01",
        "2018-07-01",
        "2018-08-01",
        "2018-09-01",
        "2018-10-01",
        "2018-11-01",
        "2018-12-01",
    ],
    "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
    "x1": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
}


def _l1_minimal(
    target: str, horizon: int, panel: dict[str, list[Any]] | None
) -> dict[str, Any]:
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
            {
                "id": "src_X",
                "type": "source",
                "selector": {
                    "layer_ref": "l2",
                    "sink_name": "l2_clean_panel_v1",
                    "subset": {"role": "predictors"},
                },
            },
            {
                "id": "src_y",
                "type": "source",
                "selector": {
                    "layer_ref": "l2",
                    "sink_name": "l2_clean_panel_v1",
                    "subset": {"role": "target"},
                },
            },
            {
                "id": "lag_x",
                "type": "step",
                "op": "lag",
                "params": {"n_lag": 1},
                "inputs": ["src_X"],
            },
            {
                "id": "y_h",
                "type": "step",
                "op": "target_construction",
                "params": {
                    "mode": "point_forecast",
                    "method": "direct",
                    "horizon": horizon,
                },
                "inputs": ["src_y"],
            },
        ],
        "sinks": {
            "l3_features_v1": {"X_final": "lag_x", "y_final": "y_h"},
            "l3_metadata_v1": "auto",
        },
    }


def _l4_single_fit(
    family: str, fit_params: dict[str, Any], fit_node_id: str = "fit"
) -> dict[str, Any]:
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
            {
                "id": "src_X",
                "type": "source",
                "selector": {
                    "layer_ref": "l3",
                    "sink_name": "l3_features_v1",
                    "subset": {"component": "X_final"},
                },
            },
            {
                "id": "src_y",
                "type": "source",
                "selector": {
                    "layer_ref": "l3",
                    "sink_name": "l3_features_v1",
                    "subset": {"component": "y_final"},
                },
            },
            fit,
            {
                "id": "predict",
                "type": "step",
                "op": "predict",
                "inputs": [fit_node_id, "src_X"],
            },
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
            {
                "id": "src_X",
                "type": "source",
                "selector": {
                    "layer_ref": "l3",
                    "sink_name": "l3_features_v1",
                    "subset": {"component": "X_final"},
                },
            },
            {
                "id": "src_y",
                "type": "source",
                "selector": {
                    "layer_ref": "l3",
                    "sink_name": "l3_features_v1",
                    "subset": {"component": "y_final"},
                },
            },
            {
                "id": "fit_benchmark",
                "type": "step",
                "op": "fit_model",
                "params": bench_params,
                "is_benchmark": True,
                "inputs": ["src_X", "src_y"],
            },
            {
                "id": "predict_benchmark",
                "type": "step",
                "op": "predict",
                "inputs": ["fit_benchmark", "src_X"],
            },
            {
                "id": "fit_cell",
                "type": "step",
                "op": "fit_model",
                "params": cell_params,
                "inputs": ["src_X", "src_y"],
            },
            {
                "id": "predict_cell",
                "type": "step",
                "op": "predict",
                "inputs": ["fit_cell", "src_X"],
            },
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

    l5: dict[str, Any] = {"fixed_axes": {}}
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
            "fixed_axes": {
                "failure_policy": "fail_fast",
                "reproducibility_mode": "seeded_reproducible",
            },
            "leaf_config": {"random_seed": seed},
        },
        "1_data": _l1_minimal(target, horizon, panel),
        "2_preprocessing": {
            "fixed_axes": {
                "transform_policy": "no_transform",
                "outlier_policy": "none",
                "imputation_policy": "none_propagate",
                "frame_edge_policy": "keep_unbalanced",
            },
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
    B: int = 100,
    inner_n_estimators: int = 1500,
    inner_learning_rate: float = 0.1,
    inner_max_depth: int = 3,
    inner_subsample: float = 0.5,
    sample_frac: float = 0.75,
    da_noise_frac: float = 1.0 / 3.0,
    da_drop_rate: float = 0.2,
    panel: dict[str, list[Any]] | None = None,
    seed: int = 42,
    n_iterations: int | None = None,
) -> dict[str, Any]:
    """Goulet Coulombe (2024) "To Bag is to Prune" -- Booging algorithm.

    **Decomposition.** Booging = outer ``B``-bagging of *intentionally
    over-fitted* inner Stochastic Gradient Boosted trees, with
    per-bag Data Augmentation. Each outer bag (i) draws a row
    sub-sample of size ``sample_frac · n`` *without replacement*,
    (ii) appends a noisy column copy ``X̃_k = X_k + N(0, (σ_k ·
    da_noise_frac)²)`` to the design (doubling the width to ``2K``),
    (iii) drops a random ``da_drop_rate`` fraction of the augmented
    columns, and (iv) fits one ``GradientBoostingRegressor`` at fixed
    high ``inner_n_estimators`` (over-fit regime, *not* CV-tuned).
    Final forecast is the mean over the ``B`` per-bag predictions.

    The bag-prune theorem (paper §2.4) replaces tuning the boosting
    depth ``S`` with outer bagging: fix ``S`` to a high (over-fitting)
    value and let the bag average prune. This is the inverse of the
    standard "tune via CV" recipe.

    **Status: operational.** Recipe dispatches to ``_BoogingWrapper``
    via the ``bagging.strategy = "booging"`` sub-axis (see
    ``runtime.py:1958``).

    Paper-faithful defaults (Appendix A.2 p.39):
    ``B = 100``, ``inner_n_estimators = 1500``,
    ``inner_learning_rate = 0.1``, ``inner_max_depth = 3``
    (paper §4.1 p.25), ``inner_subsample = 0.5``,
    ``sample_frac = 0.75``, ``da_noise_frac = 1/3``,
    ``da_drop_rate = 0.2``.

    The legacy ``n_iterations`` kwarg is retained as a deprecated alias
    for ``B`` and emits ``DeprecationWarning`` when used.

    Reference: Goulet Coulombe (2024) "To Bag is to Prune"
    arXiv:2008.07063 §2.4 + Appendix A.2 p.39.
    """

    if n_iterations is not None:
        warnings.warn(
            "`n_iterations` is deprecated; use `B` (the outer bagging count). "
            "Will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        B = int(n_iterations)

    return _base_recipe(
        target=target,
        horizon=horizon,
        panel=panel,
        seed=seed,
        l4=_l4_single_fit(
            family="bagging",
            fit_params={
                "strategy": "booging",
                "n_estimators": int(B),
                "inner_n_estimators": int(inner_n_estimators),
                "inner_learning_rate": float(inner_learning_rate),
                "inner_max_depth": int(inner_max_depth),
                "inner_subsample": float(inner_subsample),
                "max_samples": float(sample_frac),
                "da_noise_frac": float(da_noise_frac),
                "da_drop_rate": float(da_drop_rate),
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
    eta_depth_step: float = 0.01,
    eta_max_plateau: float = 0.5,
    mtry_frac: float = 0.75,
    max_depth: int | None = None,
    panel: dict[str, list[Any]] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Goulet Coulombe (2024) Slow-Growing Tree =
    ``decision_tree(split_shrinkage=η)``.

    Phase B-3 audit-fix (paper p.87-88 rule-of-thumb defaults):

    * ``eta_depth_step`` default raised from ``0.0`` to ``0.01``. Paper
      p.87 specifies "starting at η=0.1 and increasing it by 0.01 with
      depth, until an imposed plateau of 0.5". The previous default
      silently disabled the depth-step rule.
    * ``eta_max_plateau`` (paper p.87) surfaced as a first-class helper
      arg with default ``0.5`` (the paper's "imposed plateau").
    * ``mtry_frac`` (paper p.88 §2.3 "mtry = 0.75 is used throughout")
      surfaced as a first-class helper arg with default ``0.75``.

    Other gap-fixes retained: ``herfindahl_threshold`` (H̄, default 0.25
    per paper) and ``max_depth`` (safety bound, default None) remain
    first-class helper args. If you need η=0 (which routes to plain
    sklearn ``DecisionTreeRegressor`` and bypasses the SGT mechanism)
    the helper emits a ``UserWarning``; pick a non-zero η or a recipe
    from :func:`slow_growing_tree_grid` instead.

    **Status: pre-promotion** -- ``decision_tree.split_shrinkage`` is
    future. Reference: Goulet Coulombe (2024)
    doi:10.1007/978-3-031-43601-7_4.
    """

    if float(split_shrinkage) == 0.0:
        import warnings

        warnings.warn(
            "split_shrinkage=0.0 disables the SGT mechanism; routes to "
            "sklearn DecisionTreeRegressor (CART). The slow_growing_tree "
            "helper expects eta > 0; use eta in (0, 1] or pick a recipe "
            "in slow_growing_tree_grid().",
            UserWarning,
            stacklevel=2,
        )

    fit_params: dict[str, Any] = {
        "split_shrinkage": split_shrinkage,
        "herfindahl_threshold": herfindahl_threshold,
        "eta_depth_step": eta_depth_step,
        "eta_max_plateau": eta_max_plateau,
        "mtry_frac": mtry_frac,
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
    panel: dict[str, list[Any]] | None = None,
    seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """Slow-Growing Tree paper-faithful grid: the three (η, H̄) recipe
    lines from Goulet Coulombe (2024) §3 p.90 (Figure 2 SGT row).

    Phase B-3 audit-fix: replaces the prior 4×3 Cartesian product
    (which both omitted the paper's η=0.01 line and silently routed
    η=0 to sklearn ``DecisionTreeRegressor``) with the exact 3-line
    set from the paper:

    * ``(η = 0.5,  H̄ = 0.25)`` -- higher learning rate, mid early-stopping
    * ``(η = 0.1,  H̄ = 0.25)``
    * ``(η = 0.01, H̄ = 0.05)`` -- low learning rate + tighter early-stopping

    Each cell uses the paper p.87-88 rule-of-thumb defaults
    (``eta_depth_step=0.01``, ``eta_max_plateau=0.5``, ``mtry_frac=0.75``).
    Returns a dict keyed by ``"eta<η>__h<H̄>"`` so users can run each
    cell and aggregate via L5 / L6.

    Reference: Goulet Coulombe (2024) §3 p.90, Figure 2 (Slow-Growing
    Tree row).
    """

    paper_grid = (
        (0.5, 0.25),
        (0.1, 0.25),
        (0.01, 0.05),
    )
    grid: dict[str, dict[str, Any]] = {}
    for eta, h_bar in paper_grid:
        recipe = slow_growing_tree(
            target=target,
            horizon=horizon,
            split_shrinkage=eta,
            herfindahl_threshold=h_bar,
            eta_depth_step=0.01,
            eta_max_plateau=0.5,
            mtry_frac=0.75,
            panel=panel,
            seed=seed,
        )
        grid[f"eta{eta}__h{h_bar}"] = recipe
    return grid


def two_step_ridge(
    *,
    target: str = "y",
    horizon: int = 1,
    alpha_step2: float = 0.1,
    vol_model: str = "garch11",
    alpha_strategy: str = "second_cv",
    alpha_grid: list[float] | None = None,
    cv_folds: int = 5,
    panel: dict[str, list[Any]] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Goulet Coulombe (2025 IJF) 2SRR = ``ridge(prior=random_walk)``.

    Implements paper §2.5 Algorithm 1: warm-start ridge on a homogeneous
    Ω, recover heterogeneous Ω_θ / Ω_ε from the step-1 residuals + θ̂_1
    sample variance, then **rerun CV** at step 4 to pick the final λ
    before a closed-form heterogeneous-Ω solve. The second CV is the
    paper's "crucial" step (§2.5 footnote 4 + §2.4.1): heterogeneous
    variance changes the effective regularization, so the warm-start λ
    is no longer the optimum. Default ``alpha_strategy="second_cv"``
    enables the K-fold CV; set to ``"fixed"`` to bypass it.

    Phase B-8 audit-fix:

    * Dropped ``alpha_step1`` (paper-faithful 2SRR uses ONE λ across
      step 1 + step 2, picked by the second CV; the previous wired
      ``fit_step1`` ridge node was unused dead weight).
    * Surfaced ``alpha_strategy`` (``"second_cv"`` default,
      ``"fixed"``), ``alpha_grid`` (default
      ``[0.01, 0.1, 1.0, 10.0, 100.0]``), and ``cv_folds`` (default
      ``5``) as first-class kwargs for paper §2.5 step 4.
    * ``alpha_step2`` retained as the fallback / fixed-strategy λ.

    ``vol_model`` (default ``"garch11"``) controls the residual-variance
    estimator used in step 2's heterogeneous-Ω solve (paper §4 Eq. 11).
    Previous default ``"ewma"`` was the lighter RiskMetrics decay; the
    paper specifies GARCH(1,1) (requires the optional ``arch`` package;
    falls back to EWMA if unavailable).

    **Status: pre-promotion** -- ``ridge.prior=random_walk`` is future.
    Reference: doi:10.1016/j.ijforecast.2024.08.006.
    """

    step2_params: dict[str, Any] = {
        "family": "ridge",
        "alpha": alpha_step2,
        "prior": "random_walk",
        "vol_model": vol_model,
        "alpha_strategy": alpha_strategy,
        "cv_folds": cv_folds,
        "forecast_strategy": "direct",
        "training_start_rule": "expanding",
        "refit_policy": "every_origin",
        "search_algorithm": "none",
        "min_train_size": 6,
    }
    if alpha_grid is not None:
        step2_params["alpha_grid"] = list(alpha_grid)
    l4 = {
        "nodes": [
            {
                "id": "src_X",
                "type": "source",
                "selector": {
                    "layer_ref": "l3",
                    "sink_name": "l3_features_v1",
                    "subset": {"component": "X_final"},
                },
            },
            {
                "id": "src_y",
                "type": "source",
                "selector": {
                    "layer_ref": "l3",
                    "sink_name": "l3_features_v1",
                    "subset": {"component": "y_final"},
                },
            },
            {
                "id": "fit_step2",
                "type": "step",
                "op": "fit_model",
                "params": step2_params,
                "inputs": ["src_X", "src_y"],
            },
            {
                "id": "predict",
                "type": "step",
                "op": "predict",
                "inputs": ["fit_step2", "src_X"],
            },
        ],
        "sinks": {
            "l4_forecasts_v1": "predict",
            "l4_model_artifacts_v1": "fit_step2",
            "l4_training_metadata_v1": "auto",
        },
    }
    return _base_recipe(target=target, horizon=horizon, panel=panel, seed=seed, l4=l4)


def hemisphere_neural_network(
    *,
    target: str = "y",
    horizon: int = 1,
    panel: dict[str, list[Any]] | None = None,
    seed: int = 42,
    B: int = 1000,
    neurons: int = 400,
    lc: int = 2,
    lm: int = 2,
    lv: int = 2,
    nu: float | None = None,
    lambda_emphasis: float = 1.0,
    n_epochs: int = 200,
    dropout: float = 0.2,
    lr: float = 0.01,
    sub_rate: float = 0.80,
    quantile_levels: tuple[float, ...] = (0.05, 0.16, 0.84, 0.95),
    forecast_object: Literal["mean", "quantile", "density"] = "density",
) -> dict[str, Any]:
    """Goulet Coulombe / Frenette / Klieber (2025 JAE) HNN =
    ``mlp(architecture=hemisphere, loss=volatility_emphasis)``.

    HNN models ``y_{t+h} ~ N(h_m(X_t), h_v(X_t))`` (paper Eq. 1) with a
    shared ReLU common core feeding two hemispheres -- a mean head
    ``h_m`` and a softplus variance head ``h_v``. The ensemble of ``B``
    blocked-OOB subsamples (paper Eq. 8 / Ingredient 3) is averaged to
    yield calibrated mean + variance, and a log-linear "reality check"
    on the OOB residual variance (paper Eq. 9-10) corrects the variance
    head before quantile/density emission.

    **Hyperparameter defaults are paper-faithful:**

    * ``B = 1000`` (paper p.12, ensemble size).
    * ``neurons = 400`` (paper §3 hidden-layer width).
    * ``lc = lm = lv = 2`` (paper §3 symmetric structure: two hidden
      layers per hemisphere and common block).
    * ``nu = None`` -- triggers data-driven plain-NN OOB residual proxy
      for the variance-emphasis target (paper p.11 footnote 2).
    * ``lambda_emphasis = 1.0`` (Lagrangian multiplier on the volatility-
      emphasis penalty added to the MLE loss; paper §3.2 Ingredient 2).
    * ``n_epochs = 200``, ``dropout = 0.2``, ``lr = 0.01`` (paper §3 + §4
      training schedule).
    * ``sub_rate = 0.80`` (per-bag block fraction; paper Eq. 8 example).
    * ``quantile_levels = (0.05, 0.16, 0.84, 0.95)`` (paper Figure 3
      fan-chart bands).
    * ``forecast_object = 'density'`` -- the paper headline is density
      forecasting, so ``macroforecast.run`` populates
      ``forecast_intervals`` with mean + variance + Gaussian quantile
      bands derived from the Eq. 10-corrected variance head.

    For laptops / smoke tests, dial ``B`` and ``neurons`` down (e.g.
    ``B=10, neurons=16, n_epochs=20``) -- the paper's defaults are
    intentionally heavy.

    **Status: operational; Eq. 10 reality-check active on public path
    post-Phase-B9 fix.** ``predict_quantiles`` and
    ``predict_distribution`` were unreachable from ``macroforecast.run``
    in v0.9.x because ``_emit_quantile_intervals`` routed by family
    string ("mlp" had no native quantile engine) and the density branch
    had no L4 dispatch. Phase B-9 paper-9 F1+F2 added the HNN dispatch
    in ``runtime._emit_quantile_intervals`` /
    ``_emit_density_intervals`` so the paper's distributional head
    drives ``forecast_intervals`` directly.

    Reference: Goulet Coulombe / Frenette / Klieber (2025) "Hemisphere
    Neural Networks", JAE.
    """

    fit_params: dict[str, Any] = {
        "architecture": "hemisphere",
        "loss": "volatility_emphasis",
        "B": int(B),
        "neurons": int(neurons),
        "lc": int(lc),
        "lm": int(lm),
        "lv": int(lv),
        "lambda_emphasis": float(lambda_emphasis),
        "n_epochs": int(n_epochs),
        "dropout": float(dropout),
        "lr": float(lr),
        "sub_rate": float(sub_rate),
        "quantile_levels": list(quantile_levels),
        "forecast_object": str(forecast_object),
    }
    if nu is not None:
        fit_params["nu"] = float(nu)

    return _base_recipe(
        target=target,
        horizon=horizon,
        panel=panel,
        seed=seed,
        l4=_l4_single_fit(
            family="mlp",
            fit_params=fit_params,
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

    # Phase B-1 F1 fix: paper Eq. (3) is `y_{t+h} = ν_i + γ_i X_{i,t} +
    # u_{i,t+h}`, so the supervised slope used to scale each predictor must
    # regress the *h-shifted* target on `X_{i,t}`. The target_construction
    # step (`y_h`) emits that shift; the `scaled_pca` step now consumes
    # `[src_X, y_h]` so `_first_series` resolves the shifted target as the
    # `target_signal` input to the L3 op (instead of the unshifted `src_y`,
    # which would yield a contemporaneous-correlation slope rather than a
    # predictive one).
    l3 = {
        "nodes": [
            {
                "id": "src_X",
                "type": "source",
                "selector": {
                    "layer_ref": "l2",
                    "sink_name": "l2_clean_panel_v1",
                    "subset": {"role": "predictors"},
                },
            },
            {
                "id": "src_y",
                "type": "source",
                "selector": {
                    "layer_ref": "l2",
                    "sink_name": "l2_clean_panel_v1",
                    "subset": {"role": "target"},
                },
            },
            {
                "id": "y_h",
                "type": "step",
                "op": "target_construction",
                "params": {
                    "mode": "point_forecast",
                    "method": "direct",
                    "horizon": horizon,
                },
                "inputs": ["src_y"],
            },
            {
                "id": "spca",
                "type": "step",
                "op": "scaled_pca",
                "params": {
                    "n_components": n_components,
                    "temporal_rule": "expanding_window_per_origin",
                },
                "inputs": ["src_X", "y_h"],
            },
        ],
        "sinks": {
            "l3_features_v1": {"X_final": "spca", "y_final": "y_h"},
            "l3_metadata_v1": "auto",
        },
    }
    recipe = _base_recipe(
        target=target,
        horizon=horizon,
        panel=panel,
        seed=seed,
        l4=_l4_single_fit("ridge", {"alpha": 1.0}),
    )
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
    ridge_lambda: float = 0.1,
    rw_regul: float = 0.75,
    mtry_frac: float = 1.0 / 3.0,
    subsampling_rate: float = 0.75,
    quantile_rate: float = 0.3,
    trend_push: float = 1.0,
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
    upstream 12 to match the paper's monthly-data spec (Coulombe 2024
    JAE §2 footnote 7 for monthly macro panels; upstream default 12
    is for higher-frequency data). Override to 8 for quarterly panels
    or 52 for weekly panels.

    **Phase B-5 paper-5 (v0.9.0a0) helper expansion.** The paper-relevant
    hyperparameters (Goulet Coulombe 2024 JAE p.7-10) are now first-class
    helper kwargs so the public API exposes the full paper calibration
    rather than forcing callers to hand-edit ``fit_params``. Defaults
    match the paper:

    * ``ridge_lambda = 0.1`` -- per-leaf ridge penalty λ (paper p.9).
    * ``rw_regul = 0.75`` -- random-walk shrinkage ζ on the time-varying
      coefficients (paper p.10, fixed across all simulations).
    * ``mtry_frac = 1/3`` -- fraction of features sampled at each split
      (paper p.7, "mtry = 1/3" matching the Breiman default).
    * ``subsampling_rate = 0.75`` -- per-tree subsample fraction inside
      the Block Bayesian Bootstrap (paper p.10).
    * ``quantile_rate = 0.3`` -- vendored MRF default for the
      Extremely-Randomised-Trees split-quantile rate (paper §3.2 ERT).
    * ``trend_push = 1.0`` -- vendored MRF default trend-axis split
      preference (paper §3.2; values >1 bias splits toward the trend
      column for explicit time-axis non-stationarity handling).

    All six are forwarded into the L4 ``fit_model`` ``params`` dict and
    plumbed through ``_MRFExternalWrapper`` to the vendored
    ``MacroRandomForest`` constructor.

    **Status: operational.**
    """

    return _base_recipe(
        target=target,
        horizon=horizon,
        panel=panel,
        seed=seed,
        l4=_l4_single_fit(
            family="macroeconomic_random_forest",
            fit_params={
                "n_estimators": n_estimators,
                "block_size": block_size,
                "ridge_lambda": ridge_lambda,
                "rw_regul": rw_regul,
                "mtry_frac": mtry_frac,
                "subsampling_rate": subsampling_rate,
                "quantile_rate": quantile_rate,
                "trend_push": trend_push,
            },
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
    n_estimators: int = 500,
    min_samples_leaf: int = 40,
    sided: Literal["one", "two"] = "two",
    panel: dict[str, list[Any]] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Goulet Coulombe & Klieber (2025) AlbaMA = ``adaptive_ma_rf`` L3 op
    feeding a downstream ridge.

    **Decomposition.** Paper §2.1-2.3 defines AlbaMA as a random forest
    fit with a *single* regressor (the time index) on the target series
    ``y`` -- ``RF(y_t ~ t)``. Per-observation leaf membership induces
    a weight matrix ``w_τt`` whose row sums to 1, so the smoother is a
    learned-bandwidth moving average of ``y`` (not of the predictor
    panel). Paper p.8 fixes ``B = n_estimators = 500`` and
    ``min_samples_leaf = 40`` as defaults; paper §3.3 selects the
    two-sided variant for retrospective smoothing and the one-sided
    expanding-window variant for real-time nowcasting.

    The L3 ``adaptive_ma_rf`` step consumes ``src_y`` (the target
    series) -- predictors are *not* inputs to AlbaMA. The smoothed
    target series feeds downstream ridge as the canonical feature.

    **Status: operational** -- ``adaptive_ma_rf`` runtime ships in
    v0.9 (paper-faithful K=1 RF, ``max_features=1``). Reference:
    arXiv:2501.13222 §2.1-2.3 + p.7-8 + §3.
    """

    l3 = {
        "nodes": [
            {
                "id": "src_X",
                "type": "source",
                "selector": {
                    "layer_ref": "l2",
                    "sink_name": "l2_clean_panel_v1",
                    "subset": {"role": "predictors"},
                },
            },
            {
                "id": "src_y",
                "type": "source",
                "selector": {
                    "layer_ref": "l2",
                    "sink_name": "l2_clean_panel_v1",
                    "subset": {"role": "target"},
                },
            },
            {
                "id": "alba",
                "type": "step",
                "op": "adaptive_ma_rf",
                "params": {
                    "n_estimators": n_estimators,
                    "min_samples_leaf": min_samples_leaf,
                    "sided": sided,
                },
                "inputs": ["src_y"],
            },
            {
                "id": "y_h",
                "type": "step",
                "op": "target_construction",
                "params": {
                    "mode": "point_forecast",
                    "method": "direct",
                    "horizon": horizon,
                },
                "inputs": ["src_y"],
            },
        ],
        "sinks": {
            "l3_features_v1": {"X_final": "alba", "y_final": "y_h"},
            "l3_metadata_v1": "auto",
        },
    }
    recipe = _base_recipe(
        target=target,
        horizon=horizon,
        panel=panel,
        seed=seed,
        l4=_l4_single_fit("ridge", {"alpha": 1.0}),
    )
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
    """Goulet Coulombe (2026) OLS-as-Attention.

    Returns OLS forecast + attention-weight matrix ``Omega`` (paper
    Eq. 3) + per-test-point proximity diagnostics. The L7 block invokes
    the ``attention_weights`` op which builds
    ``Omega = X_test (X'_train X_train)^{-1} X'_train`` so that
    ``y_hat_test = Omega @ y_train`` (paper §3 representer identity).

    Phase B-10 paper-10 promotion: the helper previously returned
    generic OLS forecasts only -- ``Omega`` was never computed and the
    paper's central interpretive object was inaccessible. The L7
    ``attention_weights`` op is now operational (closed-form, ~10 lines
    of NumPy) and the helper appends a ``7_interpretation`` block by
    default so ``macroforecast.run`` exposes the attention matrix on
    the public L7 sink.

    **Status: operational.** Reference: SSRN 5200864.
    """

    recipe = _base_recipe(
        target=target,
        horizon=horizon,
        panel=panel,
        seed=seed,
        l4=_l4_single_fit(family="ols", fit_params={}, fit_node_id="fit_ols_attn_demo"),
    )
    # Phase B-10 F1 fix: attach the L7 attention-weights interpretation
    # block by default. Mirrors the paper-4 ``arctic_var`` pattern -- the
    # paper's central object (Omega) is now reachable via
    # ``result.cells[0].runtime_result.artifacts["l7_importance_v1"]``.
    recipe["7_interpretation"] = {
        "enabled": True,
        "nodes": [
            {
                "id": "src_model",
                "type": "source",
                "selector": {
                    "layer_ref": "l4",
                    "sink_name": "l4_model_artifacts_v1",
                    "subset": {"model_id": "fit_ols_attn_demo"},
                },
            },
            {
                "id": "src_X",
                "type": "source",
                "selector": {
                    "layer_ref": "l3",
                    "sink_name": "l3_features_v1",
                    "subset": {"component": "X_final"},
                },
            },
            {
                "id": "attention",
                "type": "step",
                "op": "attention_weights",
                "params": {"add_intercept": True, "model_family": "ols"},
                "inputs": ["src_model", "src_X"],
            },
        ],
        "sinks": {
            "l7_importance_v1": {"global": "attention"},
        },
        "fixed_axes": {
            "output_table_format": "long",
            "figure_type": "heatmap",
            "top_k_features_to_show": 20,
            "precision_digits": 4,
            "figure_format": "pdf",
            "latex_table_export": False,
        },
    }
    return recipe


# ---------------------------------------------------------------------------
# 8. Anatomy of OOS Forecasting (Borup et al. 2022)
# ---------------------------------------------------------------------------


def anatomy_oos(
    *,
    target: str = "y",
    horizon: int = 1,
    initial_window: int | None = None,
    n_iterations: int = 500,
    metric: Literal["oshapley_vi", "pbsv"] = "oshapley_vi",
    panel: dict[str, list[Any]] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Borup et al. (2022) Anatomy of OOS Forecasting Accuracy.

    **Decomposition.** Baseline forecast (ridge on lagged X) plus an
    L7 ``oshapley_vi`` (or ``pbsv``) interpretation op. Helper returns
    the L0-L4 baseline plus a ``7_interpretation`` block whose
    interpretation step carries ``initial_window`` and ``n_iterations``
    as op params — the anatomy adapter (``_l7_anatomy_op``) reads
    ``params["initial_window"]`` to take the paper-faithful Path A
    (per-origin refit via ``AnatomySubsets.generate``) instead of the
    final-window-fit Path B.

    Phase B-11 paper-11 F1 fix: previous helper stamped
    ``anatomy_initial_window`` into ``0_meta.leaf_config`` but no
    consumer read that key, so users following the helper always
    routed to Path B (degraded). The L7 block now wires the params
    onto the op step where the runtime actually picks them up.

    The default metric is ``oshapley_vi`` per paper §2.4 (the local
    per-OOS-instance Shapley values are the headline contribution);
    pass ``metric="pbsv"`` for the global squared-error PBSV
    (paper Eq. 24).

    Recommended ``initial_window`` ≈ ``int(T * 0.6)`` (60% expanding-
    window seed). ``n_iterations=500`` matches paper p.16 footnote 16.

    iShapley-VI (paper Eq. 10, in-sample analogue) is **deferred** to
    a later phase — anatomy 0.1.6 has no native iShapley adapter and
    the in-sample formulation requires additional plumbing beyond
    this Phase B-11 scope.

    **Status: pre-promotion** for the full L7 workflow (oshapley_vi /
    pbsv via anatomy package, depends on the optional ``anatomy``
    extra); the baseline forecast is operational.
    Reference: SSRN 4278745.
    """

    fit_node_id = "fit_anatomy_baseline"
    recipe = _base_recipe(
        target=target,
        horizon=horizon,
        panel=panel,
        seed=seed,
        l4=_l4_single_fit(
            family="ridge", fit_params={"alpha": 1.0}, fit_node_id=fit_node_id
        ),
    )

    # Op params: ``initial_window`` enables Path A; omit when None so
    # the runtime takes Path B and emits its degraded-routing warning.
    op_params: dict[str, Any] = {"n_iterations": int(n_iterations)}
    if initial_window is not None:
        op_params["initial_window"] = int(initial_window)

    recipe["7_interpretation"] = {
        "enabled": True,
        "nodes": [
            {
                "id": "src_model",
                "type": "source",
                "selector": {
                    "layer_ref": "l4",
                    "sink_name": "l4_model_artifacts_v1",
                    "subset": {"model_id": fit_node_id},
                },
            },
            {
                "id": "src_X",
                "type": "source",
                "selector": {
                    "layer_ref": "l3",
                    "sink_name": "l3_features_v1",
                    "subset": {"component": "X_final"},
                },
            },
            {
                "id": "src_y",
                "type": "source",
                "selector": {
                    "layer_ref": "l3",
                    "sink_name": "l3_features_v1",
                    "subset": {"component": "y_final"},
                },
            },
            {
                "id": "anatomy_explain",
                "type": "step",
                "op": metric,
                "params": op_params,
                "inputs": ["src_model", "src_X", "src_y"],
            },
        ],
        "sinks": {
            "l7_importance_v1": {"global": "anatomy_explain"},
        },
        "fixed_axes": {
            "output_table_format": "long",
            "figure_type": "bar_global",
            "top_k_features_to_show": 20,
            "precision_digits": 4,
            "figure_format": "pdf",
            "latex_table_export": False,
        },
    }
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

    **Decomposition.** Baseline forecast (ridge on lagged X) plus an L7
    ``dual_decomposition`` interpretation op (paper §2 Eqs. 5-6, the
    representer-theorem identity ``ŷₜ = Σⱼ wⱼ(xₜ) · yⱼ``). The op's
    output artifact also carries the four paper §2.8 portfolio
    diagnostics (FC = forecast concentration, FSP = forecast short
    position, FL = forecast leverage, FT = forecast turnover) inline as
    a ``portfolio_metrics`` frame on the L7 sink — trivial numpy
    reductions on the dual weights that do not warrant a separate L7
    op (decomposition discipline).

    Phase B-12 paper-12 F1/F2 fix: the previous helper returned only
    the L4 ridge baseline (no L7 block), so users following the helper
    got ``forecasts`` but never reached the paper's central object —
    the ``(n_test × n_train)`` dual-weight matrix and per-row
    portfolio metrics. The helper now appends a ``7_interpretation``
    block invoking ``dual_decomposition`` so ``macroforecast.run``
    exposes ``frame.attrs['dual_weights']`` and
    ``frame.attrs['portfolio_metrics']`` on the public L7 sink.

    **Status: operational.** Linear ridge / OLS / lasso, RF / ExtraTrees
    tree-bagging, kernel SVR, and KernelRidge (paper §2.2 headline
    application) are paper-faithful. NN ridge-representation (paper
    §2.3 Eqs. 9-12) is deferred to a later phase.
    Reference: arXiv:2412.13076.
    """

    fit_node_id = "fit_dual_baseline"
    recipe = _base_recipe(
        target=target,
        horizon=horizon,
        panel=panel,
        seed=seed,
        l4=_l4_single_fit(
            family="ridge", fit_params={"alpha": 1.0}, fit_node_id=fit_node_id
        ),
    )
    # Phase B-12 F1/F2 fix: attach the L7 dual_decomposition block by
    # default. Mirrors the paper-10 ``ols_attention_demo`` and paper-11
    # ``anatomy_oos`` patterns.
    recipe["7_interpretation"] = {
        "enabled": True,
        "nodes": [
            {
                "id": "src_model",
                "type": "source",
                "selector": {
                    "layer_ref": "l4",
                    "sink_name": "l4_model_artifacts_v1",
                    "subset": {"model_id": fit_node_id},
                },
            },
            {
                "id": "src_X",
                "type": "source",
                "selector": {
                    "layer_ref": "l3",
                    "sink_name": "l3_features_v1",
                    "subset": {"component": "X_final"},
                },
            },
            {
                "id": "src_y",
                "type": "source",
                "selector": {
                    "layer_ref": "l3",
                    "sink_name": "l3_features_v1",
                    "subset": {"component": "y_final"},
                },
            },
            {
                "id": "dual_explain",
                "type": "step",
                "op": "dual_decomposition",
                "params": {},
                "inputs": ["src_model", "src_X", "src_y"],
            },
        ],
        "sinks": {
            "l7_importance_v1": {"global": "dual_explain"},
        },
        "fixed_axes": {
            "output_table_format": "long",
            "figure_type": "heatmap",
            "top_k_features_to_show": 20,
            "precision_digits": 4,
            "figure_format": "pdf",
            "latex_table_export": False,
        },
    }
    return recipe


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
    smooth_window: int = 3,
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

    Phase B-13 (Round 5 F3 + F8) updates:

    * ``smooth_window`` (default ``3``): paper §3 explicitly applies a
      3-month centred moving average to the order-statistic time series
      before they enter the fused-difference ridge ("the order statistics
      time series are smoothed using the 3 months moving average"). This
      is now the helper default and is forwarded into the L3
      ``asymmetric_trim`` step's ``smooth_window`` param. Pass
      ``smooth_window=0`` to recover the un-smoothed pre-A4 behaviour.
    * Both Variant A (``_ShrinkToTargetRidge``) and Variant B
      (``_FusedDifferenceRidge``) now solve their convex QPs via
      ``cvxpy + OSQP``, matching the paper §2 "Implementation Details"
      stated solver (CVXR / convex programming, OSQP/ECOS backends).

    **Status: operational (v0.9.0a0 Phase A2 + Phase A3 + Phase B-13).**
    Both ``asymmetric_trim`` (operational since v0.8.9 B-6) and
    ``ridge(coefficient_constraint=nonneg, prior={fused_difference,
    shrink_to_target})`` (operational since v0.9.0a0) are
    runtime-implemented. Reference: arXiv:2501.13... (technical report).

    Phase D-1 (Round 5 F2) gap-fix: both variants previously set
    ``"mode": "point_forecast"`` in the ``target_construction`` step,
    training weights to predict the single h-step-ahead inflation value
    ``π_{t+h}``. Paper §2 Eq. (1) and §3 define the target as
    ``π_{t+1:t+h}`` — the **average** of headline inflation between t+1
    and t+h. Both branches now set ``"mode": "cumulative_average"``,
    which routes to ``_cumulative_average_target`` in the runtime
    (``(1/h) Σ_{j=1}^{h} π_{t+j}``).
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
                {
                    "id": "src_X",
                    "type": "source",
                    "selector": {
                        "layer_ref": "l2",
                        "sink_name": "l2_clean_panel_v1",
                        "subset": {"role": "predictors"},
                    },
                },
                {
                    "id": "src_y",
                    "type": "source",
                    "selector": {
                        "layer_ref": "l2",
                        "sink_name": "l2_clean_panel_v1",
                        "subset": {"role": "target"},
                    },
                },
                {
                    "id": "trim_x",
                    "type": "step",
                    "op": "asymmetric_trim",
                    "params": {"smooth_window": int(smooth_window)},
                    "inputs": ["src_X"],
                },
                {
                    "id": "lag_x",
                    "type": "step",
                    "op": "lag",
                    "params": {"n_lag": 1},
                    "inputs": ["trim_x"],
                },
                {
                    "id": "y_h",
                    "type": "step",
                    "op": "target_construction",
                    "params": {
                        "mode": "cumulative_average",
                        "horizon": horizon,
                    },
                    "inputs": ["src_y"],
                },
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
                {
                    "id": "src_X",
                    "type": "source",
                    "selector": {
                        "layer_ref": "l2",
                        "sink_name": "l2_clean_panel_v1",
                        "subset": {"role": "predictors"},
                    },
                },
                {
                    "id": "src_y",
                    "type": "source",
                    "selector": {
                        "layer_ref": "l2",
                        "sink_name": "l2_clean_panel_v1",
                        "subset": {"role": "target"},
                    },
                },
                {
                    "id": "lag_x",
                    "type": "step",
                    "op": "lag",
                    "params": {"n_lag": 1},
                    "inputs": ["src_X"],
                },
                {
                    "id": "y_h",
                    "type": "step",
                    "op": "target_construction",
                    "params": {
                        "mode": "cumulative_average",
                        "horizon": horizon,
                    },
                    "inputs": ["src_y"],
                },
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
        target=target,
        horizon=horizon,
        panel=panel,
        seed=seed,
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
    n_components: int = 12,
    panel: dict[str, list[Any]] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Rapach & Zhou (2025) Sparse Macro-Finance Factors helper recipe.

    **Decomposition (paper §2.1).** Sparse Component Analysis (SCA) per
    Chen & Rohe (2023) — the bilinear convex-hull form

        ``max ‖Z' X Θ‖_F  s.t.  Z ∈ H(T,J),  Θ ∈ H(M,J),  ‖Θ‖_1 ≤ ζ``

    routed through the ``sparse_pca_chen_rohe`` op (NOT the sklearn
    Zou-Hastie-Tibshirani 2006 variant). The default ``n_components=12``
    matches the paper's headline ``J = 12`` factor count; the L1
    budget defaults to the binding boundary ``ζ = J`` (the paper's
    cross-validated optimum, §2.3).

    **Factor dynamics (paper §2.3).** ``var_innovations=True`` fits a
    first-order vector autoregression ``S_t = B S_{t-1} + e_t`` on the
    SCA scores and returns the fitted residuals as the *sparse macro-
    finance factors* of the paper title — Strategy step 2.

    Downstream estimator: ``ridge`` regression on the J factors.

    **Status: operational.** Faithful to Rapach & Zhou (2025) §2.1 +
    §2.3; does not include the §3 supervised-PCA risk-premium chain
    (separate phase).
    """

    l3 = {
        "nodes": [
            {
                "id": "src_X",
                "type": "source",
                "selector": {
                    "layer_ref": "l2",
                    "sink_name": "l2_clean_panel_v1",
                    "subset": {"role": "predictors"},
                },
            },
            {
                "id": "src_y",
                "type": "source",
                "selector": {
                    "layer_ref": "l2",
                    "sink_name": "l2_clean_panel_v1",
                    "subset": {"role": "target"},
                },
            },
            {
                "id": "spca",
                "type": "step",
                "op": "sparse_pca_chen_rohe",
                "params": {
                    "n_components": n_components,
                    "var_innovations": True,
                    "temporal_rule": "expanding_window_per_origin",
                },
                "inputs": ["src_X"],
            },
            {
                "id": "y_h",
                "type": "step",
                "op": "target_construction",
                "params": {
                    "mode": "point_forecast",
                    "method": "direct",
                    "horizon": horizon,
                },
                "inputs": ["src_y"],
            },
        ],
        "sinks": {
            "l3_features_v1": {"X_final": "spca", "y_final": "y_h"},
            "l3_metadata_v1": "auto",
        },
    }
    recipe = _base_recipe(
        target=target,
        horizon=horizon,
        panel=panel,
        seed=seed,
        l4=_l4_single_fit("ridge", {"alpha": 1.0}),
    )
    recipe["3_feature_engineering"] = l3
    return recipe


def sparse_macro_factors_risk_premia(
    sparse_factors,           # np.ndarray | pd.DataFrame, shape (T, J)
    test_asset_returns,       # np.ndarray | pd.DataFrame, shape (T, N)
    *,
    q_grid=(0.05, 0.10, 0.15, 0.20, 0.25, 0.50, 0.75, 1.00),
    cv_folds: int = 5,
    random_state: int | None = None,
) -> dict:
    """Rapach & Zhou (2025) §2.3 Strategy Step 3 — SPCA risk-premia estimation.

    Standalone analysis function (not a :func:`macroforecast.run`-compatible
    recipe). Computes Supervised PCA (SPCA) risk-premia estimates (γ̂) from
    sparse macro-finance factor innovations and equity test asset returns.

    This is the sibling of :func:`sparse_macro_factors`, which covers §2.1
    (SCA factor extraction) and §2.2 (VAR(1) innovation filtering). Strategy
    step 3 — factor-mimicking portfolio construction + SPCA γ̂ estimation —
    is an asset-pricing procedure and is therefore implemented here as a
    standalone analysis function rather than a forecasting recipe.

    **Status: operational (phase-f15).** Reference: Rapach & Zhou (2025)
    §2.3 Strategy Step 3. Closes audit-paper-15.md F10 LOW / R2.

    Parameters
    ----------
    sparse_factors : np.ndarray or pd.DataFrame, shape (T, J)
        Sparse macro-finance factor innovations produced by
        :func:`sparse_macro_factors` / ``_sparse_pca_chen_rohe`` with
        ``var_innovations=True``.
        **Row 0 is the VAR(1) boundary zero-fill placeholder** (per §2.3
        footnote 6) and will be dropped internally. Usable rows: index 1..T-1.
    test_asset_returns : np.ndarray or pd.DataFrame, shape (T, N)
        Excess returns of N ≥ 1 equity test assets, aligned to the
        ``sparse_factors`` time index. Paper uses N > 600 CRSP test portfolios.
    q_grid : tuple of float, optional
        Screening proportion candidates, each in (0, 1]. Default matches
        paper §2.3 5-fold CV grid: {0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1}.
    cv_folds : int, optional
        Number of cross-validation folds for q selection. Default 5 (paper §2.3
        "we use 5-fold CV").
    random_state : int or None, optional
        RNG seed for CV fold splitting (passed to :class:`sklearn.model_selection.KFold`).

    Returns
    -------
    dict with keys:

    ``"gamma_hat"`` : np.ndarray, shape (J,)
        SPCA risk-premia estimates (the paper's γ̂^SPCA).
    ``"beta_hat"`` : np.ndarray, shape (N, J)
        Factor-loading matrix from time-series OLS of R on FMP returns.
    ``"fmp_returns"`` : np.ndarray, shape (T_eff, J)
        Factor-mimicking portfolio return panel (after boundary drop).
    ``"fmp_weights"`` : list of np.ndarray
        One weight vector per factor; shape (N_screened,).
    ``"screened_assets"`` : list of list of int
        Screened asset column indices (per-factor, union over all factors).
    ``"q_selected"`` : float
        CV-selected screening proportion.

    Raises
    ------
    ValueError
        If ``sparse_factors`` and ``test_asset_returns`` have incompatible
        shapes, if there are insufficient rows for CV, or if any q in
        ``q_grid`` is outside (0, 1].
    """
    import math
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import KFold

    # ------------------------------------------------------------------
    # 0. Input coercion
    # ------------------------------------------------------------------
    if isinstance(sparse_factors, pd.DataFrame):
        G_full = sparse_factors.to_numpy(dtype=float)
    else:
        G_full = np.asarray(sparse_factors, dtype=float)

    if isinstance(test_asset_returns, pd.DataFrame):
        R_full = test_asset_returns.to_numpy(dtype=float)
    else:
        R_full = np.asarray(test_asset_returns, dtype=float)

    # Validate q_grid
    q_grid = tuple(q_grid)
    for q in q_grid:
        if not (0 < q <= 1.0):
            raise ValueError(
                f"All q values in q_grid must be in (0, 1]; got {q!r}."
            )

    # ------------------------------------------------------------------
    # 1. Drop boundary row (VAR(1) zero-fill, §2.3 footnote 6)
    # ------------------------------------------------------------------
    G = G_full[1:]   # shape (T_eff, J)
    R = R_full[1:]   # shape (T_eff, N)

    T_eff, J = G.shape
    N = R.shape[1] if R.ndim == 2 else 1
    if R.ndim == 1:
        R = R.reshape(-1, 1)

    # Shape compatibility check
    if R.shape[0] != T_eff:
        raise ValueError(
            f"sparse_factors and test_asset_returns row mismatch after boundary drop: "
            f"G has {T_eff} rows, R has {R.shape[0]} rows. "
            "Both must have the same length T."
        )

    # Sufficient rows for CV
    if T_eff < cv_folds * 2:
        raise ValueError(
            f"Insufficient rows for {cv_folds}-fold CV: need at least "
            f"{cv_folds * 2} rows after boundary drop, got {T_eff}."
        )

    # ------------------------------------------------------------------
    # Helper: screen + FMP weights for a given q on given (G_sub, R_sub)
    # ------------------------------------------------------------------
    def _screen_and_fmp(G_sub, R_sub, q):
        """Compute screened asset union and per-factor FMP weights."""
        T_sub, N_sub = R_sub.shape
        screened_union: set[int] = set()
        per_factor_screened: list[list[int]] = []
        for f in range(J):
            n_keep = max(1, math.ceil(q * N_sub))
            corr_vals = np.array([
                abs(np.corrcoef(R_sub[:, j], G_sub[:, f])[0, 1])
                if np.std(R_sub[:, j]) > 1e-12 and np.std(G_sub[:, f]) > 1e-12
                else 0.0
                for j in range(N_sub)
            ])
            top_idx = np.argsort(corr_vals)[-n_keep:].tolist()
            per_factor_screened.append(sorted(top_idx))
            screened_union.update(top_idx)
        all_screened = sorted(screened_union)
        return all_screened, per_factor_screened

    def _fmp_weights_and_returns(G_sub, R_sub, all_screened):
        """Compute FMP weights and return panel."""
        R_s = R_sub[:, all_screened]   # (T_sub, N_s)
        fmp_ret = np.zeros((R_sub.shape[0], J), dtype=float)
        fmp_w_list: list[np.ndarray] = []
        for f in range(J):
            w_dot = np.linalg.lstsq(R_s, G_sub[:, f], rcond=None)[0]
            denom = float(np.sum(w_dot))
            if abs(denom) > 1e-12:
                w_mp = w_dot / denom
            else:
                w_mp = w_dot  # degenerate case: skip normalization
            fmp_ret[:, f] = R_s @ w_mp
            fmp_w_list.append(w_mp)
        return fmp_ret, fmp_w_list

    # ------------------------------------------------------------------
    # 2. CV over q_grid to select q_selected
    # ------------------------------------------------------------------
    kf = KFold(n_splits=cv_folds, shuffle=False)
    q_mse: dict[float, float] = {}

    for q in q_grid:
        fold_mses: list[float] = []
        for train_idx, val_idx in kf.split(np.arange(T_eff)):
            G_tr, G_val = G[train_idx], G[val_idx]
            R_tr, R_val = R[train_idx], R[val_idx]

            all_sc, _ = _screen_and_fmp(G_tr, R_tr, q)
            if not all_sc:
                fold_mses.append(float("inf"))
                continue

            fmp_tr, _ = _fmp_weights_and_returns(G_tr, R_tr, all_sc)
            # beta_hat on train: OLS of R_tr[:, j] on fmp_tr (with intercept)
            X_tr = np.column_stack([np.ones(len(train_idx)), fmp_tr])
            beta_tr = np.zeros((R_tr.shape[1], J), dtype=float)
            for j in range(R_tr.shape[1]):
                coefs = np.linalg.lstsq(X_tr, R_tr[:, j], rcond=None)[0]
                beta_tr[j] = coefs[1:]  # drop intercept

            # Predict on val: R̂_val = fmp_val @ beta_tr.T
            # Need fmp_val on val set (use train weights, re-project)
            R_s_val = R_val[:, all_sc]
            # Re-project val returns through train FMP weights
            fmp_val = np.column_stack([
                R_s_val @ _.copy() for _ in _fmp_weights_and_returns(G_tr, R_tr, all_sc)[1]
            ])
            R_hat_val = fmp_val @ beta_tr.T  # (T_val, N)
            mse = float(np.mean((R_val - R_hat_val) ** 2))
            fold_mses.append(mse)
        q_mse[q] = float(np.mean(fold_mses))

    q_selected = min(q_mse, key=q_mse.get)

    # ------------------------------------------------------------------
    # 3. Screening on full data with q_selected
    # ------------------------------------------------------------------
    all_screened, per_factor_screened_full = _screen_and_fmp(G, R, q_selected)

    # ------------------------------------------------------------------
    # 4. Factor-mimicking portfolio on full data
    # ------------------------------------------------------------------
    fmp_returns_full, fmp_weights_full = _fmp_weights_and_returns(G, R, all_screened)

    # ------------------------------------------------------------------
    # 5. Time-series OLS for beta_hat on full data
    # ------------------------------------------------------------------
    X_full = np.column_stack([np.ones(T_eff), fmp_returns_full])  # (T_eff, J+1)
    beta_hat = np.zeros((N, J), dtype=float)
    for j in range(N):
        coefs = np.linalg.lstsq(X_full, R[:, j], rcond=None)[0]
        beta_hat[j] = coefs[1:]  # drop intercept

    # ------------------------------------------------------------------
    # 6. SPCA cross-sectional risk premia
    # ------------------------------------------------------------------
    mu_hat = R.mean(axis=0)   # shape (N,)
    gamma_hat = np.linalg.lstsq(beta_hat, mu_hat, rcond=None)[0]  # shape (J,)

    return {
        "gamma_hat": gamma_hat,
        "beta_hat": beta_hat,
        "fmp_returns": fmp_returns_full,
        "fmp_weights": fmp_weights_full,
        "screened_assets": per_factor_screened_full,
        "q_selected": q_selected,
    }


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
            {
                "id": "src_X",
                "type": "source",
                "selector": {
                    "layer_ref": "l2",
                    "sink_name": "l2_clean_panel_v1",
                    "subset": {"role": "predictors"},
                },
            },
            {
                "id": "src_y",
                "type": "source",
                "selector": {
                    "layer_ref": "l2",
                    "sink_name": "l2_clean_panel_v1",
                    "subset": {"role": "target"},
                },
            },
            {
                "id": "marx",
                "type": "step",
                "op": "ma_increasing_order",
                "params": {"max_order": max_order},
                "inputs": ["src_X"],
            },
            {
                "id": "rotated",
                "type": "step",
                "op": "pca",
                "params": {"n_components": 4},
                "inputs": ["marx"],
            },
            {
                "id": "y_h",
                "type": "step",
                "op": "target_construction",
                "params": {
                    "mode": "point_forecast",
                    "method": "direct",
                    "horizon": horizon,
                },
                "inputs": ["src_y"],
            },
        ],
        "sinks": {
            "l3_features_v1": {"X_final": "rotated", "y_final": "y_h"},
            "l3_metadata_v1": "auto",
        },
    }
    recipe = _base_recipe(
        target=target,
        horizon=horizon,
        panel=panel,
        seed=seed,
        l4=_l4_single_fit("random_forest", {"n_estimators": 200}),
    )
    recipe["3_feature_engineering"] = l3
    return recipe


# v0.9.0F gap-closure: 16-cell horse race enumeration for the
# paper's Table 1 (Coulombe et al. 2021 §3 + Recap p.11).

_DATA_TRANSFORM_CELLS_16 = (
    "F",
    "F-X",
    "F-MARX",
    "F-MAF",
    "F-Level",
    "F-X-MARX",
    "F-X-MAF",
    "F-X-Level",
    "F-X-MARX-Level",
    "X",
    "MARX",
    "MAF",
    "X-MARX",
    "X-MAF",
    "X-Level",
    "X-MARX-Level",
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

    Phase D-1 gap-fix (Round 5 F7): ``pca`` nodes require
    ``temporal_rule="expanding_window_per_origin"`` (hard rule of
    ``_factor_op``). Added to F-branch (``feat_F``) and MAF-branch
    (``feat_MAF``). F-branch output is now lagged (``feat_F_lag``,
    ``n_lag=4``) before concat, matching Table 1's
    ``{L^{i-1} F_t}_{i=1}^{p_f}`` structure (p_f=4).
    """

    components = cell.split("-")
    nodes: list[dict[str, Any]] = [
        {
            "id": "src_X",
            "type": "source",
            "selector": {
                "layer_ref": "l2",
                "sink_name": "l2_clean_panel_v1",
                "subset": {"role": "predictors"},
            },
        },
        {
            "id": "src_y",
            "type": "source",
            "selector": {
                "layer_ref": "l2",
                "sink_name": "l2_clean_panel_v1",
                "subset": {"role": "target"},
            },
        },
    ]
    feature_nodes: list[str] = []
    if "F" in components:
        nodes.append(
            {
                "id": "feat_F",
                "type": "step",
                "op": "pca",
                "params": {
                    "n_components": 4,
                    "temporal_rule": "expanding_window_per_origin",
                },
                "inputs": ["src_X"],
            }
        )
        nodes.append(
            {
                "id": "feat_F_lag",
                "type": "step",
                "op": "lag",
                "params": {"n_lag": 4},
                "inputs": ["feat_F"],
            }
        )
        feature_nodes.append("feat_F_lag")
    if "X" in components:
        nodes.append(
            {
                "id": "feat_X",
                "type": "step",
                "op": "lag",
                "params": {"max_lag": 4},
                "inputs": ["src_X"],
            }
        )
        feature_nodes.append("feat_X")
    if "MARX" in components:
        nodes.append(
            {
                "id": "feat_MARX",
                "type": "step",
                "op": "ma_increasing_order",
                "params": {"max_order": max_order},
                "inputs": ["src_X"],
            }
        )
        feature_nodes.append("feat_MARX")
    if "MAF" in components:
        # MAF = MA-of-X then PCA (paper: "moving average factors")
        nodes.append(
            {
                "id": "feat_MAF_ma",
                "type": "step",
                "op": "ma_increasing_order",
                "params": {"max_order": max_order},
                "inputs": ["src_X"],
            }
        )
        nodes.append(
            {
                "id": "feat_MAF",
                "type": "step",
                "op": "pca",
                "params": {
                    "n_components": 4,
                    "temporal_rule": "expanding_window_per_origin",
                },
                "inputs": ["feat_MAF_ma"],
            }
        )
        feature_nodes.append("feat_MAF")
    if "Level" in components:
        nodes.append(
            {
                "id": "feat_Level",
                "type": "step",
                "op": "lag",
                "params": {"max_lag": 4},
                "inputs": ["src_y"],
            }
        )
        feature_nodes.append("feat_Level")
    nodes.append(
        {
            "id": "X_final",
            "type": "step",
            "op": "weighted_concat",
            "params": {},
            "inputs": feature_nodes,
        }
    )
    # Runtime switches on ``mode`` (not ``method``); map target_method to
    # the right mode so ``path_average`` actually triggers the cumulative-
    # average target (paper §2.2 + Table 2).
    target_mode = (
        "path_average"
        if target_method in {"path_average", "cumulative_average"}
        else "point_forecast"
    )
    nodes.append(
        {
            "id": "y_h",
            "type": "step",
            "op": "target_construction",
            "params": {
                "mode": target_mode,
                "method": target_method,
                "horizon": horizon,
            },
            "inputs": ["src_y"],
        }
    )
    return {
        "nodes": nodes,
        "sinks": {
            "l3_features_v1": {"X_final": "X_final", "y_final": "y_h"},
            "l3_metadata_v1": "auto",
        },
    }


# Paper Table 1 family list (7 entries): AR / FM / Adaptive Lasso / Elastic
# Net / Linear Boosting / Random Forest / Boosted Trees. macroforecast L4
# mappings: FM → ``factor_augmented_ar`` (Stock-Watson 2002a ARDI), LB →
# ``glmboost`` (component-wise L2 boosting with linear base learners).
_DATA_TRANSFORM_FAMILIES_DEFAULT = (
    "ar_p",  # AR
    "factor_augmented_ar",  # FM (Stock-Watson 2002a) — audit-fix
    "lasso_path",  # AL (Adaptive Lasso proxy)
    "elastic_net",  # EN
    "glmboost",  # LB (component-wise linear boosting) — audit-fix
    "random_forest",  # RF
    "gradient_boosting",  # BT
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
    attach_eval_blocks: bool = False,
    benchmark_family: str = "factor_augmented_ar",
    benchmark_params: dict[str, Any] | None = None,
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

    Phase B-15 paper-15 F5: ``attach_eval_blocks=True`` mirrors the
    sister helper ``ml_useful_macro_horse_race`` -- it routes each cell
    through ``_l4_with_benchmark`` (so each recipe has BOTH the cell
    family AND a benchmark fit_node flagged ``is_benchmark: true``) and
    stamps the canonical paper-16 L5 / L6 dicts so the §4.4 DM-vs-FM
    comparison is one kwarg away. The default benchmark is
    ``factor_augmented_ar`` with ``search_algorithm="bic"`` and
    ``n_factors=4``, matching paper §4.4 ("DM tested against the
    reference (ARDI, BIC)").

    **Status: operational (v0.9.0a0 audit gap-fix).** Reference:
    arXiv:2008.01714.
    """

    grid: dict[str, dict[str, Any]] = {}
    horizons_iter = tuple(horizons) if horizons else (horizon,)
    eval_l5, eval_l6 = _l5_l6_paper16_eval_blocks() if attach_eval_blocks else ({}, {})
    for cell in cells:
        for family in families:
            for h in horizons_iter:
                for method in target_methods:
                    if attach_eval_blocks:
                        l4 = _l4_with_benchmark(
                            family,
                            {},
                            benchmark_family=benchmark_family,
                            benchmark_params=benchmark_params,
                        )
                    else:
                        l4 = _l4_single_fit(family, {})
                    recipe = _base_recipe(
                        target=target,
                        horizon=h,
                        panel=panel,
                        seed=seed,
                        l4=l4,
                    )
                    recipe["3_feature_engineering"] = _l3_data_transforms_cell(
                        cell, h, max_order, target_method=method
                    )
                    if attach_eval_blocks:
                        recipe["5_evaluation"] = dict(eval_l5)
                        recipe["6_statistical_tests"] = dict(eval_l6)
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
        l4=_l4_single_fit(
            family="ridge", fit_params={"alpha": 1.0}, fit_node_id="fit_baseline"
        ),
    )


# v0.9.0F gap-closure: 4-feature horse race for the paper's central
# decomposition (Coulombe et al. 2022 JAE §3 + Eq. 16).

_ML_USEFUL_FEATURES: dict[str, tuple[str, dict[str, Any]]] = {
    # Feature 1 — nonlinearity
    "linear_baseline": ("ar_p", {"n_lag": 4}),
    "krr_rbf": ("kernel_ridge", {"kernel": "rbf", "alpha": 1.0}),
    "rf": ("random_forest", {"n_estimators": 200, "max_depth": 8}),
    # Feature 2 — regularization
    "lasso": ("lasso", {"alpha": 0.05}),
    "elastic_net": ("elastic_net", {"alpha": 0.05, "l1_ratio": 0.5}),
    "ridge": ("ridge", {"alpha": 1.0}),
    # Feature 4 — loss function
    "svr_rbf": ("svr_rbf", {"C": 1.0, "epsilon": 0.1, "gamma": "scale"}),
    "svr_linear": ("svr_linear", {"C": 1.0, "epsilon": 0.1}),
    # Reference baselines
    "fm": ("factor_augmented_ar", {"n_factors": 3, "n_lag": 2}),
    "ols": ("ols", {}),
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
    cases_iter: tuple[str, ...] = (
        cases if cases is not None else tuple(_ML_USEFUL_FEATURES.keys())
    )
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
                            family,
                            cell_fit_params,
                            benchmark_family=benchmark_family,
                            benchmark_params=benchmark_params,
                        )
                    else:
                        l4 = _l4_single_fit(family, cell_fit_params)
                    recipe = _base_recipe(
                        target=tgt,
                        horizon=h,
                        panel=panel,
                        seed=seed,
                        l4=l4,
                    )
                    # H_minus: paper §3.2 data-poor (own-lag y only).
                    # H_plus: data-rich (PCA factors of X plus lagged y).
                    recipe["3_feature_engineering"] = _l3_h_axis(
                        data_richness,
                        h,
                        n_factors=n_factors,
                    )
                    if attach_eval_blocks:
                        recipe["5_evaluation"] = dict(eval_l5)
                        recipe["6_statistical_tests"] = dict(eval_l6)
                    grid[f"{tgt}__{case}__h{h}__{cv}__{data_richness}"] = recipe
    return grid


def _l3_h_axis(
    richness: str, horizon: int, *, n_factors: int = 4, n_lag: int = 4
) -> dict[str, Any]:
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
        {
            "id": "src_X",
            "type": "source",
            "selector": {
                "layer_ref": "l2",
                "sink_name": "l2_clean_panel_v1",
                "subset": {"role": "predictors"},
            },
        },
        {
            "id": "src_y",
            "type": "source",
            "selector": {
                "layer_ref": "l2",
                "sink_name": "l2_clean_panel_v1",
                "subset": {"role": "target"},
            },
        },
    ]
    if richness == "H_minus":
        if n_lag == 0:
            nodes.append(
                {
                    "id": "X_final",
                    "type": "step",
                    "op": "identity",
                    "params": {},
                    "inputs": ["src_y"],
                }
            )
            x_final = "X_final"
        else:
            nodes.append(
                {
                    "id": "lag_y",
                    "type": "step",
                    "op": "lag",
                    "params": {"n_lag": n_lag},
                    "inputs": ["src_y"],
                }
            )
            x_final = "lag_y"
    else:  # H_plus
        nodes.append(
            {
                "id": "feat_F",
                "type": "step",
                "op": "pca",
                "params": {
                    "n_components": n_factors,
                    "temporal_rule": "expanding_window_per_origin",
                },
                "inputs": ["src_X"],
            }
        )
        if n_lag == 0:
            # No-lag DAG: identity(feat_F) ⊕ identity(src_y) via weighted_concat.
            nodes.append(
                {
                    "id": "id_y",
                    "type": "step",
                    "op": "identity",
                    "params": {},
                    "inputs": ["src_y"],
                }
            )
            nodes.append(
                {
                    "id": "X_final",
                    "type": "step",
                    "op": "weighted_concat",
                    "params": {},
                    "inputs": ["feat_F", "id_y"],
                }
            )
        else:
            nodes.append(
                {
                    "id": "lag_F",
                    "type": "step",
                    "op": "lag",
                    "params": {"n_lag": n_lag},
                    "inputs": ["feat_F"],
                }
            )
            nodes.append(
                {
                    "id": "lag_y",
                    "type": "step",
                    "op": "lag",
                    "params": {"n_lag": n_lag},
                    "inputs": ["src_y"],
                }
            )
            nodes.append(
                {
                    "id": "X_final",
                    "type": "step",
                    "op": "weighted_concat",
                    "params": {},
                    "inputs": ["lag_F", "lag_y"],
                }
            )
        x_final = "X_final"
    nodes.append(
        {
            "id": "y_h",
            "type": "step",
            "op": "target_construction",
            "params": {
                "mode": "point_forecast",
                "method": "direct",
                "horizon": horizon,
            },
            "inputs": ["src_y"],
        }
    )
    return {
        "nodes": nodes,
        "sinks": {
            "l3_features_v1": {"X_final": x_final, "y_final": "y_h"},
            "l3_metadata_v1": "auto",
        },
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
                target=target,
                horizon=horizon,
                panel=panel,
                seed=seed,
                l4=_l4_single_fit(family, {}),
            )
            recipe["3_feature_engineering"] = _l3_b_rotation(
                rot, horizon, n_factors=n_factors, n_lag=n_lag
            )
            grid[f"{rot}__{family}"] = recipe
    return grid


def _l3_b_rotation(
    rotation: str, horizon: int, *, n_factors: int = 4, n_lag: int = 4
) -> dict[str, Any]:
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
        {
            "id": "src_X",
            "type": "source",
            "selector": {
                "layer_ref": "l2",
                "sink_name": "l2_clean_panel_v1",
                "subset": {"role": "predictors"},
            },
        },
        {
            "id": "src_y",
            "type": "source",
            "selector": {
                "layer_ref": "l2",
                "sink_name": "l2_clean_panel_v1",
                "subset": {"role": "target"},
            },
        },
    ]
    if rotation == "B1":
        # Identity: pass X through unchanged.
        if n_lag == 0:
            nodes.append(
                {
                    "id": "X_final",
                    "type": "step",
                    "op": "identity",
                    "params": {},
                    "inputs": ["src_X"],
                }
            )
        else:
            nodes.append(
                {
                    "id": "X_final",
                    "type": "step",
                    "op": "lag",
                    "params": {"n_lag": n_lag},
                    "inputs": ["src_X"],
                }
            )
    elif rotation == "B2":
        # Phase A2 fix: paper §3.2 Eq. (18) keeps ALL N factors ("we do
        # not select F_t … we keep them all"). Phase A3 fix: pass the
        # explicit ``n_components="all"`` sentinel rather than relying on
        # the schema default (4) — runtime resolves "all" → min(T, N)
        # at PCA fit time.
        if n_lag == 0:
            # Phase A4c: no-lag identity path. PCA-of-X is the X_final.
            nodes.append(
                {
                    "id": "X_final",
                    "type": "step",
                    "op": "pca",
                    "params": {
                        "n_components": "all",
                        "temporal_rule": "expanding_window_per_origin",
                    },
                    "inputs": ["src_X"],
                }
            )
        else:
            nodes.append(
                {
                    "id": "feat_pca",
                    "type": "step",
                    "op": "pca",
                    "params": {
                        "n_components": "all",
                        "temporal_rule": "expanding_window_per_origin",
                    },
                    "inputs": ["src_X"],
                }
            )
            # Phase A3 fix: rename ``max_lag`` → ``n_lag`` (the lag op reads
            # ``n_lag``; the typo silently fell back to the default 4).
            nodes.append(
                {
                    "id": "X_final",
                    "type": "step",
                    "op": "lag",
                    "params": {"n_lag": n_lag},
                    "inputs": ["feat_pca"],
                }
            )
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
            nodes.append(
                {
                    "id": "X_final",
                    "type": "step",
                    "op": "pca",
                    "params": {
                        "n_components": "all",
                        "temporal_rule": "expanding_window_per_origin",
                    },
                    "inputs": ["src_X"],
                }
            )
        else:
            nodes.append(
                {
                    "id": "lag_x",
                    "type": "step",
                    "op": "lag",
                    "params": {"n_lag": n_lag},
                    "inputs": ["src_X"],
                }
            )
            nodes.append(
                {
                    "id": "lag_y",
                    "type": "step",
                    "op": "lag",
                    "params": {"n_lag": n_lag},
                    "inputs": ["src_y"],
                }
            )
            nodes.append(
                {
                    "id": "h_plus",
                    "type": "step",
                    "op": "weighted_concat",
                    "params": {},
                    "inputs": ["lag_y", "lag_x"],
                }
            )
            nodes.append(
                {
                    "id": "X_final",
                    "type": "step",
                    "op": "pca",
                    "params": {
                        "n_components": "all",
                        "temporal_rule": "expanding_window_per_origin",
                    },
                    "inputs": ["h_plus"],
                }
            )
    else:
        raise ValueError(f"unknown rotation {rotation!r}; expected B1/B2/B3")
    nodes.append(
        {
            "id": "y_h",
            "type": "step",
            "op": "target_construction",
            "params": {
                "mode": "point_forecast",
                "method": "direct",
                "horizon": horizon,
            },
            "inputs": ["src_y"],
        }
    )
    return {
        "nodes": nodes,
        "sinks": {
            "l3_features_v1": {"X_final": "X_final", "y_final": "y_h"},
            "l3_metadata_v1": "auto",
        },
    }


# ---------------------------------------------------------------------------
# 16. Arctic Amplification VAR / VARCTIC (Coulombe & Goebel 2021)
# ---------------------------------------------------------------------------


def arctic_var(
    *,
    target: str = "y",
    horizon: int = 1,
    n_lag: int | None = None,
    panel: dict[str, list[Any]] | None = None,
    seed: int = 42,
    b_AR: float = 0.9,
    lambda_1: float = 0.3,
    lambda_cross: float = 0.5,
    lambda_decay: float = 1.5,
    n_posterior_draws: int = 2000,
    posterior_irf_periods: int = 36,
    ordering: tuple[str, ...] | None = None,
    n_lags: int | None = None,
) -> dict[str, Any]:
    """Goulet Coulombe & Goebel (2021) VARCTIC = ``bvar_minnesota`` family
    + L7 ``orthogonalised_irf`` / ``historical_decomposition`` / ``fevd``
    ops for shock decomposition + Bayesian credible bands.

    **Phase B-4 paper-4 rewire (2026-05-08).** The helper now routes to
    the multi-equation Minnesota BVAR (paper §3.c, Eq. 5) rather than
    the OLS-VAR proxy used in v0.9.0a0. Default hyperparameters track
    paper Appx-A.3 VARCTIC 8 optimum: ``b_AR = 0.9`` (random-walk
    anchor), ``λ₁ = 0.3`` (overall tightness), ``λ_cross = 0.5``
    (cross-equation shrinkage), ``λ_decay = 1.5`` (lag decay),
    ``n_posterior_draws = 2000`` (paper Appx-A footnote A5),
    ``n_lag = 12`` (paper VARCTIC 8 P=12).

    **Cholesky identification.** Pass ``ordering = ("co2", "tcc",
    "pr", "at", "sst", "sie", "sit", "albedo")`` (or the panel-
    appropriate column names) to anchor the Cholesky decomposition to
    the paper §3.c (footnote 12) ordering: CO₂ first (most exogenous),
    surface-feedback channels last. With ``ordering = None`` the BVAR
    falls back to pandas-concat column order.

    **Backward-compatibility.** ``n_lags`` (plural) is accepted as a
    legacy alias and emits ``DeprecationWarning``; the recipe always
    writes the canonical ``n_lag`` key consumed by ``_BayesianVAR``.
    Earlier releases silently routed ``n_lags`` to nowhere because the
    L4 BVAR factory reads ``n_lag``.

    **Decomposition.** Helper returns L0-L4 + an L7 block that wires
    ``orthogonalised_irf`` and ``historical_decomposition`` against the
    fitted BVAR so the artifact carries posterior-mean impulse / HD
    paths plus 16/84 percentile credible bands per the paper §3.

    **Status: operational** (BVAR Minnesota + posterior IRF / HD bands).
    Reference: doi:10.1175/JCLI-D-20-0324.1.
    """

    # F2 backward-compat: accept legacy ``n_lags`` plural arg with a
    # DeprecationWarning. The canonical key is ``n_lag`` (singular) to
    # match the L4 factory at runtime.py and the rest of the codebase.
    if n_lags is not None:
        import warnings

        warnings.warn(
            "arctic_var: 'n_lags' is deprecated; use the canonical "
            "'n_lag' (singular). Earlier releases silently dropped "
            "'n_lags' because the L4 BVAR factory reads 'n_lag'.",
            DeprecationWarning,
            stacklevel=2,
        )
        if n_lag is None:
            n_lag = int(n_lags)
        elif int(n_lags) != int(n_lag):
            raise ValueError(
                f"arctic_var: conflicting n_lag={n_lag} and "
                f"n_lags={n_lags}; pass exactly one (prefer n_lag)."
            )
    if n_lag is None:
        n_lag = 12  # paper VARCTIC 8 default

    fit_params: dict[str, Any] = {
        "n_lag": int(n_lag),
        "b_AR": float(b_AR),
        "lambda_1": float(lambda_1),
        "lambda_cross": float(lambda_cross),
        "lambda_decay": float(lambda_decay),
        "n_posterior_draws": int(n_posterior_draws),
        "posterior_irf_periods": int(posterior_irf_periods),
    }
    if ordering is not None:
        fit_params["ordering"] = list(ordering)

    recipe = _base_recipe(
        target=target,
        horizon=horizon,
        panel=panel,
        seed=seed,
        l4=_l4_single_fit(
            family="bvar_minnesota",
            fit_params=fit_params,
            fit_node_id="fit_varctic",
        ),
    )
    # Phase B-4 F7: attach the L7 IRF / HD block by default so the
    # public artifact carries posterior-mean responses + 16/84 bands.
    recipe["7_interpretation"] = {
        "enabled": True,
        "nodes": [
            {
                "id": "src_var",
                "type": "source",
                "selector": {
                    "layer_ref": "l4",
                    "sink_name": "l4_model_artifacts_v1",
                    "subset": {"model_id": "fit_varctic"},
                },
            },
            {
                "id": "src_X",
                "type": "source",
                "selector": {
                    "layer_ref": "l3",
                    "sink_name": "l3_features_v1",
                    "subset": {"component": "X_final"},
                },
            },
            {
                "id": "irf",
                "type": "step",
                "op": "orthogonalised_irf",
                "params": {
                    "n_periods": int(posterior_irf_periods),
                    "model_family": "bvar_minnesota",
                },
                "inputs": ["src_var", "src_X"],
            },
            {
                "id": "hd",
                "type": "step",
                "op": "historical_decomposition",
                "params": {
                    "n_periods": int(posterior_irf_periods),
                    "model_family": "bvar_minnesota",
                },
                "inputs": ["src_var", "src_X"],
            },
        ],
        "sinks": {
            "l7_importance_v1": {
                "global": ["irf", "hd"],
            },
        },
        "fixed_axes": {
            "output_table_format": "long",
            "figure_type": "auto",
            "top_k_features_to_show": 20,
            "precision_digits": 4,
            "figure_format": "pdf",
            "latex_table_export": False,
        },
    }
    return recipe


# ---------------------------------------------------------------------------
# Phase C top-6 net-new method helpers (2026-05-08).
# ---------------------------------------------------------------------------


def u_midas(
    *,
    target: str = "y",
    horizon: int = 1,
    freq_ratio: int = 3,
    n_lags_high: "int | Literal['bic', 'aic']" = "bic",
    include_y_lag: bool = True,
    regularization: "Literal['none', 'ridge']" = "none",
    alpha: float = 1.0,
    panel: "dict[str, list[Any]] | None" = None,
    seed: int = 42,
) -> "dict[str, Any]":
    """Foroni-Marcellino-Schumacher (2015) Unrestricted MIDAS recipe.
    Published as Bundesbank Discussion Paper Series 1, No 35/2011;
    JRSS-A 178(1), 57-82, DOI 10.1111/rssa.12043.

    Model (paper §3.2 eq.(20)):
        y_{t*k} = mu0 + mu1 y_{t*k-k} + psi(L) x_{t*k-1} + eps_{t*k}

    where psi(L) = psi0 + psi1*L + ... + psi_K*L^K is the unrestricted HF lag
    polynomial. mu0, mu1, and psi(L) are estimated by OLS (paper §3.2 p.11).

    Lag order K is selected by BIC with K_max = ceil(1.5 * freq_ratio)
    (paper §3.2 p.11 + §3.5). Pass n_lags_high as an integer to fix K.

    Design matrix (paper §2.1 eq.(8)):
        Stack columns: [y_lag1 | x_lag0 | x_lag1 | ... | x_lag{K-1}]
        indexed at LF dates (stock-variable aggregation, §3.1 p.9).

    Decomposition:
      L3 op "u_midas": lag-stack HF predictors + optional AR(1) y-lag;
        BIC selects K if n_lags_high in {"bic", "aic"}.
      L4 op "ols": fit unrestricted polynomial by OLS (paper §3.2 p.11).

    Parameters
    ----------
    n_lags_high : int or {"bic", "aic"}, default "bic"
        Number of HF lags. "bic"/"aic" triggers information-criterion
        selection over K in {1, ..., ceil(1.5*freq_ratio)} per paper §3.5.
    include_y_lag : bool, default True
        Include AR(1) y-lag term mu1 y_{t*k-k} per paper §3.2 eq.(20).
        Set False to match simplified §2.3 eq.(14) (no-AR form).
    regularization : {"none", "ridge"}, default "none"
        "none" = OLS (paper-faithful). "ridge" = ridge regression with
        penalty alpha; deviates from paper §3.2 estimator choice.
    alpha : float, default 1.0
        Ridge penalty (only used when regularization="ridge").

    Monte Carlo anchor (paper §3.4 Table 2, recursive HF VAR DGP):
      OOS MSE ratio U-MIDAS/MIDAS approx 0.91-0.94 for k=3, rho >= 0.5.
      OOS MSE ratio approx 1.07-1.24 for k=12, k=60 (MIDAS wins).
    """
    # Input validation
    if regularization not in ("none", "ridge"):
        raise ValueError(f"regularization must be 'none' or 'ridge'; got {regularization!r}")
    if regularization == "ridge" and alpha <= 0:
        raise ValueError(f"alpha must be > 0 when regularization='ridge'; got {alpha}")

    # Build L3 DAG
    umidas_inputs = ["src_X", "src_y"] if include_y_lag else ["src_X"]

    l3 = {
        "nodes": [
            {
                "id": "src_X",
                "type": "source",
                "selector": {
                    "layer_ref": "l2",
                    "sink_name": "l2_clean_panel_v1",
                    "subset": {"role": "predictors"},
                },
            },
            {
                "id": "src_y",
                "type": "source",
                "selector": {
                    "layer_ref": "l2",
                    "sink_name": "l2_clean_panel_v1",
                    "subset": {"role": "target"},
                },
            },
            {
                "id": "y_h",
                "type": "step",
                "op": "target_construction",
                "params": {
                    "mode": "point_forecast",
                    "method": "direct",
                    "horizon": horizon,
                },
                "inputs": ["src_y"],
            },
            {
                "id": "umidas",
                "type": "step",
                "op": "u_midas",
                "params": {
                    "freq_ratio": freq_ratio,
                    "n_lags_high": n_lags_high,
                    "target_freq": "low",
                    "include_y_lag": include_y_lag,
                    "temporal_rule": "expanding_window_per_origin",
                },
                "inputs": umidas_inputs,
            },
        ],
        "sinks": {
            "l3_features_v1": {"X_final": "umidas", "y_final": "y_h"},
            "l3_metadata_v1": "auto",
        },
    }
    # R1: OLS default (paper §3.2 p.11); ridge opt-in via regularization="ridge"
    if regularization == "ridge":
        l4 = _l4_single_fit("ridge", {"alpha": alpha})
    else:
        l4 = _l4_single_fit("ols", {})
    recipe = _base_recipe(
        target=target,
        horizon=horizon,
        panel=panel,
        seed=seed,
        l4=l4,
    )
    recipe["3_feature_engineering"] = l3
    return recipe


def factor_midas_nowcast(
    *,
    target: str = "y",
    horizon: int = 1,
    freq_ratio: int = 3,
    n_factors: int = 1,
    n_lags_high: "int | Literal['bic', 'aic']" = "bic",
    regularization: "Literal['none', 'ridge']" = "none",
    alpha: float = 1.0,
    panel: "dict[str, list[Any]] | None" = None,
    seed: int = 42,
) -> "dict[str, Any]":
    """Marcellino & Schumacher (2010) Factor MIDAS recipe.
    Oxford Bulletin of Economics and Statistics 72(4), 518-550.
    DOI 10.1111/j.1468-0084.2010.00591.x.
    EUI preprint ECO-2008-16 (cadmus.eui.eu handle 1814/8087).

    Implementation reconstructed from abstract and established Factor MIDAS
    literature; PDF §-references pending post-acquisition verification.

    Model (paper Fig. 1 flowchart):
        Step 1: Extract r common factors from HF panel via static PCA (paper §2.1).
                F = PCA(X_hf, r)
        Step 2: Apply MIDAS lag aggregation to factors (paper §2.2 + §3 U-MIDAS variant).
                Z_lf = midas_lag_stack(F, m, K)
        Step 3: OLS regression of LF target on aggregated factor features (paper §2.2).
                y_h = alpha + Z_lf @ beta + epsilon

    Decomposition:
      L3 "dfm": static PCA factor extraction from HF predictors (paper §2.1 Method B).
      L3 "u_midas": unrestricted lag-stacking of extracted factors (paper §3 parsimonious).
      L4 "ols": OLS fit of LF target on MIDAS-aggregated factor features.

    Explicit implementation assumptions (PDF unavailable; reconstruction from abstract):
      1. Factor extraction = static PCA (paper Method B), not Kalman smoother (Method D).
         Paper states extraction methods do not significantly differ in nowcast accuracy.
      2. MIDAS variant = unrestricted (U-MIDAS) lag stacking, not parametric exp-Almon.
         Maximally reuses existing ``_u_midas`` / ``op: "u_midas"`` infrastructure.
      3. Default n_factors=1 (paper uses r=1 or r=2 for German GDP).
      4. Default n_lags_high="bic" (BIC selection over K in {1,...,ceil(1.5*freq_ratio)}).
         Paper §3 recommends K=1 (contemporaneous factor only) as best overall.

    Parameters
    ----------
    target : str, default "y"
        Name of the target column in the panel.
    horizon : int, default 1
        Forecast horizon in LF periods (h in paper notation).
    freq_ratio : int, default 3
        HF periods per LF period (3 = monthly HF / quarterly LF, as in paper).
        Must be >= 1.
    n_factors : int, default 1
        Number of common factors r extracted from HF panel. Paper uses r=1 or r=2
        for German GDP nowcasting. Must be >= 1 and < number of HF predictors.
    n_lags_high : int or {"bic", "aic"}, default "bic"
        Number of HF lags K in the MIDAS polynomial. "bic"/"aic" triggers
        BIC/AIC selection over K in {1, ..., ceil(1.5*freq_ratio)} (consistent
        with Foroni-Marcellino-Schumacher 2015 §3.5 lag selection). Paper §3
        results: K=1 (single-lag / contemporaneous factor) performs best overall.
    regularization : {"none", "ridge"}, default "none"
        "none" = OLS (paper-faithful). "ridge" = ridge; deviates from paper.
    alpha : float, default 1.0
        Ridge penalty (only used when regularization="ridge"). Must be > 0.
    panel : dict or None, default None
        Inline custom panel dict ``{"date": [...], "y": [...], "x1": [...], ...}``.
        If None, uses the package default 12-row synthetic panel.
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    dict[str, Any]
        A valid macroforecast recipe dict ready for ``macroforecast.run(recipe)``.

    Status: **operational** -- uses existing ``op: "dfm"`` (L3 PCA factor
    extraction) and ``op: "u_midas"`` (L3 MIDAS lag aggregation). Both ops
    were promoted operational in v0.9.0a0.
    """
    # --- Input validation ---
    if not (isinstance(n_factors, int) and n_factors >= 1):
        raise ValueError(
            f"n_factors must be a positive integer; got {n_factors!r}"
        )
    if not (isinstance(freq_ratio, int) and freq_ratio >= 1):
        raise ValueError(
            f"freq_ratio must be a positive integer; got {freq_ratio!r}"
        )
    if not (
        (isinstance(n_lags_high, int) and n_lags_high >= 1)
        or n_lags_high in ("bic", "aic")
    ):
        raise ValueError(
            f"n_lags_high must be a positive integer or one of 'bic', 'aic';"
            f" got {n_lags_high!r}"
        )
    if regularization not in ("none", "ridge"):
        raise ValueError(
            f"regularization must be 'none' or 'ridge'; got {regularization!r}"
        )
    if regularization == "ridge" and alpha <= 0:
        raise ValueError(
            f"alpha must be > 0 when regularization='ridge'; got {alpha}"
        )

    # --- Build L3 DAG (two-step: dfm factor extraction -> u_midas aggregation) ---
    l3 = {
        "nodes": [
            {
                "id": "src_X",
                "type": "source",
                "selector": {
                    "layer_ref": "l2",
                    "sink_name": "l2_clean_panel_v1",
                    "subset": {"role": "predictors"},
                },
            },
            {
                "id": "src_y",
                "type": "source",
                "selector": {
                    "layer_ref": "l2",
                    "sink_name": "l2_clean_panel_v1",
                    "subset": {"role": "target"},
                },
            },
            {
                "id": "y_h",
                "type": "step",
                "op": "target_construction",
                "params": {
                    "mode": "point_forecast",
                    "method": "direct",
                    "horizon": horizon,
                },
                "inputs": ["src_y"],
            },
            {
                "id": "factors",
                "type": "step",
                "op": "dfm",
                "params": {
                    "n_factors": n_factors,
                    "temporal_rule": "expanding_window_per_origin",
                },
                "inputs": ["src_X"],
            },
            {
                "id": "fmidas",
                "type": "step",
                "op": "u_midas",
                "params": {
                    "freq_ratio": freq_ratio,
                    "n_lags_high": n_lags_high,
                    "target_freq": "low",
                    "include_y_lag": False,
                    "temporal_rule": "expanding_window_per_origin",
                },
                "inputs": ["factors"],
            },
        ],
        "sinks": {
            "l3_features_v1": {"X_final": "fmidas", "y_final": "y_h"},
            "l3_metadata_v1": "auto",
        },
    }

    # --- Build L4 (OLS default; ridge opt-in) ---
    if regularization == "ridge":
        l4 = _l4_single_fit("ridge", {"alpha": alpha})
    else:
        l4 = _l4_single_fit("ols", {})

    recipe = _base_recipe(
        target=target,
        horizon=horizon,
        panel=panel,
        seed=seed,
        l4=l4,
    )
    recipe["3_feature_engineering"] = l3
    return recipe


def midas_almon(
    *,
    target: str = "y",
    horizon: int = 1,
    weighting: str = "exp_almon",
    polynomial_order: int = 2,
    freq_ratio: int = 3,
    n_lags_high: int = 12,
    panel: dict[str, list[Any]] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Ghysels-Sinko-Valkanov (2007) MIDAS with parametric weighted lag polynomial.

    **Decomposition.** Single ``midas`` L3 op that internally fits the
    NLS optimisation (``scipy.optimize.minimize``) and emits one
    aggregated column per HF predictor. Downstream linear regression
    then maps the weighted aggregates to the target.

    **Status: operational** -- ``midas`` op operational v0.9.1 dev-stage
    Phase C (2026-05-08).
    """

    l3 = {
        "nodes": [
            {
                "id": "src_X",
                "type": "source",
                "selector": {
                    "layer_ref": "l2",
                    "sink_name": "l2_clean_panel_v1",
                    "subset": {"role": "predictors"},
                },
            },
            {
                "id": "src_y",
                "type": "source",
                "selector": {
                    "layer_ref": "l2",
                    "sink_name": "l2_clean_panel_v1",
                    "subset": {"role": "target"},
                },
            },
            {
                "id": "y_h",
                "type": "step",
                "op": "target_construction",
                "params": {
                    "mode": "point_forecast",
                    "method": "direct",
                    "horizon": horizon,
                },
                "inputs": ["src_y"],
            },
            {
                "id": "midas",
                "type": "step",
                "op": "midas",
                "params": {
                    "weighting": weighting,
                    "polynomial_order": polynomial_order,
                    "freq_ratio": freq_ratio,
                    "n_lags_high": n_lags_high,
                    "temporal_rule": "expanding_window_per_origin",
                },
                "inputs": ["src_X", "y_h"],
            },
        ],
        "sinks": {
            "l3_features_v1": {"X_final": "midas", "y_final": "y_h"},
            "l3_metadata_v1": "auto",
        },
    }
    recipe = _base_recipe(
        target=target,
        horizon=horizon,
        panel=panel,
        seed=seed,
        l4=_l4_single_fit("ridge", {"alpha": 1.0}),
    )
    recipe["3_feature_engineering"] = l3
    return recipe


def sliced_inverse_regression(
    *,
    target: str = "y",
    horizon: int = 1,
    n_components: int = 2,
    n_slices: int = 10,
    scaling_method: str = "scaled_pca",
    panel: dict[str, list[Any]] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Fan-Xue-Yao (2017) sliced inverse regression with optional Huang-Zhou
    (2022) predictive scaling (``scaling_method='scaled_pca'`` = sSUFF).

    **Decomposition.** Single ``sliced_inverse_regression`` L3 op (paper-
    faithful: standardise → optional column-wise scaling → sort by y →
    H slices → between-slice covariance → top-K eigenvectors → project)
    feeding a downstream ridge.

    **Status: operational** -- ``sliced_inverse_regression`` op
    operational v0.9.1 dev-stage Phase C (2026-05-08).
    """

    l3 = {
        "nodes": [
            {
                "id": "src_X",
                "type": "source",
                "selector": {
                    "layer_ref": "l2",
                    "sink_name": "l2_clean_panel_v1",
                    "subset": {"role": "predictors"},
                },
            },
            {
                "id": "src_y",
                "type": "source",
                "selector": {
                    "layer_ref": "l2",
                    "sink_name": "l2_clean_panel_v1",
                    "subset": {"role": "target"},
                },
            },
            {
                "id": "y_h",
                "type": "step",
                "op": "target_construction",
                "params": {
                    "mode": "point_forecast",
                    "method": "direct",
                    "horizon": horizon,
                },
                "inputs": ["src_y"],
            },
            {
                "id": "sir",
                "type": "step",
                "op": "sliced_inverse_regression",
                "params": {
                    "n_components": n_components,
                    "n_slices": n_slices,
                    "scaling_method": scaling_method,
                    "temporal_rule": "expanding_window_per_origin",
                },
                "inputs": ["src_X", "y_h"],
            },
        ],
        "sinks": {
            "l3_features_v1": {"X_final": "sir", "y_final": "y_h"},
            "l3_metadata_v1": "auto",
        },
    }
    recipe = _base_recipe(
        target=target,
        horizon=horizon,
        panel=panel,
        seed=seed,
        l4=_l4_single_fit("ridge", {"alpha": 1.0}),
    )
    recipe["3_feature_engineering"] = l3
    return recipe


def garch_volatility(
    *,
    target: str = "y",
    horizon: int = 1,
    family: str = "garch11",
    min_train_size: int | None = None,
    panel: dict[str, list[Any]] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """GARCH-family univariate volatility recipe.

    **Decomposition.** Standard L3 lag + target_construction; L4 fit
    dispatches to ``_GARCHFamily`` which wraps the ``arch`` package.
    Three variants supported via ``family``:
    ``garch11`` (Bollerslev 1986), ``egarch`` (Nelson 1991),
    ``realized_garch_with_rv_exog`` (RV as exogenous regressor in a
    vanilla GARCH(1,1); honest rename of the previous
    ``realized_garch`` family — **NOT** the Hansen-Huang-Shek 2012
    joint MLE).

    Phase C-3 audit-fix (M9): the helper auto-sets ``min_train_size``
    based on the family. The ``arch`` package requires ≥30 observations
    to fit a GARCH spec, so the helper now defaults to a paper-
    conservative ``min_train_size = 60`` for any GARCH family. The
    pre-fix default of 6 (inherited from ``_l4_single_fit``) made the
    recipe unrunnable without manual override. Callers may override
    via the explicit ``min_train_size`` keyword.

    The legacy family name ``realized_garch`` is now FUTURE (reserved
    for the proper Hansen-Huang-Shek 2012 joint MLE); the helper raises
    ``ValueError`` if used and points at ``realized_garch_with_rv_exog``.

    **Status: operational** when ``arch>=6.0`` is installed (optional
    extra ``pip install macroforecast[arch]``); otherwise the L4 fit
    raises ``NotImplementedError`` with an install hint.

    **Panel size requirement.** This recipe requires the training panel to have
    at least 60 observations (per series). Calling with fewer observations will
    raise an error from the ``arch`` package. Override via
    ``min_train_size=K`` for K ≥ 30 (the absolute floor).
    """

    operational_garch_families = {
        "garch11",
        "egarch",
        "realized_garch_with_rv_exog",
    }
    if family == "realized_garch":
        raise ValueError(
            "family='realized_garch' is now reserved (FUTURE) for the "
            "Hansen-Huang-Shek (2012) joint return + measurement-equation "
            "MLE. The previous v0.9.0F runtime fed RV as the GARCH(1,1) "
            "exogenous regressor — that approximation is now exposed "
            "honestly under family='realized_garch_with_rv_exog'."
        )
    if family not in operational_garch_families:
        raise ValueError(
            f"family must be one of {sorted(operational_garch_families)}; "
            f"got {family!r}"
        )
    # Phase C-3 audit-fix (M9): GARCH requires ≥30 obs (arch package);
    # paper-conservative buffer = 60. Helper users can override via
    # explicit ``min_train_size`` kwarg.
    resolved_min_train_size = (
        max(60, int(min_train_size)) if min_train_size is not None else 60
    )
    return _base_recipe(
        target=target,
        horizon=horizon,
        panel=panel,
        seed=seed,
        l4=_l4_single_fit(
            family=family,
            fit_params={"min_train_size": resolved_min_train_size},
            fit_node_id="fit_garch",
        ),
    )


def ets(
    *,
    target: str = "y",
    horizon: int = 1,
    error_trend_seasonal: str = "AAN",
    seasonal_periods: int = 12,
    panel: dict[str, list[Any]] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Hyndman-Koehler-Ord-Snyder (2008) ETS state-space recipe.

    Wraps :class:`statsmodels.tsa.exponential_smoothing.ets.ETSModel`.

    **Status: operational** -- ``ets`` family operational v0.9.1
    dev-stage Phase C (2026-05-08).
    """

    return _base_recipe(
        target=target,
        horizon=horizon,
        panel=panel,
        seed=seed,
        l4=_l4_single_fit(
            family="ets",
            fit_params={
                "error_trend_seasonal": error_trend_seasonal,
                "seasonal_periods": seasonal_periods,
            },
            fit_node_id="fit_ets",
        ),
    )


def theta_method(
    *,
    target: str = "y",
    horizon: int = 1,
    theta: float = 2.0,
    panel: dict[str, list[Any]] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Assimakopoulos-Nikolopoulos (2000) Theta(2) closed-form recipe.

    **Decomposition.** Linear-trend OLS + simple exponential smoothing
    blended 0.5 / 0.5 at theta=2 (the M3-winner setup).

    **Status: operational** -- ``theta_method`` family operational
    v0.9.1 dev-stage Phase C (2026-05-08).
    """

    return _base_recipe(
        target=target,
        horizon=horizon,
        panel=panel,
        seed=seed,
        l4=_l4_single_fit(
            family="theta_method",
            fit_params={"theta": theta},
            fit_node_id="fit_theta",
        ),
    )


def holt_winters(
    *,
    target: str = "y",
    horizon: int = 1,
    seasonal_periods: int = 12,
    trend: str | None = "add",
    seasonal: str | None = "add",
    panel: dict[str, list[Any]] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Hyndman-Athanasopoulos (2018) §7 Holt-Winters recipe.

    Wraps :class:`statsmodels.tsa.holtwinters.ExponentialSmoothing`. The
    runtime auto-disables seasonality when the training series is
    shorter than ``2 * seasonal_periods``.

    **Status: operational** -- ``holt_winters`` family operational
    v0.9.1 dev-stage Phase C (2026-05-08).
    """

    return _base_recipe(
        target=target,
        horizon=horizon,
        panel=panel,
        seed=seed,
        l4=_l4_single_fit(
            family="holt_winters",
            fit_params={
                "seasonal_periods": seasonal_periods,
                "trend": trend,
                "seasonal": seasonal,
            },
            fit_node_id="fit_hw",
        ),
    )


def bai_ng_corrected_factor_ar(
    *,
    target: str = "y",
    horizon: int = 1,
    n_factors: int = 4,
    n_lag: int = 1,
    quantile_levels: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95),
    panel: dict[str, list[Any]] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Bai-Ng (2006) generated-regressor PI correction on factor_augmented_ar.

    **Decomposition.** Standard L3 lag + target_construction; L4 fit on
    ``factor_augmented_ar`` family with ``forecast_object='quantile'``;
    predict node sets ``pi_correction='bai_ng'`` so the runtime applies
    the Theorem 3 / Corollary 1 correction (V₂/N + V₁/T + σ²_ε) to the
    quantile-band sigma.

    **Status: operational** -- ``predict.pi_correction`` axis operational
    v0.9.1 dev-stage Phase C (2026-05-08).
    """

    fit_params = {
        "family": "factor_augmented_ar",
        "n_factors": n_factors,
        "n_lag": n_lag,
        "forecast_object": "quantile",
        "quantile_levels": list(quantile_levels),
        "forecast_strategy": "direct",
        "training_start_rule": "expanding",
        "refit_policy": "every_origin",
        "search_algorithm": "none",
        "min_train_size": 6,
    }
    l4 = {
        "nodes": [
            {
                "id": "src_X",
                "type": "source",
                "selector": {
                    "layer_ref": "l3",
                    "sink_name": "l3_features_v1",
                    "subset": {"component": "X_final"},
                },
            },
            {
                "id": "src_y",
                "type": "source",
                "selector": {
                    "layer_ref": "l3",
                    "sink_name": "l3_features_v1",
                    "subset": {"component": "y_final"},
                },
            },
            {
                "id": "fit",
                "type": "step",
                "op": "fit_model",
                "params": fit_params,
                "inputs": ["src_X", "src_y"],
            },
            {
                "id": "predict",
                "type": "step",
                "op": "predict",
                "params": {"pi_correction": "bai_ng"},
                "inputs": ["fit", "src_X"],
            },
        ],
        "forecast_object": "quantile",
        "sinks": {
            "l4_forecasts_v1": "predict",
            "l4_model_artifacts_v1": "fit",
            "l4_training_metadata_v1": "auto",
        },
    }
    return _base_recipe(target=target, horizon=horizon, panel=panel, seed=seed, l4=l4)


__all__ = [
    # Operational (runs end-to-end on current main):
    "perfectly_random_forest",  # #3 PRF baseline
    "scaled_pca",  # #1
    "macroeconomic_random_forest",  # #2
    "ols_attention_demo",  # #7 conceptual
    "sparse_macro_factors",  # #11
    "sparse_macro_factors_risk_premia",  # #11 step-3 sibling (phase-f15)
    "macroeconomic_data_transformations",  # #12 MARX
    "ml_useful_macro",  # #13
    "ml_useful_macro_horse_race",  # #16 paper-16 grid helper
    "ml_useful_macro_b_grid",  # #16 §3.2 B₁/B₂/B₃ rotation grid
    "macroeconomic_data_transformations_horse_race",  # #15 16-cell helper
    "arctic_var",  # #16
    # Phase C top-6 net-new methods:
    "u_midas",  # M1 Foroni-Marcellino-Schumacher 2015
    "factor_midas_nowcast",  # F-02 Marcellino-Schumacher 2010 OBES 72(4)
    "midas_almon",  # M2 Ghysels-Sinko-Valkanov 2007
    "sliced_inverse_regression",  # M3 Fan-Xue-Yao 2017 + Huang-Zhou 2022 sSUFF
    "garch_volatility",  # M9 Bollerslev/Nelson/Hansen
    "ets",  # M16 Hyndman-Koehler-Ord-Snyder 2008
    "theta_method",  # M16 Assimakopoulos-Nikolopoulos 2000
    "holt_winters",  # M16 Hyndman-Athanasopoulos 2018
    "bai_ng_corrected_factor_ar",  # M12 Bai-Ng 2006 PI correction
    # Pre-promotion (depend on future-status atomic primitives):
    "booging",  # #3b bagging.strategy
    "marsquake",  # #3c mars family
    "adaptive_ma",  # #4 adaptive_ma_rf
    "two_step_ridge",  # #5 ridge.prior=random_walk
    "hemisphere_neural_network",  # #6 mlp.architecture
    "anatomy_oos",  # #8 oshapley_vi/pbsv
    "dual_interpretation",  # #9 dual_decomposition
    "maximally_forward_looking",  # #10 asymmetric_trim + ridge.constraint
    "slow_growing_tree",  # #14 decision_tree.split_shrinkage
    "assemblage_regression",  # repackaged ridge.constraint
]
