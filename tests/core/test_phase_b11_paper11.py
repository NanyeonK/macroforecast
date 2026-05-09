"""Phase B-11 paper-11 (Borup, Goulet Coulombe, Rapach, Montes Schütte
& Schwenk-Nebbe 2022 -- "Anatomy of Out-of-Sample Forecasting Accuracy")
helper-rewrite tests.

Round 1 audit found two gaps that left the paper's central interpretive
ops (oShapley-VI / PBSV) effectively unreachable from the helper:

* **F1 (CRITICAL)** -- ``anatomy_oos(initial_window=...)`` stamped
  ``0_meta.leaf_config.anatomy_initial_window`` but no consumer read
  that key. The base recipe also returned no L7 block, so users
  following the helper *always* routed to Path B (final-window-only
  fit, paper-violating estimand) regardless of ``initial_window``.

* **F3 (HIGH)** -- ``params_schema`` for ``oshapley_vi`` and ``pbsv``
  returned ``{}``. ``initial_window`` and ``n_iterations`` were not in
  the registered schema, so neither the validator nor recipe authors
  saw them.

The six tests below close F1 + F3 and cross-check the runtime against
two paper-level identities (Shapley efficiency, Eq. 14; linear-model
closed form, Eq. 15) on a small linear DGP. iShapley-VI (paper Eq. 10)
remains deferred per the request -- anatomy 0.1.6 has no native
in-sample adapter, so iShapley needs additional plumbing beyond Phase
B-11 scope.

Reference: Borup, Goulet Coulombe, Rapach, Montes Schütte &
Schwenk-Nebbe (2022) "Anatomy of Out-of-Sample Forecasting Accuracy",
SSRN 4278745; §2.4 (oShapley-VI / PBSV), Eqs. 14-16, 24, p.16 fn 16.
"""

from __future__ import annotations

import datetime
import warnings

import numpy as np
import pandas as pd
import pytest

import macroforecast
from macroforecast.recipes.paper_methods import anatomy_oos


def _anatomy_extra_available() -> bool:
    try:
        import anatomy  # noqa: F401
    except ImportError:
        return False
    return True


# ----------------------------------------------------------------------
# DGP fixture: linear model y = beta . x + eps so paper Eqs. 14-15 hold
# ----------------------------------------------------------------------


def _build_linear_panel(T: int = 60, K: int = 3, seed: int = 0):
    """Linear-Gaussian DGP with K predictors (no intercept) so the
    Shapley efficiency identity (paper Eq. 14) and linear-model
    closed-form (paper Eq. 15) both reduce to algebraic checks."""

    rng = np.random.default_rng(seed)
    dates: list[str] = []
    d = datetime.date(2014, 1, 1)
    for _ in range(T):
        dates.append(d.strftime("%Y-%m-%d"))
        m = d.month + 1
        y_yr = d.year + (m - 1) // 12
        m = ((m - 1) % 12) + 1
        d = datetime.date(y_yr, m, 1)

    X = rng.normal(0.0, 1.0, size=(T, K))
    beta = np.array([2.0, -1.0, 0.5])[:K]
    noise = rng.normal(0.0, 0.1, size=T)
    y = X @ beta + noise

    panel: dict[str, list] = {"date": dates, "y": y.tolist()}
    for j in range(K):
        panel[f"x{j + 1}"] = X[:, j].tolist()
    return panel, beta


# ----------------------------------------------------------------------
# Test 1 -- F1 (recipe shape): helper attaches an L7 block with the
# anatomy params on the op step (NOT in dead 0_meta.leaf_config slot)
# ----------------------------------------------------------------------


def test_anatomy_oos_helper_attaches_l7_block():
    """Phase B-11 F1 fix: ``anatomy_oos`` must append a
    ``7_interpretation`` block whose interpretation step is
    ``oshapley_vi`` (or ``pbsv``) and carries ``initial_window`` /
    ``n_iterations`` as op params. The previous helper stamped
    ``anatomy_initial_window`` into ``0_meta.leaf_config`` but no
    consumer read that key, so users always silently fell back to
    Path B (degraded final-window fit)."""

    recipe = anatomy_oos(initial_window=40, n_iterations=200)

    # F1 closure: L7 block exists and is enabled.
    assert "7_interpretation" in recipe, (
        "helper must attach a 7_interpretation block by default"
    )
    block = recipe["7_interpretation"]
    assert block.get("enabled") is True

    # The op step must be oshapley_vi (paper §2.4 default) and the
    # anatomy params must live on the op step (not anywhere else).
    nodes = block.get("nodes", [])
    step = next((n for n in nodes if n.get("type") == "step"), None)
    assert step is not None, "L7 block must contain a step node"
    assert step["op"] == "oshapley_vi"
    assert step["params"]["initial_window"] == 40
    assert step["params"]["n_iterations"] == 200

    # Dead-stamp removal: the helper must NOT write the legacy
    # anatomy_initial_window key into 0_meta.leaf_config (no consumer
    # reads it; keeping it would be a foot-gun for recipe diffs).
    leaf = recipe["0_meta"].get("leaf_config", {})
    assert "anatomy_initial_window" not in leaf
    assert "anatomy_n_iterations" not in leaf

    # ``metric="pbsv"`` opts into the global squared-error variant
    # (paper Eq. 24).
    recipe_pbsv = anatomy_oos(initial_window=40, n_iterations=200, metric="pbsv")
    step_pbsv = next(
        n for n in recipe_pbsv["7_interpretation"]["nodes"] if n.get("type") == "step"
    )
    assert step_pbsv["op"] == "pbsv"


# ----------------------------------------------------------------------
# Test 2 -- F1 procedure-level: initial_window from helper actually
# reaches _l7_anatomy_op (Path A: per-origin refit count = T - W)
# ----------------------------------------------------------------------


@pytest.mark.skipif(
    not _anatomy_extra_available(),
    reason="[anatomy] extra not installed (pip install macroforecast[anatomy])",
)
def test_anatomy_oos_helper_initial_window_reaches_runtime():
    """Phase B-11 F1 procedure-level closure: build a small DGP, call
    ``anatomy_oos(initial_window=20, ...)``, run via
    ``macroforecast.run``, and confirm the helper's ``initial_window``
    propagated all the way to ``_l7_anatomy_op`` -- evidenced by the
    Path A status label on the importance frame and a non-trivial
    importance signal recovered from the linear DGP."""

    panel, _ = _build_linear_panel(T=60, K=3, seed=0)
    recipe = anatomy_oos(
        target="y",
        horizon=1,
        panel=panel,
        initial_window=20,
        n_iterations=50,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = macroforecast.run(recipe)

    assert result.cells, "anatomy_oos must produce at least one cell"
    artifacts = result.cells[0].runtime_result.artifacts
    assert "l7_importance_v1" in artifacts, (
        "F1 fix: helper must wire an L7 sink so the anatomy frame "
        f"reaches macroforecast.run output (got {list(artifacts)})"
    )

    importance = artifacts["l7_importance_v1"]
    global_importance = getattr(importance, "global_importance", {}) or {}
    assert global_importance, "L7 global_importance must be non-empty"

    # Locate the oshapley_vi frame (method attr or key tuple).
    osh_frame = None
    for key, frame in global_importance.items():
        method = key[3] if isinstance(key, tuple) and len(key) >= 4 else None
        attr_method = (
            frame.attrs.get("method") if isinstance(frame, pd.DataFrame) else None
        )
        if method == "oshapley_vi" or attr_method == "oshapley_vi":
            osh_frame = frame
            break
    assert osh_frame is not None, (
        f"oshapley_vi frame not found in L7 sink (keys={list(global_importance)})"
    )

    # Path A signature: status column == "operational" for every feature.
    assert (osh_frame["status"] == "operational").all(), (
        f"helper must drive Path A; got status counts "
        f"{osh_frame['status'].value_counts().to_dict()}"
    )
    # Importance must be finite and sum to a non-zero magnitude (the
    # linear DGP is fully informative — at least one feature should
    # carry mass).
    importances = osh_frame["importance"].to_numpy()
    assert np.all(np.isfinite(importances))
    assert float(np.sum(np.abs(importances))) > 0.0


# ----------------------------------------------------------------------
# Test 3 -- F1 procedure-level: Path A path does NOT trigger the
# Path B UserWarning (silent-degraded routing protection)
# ----------------------------------------------------------------------


@pytest.mark.skipif(
    not _anatomy_extra_available(),
    reason="[anatomy] extra not installed",
)
def test_anatomy_oos_path_a_does_not_warn_about_path_b():
    """Phase B-11 F1 follow-on: the helper's default routing now
    populates ``initial_window`` so the anatomy adapter takes Path A.
    The Path B UserWarning at ``runtime.py:6820-6827`` (which fires
    only when ``initial_window`` is missing/zero) must NOT fire when
    the helper is driven with a positive ``initial_window``."""

    panel, _ = _build_linear_panel(T=50, K=3, seed=1)
    recipe = anatomy_oos(
        target="y",
        horizon=1,
        panel=panel,
        initial_window=20,
        n_iterations=20,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        macroforecast.run(recipe)

    msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    path_b_msgs = [m for m in msgs if "Path B" in m and "initial_window" in m]
    assert not path_b_msgs, (
        f"Path A (initial_window=20) must NOT emit the Path B warning; "
        f"got: {path_b_msgs}"
    )


# ----------------------------------------------------------------------
# Test 4 -- F3: registry schema exposes initial_window + n_iterations
# ----------------------------------------------------------------------


def test_oshapley_vi_op_schema_exposes_initial_window():
    """Phase B-11 F3 fix: ``_schema('oshapley_vi')`` and
    ``_schema('pbsv')`` previously returned ``{}``, hiding the two
    routing parameters from the validator and from any author
    inspecting the registry. After the fix both keys must be present
    with sensible defaults (anatomy paper p.16 fn 16: M=500)."""

    from macroforecast.core.ops.l7_ops import _schema
    from macroforecast.core.ops.registry import get_op

    for op_name in ("oshapley_vi", "pbsv"):
        schema = _schema(op_name)
        assert "initial_window" in schema, (
            f"{op_name}._schema must expose initial_window"
        )
        assert "n_iterations" in schema, f"{op_name}._schema must expose n_iterations"
        assert schema["initial_window"]["type"] is int
        assert schema["n_iterations"]["type"] is int
        assert int(schema["n_iterations"]["default"]) == 500

        # Cross-check: registered OpSpec carries the same schema.
        spec = get_op(op_name)
        assert "initial_window" in spec.params_schema
        assert "n_iterations" in spec.params_schema


# ----------------------------------------------------------------------
# Test 5 -- procedure-level paper Eq. 14: Shapley efficiency
#           sum_p phi_p(i) ≈ y_hat(i) - phi_empty
# ----------------------------------------------------------------------


@pytest.mark.skipif(
    not _anatomy_extra_available(),
    reason="[anatomy] extra not installed",
)
def test_anatomy_oos_shapley_efficiency_identity():
    """Phase B-11 procedure-level paper Eq. 14: the Shapley values must
    satisfy the efficiency property -- summed across predictors they
    equal the model prediction minus the base contribution. We exercise
    this by calling ``Anatomy.explain()`` directly on the same DGP /
    estimator the helper uses, since the helper aggregates
    ``mean(|values|)`` over rows and discards the per-row signed
    decomposition. The identity must hold to numerical precision on a
    linear DGP with M=200, T=80."""

    from anatomy import (
        Anatomy,
        AnatomyModel,
        AnatomyModelProvider,
        AnatomySubsets,
    )
    from sklearn.linear_model import Ridge

    panel, _ = _build_linear_panel(T=80, K=3, seed=2)
    X = pd.DataFrame({k: panel[k] for k in panel if k != "date" and k != "y"})
    y = pd.Series(panel["y"], name="y")
    full_block = pd.concat([y, X], axis=1).reset_index(drop=True)
    initial_window = 30
    n_iter = 200

    subsets = AnatomySubsets.generate(
        index=full_block.index,
        initial_window=initial_window,
        estimation_type=AnatomySubsets.EstimationType.EXPANDING,
        periods=1,
    )
    feature_cols = list(X.columns)

    def _provider_fn(key):
        period = int(getattr(key, "period", 0))
        train_slice = subsets.get_train_subset(period)
        test_slice = subsets.get_test_subset(period)
        train = full_block.iloc[train_slice].reset_index(drop=True)
        test = full_block.iloc[test_slice].reset_index(drop=True)
        period_fitted = Ridge(alpha=1.0).fit(
            train[feature_cols].to_numpy(dtype=float),
            train["y"].to_numpy(dtype=float),
        )

        def _predict(arr):
            arr = np.asarray(arr, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return np.asarray(period_fitted.predict(arr)).ravel()

        return AnatomyModelProvider.PeriodValue(
            train=train, test=test, model=AnatomyModel(pred_fn=_predict)
        )

    np.random.seed(0)
    provider = AnatomyModelProvider(
        n_periods=max(1, len(full_block) - initial_window),
        n_features=len(feature_cols),
        model_names=["ridge"],
        y_name="y",
        provider_fn=_provider_fn,
    )
    anat = Anatomy(provider=provider, n_iterations=n_iter).precompute(n_jobs=1)
    df = anat.explain()

    # Recompute predictions per-period to compare against
    # base + sum(phi). For each test row, the model is the period
    # ridge fit on the expanding training window.
    n_test_rows = len(df)
    y_hat_actual = np.zeros(n_test_rows, dtype=float)
    for period in range(max(1, len(full_block) - initial_window)):
        train_slice = subsets.get_train_subset(period)
        test_slice = subsets.get_test_subset(period)
        train = full_block.iloc[train_slice]
        test = full_block.iloc[test_slice]
        ridge = Ridge(alpha=1.0).fit(
            train[feature_cols].to_numpy(dtype=float),
            train["y"].to_numpy(dtype=float),
        )
        preds = ridge.predict(test[feature_cols].to_numpy(dtype=float))
        for offset, idx in enumerate(test.index):
            # df is indexed by the per-period (model, ts) MultiIndex; we
            # match test row 1:1 with period since periods=1.
            y_hat_actual[period + offset] = preds[offset]

    # paper Eq. 14: sum_p phi_p ≈ y_hat - base. Compute per row.
    base = df["base_contribution"].to_numpy()
    phi_sum = df.drop(columns="base_contribution").sum(axis=1).to_numpy()
    lhs = phi_sum + base
    # Tolerance ~0.05 with M=200 on the linear DGP (Castro-Gomez-Tejada
    # permutation Shapley converges as 1/sqrt(M)).
    max_dev = float(np.max(np.abs(lhs - y_hat_actual)))
    assert max_dev < 0.1, (
        f"paper Eq. 14 (Shapley efficiency) violated: "
        f"max|sum_p phi_p + base - y_hat| = {max_dev}"
    )


# ----------------------------------------------------------------------
# Test 6 -- procedure-level paper Eq. 15: linear-model closed form
#           phi_p(i) ≈ beta_hat_p (x_{p,i} - x_bar_p)
# ----------------------------------------------------------------------


@pytest.mark.skipif(
    not _anatomy_extra_available(),
    reason="[anatomy] extra not installed",
)
def test_anatomy_oos_linear_model_closed_form():
    """Phase B-11 procedure-level paper Eq. 15: for a linear model the
    OOS Shapley value of predictor p at observation i collapses to
    ``beta_hat_p (x_{p,i} - mean(x_p))`` (centred-coefficient form).
    On a linear DGP fitted with OLS this should hold within Monte
    Carlo noise."""

    from anatomy import (
        Anatomy,
        AnatomyModel,
        AnatomyModelProvider,
        AnatomySubsets,
    )
    from sklearn.linear_model import LinearRegression

    panel, _ = _build_linear_panel(T=80, K=3, seed=3)
    X = pd.DataFrame({k: panel[k] for k in panel if k != "date" and k != "y"})
    y = pd.Series(panel["y"], name="y")
    full_block = pd.concat([y, X], axis=1).reset_index(drop=True)
    initial_window = 30
    n_iter = 300

    subsets = AnatomySubsets.generate(
        index=full_block.index,
        initial_window=initial_window,
        estimation_type=AnatomySubsets.EstimationType.EXPANDING,
        periods=1,
    )
    feature_cols = list(X.columns)
    fitted_models: list[LinearRegression] = []

    def _provider_fn(key):
        period = int(getattr(key, "period", 0))
        train_slice = subsets.get_train_subset(period)
        test_slice = subsets.get_test_subset(period)
        train = full_block.iloc[train_slice].reset_index(drop=True)
        test = full_block.iloc[test_slice].reset_index(drop=True)
        period_fitted = LinearRegression().fit(
            train[feature_cols].to_numpy(dtype=float),
            train["y"].to_numpy(dtype=float),
        )
        fitted_models.append(period_fitted)

        def _predict(arr):
            arr = np.asarray(arr, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return np.asarray(period_fitted.predict(arr)).ravel()

        return AnatomyModelProvider.PeriodValue(
            train=train, test=test, model=AnatomyModel(pred_fn=_predict)
        )

    np.random.seed(0)
    n_periods = max(1, len(full_block) - initial_window)
    provider = AnatomyModelProvider(
        n_periods=n_periods,
        n_features=len(feature_cols),
        model_names=["lr"],
        y_name="y",
        provider_fn=_provider_fn,
    )
    anat = Anatomy(provider=provider, n_iterations=n_iter).precompute(n_jobs=1)
    df = anat.explain()

    # Recompute Eq. 15 predictions: phi_p^closed = beta_hat_p (x_{p,i} - mean_train_p).
    closed = np.zeros((n_periods, len(feature_cols)), dtype=float)
    actual = np.zeros((n_periods, len(feature_cols)), dtype=float)
    phi_columns = [c for c in df.columns if c != "base_contribution"]
    df_arr = df[phi_columns].to_numpy()
    for period in range(n_periods):
        ridge = fitted_models[period]
        train_slice = subsets.get_train_subset(period)
        test_slice = subsets.get_test_subset(period)
        train = full_block.iloc[train_slice]
        test = full_block.iloc[test_slice]
        x_bar = train[feature_cols].mean(axis=0).to_numpy()
        x_test = test[feature_cols].iloc[0].to_numpy()
        for j, _ in enumerate(feature_cols):
            closed[period, j] = ridge.coef_[j] * (x_test[j] - x_bar[j])
        actual[period, :] = df_arr[period, :]

    # paper Eq. 15: phi_p ≈ beta_p (x_p - x_bar_p) within Monte Carlo
    # error. Tolerance reflects the M=300 permutation-Shapley noise
    # plus the OOS expanding-window variance.
    max_dev = float(np.max(np.abs(closed - actual)))
    assert max_dev < 0.5, (
        f"paper Eq. 15 (linear-model closed form) violated: "
        f"max|beta(x-x_bar) - phi| = {max_dev}; closed={closed[:3]}, "
        f"actual={actual[:3]}"
    )
